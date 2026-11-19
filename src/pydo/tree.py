"""DecisionTree: distil the account-level optimum into a policy a human can read, audit and deploy.

How it works
------------
1. Fit a CART on RAW portfolio fields to the decisions of a solved ``constrained`` scenario (accounts weighted by what is at
   stake in their decision, so the tree spends its splits where the money is).
2. Keep only the PARTITION the CART found: every leaf is a group of accounts.
3. Re-optimise ONE shared action per leaf against the SAME constraints: a small mixed-integer program whose variables are the
   leaves, not the accounts. The result is a handful of rules that is still feasible (or an explicit "infeasible: grow the
   tree" error, never a silently violated rule).

The fitted tree can be saved as JSON, re-loaded without scikit-learn and applied to any frame that has the same columns.
"""
import copy
import json
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .errors import InfeasibleModelError, ProjectError, SolverError


@dataclass
class LeafRule:
    leaf: int
    description: str
    conditions: dict
    action: object
    n_apps: int

    def __str__(self):
        return f"[leaf {self.leaf:>2}] IF {self.description}  THEN action = {self.action}   ({self.n_apps:,} accounts)"


def _fmt(x):
    return f"{x:.6g}"


class DecisionTree:
    """CART distillation of a constrained scenario, with one re-optimised action per leaf.

    Parameters
    ----------
    max_depth      : depth of the CART.
    max_splits     : maximum number of splits of the whole tree (so at most ``max_splits + 1`` leaves); ``None`` = no cap.
    min_weight     : minimum number of accounts in a leaf.
    min_gini_gain  : minimum impurity decrease for a split (scikit-learn's ``min_impurity_decrease``).
    sample_weight  : ``"stake"`` (default) weights every account by the expected profit that its decision is worth
                     (|profit if approved - profit if rejected|); ``None`` weights all accounts equally.
    """

    def __init__(self, max_depth=4, max_splits=None, min_weight=100, min_gini_gain=0.0001, sample_weight="stake",
                 random_state=42):
        self.max_depth, self.max_splits, self.min_weight = max_depth, max_splits, min_weight
        self.min_gini_gain, self.sample_weight, self.random_state = min_gini_gain, sample_weight, random_state
        self.policy_ = None
        self._fitted = False

    # ------------------------------------------------------------------ encoding
    def _make_encoder(self, frame, variables):
        spec = []
        for v in variables:
            col = frame[v]
            if pd.api.types.is_numeric_dtype(col):
                spec.append({"feature": v, "kind": "numeric", "integer": bool(pd.api.types.is_integer_dtype(col))})
            else:
                spec.append({"feature": v, "kind": "categorical",
                             "categories": sorted(col.dropna().astype(str).unique().tolist())})
        return spec

    def _encode(self, frame):
        cols, names = [], []
        for sp in self._spec:
            col = frame[sp["feature"]]
            if sp["kind"] == "numeric":
                cols.append(col.to_numpy(float)); names.append((sp["feature"], None))
            else:
                s = col.astype(str)
                for cat in sp["categories"]:
                    cols.append((s == cat).to_numpy(float)); names.append((sp["feature"], cat))
        return np.column_stack(cols), names

    # ------------------------------------------------------------------ fit
    def fit(self, scenario, variables, solver="highs", time_limit=300, gap=0.05, constrained_growth="posthoc",
            n_jobs=None, test_size=0.3):
        """Grow the CART on the solved ``scenario`` and re-optimise one action per leaf against its constraints."""
        from sklearn.tree import DecisionTreeClassifier
        if constrained_growth != "posthoc":
            raise NotImplementedError("only constrained_growth='posthoc' is implemented: grow first, then re-optimise the leaves")
        if solver != "highs":
            raise SolverError(f"solver '{solver}' is not available; use 'highs'")
        if scenario.flow != "constrained" or scenario.selected is None:
            raise ProjectError("fit() needs a solved CONSTRAINED scenario (call optimize() first)")
        proj, portfolio = scenario.project, scenario.project.portfolio
        if len(proj.variables) != 1 or len(proj.variables.treatments()) != 2:
            raise NotImplementedError("distillation supports one decision variable with two treatments (approve / reject)")
        frame, key = portfolio.data, portfolio.key_column
        variables = list(variables)
        absent = [v for v in variables if v not in frame.columns]
        calc = [v for v in variables if v.startswith("v_")]
        if absent or calc:
            raise ProjectError(f"predictors must be RAW portfolio columns; not usable: {absent + calc}")

        self.source_name, self.objective = scenario.name, proj.objective
        self.var = proj.variables.names[0]
        self.action_values = [t[self.var] for t in proj.variables.treatments()]         # [reference (reject), approve]
        self.attempted_features = variables
        self._spec = self._make_encoder(frame, variables)
        X, self._enc_names = self._encode(frame)
        n = len(frame)
        y = (scenario.selected[self.var].to_numpy() == self.action_values[1]).astype(int)

        # profit of the two actions for every account (from the pairs the scenario evaluated), and the stake of the decision
        obj = scenario.pairs[self.objective].to_numpy(float).reshape(n, 2)
        stake = np.abs(obj[:, 1] - obj[:, 0])
        w = None
        if self.sample_weight == "stake":
            w = (stake + 1e-9) / (stake.mean() + 1e-9)

        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(n)
        n_test = int(round(test_size * n))
        self._test_idx, self._train_idx = np.sort(order[:n_test]), np.sort(order[n_test:])

        no_split = self.max_splits is not None and self.max_splits < 1                 # a single-leaf tree: no split allowed
        sk = DecisionTreeClassifier(max_depth=self.max_depth,
                                    min_samples_leaf=len(self._train_idx) if no_split else self.min_weight,
                                    min_impurity_decrease=self.min_gini_gain,
                                    max_leaf_nodes=None if (self.max_splits is None or no_split) else self.max_splits + 1,
                                    random_state=self.random_state)
        sk.fit(X[self._train_idx], y[self._train_idx], sample_weight=None if w is None else w[self._train_idx])
        self._sk = sk

        # --- the partition: leaf of every account, then the rules (conditions per leaf)
        raw_leaf = sk.apply(X)
        leaf_nodes = np.unique(raw_leaf)
        remap = {node: i for i, node in enumerate(leaf_nodes)}
        leaf_of = np.array([remap[v] for v in raw_leaf])
        self.n_leaves = len(leaf_nodes)
        majority = {remap[node]: int(sk.tree_.value[node][0].argmax()) for node in leaf_nodes}
        self._conditions = self._extract_conditions(remap)

        # --- re-optimise one action per leaf, same constraints as the account-level problem
        scenario._compile()
        model = scenario.model
        actions, solve_info = self._solve_leaves(model, leaf_of, time_limit, gap)

        # --- bookkeeping
        self.leaf_of_ = leaf_of
        self.actions_ = {i: int(a) for i, a in enumerate(actions)}
        self.majority_ = majority
        self._build_rules(frame, leaf_of, y, obj)
        self.info = solve_info
        self.info.update(leaves=self.n_leaves, selected_features=self.selected_features)
        self._fitted = True
        self._scenario = scenario
        self._source_decision = y
        self._obj = obj
        self._weights = w
        self.policy_ = scenario.apply_policy(self.predict(frame), name="Distilled approval policy")
        self._majority_policy = scenario.apply_policy(
            np.array([self.action_values[majority[i]] for i in leaf_of]), name="_tree_majority")
        return self

    # ------------------------------------------------------------------ rules
    def _extract_conditions(self, remap):
        tr = self._sk.tree_
        conds = {}

        def new():
            return {sp["feature"]: ({"lo": -math.inf, "hi": math.inf} if sp["kind"] == "numeric"
                                    else {"in": None, "not_in": []}) for sp in self._spec}

        def walk(node, state):
            if tr.children_left[node] == -1:
                conds[remap[node]] = state
                return
            feat, cat = self._enc_names[tr.feature[node]]
            t = float(tr.threshold[node])
            left, right = copy.deepcopy(state), copy.deepcopy(state)
            if cat is None:
                left[feat]["hi"] = min(left[feat]["hi"], t)
                right[feat]["lo"] = max(right[feat]["lo"], t)
            else:
                left[feat]["not_in"].append(cat)                       # one-hot column <= 0.5  ->  not this category
                right[feat]["in"] = [cat]
            walk(tr.children_left[node], left)
            walk(tr.children_right[node], right)

        walk(0, new())
        for cond in conds.values():
            for sp in self._spec:
                if sp["kind"] != "categorical":
                    continue
                d = cond[sp["feature"]]
                if d["in"]:
                    d["not_in"] = []
                else:
                    remaining = [c for c in sp["categories"] if c not in set(d["not_in"])]
                    if len(remaining) == 1:
                        d["in"], d["not_in"] = remaining, []
        return conds

    def _describe(self, cond):
        parts = []
        for sp in self._spec:
            d = cond[sp["feature"]]
            f = sp["feature"]
            if sp["kind"] == "numeric":
                lo, hi = d["lo"], d["hi"]
                if sp["integer"]:
                    lo_i = None if math.isinf(lo) else math.floor(lo) + 1
                    hi_i = None if math.isinf(hi) else math.floor(hi)
                    if lo_i is not None and hi_i is not None:
                        parts.append(f"{f} = {lo_i}" if lo_i == hi_i else f"{lo_i} <= {f} <= {hi_i}")
                    elif lo_i is not None:
                        parts.append(f"{f} >= {lo_i}")
                    elif hi_i is not None:
                        parts.append(f"{f} <= {hi_i}")
                else:
                    if not math.isinf(lo) and not math.isinf(hi):
                        parts.append(f"{_fmt(lo)} < {f} <= {_fmt(hi)}")
                    elif not math.isinf(lo):
                        parts.append(f"{f} > {_fmt(lo)}")
                    elif not math.isinf(hi):
                        parts.append(f"{f} <= {_fmt(hi)}")
            else:
                if d["in"]:
                    parts.append(f"{f} = '{d['in'][0]}'")
                elif d["not_in"]:
                    ex = sorted(set(d["not_in"]))
                    parts.append(f"{f} != '{ex[0]}'" if len(ex) == 1 else f"{f} not in ({', '.join(repr(c) for c in ex)})")
        return " AND ".join(parts) if parts else "ALL ACCOUNTS"

    def _build_rules(self, frame, leaf_of, y, obj):
        names = {0: "reject", 1: "approve"}
        self.rules_ = []
        rows = []
        for i in range(self.n_leaves):
            members = leaf_of == i
            act = self.actions_[i]
            desc = self._describe(self._conditions[i])
            self.rules_.append(LeafRule(i, desc, self._conditions[i], self.action_values[act], int(members.sum())))
            src_profit = np.where(y[members] == 1, obj[members, 1], obj[members, 0]).sum()
            leaf_profit = obj[members, act].sum()
            rows.append({"leaf": i, "rule": desc, "action": names[act], "majority_of_source": names[self.majority_[i]],
                         "accounts": int(members.sum()), "source_approval_share": float(y[members].mean()),
                         "profit_source_bn": src_profit / 1e9, "profit_leaf_bn": leaf_profit / 1e9,
                         "loss_vs_source_bn": (src_profit - leaf_profit) / 1e9})
        self.leaves_ = pd.DataFrame(rows).set_index("leaf")
        used = {f for c in self._conditions.values() for f, d in c.items()
                if ("lo" in d and (not math.isinf(d["lo"]) or not math.isinf(d["hi"]))) or ("in" in d and (d["in"] or d["not_in"]))}
        self.selected_features = [f for f in self.attempted_features if f in used]

    # ------------------------------------------------------------------ leaf-level optimisation
    def _solve_leaves(self, model, leaf_of, time_limit, gap):
        if model.T != 2:
            raise NotImplementedError("distillation supports two treatments")
        n, L = len(leaf_of), self.n_leaves
        M = sparse.csr_array((np.ones(n), (np.arange(n), leaf_of)), shape=(n, L))
        A = (model.A @ M) if model.A.shape[0] else None
        sign = -1.0 if model.maximize else 1.0
        c = sign * (model.obj_coef @ M)
        lb = ((M.T @ model.lbv) > 0).astype(float)                     # a leaf holding a forced-approve account is approve
        ub = ((M.T @ (1.0 - model.ubv)) == 0).astype(float)            # ... a leaf holding a never-approve account is reject
        if (lb > ub).any():
            raise InfeasibleModelError("a leaf mixes accounts that must be approved with accounts that cannot be")
        cons = [LinearConstraint(A, model.lo, model.hi)] if A is not None else None
        t0 = time.time()
        res = milp(c=c, constraints=cons, integrality=np.ones(L), bounds=Bounds(lb, ub),
                   options=dict(time_limit=float(time_limit), mip_rel_gap=float(gap)))
        if res.x is None:
            if res.status == 2:
                raise InfeasibleModelError(
                    f"no assignment of whole leaves satisfies every rule ({L} leaves). The partition is too coarse for "
                    "the constraints: grow a deeper tree (max_depth / max_splits) or offer more predictors")
            raise SolverError(f"solver stopped without a solution (status {res.status}): {res.message}")
        y = np.round(res.x).astype(int)
        total = model.obj_const + float(model.obj_coef @ (M @ y))
        info = dict(status="optimal" if res.status == 0 else "time_limit (feasible, not proven optimal)",
                    objective_total=total, gap_achieved=getattr(res, "mip_gap", None), time_seconds=time.time() - t0)
        return y, info

    # ------------------------------------------------------------------ prediction
    def _leaf_index(self, frame):
        n = len(frame)
        leaf = np.full(n, -1)
        for i, cond in self._conditions.items():
            mask = np.ones(n, dtype=bool)
            for sp in self._spec:
                d = cond[sp["feature"]]
                col = frame[sp["feature"]]
                if sp["kind"] == "numeric":
                    x = col.to_numpy(float)
                    mask &= (x > d["lo"]) & (x <= d["hi"])
                else:
                    s = col.astype(str)
                    if d["in"]:
                        mask &= (s == d["in"][0]).to_numpy()
                    if d["not_in"]:
                        mask &= ~s.isin(d["not_in"]).to_numpy()
            leaf[mask & (leaf == -1)] = i
        if (leaf == -1).any():
            raise ProjectError(f"{int((leaf == -1).sum())} accounts fall in no leaf (a category or value the tree never saw)")
        return leaf

    def predict(self, frame):
        """The decision (a value of the decision variable) for every row of ``frame``, from the rules only."""
        self._check_fitted()
        act = np.array([self.action_values[self.actions_[i]] for i in range(self.n_leaves)])
        return act[self._leaf_index(frame)]

    def _check_fitted(self):
        if not self._fitted:
            raise ProjectError("fit() the tree first")

    # ------------------------------------------------------------------ reports
    def performance(self):
        """Agreement with the account-level policy on the accounts the CART was grown on (train) and on accounts it never saw
        (test), and the expected profit the tree policy keeps per account against the account-level optimum.

        Only May 2015 carries a PD and a decision, so there is no out-of-time sample: train / test is a random split of accounts."""
        self._check_fitted()
        rows = {}
        act = np.array([self.actions_[i] for i in range(self.n_leaves)])[self.leaf_of_]
        for name, idx in (("train", self._train_idx), ("test", self._test_idx)):
            src, mine = self._source_decision[idx], act[idx]
            p_src = np.where(src == 1, self._obj[idx, 1], self._obj[idx, 0]).sum()
            p_tree = self._obj[idx, mine].sum() if len(idx) else 0.0
            rows[name] = {"accounts": len(idx), "agreement with account-level policy": float((src == mine).mean()),
                          "approval rate, account-level": float(src.mean()), "approval rate, tree": float(mine.mean()),
                          "profit kept vs account-level": float(p_tree / p_src) if p_src else float("nan")}
        return pd.DataFrame(rows).T

    def objective_report(self):
        """What did deployability cost? The account-level optimum, the tree with the CART's majority actions (before
        re-optimisation) and the final tree (one re-optimised action per leaf), each scored by the same evaluator."""
        self._check_fitted()
        src = self._scenario
        rows = {}
        for label, sc in (("account-level optimum (source scenario)", src),
                          ("tree, majority action per leaf (before re-optimisation)", self._majority_policy),
                          ("distilled tree (leaf actions re-optimised)", self.policy_)):
            viol = None
            if sc.constraints is not None:
                rep = sc.constraint_report()
                viol = int((~rep["satisfied"]).sum())
            rows[label] = {"expected profit (bn VND)": sc.total_portfolio / 1e9, "rules violated": viol}
        out = pd.DataFrame(rows).T
        base = out.iloc[0, 0]
        out["vs account-level (bn VND)"] = out["expected profit (bn VND)"] - base
        out["% of account-level optimum"] = out["expected profit (bn VND)"] / base
        return out

    def print_objectives(self):
        rep = self.objective_report()
        print(f"{self.n_leaves} leaves from {len(self.selected_features)} predictors {self.selected_features}")
        for label, r in rep.iterrows():
            v = "" if r["rules violated"] is None else f" | rules violated: {int(r['rules violated'])}"
            print(f"  {label:<58} {r['expected profit (bn VND)']:>9.3f} bn VND  ({r['% of account-level optimum']:.2%} of the optimum){v}")
        cost = rep.iloc[0, 0] - rep.iloc[2, 0]
        print(f"=> simplifying {self._obj.shape[0]:,} account decisions into {self.n_leaves} rules cost "
              f"{cost:.3f} bn VND, {cost / rep.iloc[0, 0]:.2%} of the account-level optimum")

    def visualize(self, figsize=None, node=0, fontsize=9.5, title=True):
        """The distilled policy as a left-to-right tree: one row per leaf, no overlapping boxes.

        Every internal node shows the split (upper branch = the condition is true, lower branch = false). Every leaf shows its
        number, the FINAL action (after the leaf-level re-optimisation) and the number of accounts. A dashed border marks a leaf
        whose final action differs from the CART's majority decision.

        ``node`` draws only the subtree below that CART node (0 = the whole tree), which keeps big trees readable on a slide.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        self._check_fitted()
        tr = self._sk.tree_
        integer = {sp["feature"]: sp.get("integer", False) for sp in self._spec}

        leaf_nodes = sorted(i for i in range(tr.node_count) if tr.children_left[i] == -1)
        remap = {node: i for i, node in enumerate(leaf_nodes)}
        pos, depth = {}, {}
        counter = [0]

        def place(node, d):
            depth[node] = d
            if tr.children_left[node] == -1:
                pos[node] = counter[0]
                counter[0] += 1
            else:
                a, b = place(tr.children_left[node], d + 1), place(tr.children_right[node], d + 1)
                pos[node] = (a + b) / 2
            return pos[node]

        place(node, 0)
        max_d = max(depth.values())
        n_leaves = len(pos_leaves := [n for n in depth if tr.children_left[n] == -1])
        first = min(pos[n] for n in pos_leaves)
        for k in pos:
            pos[k] -= first
        if figsize is None:
            figsize = (2.6 * max_d + 4.6, 0.42 * n_leaves + 1.6)
        fig, ax = plt.subplots(figsize=figsize)
        col = {"approve": ("#0F8B8D", "#DCEFEF"), "reject": ("#D9822B", "#FBE9D7")}
        step = 1.0

        def label(node):
            feat, cat = self._enc_names[tr.feature[node]]
            if cat is not None:
                return f"{feat} != {cat}"
            t = float(tr.threshold[node])
            return f"{feat} <= {math.floor(t)}" if integer.get(feat) else f"{feat} <= {t:.4g}"

        def draw(node):
            x, y = depth[node] * step, pos[node]
            if tr.children_left[node] == -1:
                i = remap[node]
                act = "approve" if self.actions_[i] == 1 else "reject"
                diff = self.actions_[i] != self.majority_[i]
                text = f"leaf {i}: {act.upper()}  ({self.rules_[i].n_apps:,} accounts){'  *' if diff else ''}"
                ax.text(x + 0.04, y, text, va="center", ha="left", fontsize=fontsize, color="#1F2A44",
                        bbox=dict(boxstyle="round,pad=0.3", fc=col[act][1], ec=col[act][0], lw=1.6, ls="--" if diff else "-"))
                return
            for child, dy_label in ((tr.children_left[node], "yes"), (tr.children_right[node], "no")):
                cx, cy = depth[child] * step, pos[child]
                ax.plot([x, x, cx], [y, cy, cy], color="#8A94A3", lw=1.2, zorder=1)
                ax.text(x + 0.03, cy + (-0.18 if dy_label == "yes" else 0.18), dy_label, fontsize=7.5, color="#5B6577",
                        va="center", ha="left")
                draw(child)
            ax.text(x, y, label(node), va="center", ha="center", fontsize=fontsize, color="#1F2A44", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#5B6577", lw=1.2))

        draw(node)
        ax.set_xlim(-0.55, max_d * step + 2.6)
        ax.set_ylim(n_leaves - 0.4, -0.8)
        ax.axis("off")
        if title:
            ax.set_title(f"Distilled policy: {n_leaves} leaves.  Upper branch = condition true.  "
                         "Colour = final action; dashed border (*) = differs from the CART majority.", fontsize=11, loc="left")
        fig.tight_layout()
        return fig

    def explain(self, plot_type="beeswarm", max_background=100, max_samples=250):
        """SHAP over the SELECTED predictors only: which of them drive the probability of 'approve', and in which direction."""
        import matplotlib.pyplot as plt
        import shap
        self._check_fitted()
        X, names = self._encode(self._scenario.project.portfolio.data)
        keep = [j for j, (f, _) in enumerate(names) if f in self.selected_features]
        label = [f if c is None else f"{f}={c}" for f, c in (names[j] for j in keep)]
        rng = np.random.default_rng(self.random_state)
        bg = X[rng.choice(len(X), min(max_background, len(X)), replace=False)]
        xs = X[rng.choice(len(X), min(max_samples, len(X)), replace=False)]
        expl = shap.TreeExplainer(self._sk, data=bg, feature_perturbation="interventional", model_output="probability")
        sv = expl.shap_values(xs, check_additivity=False)
        sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
        fig = plt.figure()
        shap.summary_plot(sv[:, keep], xs[:, keep], feature_names=label, plot_type="dot" if plot_type == "beeswarm" else "bar",
                          show=False)
        imp = pd.Series(np.abs(sv[:, keep]).mean(0), index=label).sort_values(ascending=False)
        return {"figure": plt.gcf(), "mean_abs_shap": imp}

    # ------------------------------------------------------------------ persistence
    def to_dict(self):
        self._check_fitted()
        def clean(c):
            return {f: ({"lo": None if math.isinf(d["lo"]) and d["lo"] < 0 else d["lo"],
                         "hi": None if math.isinf(d["hi"]) else d["hi"]} if "lo" in d else
                        {"in": d["in"], "not_in": sorted(set(d["not_in"]))}) for f, d in c.items()}
        return {"format": "pydo-decision-tree", "version": 1, "source": self.source_name, "variable": self.var,
                "action_values": self.action_values, "features": self._spec,
                "params": {"max_depth": self.max_depth, "max_splits": self.max_splits, "min_weight": self.min_weight,
                           "min_gini_gain": self.min_gini_gain},
                "leaves": [{"leaf": r.leaf, "rule": r.description, "action": self.actions_[r.leaf], "accounts": r.n_apps,
                            "conditions": clean(self._conditions[r.leaf])} for r in self.rules_]}

    def save(self, path, overwrite=False):
        """Write the rules as JSON (a human-readable, scikit-learn-free artefact). Returns the path."""
        import os
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path):
        """Re-load a saved tree: ``predict`` and ``rules_`` work, no scikit-learn and no scenario needed."""
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("format") != "pydo-decision-tree":
            raise ProjectError(f"{path} is not a pydo decision tree")
        t = cls(**d["params"])
        t.source_name, t.var, t.action_values, t._spec = d["source"], d["variable"], d["action_values"], d["features"]
        t._conditions, t.actions_, t.rules_ = {}, {}, []
        for lf in d["leaves"]:
            i = lf["leaf"]
            cond = {}
            for f, c in lf["conditions"].items():
                cond[f] = ({"lo": -math.inf if c["lo"] is None else c["lo"], "hi": math.inf if c["hi"] is None else c["hi"]}
                           if "lo" in c else {"in": c["in"], "not_in": list(c["not_in"])})
            t._conditions[i], t.actions_[i] = cond, lf["action"]
            t.rules_.append(LeafRule(i, lf["rule"], cond, t.action_values[lf["action"]], lf["accounts"]))
        t.n_leaves = len(t.rules_)
        t.attempted_features = [sp["feature"] for sp in t._spec]
        t.selected_features = t.attempted_features
        t._fitted = True
        return t

    def to_sql(self, table="applications"):
        """The rules as a SQL CASE expression, for a rule engine that speaks SQL."""
        self._check_fitted()
        lines = []
        for r in self.rules_:
            conds = []
            integer = {sp["feature"]: sp.get("integer", False) for sp in self._spec}
            for f, c in self._conditions[r.leaf].items():
                if "lo" in c and integer.get(f):
                    lo = None if math.isinf(c["lo"]) else math.floor(c["lo"]) + 1
                    hi = None if math.isinf(c["hi"]) else math.floor(c["hi"])
                    if lo is not None and hi is not None:
                        conds.append(f"{f} = {lo}" if lo == hi else f"{f} BETWEEN {lo} AND {hi}")
                    elif lo is not None:
                        conds.append(f"{f} >= {lo}")
                    elif hi is not None:
                        conds.append(f"{f} <= {hi}")
                elif "lo" in c:
                    if not math.isinf(c["lo"]):
                        conds.append(f"{f} > {_fmt(c['lo'])}")
                    if not math.isinf(c["hi"]):
                        conds.append(f"{f} <= {_fmt(c['hi'])}")
                else:
                    if c["in"]:
                        conds.append(f"{f} = '{c['in'][0]}'")
                    if c["not_in"]:
                        conds.append(f"{f} NOT IN ({', '.join(repr(x) for x in sorted(set(c['not_in'])))})")
            lines.append(f"  WHEN {' AND '.join(conds) or 'TRUE'} THEN {self.action_values[self.actions_[r.leaf]]}"
                         f"   -- leaf {r.leaf}: {r.description}")
        return f"SELECT CASE\n" + "\n".join(lines) + f"\nEND AS {self.var}\nFROM {table};"

    def __repr__(self):
        if not self._fitted:
            return "DecisionTree(not fitted)"
        return f"DecisionTree({self.n_leaves} leaves, predictors {self.selected_features})"
