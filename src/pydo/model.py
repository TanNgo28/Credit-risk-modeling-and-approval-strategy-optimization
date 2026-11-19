"""From rules to a mixed-integer program, and the solver call.

Model
-----
Every account picks exactly one treatment. The first treatment of the domain is the *reference*: its indicator is
eliminated (``x_i0 = 1 - sum_t y_it``) so that, for approve / reject, the model has ONE binary variable ``y_i`` per
account and no assignment rows. Every aggregate is then linear in ``y``:

    SUM(v)  WHERE c   ->  sum_i c_i * ( v_i0 + sum_t (v_it - v_i0) y_it )
    AVG(v)  WHERE c   ->  the same divided by the (fixed) number of accounts matching c
    WEIGHTED_AVG(v,w) <= K   ->  sum_i c_i * w * (v - K) * x   <= 0      (ratio of sums, made linear)
    A <= B * k        ->  A - k B <= 0

The problem is handed to HiGHS through ``scipy.optimize.milp``.
"""
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .errors import ConstraintError, InfeasibleModelError, SolverError
from .rules import cond_columns, eval_cond

_OPS = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "=": lambda a, b: np.isclose(a, b),
        "<": lambda a, b: a < b, ">": lambda a, b: a > b, "!=": lambda a, b: a != b}


def check_variable(name, calc_vars, data):
    """A rule variable is a calculated variable or a numeric raw portfolio column."""
    if name in calc_vars:
        return "calc"
    if name in data.columns and pd.api.types.is_numeric_dtype(data[name]):
        return "raw"
    raise ConstraintError(f"'{name}' is neither a calculated variable nor a numeric portfolio column. "
                          f"Calculated variables available: {sorted(calc_vars)}")


class BoundRule:
    """A parsed rule bound to a portfolio: variables checked, WHERE masks computed (and never empty)."""

    def __init__(self, rule, data, calc_vars):
        self.rule, self.n = rule, len(data)
        self.masks = {}
        try:
            if rule.kind == "account":
                check_variable(rule.account[0], calc_vars, data)
            else:
                for side, term in (("l", rule.lhs), ("r", rule.rhs)):
                    if term.kind == "num":
                        self.masks[side] = None
                        continue
                    for v in (term.var, term.weight):
                        if v:
                            check_variable(v, calc_vars, data)
                    self.masks[side] = self._mask(term, data, calc_vars)
        except ConstraintError as exc:
            raise ConstraintError(f"constraint '{rule.name}': {exc}") from exc

    def _mask(self, term, data, calc_vars):
        if term.where is None:
            return np.ones(self.n, dtype=bool)
        for c in cond_columns(term.where):
            if c in calc_vars or c.startswith("v_"):
                raise ConstraintError(f"WHERE references the calculated variable {c!r}. A WHERE clause may only use RAW "
                                      "portfolio columns, because the aggregate denominators must be fixed before the "
                                      "solver runs. Build the field as a column of the portfolio first")
            if c not in data.columns:
                raise ConstraintError(f"WHERE references {c!r}, which is not a column of the portfolio")
        mask = eval_cond(term.where, data)
        if not mask.any():
            raise ConstraintError("WHERE matches zero accounts in the active portfolio (an empty segment); "
                                  "raise the sampling fraction, stratify on this column or widen the segment")
        return mask

    @property
    def scope(self):
        """Accounts the rule talks about (the larger of its two sides)."""
        sizes = [int(m.sum()) for m in self.masks.values() if m is not None]
        return max(sizes) if sizes else self.n

    # ------------------------------------------------------------------ evaluation of a given selection
    @staticmethod
    def _term_value(term, mask, sel):
        if term.kind == "num":
            return term.value
        v = sel[term.var]
        if term.kind == "sum":
            return term.mult * float(v[mask].sum())
        if term.kind == "avg":
            return term.mult * float(v[mask].sum()) / int(mask.sum())
        w = sel[term.weight]
        den = float(w[mask].sum())
        return float(np.dot(v[mask], w[mask]) / den) if den != 0 else float("nan")

    def evaluate(self, sel):
        """``sel``: dict name -> array (one value per account, the SELECTED treatment). Returns lhs, rhs, slack, violation."""
        r = self.rule
        if r.kind == "account":
            col, op, val = r.account
            bad = int((~_OPS[op](sel[col], val)).sum())
            return dict(lhs=float(bad), rhs=0.0, slack=float(-bad), violation=float(bad))
        lhs = self._term_value(r.lhs, self.masks["l"], sel)
        rhs = self._term_value(r.rhs, self.masks["r"], sel)
        if r.op == "<=":
            slack = rhs - lhs
        elif r.op == ">=":
            slack = lhs - rhs
        else:
            slack = -abs(lhs - rhs)
        violation = float("nan") if np.isnan(slack) else max(0.0, -slack)
        return dict(lhs=lhs, rhs=rhs, slack=slack, violation=violation)


class Model:
    """The compiled MIP of a scenario. Build once, solve once (or re-use ``rows`` for diagnostics)."""

    VALIDITY_EPS = 1e-6            # smallest positive denominator of a plain WEIGHTED_AVG

    def __init__(self, data, pairs, n_treat, objective, maximize, bound_rules, calc_vars, key):
        self.n, self.T = len(data), n_treat
        self.nv = self.n * (self.T - 1)
        self.data, self.pairs, self.key = data, pairs, key
        self.objective, self.maximize, self.calc_vars = objective, maximize, calc_vars
        self.rules = bound_rules
        self._V = {}
        self.rows = []                                    # dict(name, kind, coef, lo, hi, scope)
        self.eligible = np.ones((self.n, self.T), dtype=bool)

        M = self.values(objective)
        self.obj_coef = (M[:, 1:] - M[:, [0]]).ravel()
        self.obj_const = float(M[:, 0].sum())
        for br in bound_rules:
            self._compile(br)
        self._finish()

    # ------------------------------------------------------------------ values
    def values(self, name):
        """(accounts x treatments) matrix of a calculated variable or of a raw numeric column."""
        if name not in self._V:
            if name in self.calc_vars or name in self.pairs.columns:
                self._V[name] = self.pairs[name].to_numpy(float).reshape(self.n, self.T)
            else:
                self._V[name] = np.repeat(self.data[name].to_numpy(float)[:, None], self.T, axis=1)
        return self._V[name]

    def _lin(self, M, mask, scale=1.0):
        delta = (M[:, 1:] - M[:, [0]]) * mask[:, None]
        return scale * delta.ravel(), scale * float((mask * M[:, 0]).sum())

    def _term_lin(self, term, mask):
        if term.kind == "num":
            return np.zeros(self.nv), float(term.value)
        scale = term.mult if term.kind == "sum" else term.mult / int(mask.sum())
        return self._lin(self.values(term.var), mask, scale)

    # ------------------------------------------------------------------ rules -> rows
    def _add_row(self, name, kind, coef, const, op, scope):
        if not coef.any():                                # independent of every decision: decide it now
            ok = {"<=": const <= 1e-9, ">=": const >= -1e-9, "=": abs(const) <= 1e-9}[op]
            if not ok:
                raise InfeasibleModelError(f"constraint '{name}' does not depend on any decision and is violated "
                                           f"({const:+.6g} {op} 0)")
        lo, hi = {"<=": (-np.inf, -const), ">=": (-const, np.inf), "=": (-const, -const)}[op]
        self.rows.append(dict(name=name, kind=kind, coef=coef, lo=lo, hi=hi, scope=scope))

    def _compile(self, br):
        r = br.rule
        try:
            if r.kind == "account":
                col, op, val = r.account
                if isinstance(val, str):
                    raise ConstraintError("an account-level rule compares a numeric variable with a number")
                self.eligible &= _OPS[op](self.values(col), val)
                return
            lhs, rhs, op = r.lhs, r.rhs, r.op
            wavg = [t for t in (lhs, rhs) if t.kind in ("wavg", "qwavg")]
            if wavg:
                term = wavg[0]
                other = rhs if term is lhs else lhs
                if term is rhs:                           # K op WAVG  ->  WAVG flipped-op K
                    op = {"<=": ">=", ">=": "<=", "=": "="}[op]
                mask = br.masks["l" if term is lhs else "r"]
                W, K = self.values(term.weight), other.value
                if term.kind == "qwavg" and (W[mask] <= 0).any():
                    n_bad = int((W[mask] <= 0).sum())
                    raise ConstraintError(
                        f"QUICK_WEIGHTED_AVG needs strictly positive weights but {n_bad} account-treatment pairs have "
                        f"{term.weight} <= 0 (a rejected account has a disbursed amount of zero). Use WEIGHTED_AVG")
                coef, const = self._lin(self.values(term.var) * W - K * W, mask)
                self._add_row(r.name, r.kind, coef, const, op, br.scope)
                if term.kind == "wavg":                   # the denominator must be strictly positive
                    wc, wk = self._lin(W, mask)
                    self._add_row(r.name + " [denominator > 0]", "validity", wc, wk - self.VALIDITY_EPS, ">=", br.scope)
            else:
                cl, kl = self._term_lin(lhs, br.masks["l"])
                cr, kr = self._term_lin(rhs, br.masks["r"])
                self._add_row(r.name, r.kind, cl - cr, kl - kr, op, br.scope)
        except ConstraintError as exc:
            raise ConstraintError(f"constraint '{r.name}': {exc}") from exc

    # ------------------------------------------------------------------ matrix, bounds
    def _finish(self):
        n, T, nv = self.n, self.T, self.nv
        dead = ~self.eligible.any(axis=1)
        if dead.any():
            ids = self.data[self.key].to_numpy()[dead][:3].tolist()
            raise InfeasibleModelError(f"{int(dead.sum())} accounts have no eligible treatment under the account-level "
                                       f"constraints (e.g. {ids})")
        self.pruned_pairs = int((~self.eligible).sum())
        self.lbv, self.ubv = np.zeros(nv), np.ones(nv)
        self.ubv[(~self.eligible[:, 1:]).ravel()] = 0.0
        blocks, lo, hi = [], [], []
        for row in self.rows:
            nz = np.flatnonzero(row["coef"])
            blocks.append(sparse.csr_array((row["coef"][nz], (np.zeros(len(nz), int), nz)), shape=(1, nv)))
            lo.append(row["lo"]); hi.append(row["hi"])
        if T == 2:
            self.lbv[(~self.eligible[:, 0])] = 1.0                       # reference not allowed -> the other one is forced
        else:
            idx = np.arange(nv).reshape(n, T - 1)
            rows_i = np.repeat(np.arange(n), T - 1)
            blocks.append(sparse.csr_array((np.ones(nv), (rows_i, idx.ravel())), shape=(n, nv)))   # sum_t y <= 1
            lo += [-np.inf] * n; hi += [1.0] * n
            need = np.flatnonzero(~self.eligible[:, 0])
            if len(need):
                sub = sparse.csr_array((np.ones(len(need) * (T - 1)),
                                        (np.repeat(np.arange(len(need)), T - 1), idx[need].ravel())), shape=(len(need), nv))
                blocks.append(sub)
                lo += [1.0] * len(need); hi += [np.inf] * len(need)
        self.A = sparse.vstack(blocks, format="csr") if blocks else sparse.csr_array((0, nv))
        self.lo, self.hi = np.array(lo, float), np.array(hi, float)
        if (self.lbv > self.ubv).any():
            raise InfeasibleModelError("some account has no eligible treatment after account-level pruning")

    # ------------------------------------------------------------------ reports
    def row_table(self):
        cols = ["name", "kind", "accounts_in_scope", "nonzero_coefficients"]
        return pd.DataFrame([dict(name=r["name"], kind=r["kind"], accounts_in_scope=r["scope"],
                                  nonzero_coefficients=int(np.count_nonzero(r["coef"]))) for r in self.rows],
                            columns=cols)

    def coefficient_report(self, warn_range=1e9):
        recs = []
        allnz = []
        for r in self.rows:
            a = np.abs(r["coef"][r["coef"] != 0])
            if len(a):
                recs.append(dict(row=r["name"], nonzeros=len(a), min_abs=a.min(), max_abs=a.max(), range=a.max() / a.min()))
                allnz.append(a)
        oc = np.abs(self.obj_coef[self.obj_coef != 0])
        if len(oc):
            recs.append(dict(row="OBJECTIVE", nonzeros=len(oc), min_abs=oc.min(), max_abs=oc.max(), range=oc.max() / oc.min()))
            allnz.append(oc)
        if not allnz:
            return pd.DataFrame(recs)
        big, small = max(a.max() for a in allnz), min(a.min() for a in allnz)
        recs.append(dict(row="WHOLE MODEL", nonzeros=int(sum(len(a) for a in allnz)), min_abs=small, max_abs=big, range=big / small))
        if big / small >= warn_range:
            warnings.warn(f"numerically fragile model: coefficient dynamic range {big / small:.2e} >= {warn_range:.0e}. "
                          "Rescale the variables (for example VND -> millions of VND); PyDO does not rescale for you",
                          stacklevel=2)
        return pd.DataFrame(recs)

    # ------------------------------------------------------------------ solve
    def solve(self, time_limit=600, gap=0.01, verbose=False):
        sign = -1.0 if self.maximize else 1.0
        c = sign * self.obj_coef
        cons = [LinearConstraint(self.A, self.lo, self.hi)] if self.A.shape[0] else None
        t0 = time.time()
        res = milp(c=c, constraints=cons, integrality=np.ones(self.nv), bounds=Bounds(self.lbv, self.ubv),
                   options=dict(time_limit=float(time_limit), mip_rel_gap=float(gap), disp=bool(verbose)))
        elapsed = time.time() - t0
        if res.x is None:
            if res.status == 2:
                raise InfeasibleModelError(
                    "no assignment satisfies every constraint. Do not loosen thresholds at random: (1) check the PD "
                    "calibration, (2) add the constraints one at a time and see which one flips the model, "
                    "(3) check for empty segments, (4) compare C-approval against C-risk (the central tension).")
            raise SolverError(f"solver stopped without a solution (status {res.status}): {res.message}")
        y = np.round(res.x)
        chosen = np.zeros(self.n, dtype=int)
        if self.nv:
            Y = y.reshape(self.n, self.T - 1)
            chosen = np.where(Y.any(axis=1), Y.argmax(axis=1) + 1, 0)
        total = self.obj_const + float(self.obj_coef @ y)
        dual = getattr(res, "mip_dual_bound", None)
        bound = None if dual is None or not np.isfinite(dual) else (self.obj_const - dual if self.maximize else self.obj_const + dual)
        info = dict(status="optimal" if res.status == 0 else "time_limit (feasible, not proven optimal)",
                    message=res.message, objective_total=total, best_bound=bound,
                    gap_requested=gap, gap_achieved=getattr(res, "mip_gap", None),
                    time_seconds=elapsed, time_limit=time_limit, nodes=getattr(res, "mip_node_count", None),
                    variables=self.nv, rows=int(self.A.shape[0]), nonzeros=int(self.A.nnz) + int(np.count_nonzero(self.obj_coef)),
                    pruned_pairs=self.pruned_pairs)
        return chosen, info
