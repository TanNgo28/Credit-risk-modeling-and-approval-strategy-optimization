"""Scenario: one run of a project under a flow (simulation / unconstrained / constrained)."""
import numpy as np
import pandas as pd

from .errors import ProjectError, SolverError
from .model import BoundRule, Model
from .rules import Constraints

FLOWS = ("simulation", "unconstrained", "constrained")


class Scenario:
    """A run of a project.

    Flows
    -----
    ``simulation``    evaluate a FIXED policy. The decision is read per account from a portfolio column
                      (``variable_to_decision={"APPROVE_ACTION": "APPROVE_ACTION_BAU"}``) or is one constant for
                      every account (``decisions={"APPROVE_ACTION": 1}``) - never both.
    ``unconstrained`` the best treatment per account, no business rules: the ceiling. Ties go to the first treatment
                      of the domain (for approve/reject: reject at exactly zero expected profit).
    ``constrained``   the best assignment subject to ``constraints``, solved as a mixed-integer program (HiGHS).

    ``constraints`` may also be passed to a simulation or an unconstrained scenario: nothing is enforced, but
    ``constraint_report()`` then tells which rules that policy violates.

    Typical use: ``build()`` -> (``preflight()``) -> ``optimize()`` -> ``score`` / ``export()`` / reports.
    ``score`` is the AVERAGE objective per active account; ``total`` is the sum over active accounts and
    ``total_portfolio`` scales it to the full portfolio (matters when a sample was used).
    """

    def __init__(self, project, name, description="", flow="constrained", constraints=None,
                 variable_to_decision=None, decisions=None, solver=None):
        if flow not in FLOWS:
            raise ProjectError(f"unknown flow '{flow}'. Choose one of {FLOWS}")
        if constraints is not None and not isinstance(constraints, Constraints):
            raise ProjectError("constraints must be a pydo.Constraints")
        if flow == "constrained" and (constraints is None or len(constraints) == 0):
            raise ProjectError("the constrained flow needs a non-empty 'constraints' set")
        if solver not in (None, "highs"):
            raise SolverError(f"solver '{solver}' is not available; use 'highs'")
        self.project, self.name, self.description, self.flow = project, name, description, flow
        self.constraints, self.solver = constraints, solver or "highs"
        self.variable_to_decision, self.decisions = variable_to_decision, decisions
        self._validate_policy()

        self.pairs = self.selected = self.model = self.bound = None
        self.score = self.total = self.total_portfolio = None
        self.info = {}
        self.project._register(self.name, self)

    # ------------------------------------------------------------------ validation
    def _validate_policy(self):
        v2d, dec = self.variable_to_decision, self.decisions
        variables, portfolio = self.project.variables, self.project.portfolio
        if self.flow != "simulation":
            if v2d is not None or dec is not None:
                raise ProjectError(f"'decisions' / 'variable_to_decision' only apply to the simulation flow, "
                                   f"not to '{self.flow}'")
            return
        if v2d is not None and dec is not None:
            raise ProjectError("never pass both 'decisions' and 'variable_to_decision' to the same scenario")
        if v2d is None and dec is None:
            raise ProjectError("a simulation needs a fixed policy: pass 'variable_to_decision' (portfolio columns) "
                               "or 'decisions' (constants)")
        spec = v2d if v2d is not None else dec
        if sorted(spec) != sorted(variables.names):
            raise ProjectError(f"the policy must cover exactly the decision variables {variables.names}, "
                               f"got {sorted(spec)}")
        if v2d is not None:
            absent = [c for c in v2d.values() if c not in portfolio.columns]
            if absent:
                raise ProjectError(f"decision columns not in the portfolio: {absent}. They must exist BEFORE the "
                                   "Portfolio is constructed")
        else:
            for var, val in dec.items():
                if val not in variables.domain(var):
                    raise ProjectError(f"decision {var}={val!r} is outside the domain {variables.domain(var)}")

    def _treatment_for(self, record):
        if self.decisions is not None:
            return dict(self.decisions)
        t = {var: record[col] for var, col in self.variable_to_decision.items()}
        for var, val in t.items():
            if val not in self.project.variables.domain(var):
                raise ProjectError(f"account {record[self.project.portfolio.key_column]!r}: decision {var}={val!r} "
                                   f"is outside the domain {self.project.variables.domain(var)}")
        return t

    # ------------------------------------------------------------------ build
    def build(self, n_jobs=1):
        """Evaluate the calculation pipeline for every account (and, unless simulating, every treatment).

        Variables read by the constraints are collected too, even if they are not in ``output_variables``.
        ``n_jobs`` is accepted for API compatibility and currently ignored (single process)."""
        proj, portfolio = self.project, self.project.portfolio
        key, data = portfolio.key_column, portfolio.data
        calc = set(proj.calculated_variables)

        names = [proj.objective] + [n for n in proj.output_variables if n != proj.objective]
        if self.constraints is not None:
            self.bound = [BoundRule(r, data, calc) for r in self.constraints]         # fails early, with the rule's name
            for br in self.bound:
                for v in br.rule.variables():
                    if v in calc and v not in names:
                        names.append(v)

        treatments = proj.variables.treatments()
        rows = []
        for rec in data.to_dict("records"):
            for t in ([self._treatment_for(rec)] if self.flow == "simulation" else treatments):
                obj = proj._evaluate(rec, t, key=rec[key])
                rows.append({key: rec[key], **t, **proj.collect(obj, names, rec[key])})
        self.pairs = pd.DataFrame(rows)
        self.model = None
        self.info = {"pairs": len(self.pairs), "accounts": portfolio.n_active, "treatments": len(treatments)}
        return self

    def _need_built(self, what):
        if self.pairs is None:
            raise ProjectError(f"call build() before {what}")

    def preflight(self, verbose=True):
        """Size of the assignment problem and account-level eligibility pruning, before any solve.

        Returns a DataFrame with one row per compiled constraint row (kind, accounts in scope, non-zeros)."""
        self._need_built("preflight()")
        if self.flow != "constrained":
            raise ProjectError("preflight() applies to the constrained flow")
        self._compile()
        m = self.model
        table = m.row_table()
        if verbose:
            print(f"accounts {m.n:,} | treatments {m.T} | decision variables {m.nv:,} (reference treatment eliminated)")
            print(f"constraint rows {len(table)} ({m.A.shape[0]} matrix rows incl. assignment rows) | "
                  f"non-zeros {int(m.A.nnz):,}")
            print(f"account-level eligibility: {m.pruned_pairs:,} account-treatment pairs pruned")
            small = table[(table["kind"] != "validity") & (table["accounts_in_scope"] < 30)]
            if len(small):
                print("WARNING small segments (< 30 accounts):", ", ".join(small["name"]))
        return table

    def _compile(self):
        if self.model is None:
            proj, portfolio = self.project, self.project.portfolio
            self.model = Model(portfolio.data, self.pairs, len(proj.variables.treatments()), proj.objective,
                               proj.direction == "maximize", self.bound, set(proj.calculated_variables),
                               portfolio.key_column)

    def with_constraints(self, constraints, name=None):
        """A new CONSTRAINED scenario that reuses the already evaluated account-treatment pairs of this one.

        The economics are not recomputed, only the rules change: this is what makes shadow prices, frontiers and
        "what if this threshold moves" studies cheap. The new rules may only read variables that were already
        evaluated here (the ones this scenario's own rules, objective and output variables use)."""
        self._need_built("with_constraints()")
        if self.flow == "simulation":
            raise ProjectError("with_constraints() needs a scenario that evaluated every treatment "
                               "(an unconstrained or constrained one), not a simulation")
        final_name = name or self.name
        previous = self.project._results.get(final_name)
        new = Scenario(self.project, name=final_name, description=self.description, flow="constrained",
                       constraints=constraints, solver=self.solver)
        if name is None:                       # a what-if variant must not take over the registered name of the original
            if previous is None:
                self.project._results.pop(final_name, None)
            else:
                self.project._results[final_name] = previous
        data, calc = self.project.portfolio.data, set(self.project.calculated_variables)
        new.bound = [BoundRule(r, data, calc) for r in constraints]
        missing = sorted({v for br in new.bound for v in br.rule.variables()
                          if v in calc and v not in self.pairs.columns})
        if missing:
            raise ProjectError(f"the new rules read {missing}, which this scenario did not evaluate; "
                               "add them to output_variables (or to the original rules) and build() again")
        new.pairs = self.pairs
        new.info = {"pairs": len(self.pairs), "accounts": len(data), "treatments": self.info.get("treatments")}
        return new

    # ------------------------------------------------------------------ optimize
    def optimize(self, solver=None, time_limit=600, gap=0.01, n_jobs=None, verbose=False, **_ignored):
        """Select the treatment per account and compute the objective.

        For the constrained flow: ``solver`` ('highs'), ``time_limit`` (seconds), ``gap`` (relative MILP gap),
        ``verbose``. ``n_jobs`` is accepted and ignored. Raises InfeasibleModelError if no assignment satisfies
        every rule."""
        self._need_built("optimize()")
        if solver not in (None, "highs"):
            raise SolverError(f"solver '{solver}' is not available; use 'highs'")
        proj, key, obj = self.project, self.project.portfolio.key_column, self.project.objective
        if self.flow == "simulation":
            sel = self.pairs
            self.info.update(status="simulated")
        elif self.flow == "unconstrained":
            pick = (self.pairs.groupby(key, sort=False)[obj].idxmax() if proj.direction == "maximize"
                    else self.pairs.groupby(key, sort=False)[obj].idxmin())
            sel = self.pairs.loc[pick.to_numpy()]
            self.info.update(status="optimal")
        else:
            self._compile()
            chosen, solve_info = self.model.solve(time_limit=time_limit, gap=gap, verbose=verbose)
            T = self.model.T
            sel = self.pairs.iloc[np.arange(len(chosen)) * T + chosen]
            self.info.update(solve_info)
        self.selected = sel.reset_index(drop=True)
        self.total = float(self.selected[obj].sum())
        self.score = float(self.selected[obj].mean())
        self.total_portfolio = self.total * proj.portfolio.scale
        if self.flow == "constrained":
            check = abs(self.total - self.info["objective_total"])
            if check > 1e-6 * max(1.0, abs(self.total)):
                raise SolverError(f"internal check failed: the exported selection gives {self.total:.6g} but the "
                                  f"solver reports {self.info['objective_total']:.6g}")
        return self

    # ------------------------------------------------------------------ reports
    def solver_report(self):
        """Requested vs achieved gap, incumbent, best bound, time and model dimensions."""
        if self.selected is None:
            raise ProjectError("call optimize() before solver_report()")
        order = ["flow", "status", "objective_total", "best_bound", "gap_requested", "gap_achieved", "time_seconds",
                 "time_limit", "nodes", "variables", "rows", "nonzeros", "pruned_pairs", "message"]
        info = {"flow": self.flow, **self.info}
        return pd.Series({k: info[k] for k in order if k in info}, name=self.name)

    def _selection_arrays(self):
        data, calc = self.project.portfolio.data, set(self.project.calculated_variables)
        sel = {}
        for br in self.bound:
            for v in br.rule.variables():
                if v not in sel:
                    sel[v] = self.selected[v].to_numpy(float) if v in calc else data[v].to_numpy(float)
        return sel

    def constraint_report(self, binding_tol=1e-3):
        """Every rule against the selected assignment: both sides, signed slack (positive = margin), violation,
        binding status. ``binding_tol`` is relative to the magnitude of the bound (1e-3 of 0.45 is ~8 accounts)."""
        if self.selected is None:
            raise ProjectError("call optimize() before constraint_report()")
        if self.constraints is None:
            raise ProjectError("this scenario has no constraints to report on")
        sel, rows = self._selection_arrays(), []
        for br in self.bound:
            ev = br.evaluate(sel)
            scale = max(1.0, abs(ev["rhs"])) if np.isfinite(ev["rhs"]) else 1.0
            slack = ev["slack"]
            rows.append(dict(constraint=br.rule.name, kind=br.rule.kind, rule=br.rule.text, lhs=ev["lhs"],
                             op="=" if br.rule.kind == "account" else br.rule.op, rhs=ev["rhs"], slack=slack,
                             violation=ev["violation"],
                             satisfied=bool(np.isfinite(slack) and ev["violation"] <= 1e-6 * scale),
                             binding=bool(br.rule.kind != "account" and np.isfinite(slack) and abs(slack) <= binding_tol * scale),
                             accounts_in_scope=br.scope))
        return pd.DataFrame(rows)

    def coefficient_report(self, warn_range=1e9):
        """Numerical health of the matrix handed to the solver: smallest / largest coefficient per row and overall."""
        self._need_built("coefficient_report()")
        if self.flow != "constrained":
            raise ProjectError("coefficient_report() applies to the constrained flow")
        self._compile()
        return self.model.coefficient_report(warn_range)

    def apply_policy(self, decisions, name="policy"):
        """Score a GIVEN policy (one decision per account) on the pairs this scenario already evaluated.

        ``decisions``: a Series indexed by account id, or an array in portfolio order, holding the value of the (single)
        decision variable. Returns a simulation-like scenario, registered under ``name``, on which ``export()``,
        ``constraint_report()`` and the totals work exactly as for any other scenario. Nothing is re-optimised and the
        economics are not recomputed: this is how a rule-based policy (a distilled tree, a cut-off) is evaluated with the
        same evaluator as the solver's own solution."""
        self._need_built("apply_policy()")
        if self.flow == "simulation":
            raise ProjectError("apply_policy() needs a scenario that evaluated every treatment "
                               "(an unconstrained or constrained one)")
        variables, portfolio = self.project.variables, self.project.portfolio
        if len(variables) != 1:
            raise NotImplementedError("apply_policy() supports a single decision variable")
        var = variables.names[0]
        treatments = variables.treatments()
        key = portfolio.key_column
        ids = portfolio.data[key]
        dec = pd.Series(np.asarray(decisions), index=ids) if not isinstance(decisions, pd.Series) else decisions.reindex(ids)
        if dec.isna().any():
            raise ProjectError("the policy has no decision for some accounts of the portfolio")
        position = {t[var]: i for i, t in enumerate(treatments)}
        bad = sorted(set(dec.unique()) - set(position))
        if bad:
            raise ProjectError(f"decisions outside the domain of {var}: {bad}")
        idx = np.arange(len(dec)) * len(treatments) + dec.map(position).to_numpy()
        sel = self.pairs.iloc[idx].reset_index(drop=True)

        new = Scenario.__new__(Scenario)
        new.project, new.name, new.description, new.flow = self.project, name, f"policy applied on '{self.name}'", "simulation"
        new.constraints, new.solver = self.constraints, self.solver
        new.variable_to_decision = new.decisions = None
        new.pairs, new.selected, new.model, new.bound = sel, sel, None, self.bound
        obj = self.project.objective
        new.total = float(sel[obj].sum())
        new.score = float(sel[obj].mean())
        new.total_portfolio = new.total * portfolio.scale
        new.info = {"status": "policy applied", "accounts": len(sel)}
        self.project._register(name, new)
        return new

    # ------------------------------------------------------------------ results
    def export(self, path=None):
        """The selected treatment and the output variables for every active account (also written as CSV if ``path``)."""
        if self.selected is None:
            raise ProjectError("call optimize() before export()")
        if path is not None:
            self.selected.to_csv(path, index=False)
        return self.selected.copy()

    def summary(self):
        if self.selected is None:
            raise ProjectError("call optimize() before summary()")
        return {"scenario": self.name, "flow": self.flow, "accounts": len(self.selected),
                "objective": self.project.objective, "direction": self.project.direction,
                "average_per_account": self.score, "total_active": self.total,
                "total_full_portfolio_estimate": self.total_portfolio}

    def __repr__(self):
        state = "optimized" if self.selected is not None else ("built" if self.pairs is not None else "defined")
        return f"Scenario('{self.name}', flow={self.flow}, {state})"
