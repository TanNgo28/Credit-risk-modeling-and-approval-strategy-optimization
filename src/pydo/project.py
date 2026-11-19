"""Project: assembles portfolio, facts, variables, functions and objective; creates scenarios."""
import numbers

import numpy as np

from .errors import ProjectError
from .facts import Facts
from .functions import Functions
from .portfolio import Portfolio
from .variables import Variables


class Obj:
    """What a calculation function receives: the raw account, the treatment under evaluation and the facts."""

    def __init__(self, account, decision, facts):
        self.account = account
        self.decision = decision
        self.facts = facts


def _is_number(value):
    return isinstance(value, (numbers.Real, np.number)) and not isinstance(value, np.ndarray)


class Project:
    """A decision-optimisation project.

    Assignment order matters, exactly as the brief says: ``facts``, ``variables`` and ``portfolio`` first, then
    ``functions`` (PyDO dry-runs them on one account to discover the calculated ``v_`` variables), and only then
    ``objective`` / ``output_variables``, which must name a calculated variable that really exists.
    """

    def __init__(self, name, description=""):
        self.name, self.description = name, description
        self._facts = self._variables = self._portfolio = self._functions = None
        self._objective = None
        self._direction = "maximize"
        self._outputs = []
        self._calc_vars = []                  # numeric calculated variables found by the dry run
        self._results = {}                    # name -> scenario / attached policy: what project.report() compares
        self._trees = {}

    # ------------------------------------------------------------------ assignments
    @property
    def facts(self):
        return self._facts

    @facts.setter
    def facts(self, value):
        if not isinstance(value, Facts):
            raise ProjectError("project.facts must be a pydo.Facts")
        self._facts = value
        self._rediscover()

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, value):
        if not isinstance(value, Variables):
            raise ProjectError("project.variables must be a pydo.Variables")
        if len(value) == 0:
            raise ProjectError("project.variables is empty")
        self._variables = value
        self._rediscover()

    @property
    def portfolio(self):
        return self._portfolio

    @portfolio.setter
    def portfolio(self, value):
        if not isinstance(value, Portfolio):
            raise ProjectError("project.portfolio must be a pydo.Portfolio")
        self._portfolio = value
        self._rediscover()

    @property
    def functions(self):
        return self._functions

    @functions.setter
    def functions(self, value):
        if not isinstance(value, Functions):
            raise ProjectError("project.functions must be a pydo.Functions")
        missing = [n for n, v in [("facts", self._facts), ("variables", self._variables),
                                  ("portfolio", self._portfolio)] if v is None]
        if missing:
            raise ProjectError(f"assign {', '.join(missing)} before functions: PyDO needs them to discover which "
                               "calculated variables the functions produce")
        self._functions = value
        self._discover()
        self._objective, self._outputs = None, []         # a new pipeline invalidates earlier choices

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, name):
        self._require_functions("objective")
        if name not in self._calc_vars:
            raise ProjectError(f"objective '{name}' is not a numeric calculated variable produced by the functions. "
                               f"Available: {self._calc_vars}")
        self._objective = name

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        if value not in ("maximize", "minimize"):
            raise ProjectError("direction must be 'maximize' or 'minimize'")
        self._direction = value

    @property
    def output_variables(self):
        return list(self._outputs)

    @output_variables.setter
    def output_variables(self, names):
        self._require_functions("output_variables")
        names = list(names)
        unknown = [n for n in names if n not in self._calc_vars]
        if unknown:
            raise ProjectError(f"output variables not produced by the functions (or not numeric): {unknown}. "
                               f"Available: {self._calc_vars}")
        self._outputs = names

    @property
    def calculated_variables(self):
        return list(self._calc_vars)

    # ------------------------------------------------------------------ internals
    def _require_functions(self, what):
        if self._functions is None:
            raise ProjectError(f"assign project.functions before {what}: PyDO has not discovered the calculated "
                               "variables yet")

    def _rediscover(self):
        if self._functions is not None and None not in (self._facts, self._variables, self._portfolio):
            self._discover()
            if self._objective not in self._calc_vars:
                self._objective = None

    def _evaluate(self, account, treatment, key=None):
        obj = Obj(account, dict(treatment), self._facts)
        self._functions.run(obj, key)
        return obj

    def _discover(self):
        first = self._portfolio.data.iloc[0].to_dict()
        obj = self._evaluate(first, self._variables.treatments()[0], key=first[self._portfolio.key_column])
        self._calc_vars = [k for k, v in vars(obj).items() if k.startswith("v_") and _is_number(v)]

    def collect(self, obj, names, key=None):
        out = {}
        for n in names:
            if not hasattr(obj, n):
                raise ProjectError(f"calculated variable '{n}' is missing for account {key!r}")
            out[n] = float(getattr(obj, n))
        return out

    # ------------------------------------------------------------------ results registry and report
    def _register(self, name, result):
        self._results[name] = result

    def register(self, scenario, name=None):
        """Make ``scenario`` the one that ``report()`` finds under ``name`` (default: its own name; renames it if given)."""
        if name is not None:
            scenario.name = name
        self._results[scenario.name] = scenario
        return scenario

    def attach_tree(self, name, tree, description="", source_scenario=None):
        """Attach a fitted DecisionTree so that it appears next to the scenarios in ``report()``.

        The tree's policy (its leaf actions applied to the portfolio) is registered under ``name``."""
        if getattr(tree, "policy_", None) is None:
            raise ProjectError("fit() the tree before attaching it")
        if source_scenario is not None and tree.source_name != source_scenario:
            raise ProjectError(f"the tree was distilled from '{tree.source_name}', not from '{source_scenario}'")
        policy = tree.policy_
        policy.name, policy.description = name, description or policy.description
        self._results[name] = policy
        self._trees[name] = tree
        return policy

    def report(self, scenarios=None, reference=None):
        """A comparison of registered scenarios (and attached trees) by name."""
        from .reports import Report
        names = list(scenarios) if scenarios is not None else [n for n in self._results if not n.startswith("_")]
        missing = [n for n in names if n not in self._results]
        if missing:
            raise ProjectError(f"unknown scenarios: {missing}. Known: {list(self._results)}")
        if reference is not None and reference not in names:
            raise ProjectError("the reference must be one of the compared scenarios")
        return Report(self, names, reference)

    # ------------------------------------------------------------------ scenarios
    def scenario(self, name, description="", flow="constrained", constraints=None,
                 variable_to_decision=None, decisions=None, solver=None):
        from .scenario import Scenario
        if self._objective is None:
            raise ProjectError("assign project.objective before creating a scenario")
        return Scenario(self, name=name, description=description, flow=flow, constraints=constraints,
                        variable_to_decision=variable_to_decision, decisions=decisions, solver=solver)

    def __repr__(self):
        def n(x, f=lambda v: v):
            return "-" if x is None else f(x)
        return (f"Project('{self.name}'\n  portfolio  : {n(self._portfolio)}\n  variables  : {n(self._variables)}"
                f"\n  functions  : {n(self._functions)}\n  facts      : {n(self._facts, len)} defined"
                f"\n  objective  : {self._direction} {n(self._objective)}"
                f"\n  outputs    : {self._outputs or '-'}\n)")
