"""PyDO (Pythonic Decision Optimizer).

You describe the *economics* (plain Python functions over one account and one treatment), the *decision space* (variables) and the
*business rules* (constraints). PyDO evaluates a fixed policy, finds the best unconstrained decision per account, solves the
constrained problem as a mixed-integer program (HiGHS) and distils the account-level answer into a small, feasible rule set.

Concepts: Portfolio, Variables, Facts, Functions, Project, Scenario (flows: simulation, unconstrained, constrained),
Constraints, DecisionTree. Full documentation: docs/pydo.md.
"""
from .errors import (ConstraintError, InfeasibleModelError, ProjectError, PyDOError, SolverError,
                     UnknownFactError)
from .facts import Facts
from .functions import Functions
from .portfolio import Portfolio
from .project import Project
from .rules import Constraints
from .scenario import Scenario
from .tree import DecisionTree, LeafRule
from .variables import Variables

__version__ = "0.0.1"
__all__ = ["Constraints", "DecisionTree", "LeafRule", "Facts", "Functions", "Portfolio", "Project", "Scenario", "Variables", "PyDOError", "ProjectError",
           "ConstraintError", "SolverError", "InfeasibleModelError", "UnknownFactError", "__version__"]
