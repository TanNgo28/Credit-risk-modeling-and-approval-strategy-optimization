"""Exception hierarchy of PyDO."""


class PyDOError(Exception):
    """Base class of every error raised on purpose by PyDO."""


class ProjectError(PyDOError):
    """The project is mis-assembled (wrong assignment order, unknown objective, bad scenario arguments...)."""


class UnknownFactError(PyDOError, KeyError):
    """A function asked for a fact that was never defined."""

    def __str__(self):                       # KeyError would otherwise show the repr of the message
        return str(self.args[0])


class ConstraintError(PyDOError):
    """A constraint rule is malformed or not allowed (phase 2)."""


class SolverError(PyDOError):
    """The solver failed or is not available (phase 2)."""


class InfeasibleModelError(SolverError):
    """No assignment satisfies every constraint (phase 2)."""
