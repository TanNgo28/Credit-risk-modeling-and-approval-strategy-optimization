"""Facts: named constants / assumptions that the calculation functions read (cost of funds, recovery rate...)."""
from .errors import UnknownFactError

_MISSING = object()


class Facts:
    """A small registry of named assumptions.

    The point of keeping assumptions here instead of inside the ``.py`` function files is that a sensitivity
    study changes ONE line (``facts.set("recovery_rate", 0.10)``) and re-runs the whole project.
    """

    def __init__(self):
        self._facts = {}

    def set(self, name, value, description=""):
        if not isinstance(name, str) or not name:
            raise ValueError("a fact needs a non-empty string name")
        self._facts[name] = (value, description)
        return self

    def get(self, name, default=_MISSING):
        if name in self._facts:
            return self._facts[name][0]
        if default is not _MISSING:
            return default
        raise UnknownFactError(f"fact '{name}' is not defined. Defined facts: {sorted(self._facts)}")

    def __contains__(self, name):
        return name in self._facts

    def __len__(self):
        return len(self._facts)

    def names(self):
        return list(self._facts)

    def as_dict(self):
        return {k: v for k, (v, _) in self._facts.items()}

    def copy(self):
        new = Facts()
        new._facts = dict(self._facts)
        return new

    def __repr__(self):
        if not self._facts:
            return "Facts(empty)"
        lines = [f"  {k} = {v!r}" + (f"   # {d}" if d else "") for k, (v, d) in self._facts.items()]
        return "Facts(\n" + "\n".join(lines) + "\n)"
