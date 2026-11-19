"""Variables: the decision space. A treatment is one value of every decision variable."""
import itertools


class Variables:
    """Decision variables and their domains.

    ``variables.set("APPROVE_ACTION", [0, 1])`` defines two treatments per account. With several variables the
    treatments are the Cartesian product of the domains, in insertion order.
    """

    def __init__(self):
        self._domains = {}

    def set(self, name, domain):
        if not isinstance(name, str) or not name:
            raise ValueError("a decision variable needs a non-empty string name")
        domain = list(domain)
        if not domain:
            raise ValueError(f"the domain of '{name}' is empty")
        if len(set(domain)) != len(domain):
            raise ValueError(f"the domain of '{name}' contains duplicates: {domain}")
        self._domains[name] = domain
        return self

    @property
    def names(self):
        return list(self._domains)

    def domain(self, name):
        return list(self._domains[name])

    def treatments(self):
        """All treatments as a list of ``{variable: value}`` dicts (stable order)."""
        if not self._domains:
            raise ValueError("no decision variable defined")
        names = list(self._domains)
        return [dict(zip(names, combo)) for combo in itertools.product(*self._domains.values())]

    def __len__(self):
        return len(self._domains)

    def __repr__(self):
        if not self._domains:
            return "Variables(empty)"
        body = ", ".join(f"{k}: {v}" for k, v in self._domains.items())
        return f"Variables({body}; {len(self.treatments())} treatments)"
