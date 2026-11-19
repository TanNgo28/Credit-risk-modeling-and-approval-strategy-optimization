"""Functions: the calculation pipeline (the economics), plain Python in ``.py`` files."""
import hashlib
import importlib.util
import inspect
from pathlib import Path

from .errors import PyDOError


class Functions:
    """An ordered list of calculation steps, each with the signature ``def f(obj):``.

    ``Functions(["functions/flatten.py", "functions/economics.py"])`` loads every public function defined in each
    file, in file order and, inside a file, in definition order. PyDO runs the steps once per account-treatment
    pair; each step reads what the earlier ones wrote on ``obj`` and attaches new attributes to it.
    Plain callables are accepted too (handy for tests).

    The ``obj`` a step receives has:
      * ``obj.account``  - dict with the raw portfolio row,
      * ``obj.decision`` - dict ``{decision variable: value}`` of the treatment being evaluated,
      * ``obj.facts``    - the project's :class:`Facts` (``obj.facts.get("name")``).
    Naming convention for attributes: ``p_`` raw portfolio field, ``a_`` action, ``f_`` fact, ``v_`` calculated value.
    """

    def __init__(self, sources):
        self.steps = []                       # (label, callable)
        self.sources = list(sources)
        for src in self.sources:
            if isinstance(src, (str, Path)):
                self.steps.extend(self._load_file(Path(src)))
            elif callable(src):
                self.steps.append((getattr(src, "__name__", repr(src)), src))
            else:
                raise TypeError(f"a function source must be a path or a callable, got {type(src).__name__}")
        if not self.steps:
            raise ValueError("no calculation function found")

    @staticmethod
    def _load_file(path):
        if not path.is_file():
            raise FileNotFoundError(f"function file not found: {path}")
        tag = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:8]
        modname = f"pydo_user_{path.stem}_{tag}"
        spec = importlib.util.spec_from_file_location(modname, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)                    # always a fresh exec: an edited file is picked up
        steps = []
        for name, obj in vars(module).items():             # definition order
            if name.startswith("_") or not inspect.isfunction(obj) or obj.__module__ != modname:
                continue
            if len(inspect.signature(obj).parameters) != 1:
                continue
            steps.append((f"{path.name}:{name}", obj))
        if not steps:
            raise ValueError(f"{path} defines no public function of the form 'def f(obj):'")
        return steps

    def run(self, obj, key=None):
        for label, fn in self.steps:
            try:
                fn(obj)
            except Exception as exc:
                where = f" for account {key!r}" if key is not None else ""
                raise PyDOError(f"function '{label}' failed{where}: {type(exc).__name__}: {exc}") from exc

    def labels(self):
        return [label for label, _ in self.steps]

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return "Functions(" + " -> ".join(self.labels()) + ")"
