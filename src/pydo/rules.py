"""Constraint rules: a small SQL-like language over the selected account-treatment pairs.

Grammar (keywords are case-insensitive, column and variable names are not)::

    rule     := side OP side                      -- global (one side is a number) or relative (both are aggregates)
              | column OP value                   -- account-level: pairs that violate it become ineligible
    side     := factor { ('*' | '/') factor }     -- at most one aggregate, the other factors are numbers
    factor   := ['-'] NUMBER | aggregate
    aggregate:= SUM(var) | AVG(var) | WEIGHTED_AVG(var, weight) | QUICK_WEIGHTED_AVG(var, weight)
                [ WHERE cond ]
    cond     := cmp | cond AND cond | cond OR cond | NOT cond | '(' cond ')'
    cmp      := column OP value | column IN (value, ...)
    OP       := <= | >= | =              (strict < and > are rejected: the linear builders need non-strict bounds)

Two rules of the game that this module enforces:

* a ``WHERE`` clause may only reference RAW portfolio columns, never calculated (``v_``) variables, because the
  denominators of the aggregates must be fixed before the solver runs;
* an aggregate whose ``WHERE`` matches zero accounts is an error, not a silently empty constraint.
"""
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .errors import ConstraintError

AGGS = {"SUM", "AVG", "WEIGHTED_AVG", "QUICK_WEIGHTED_AVG"}
WEIGHTED = {"WEIGHTED_AVG", "QUICK_WEIGHTED_AVG"}

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<num>(?:\d[\d_]*(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
      | (?P<str>'(?:[^']|'')*')
      | (?P<op><=|>=|==|!=|<>|<|>|=)
      | (?P<id>[A-Za-z_][A-Za-z_0-9]*)
      | (?P<sym>[(),*/+-])
    )""", re.X)


def _tokenize(text):
    toks, pos = [], 0
    while pos < len(text):
        if not text[pos:].strip():
            break
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ConstraintError(f"unexpected character {text[pos:].strip()[:1]!r} in rule: {text!r}")
        kind = m.lastgroup
        raw = m.group(kind)
        toks.append((kind, raw, m.start(kind)))
        pos = m.end()
    return toks


@dataclass
class Term:
    """One side of a rule. kind: 'num' | 'sum' | 'avg' | 'wavg' | 'qwavg'."""
    kind: str
    value: float = 0.0                     # for 'num'
    var: Optional[str] = None
    weight: Optional[str] = None
    where: Optional[tuple] = None          # condition AST
    mult: float = 1.0


@dataclass
class ParsedRule:
    name: str
    text: str
    description: str
    kind: str                              # 'global' | 'relative' | 'account'
    lhs: Optional[Term] = None
    op: str = "<="
    rhs: Optional[Term] = None
    account: Optional[tuple] = None        # (column, op, value) for account-level rules

    def variables(self):
        """Names of every variable / weight the rule reads (columns of WHERE clauses excluded)."""
        names = []
        if self.kind == "account":
            return [self.account[0]]
        for t in (self.lhs, self.rhs):
            for n in (t.var, t.weight):
                if n and n not in names:
                    names.append(n)
        return names


# ---------------------------------------------------------------------------------------------- parser
class _Parser:
    def __init__(self, text):
        self.text, self.toks, self.i = text, _tokenize(text), 0

    def _peek(self, k=0):
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else ("end", "", len(self.text))

    def _next(self):
        t = self._peek()
        self.i += 1
        return t

    def _err(self, msg):
        raise ConstraintError(f"{msg} (rule: {self.text!r})")

    def _expect(self, kind, value=None):
        t = self._next()
        if t[0] != kind or (value is not None and t[1] != value):
            self._err(f"expected {value or kind!r} but found {t[1] or 'end of rule'!r}")
        return t

    @staticmethod
    def _is_sym(tok, sym):
        return tok[0] == "sym" and tok[1] == sym

    @staticmethod
    def _kw(tok, word):
        return tok[0] == "id" and tok[1].upper() == word

    def _op(self, tok, strict_ok):
        raw = tok[1]
        if raw in ("<", ">") and not strict_ok:
            self._err(f"strict operator {raw!r} is not allowed: use '<=' or '>=' (the linear builders need non-strict bounds)")
        if raw in ("<>", "!=") and not strict_ok:
            self._err(f"operator {raw!r} is not supported in a constraint: use '<=', '>=' or '='")
        return "=" if raw == "==" else ("!=" if raw == "<>" else raw)

    # -------------------------------------------------------------------------------- rule
    def parse_rule(self):
        t0, t1 = self._peek(0), self._peek(1)
        if t0[0] == "id" and t0[1].upper() not in AGGS and t1[0] == "op":            # account-level
            self._next()
            op = self._op(self._next(), strict_ok=False)
            if op == "!=":
                self._err("operator '!=' is not supported in a constraint")
            value = self._value()
            self._end()
            return "account", None, op, None, (t0[1], op, value)
        lhs = self._side()
        t = self._next()
        if t[0] != "op":
            self._err(f"expected a comparison operator but found {t[1] or 'end of rule'!r}")
        op = self._op(t, strict_ok=False)
        rhs = self._side()
        self._end()
        if lhs.kind == "num" and rhs.kind == "num":
            self._err("a constraint needs at least one aggregate (SUM / AVG / WEIGHTED_AVG)")
        weighted = [s for s in (lhs, rhs) if s.kind in ("wavg", "qwavg")]
        for s in weighted:
            other = rhs if s is lhs else lhs
            if other.kind != "num":
                self._err("a WEIGHTED_AVG can only be compared with a number (a ratio of two decision-dependent "
                          "sums is not linear); write the linear form 'SUM(a) <= SUM(b) * k' instead")
            if s.mult != 1.0:
                self._err("a WEIGHTED_AVG cannot be multiplied; move the factor to the number on the other side")
        kind = "global" if "num" in (lhs.kind, rhs.kind) else "relative"
        return kind, lhs, op, rhs, None

    def _end(self):
        t = self._peek()
        if t[0] != "end":
            self._err(f"unexpected {t[1]!r} after the end of the rule")

    # -------------------------------------------------------------------------------- sides
    def _side(self):
        agg, num, op = None, 1.0, "*"
        while True:
            kind, val = self._factor()
            if kind == "agg":
                if agg is not None:
                    self._err("only one aggregate per side")
                if op == "/":
                    self._err("an aggregate cannot be a divisor")
                agg = val
            elif op == "*":
                num *= val
            else:
                if val == 0:
                    self._err("division by zero")
                num /= val
            nxt = self._peek()
            if nxt[0] == "sym" and nxt[1] in "*/":
                op = nxt[1]
                self._next()
                continue
            break
        if agg is None:
            return Term(kind="num", value=num)
        agg.mult = num
        return agg

    def _factor(self):
        t = self._peek()
        if t[0] == "sym" and t[1] in "+-":
            self._next()
            n = self._expect("num")
            return "num", (-1 if t[1] == "-" else 1) * float(n[1].replace("_", ""))
        if t[0] == "num":
            self._next()
            return "num", float(t[1].replace("_", ""))
        if t[0] == "id" and t[1].upper() in AGGS and self._is_sym(self._peek(1), "("):
            return "agg", self._aggregate()
        self._err(f"unexpected {t[1] or 'end of rule'!r}: expected a number or SUM / AVG / WEIGHTED_AVG")

    def _aggregate(self):
        name = self._next()[1].upper()
        self._expect("sym", "(")
        var = self._expect("id")[1]
        weight = None
        if name in WEIGHTED:
            self._expect("sym", ",")
            weight = self._expect("id")[1]
        self._expect("sym", ")")
        where = None
        if self._kw(self._peek(), "WHERE"):
            self._next()
            where = self._cond()
        kind = {"SUM": "sum", "AVG": "avg", "WEIGHTED_AVG": "wavg", "QUICK_WEIGHTED_AVG": "qwavg"}[name]
        return Term(kind=kind, var=var, weight=weight, where=where)

    # -------------------------------------------------------------------------------- WHERE conditions
    def _cond(self):
        node = self._and()
        while self._kw(self._peek(), "OR"):
            self._next()
            node = ("or", node, self._and())
        return node

    def _and(self):
        node = self._not()
        while self._kw(self._peek(), "AND"):
            self._next()
            node = ("and", node, self._not())
        return node

    def _not(self):
        if self._kw(self._peek(), "NOT"):
            self._next()
            return ("not", self._not())
        return self._atom()

    def _atom(self):
        t = self._peek()
        if t[0] == "sym" and t[1] == "(":
            self._next()
            node = self._cond()
            self._expect("sym", ")")
            return node
        if t[0] != "id":
            self._err(f"expected a column name in the WHERE clause but found {t[1] or 'end of rule'!r}")
        col = self._next()[1]
        n = self._next()
        if n[0] == "op":
            op = "=" if n[1] == "==" else ("!=" if n[1] == "<>" else n[1])
            return ("cmp", col, op, self._value())
        if self._kw(n, "IN"):
            self._expect("sym", "(")
            vals = [self._value()]
            while self._is_sym(self._peek(), ","):
                self._next()
                vals.append(self._value())
            self._expect("sym", ")")
            return ("in", col, vals)
        self._err(f"expected a comparison or IN after column {col!r}")

    def _value(self):
        t = self._next()
        if t[0] == "str":
            return t[1][1:-1].replace("''", "'")
        sign = 1
        if t[0] == "sym" and t[1] in "+-":
            sign = -1 if t[1] == "-" else 1
            t = self._next()
        if t[0] != "num":
            self._err(f"expected a number or a quoted string but found {t[1] or 'end of rule'!r}")
        x = float(t[1].replace("_", "")) * sign
        return int(x) if x == int(x) and "." not in t[1] and "e" not in t[1].lower() else x


def parse_rule(name, text, description=""):
    kind, lhs, op, rhs, account = _Parser(text).parse_rule()
    return ParsedRule(name=name, text=text, description=description, kind=kind, lhs=lhs, op=op, rhs=rhs,
                      account=account)


# ---------------------------------------------------------------------------------------------- conditions
def cond_columns(node):
    if node is None:
        return set()
    if node[0] in ("cmp", "in"):
        return {node[1]}
    return set().union(*(cond_columns(n) for n in node[1:] if isinstance(n, tuple)))


def eval_cond(node, frame):
    """Boolean numpy mask of a WHERE condition over the portfolio frame (NaN compares as False)."""
    kind = node[0]
    if kind == "and":
        return eval_cond(node[1], frame) & eval_cond(node[2], frame)
    if kind == "or":
        return eval_cond(node[1], frame) | eval_cond(node[2], frame)
    if kind == "not":
        return ~eval_cond(node[1], frame)
    col = frame[node[1]]
    try:
        if kind == "in":
            return col.isin(node[2]).to_numpy(bool)
        _, _, op, val = node
        res = {"=": col == val, "!=": col != val, "<=": col <= val, ">=": col >= val,
               "<": col < val, ">": col > val}[op]
        return np.asarray(res.fillna(False) if hasattr(res, "fillna") else res, dtype=bool)
    except TypeError as exc:
        raise ConstraintError(f"cannot compare column {node[1]!r} with {node[-1]!r}: {exc}") from exc


class Constraints:
    """A named set of rules. ``constraints.set(name=..., rule=..., description=...)``; syntax is checked immediately."""

    def __init__(self, description=""):
        self.description = description
        self._rules = {}

    def set(self, name, rule, description=""):
        try:
            self._rules[name] = parse_rule(name, rule, description)
        except ConstraintError as exc:
            raise ConstraintError(f"constraint '{name}': {exc}") from exc
        return self

    def get(self, name):
        return self._rules[name]

    def names(self):
        return list(self._rules)

    def copy(self):
        new = Constraints(self.description)
        new._rules = dict(self._rules)
        return new

    def __iter__(self):
        return iter(self._rules.values())

    def __len__(self):
        return len(self._rules)

    def __repr__(self):
        if not self._rules:
            return "Constraints(empty)"
        lines = [f"  [{r.kind:<8}] {r.name}: {r.text}" for r in self._rules.values()]
        head = f"Constraints({self.description!r}, {len(self._rules)} rules)" if self.description else \
            f"Constraints({len(self._rules)} rules)"
        return head + "\n" + "\n".join(lines)
