"""Report: a side-by-side comparison of scenarios registered in a project."""
import numpy as np
import pandas as pd


class Report:
    def __init__(self, project, names, reference=None):
        self.project, self.names, self.reference = project, names, reference

    def scenario_summary(self):
        """One row per scenario: expected profit (full portfolio), per-account average, change against the reference,
        and the approval rate when the decision variable is the usual approve / reject."""
        proj = self.project
        var = proj.variables.names[0] if len(proj.variables) == 1 else None
        approve = proj.variables.domain(var)[-1] if var else None
        rows = {}
        for n in self.names:
            r = proj._results[n]
            if r.selected is None:
                raise ValueError(f"scenario '{n}' has not been optimized")
            row = {"flow": getattr(r, "flow", "policy"), "accounts": len(r.selected),
                   "expected profit, full portfolio (bn VND)": r.total_portfolio / 1e9,
                   "average per account (VND)": r.score}
            if var and var in r.selected:
                row["approval rate"] = float((r.selected[var] == approve).mean())
            rows[n] = row
        out = pd.DataFrame(rows).T
        if self.reference:
            ref = out.loc[self.reference, "expected profit, full portfolio (bn VND)"]
            col = out["expected profit, full portfolio (bn VND)"].astype(float)
            out["vs reference (bn VND)"] = col - ref
            out["vs reference (%)"] = np.where(ref != 0, (col - ref) / abs(ref), np.nan)
        return out

    def __repr__(self):
        return f"Report({self.names}, reference={self.reference!r})"
