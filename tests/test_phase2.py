"""Phase 2 tests: the rule language, the constrained flow and the reports.

The solver is checked against an INDEPENDENT brute force: every 0/1 (or 0/1/2) assignment of a small portfolio is
enumerated and the aggregates are recomputed with plain numpy formulas, without any code of the package.
Run with:  python -m unittest discover -s tests -v
"""
import itertools
import sys
import tempfile
import textwrap
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pydo as do                                                                   # noqa: E402
from pydo.rules import parse_rule                                                    # noqa: E402

FLATTEN = textwrap.dedent("""
    def flatten(obj):
        obj.p_amount = obj.account["AMOUNT"]
        obj.p_pd = obj.account["PD"]
        obj.a_approve = obj.decision["APPROVE"]
""")
ECON = textwrap.dedent("""
    def econ(obj):
        obj.v_exp_profit = obj.a_approve * obj.p_amount * (0.2 - 0.7 * obj.p_pd)
        obj.v_approved = obj.a_approve
        obj.v_amount = obj.a_approve * obj.p_amount
        obj.v_pd = obj.p_pd
        obj.v_ok = 0 if (obj.a_approve == 1 and obj.p_pd >= 0.45) else 1
""")
LIMIT_ECON = textwrap.dedent("""
    def flatten(obj):
        obj.p_amount = obj.account["AMOUNT"]
        obj.p_pd = obj.account["PD"]
        obj.p_limit = obj.decision["LIMIT"]

    def econ(obj):
        obj.v_exp_profit = obj.p_limit * obj.p_amount * (0.2 - 0.7 * obj.p_pd)
        obj.v_amount = obj.p_limit * obj.p_amount
        obj.v_approved = 1 if obj.p_limit > 0 else 0
        obj.v_pd = obj.p_pd
""")


def make_frame(seed, n=12):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ID": [f"I{i:02d}" for i in range(n)],
        "SEG": ["a"] * (n // 2) + ["b"] * (n - n // 2),
        "BAND": ([0, 0, 1, 1, 2, 2] * n)[:n],
        "TENOR": rng.choice(["S", "L"], n),
        "AMOUNT": rng.integers(100, 1000, n),
        "PD": rng.uniform(0.05, 0.7, n).round(3),
        "BAU": 1,
    })


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.files = []

    def write(self, name, text):
        p = Path(self.tmp.name, name)
        p.write_text(text)
        return p

    def project(self, frame, funcs=None, treatments=(0, 1), var="APPROVE", sampling=1.0, outputs=("v_exp_profit",)):
        funcs = funcs or [self.write("flatten.py", FLATTEN), self.write("econ.py", ECON)]
        pr = do.Project("p")
        pr.facts = do.Facts()
        pr.variables = do.Variables().set(var, list(treatments))
        pr.portfolio = do.Portfolio(frame, "ID", random_sampling=sampling, stratify_columns=["SEG"])
        pr.functions = do.Functions(funcs)
        pr.objective, pr.direction = "v_exp_profit", "maximize"
        pr.output_variables = list(outputs)
        return pr


class TestParser(unittest.TestCase):
    def test_valid_rules_and_kinds(self):
        cases = {
            "AVG(v_approved) >= 0.45": "global",
            "WEIGHTED_AVG(v_pd, v_amount) <= 0.22": "global",
            "SUM(v_amount) WHERE (TENOR = 'L') <= SUM(v_amount) * 0.45": "relative",
            "AVG(v_amount) <= 350_000 / 17976": "global",
            "0.45 * SUM(v_amount) >= SUM(v_amount) WHERE (TENOR = 'L')": "relative",
            "AVG(v_approved) WHERE (SEG = 'a') AND (BAND = 0) >= AVG(v_approved) WHERE (SEG = 'a') AND (BAND = 1)": "relative",
            "SUM(v_amount) WHERE SEG IN ('a', 'b') AND NOT BAND = 2 <= 1e9": "global",
            "v_ok = 1": "account",
            "quick_weighted_avg(v_pd, v_amount) <= 0.3": "global",
        }
        for text, kind in cases.items():
            self.assertEqual(parse_rule("r", text).kind, kind, text)

    def test_rejected_rules(self):
        bad = ["AVG(v_approved) > 0.45", "AVG(v_approved) < 0.45", "v_ok > 1", "AVG(v_a) != 1",
               "0.4 >= 0.2", "AVG(v_a) >=", "AVG(v_a) 0.4", "AVG(v_a) >= 0.4 junk", "AVG(v_a, v_b) >= 1",
               "WEIGHTED_AVG(v_a) <= 1", "WEIGHTED_AVG(v_a, v_b) <= SUM(v_c)", "2 * WEIGHTED_AVG(v_a, v_b) <= 1",
               "SUM(v_a) + SUM(v_b) <= 1", "AVG(v_a) @ 1", "SUM(v_a) WHERE (X =) <= 1", "SUM(v_a) / SUM(v_b) <= 1"]
        for text in bad:
            with self.assertRaises(do.ConstraintError, msg=text):
                parse_rule("r", text)

    def test_constraints_container_names_the_bad_rule(self):
        c = do.Constraints("d").set("C1", "AVG(v_a) >= 0.5")
        with self.assertRaises(do.ConstraintError) as cm:
            c.set("C2 broken", "AVG(v_a) > 0.5")
        self.assertIn("C2 broken", str(cm.exception))
        self.assertEqual(c.names(), ["C1"])
        self.assertIn("C1", repr(c))


def brute_force(frame, risk=0.30, approval=0.5, long_share=0.4, budget=250.0):
    """Independent enumeration of every approve / reject vector."""
    n = len(frame)
    amount, pd_ = frame["AMOUNT"].to_numpy(float), frame["PD"].to_numpy(float)
    long_ = (frame["TENOR"] == "L").to_numpy()
    sa0 = ((frame["SEG"] == "a") & (frame["BAND"] == 0)).to_numpy()
    sa1 = ((frame["SEG"] == "a") & (frame["BAND"] == 1)).to_numpy()
    best, best_x = None, None
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, float)
        vol = float((amount * x).sum())
        if vol <= 0 or x.mean() < approval - 1e-12:
            continue
        if float((pd_ * amount * x).sum()) / vol > risk + 1e-9:
            continue
        if float((amount * x)[long_].sum()) > long_share * vol + 1e-9:
            continue
        if x[sa0].mean() < x[sa1].mean() - 1e-12:
            continue
        if vol / n > budget + 1e-9:
            continue
        obj = float((amount * (0.2 - 0.7 * pd_) * x).sum())
        if best is None or obj > best + 1e-9:
            best, best_x = obj, x
    return best, best_x


RULES = {
    "C1 risk":     "WEIGHTED_AVG(v_pd, v_amount) <= 0.30",
    "C2 approval": "AVG(v_approved) >= 0.5",
    "C4 long":     "SUM(v_amount) WHERE (TENOR = 'L') <= SUM(v_amount) * 0.4",
    "C7 monotone": "AVG(v_approved) WHERE (SEG = 'a') AND (BAND = 0) >= AVG(v_approved) WHERE (SEG = 'a') AND (BAND = 1)",
    "C3 budget":   "AVG(v_amount) <= 250",
}


class TestSolverAgainstBruteForce(Base):
    def constraints(self):
        c = do.Constraints("all")
        for k, v in RULES.items():
            c.set(name=k, rule=v)
        return c

    def test_binary_decisions_match_enumeration(self):
        checked = 0
        for seed in range(8):
            frame = make_frame(seed)
            best, best_x = brute_force(frame)
            sc = self.project(frame).scenario("c", flow="constrained", constraints=self.constraints())
            sc.build()
            if best is None:
                with self.assertRaises(do.InfeasibleModelError):
                    sc.optimize(gap=0.0)
                continue
            sc.optimize(gap=0.0)
            self.assertAlmostEqual(sc.total, best, places=6, msg=f"seed {seed}")
            self.assertTrue(sc.constraint_report()["satisfied"].all(), f"seed {seed}")
            checked += 1
        self.assertGreaterEqual(checked, 3)                                # the test must exercise real optima

    def test_three_treatments_match_enumeration(self):
        frame = make_frame(3, n=7)
        f = self.write("limit.py", LIMIT_ECON)
        pr = self.project(frame, funcs=[f], treatments=(0, 1, 2), var="LIMIT", outputs=("v_exp_profit", "v_amount"))
        c = do.Constraints().set("budget", "AVG(v_amount) <= 450").set("risk", "WEIGHTED_AVG(v_pd, v_amount) <= 0.30") \
                            .set("approval", "AVG(v_approved) >= 0.4")
        sc = pr.scenario("c", flow="constrained", constraints=c).build()
        amount, pd_ = frame["AMOUNT"].to_numpy(float), frame["PD"].to_numpy(float)
        best = None
        for lim in itertools.product([0, 1, 2], repeat=len(frame)):
            lim = np.array(lim, float)
            vol = float((lim * amount).sum())
            if vol <= 0 or (lim > 0).mean() < 0.4 - 1e-12 or vol / len(frame) > 450 + 1e-9:
                continue
            if float((pd_ * lim * amount).sum()) / vol > 0.30 + 1e-9:
                continue
            obj = float((lim * amount * (0.2 - 0.7 * pd_)).sum())
            best = obj if best is None or obj > best else best
        self.assertIsNotNone(best)
        sc.optimize(gap=0.0)
        self.assertAlmostEqual(sc.total, best, places=6)
        self.assertEqual(sc.preflight(verbose=False)["kind"].tolist().count("relative"), 0)
        self.assertTrue(sc.constraint_report()["satisfied"].all())

    def test_ordering_simulation_constrained_unconstrained(self):
        frame = make_frame(1)
        pr = self.project(frame)
        c = self.constraints()
        bau = pr.scenario("bau", flow="simulation", variable_to_decision={"APPROVE": "BAU"}, constraints=c).build().optimize()
        unc = pr.scenario("unc", flow="unconstrained", constraints=c).build().optimize()
        con = pr.scenario("con", flow="constrained", constraints=c).build().optimize(gap=0.0)
        self.assertLessEqual(bau.total, con.total + 1e-9)
        self.assertLessEqual(con.total, unc.total + 1e-9)


class TestConstraintBehaviour(Base):
    def test_infeasible_raises_with_guidance(self):
        frame = make_frame(0)
        c = do.Constraints().set("a", "AVG(v_approved) >= 0.9").set("r", "WEIGHTED_AVG(v_pd, v_amount) <= 0.06")
        sc = self.project(frame).scenario("c", flow="constrained", constraints=c).build()
        with self.assertRaises(do.InfeasibleModelError) as cm:
            sc.optimize()
        self.assertIn("one at a time", str(cm.exception))

    def test_where_on_a_calculated_variable_is_rejected(self):
        c = do.Constraints().set("w", "SUM(v_amount) WHERE (v_approved = 1) <= 100")
        with self.assertRaises(do.ConstraintError) as cm:
            self.project(make_frame(0)).scenario("c", flow="constrained", constraints=c).build()
        self.assertIn("RAW", str(cm.exception))

    def test_empty_segment_is_an_error(self):
        c = do.Constraints().set("e", "AVG(v_approved) WHERE (SEG = 'zzz') >= 0.1")
        with self.assertRaises(do.ConstraintError) as cm:
            self.project(make_frame(0)).scenario("c", flow="constrained", constraints=c).build()
        self.assertIn("zero accounts", str(cm.exception))

    def test_unknown_variable_is_rejected(self):
        c = do.Constraints().set("u", "AVG(v_nope) >= 0.1")
        with self.assertRaises(do.ConstraintError):
            self.project(make_frame(0)).scenario("c", flow="constrained", constraints=c).build()

    def test_quick_weighted_avg_needs_positive_weights(self):
        c = do.Constraints().set("q", "QUICK_WEIGHTED_AVG(v_pd, v_amount) <= 0.5")
        sc = self.project(make_frame(0)).scenario("c", flow="constrained", constraints=c).build()
        with self.assertRaises(do.ConstraintError) as cm:
            sc.optimize()
        self.assertIn("strictly positive", str(cm.exception))

    def test_account_level_rule_prunes_treatments(self):
        frame = make_frame(2)
        frame["PD"] = [0.05, 0.1, 0.5, 0.6, 0.05, 0.1, 0.5, 0.6, 0.05, 0.1, 0.5, 0.6]       # half are >= 0.45
        c = do.Constraints().set("ok", "v_ok = 1")
        sc = self.project(frame).scenario("c", flow="constrained", constraints=c).build()
        table = sc.preflight(verbose=False)
        sc.optimize()
        approved = sc.export().set_index("ID")["APPROVE"]
        self.assertTrue((approved[frame.set_index("ID")["PD"] >= 0.45] == 0).all())
        self.assertEqual(sc.model.pruned_pairs, 6)
        self.assertEqual(int(sc.constraint_report().loc[0, "violation"]), 0)
        self.assertTrue(table.empty)                                                 # account-level rules add no matrix row

    def test_report_on_a_fixed_policy_flags_violations(self):
        frame = pd.DataFrame({"ID": list("ABCD"), "SEG": "x", "BAND": 0, "TENOR": "S",
                              "AMOUNT": [100, 100, 100, 100], "PD": [0.1, 0.2, 0.5, 0.7], "BAU": 1})
        c = do.Constraints().set("risk", "WEIGHTED_AVG(v_pd, v_amount) <= 0.3").set("appr", "AVG(v_approved) >= 0.5")
        sc = self.project(frame).scenario("bau", flow="simulation", decisions={"APPROVE": 1}, constraints=c).build().optimize()
        rep = sc.constraint_report().set_index("constraint")
        self.assertAlmostEqual(rep.loc["risk", "lhs"], 0.375)                        # (10+20+50+70)/400
        self.assertAlmostEqual(rep.loc["risk", "slack"], -0.075)
        self.assertFalse(rep.loc["risk", "satisfied"])
        self.assertTrue(rep.loc["appr", "satisfied"])
        self.assertAlmostEqual(rep.loc["appr", "slack"], 0.5)

    def test_binding_flag_and_solver_report(self):
        frame = make_frame(4)
        c = do.Constraints().set("approval", "AVG(v_approved) >= 0.75")
        sc = self.project(frame).scenario("c", flow="constrained", constraints=c).build(n_jobs=-1)
        sc.preflight(verbose=False)
        sc.optimize(solver="highs", time_limit=60, gap=0.0, n_jobs=8, verbose=False)
        rep = sc.constraint_report()
        self.assertTrue(rep.loc[0, "binding"])                                       # 9 of 12 = exactly 0.75
        sr = sc.solver_report()
        self.assertEqual(sr["status"], "optimal")
        self.assertEqual(sr["variables"], 12)
        self.assertLessEqual(sr["gap_achieved"], 1e-9)

    def test_coefficient_report_warns_on_a_fragile_matrix(self):
        frame = make_frame(0)
        frame["AMOUNT"] = frame["AMOUNT"] * 10_000_000                                # raw VND magnitudes
        c = do.Constraints().set("approval", "AVG(v_approved) >= 0.5")
        sc = self.project(frame).scenario("c", flow="constrained", constraints=c).build()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = sc.coefficient_report()
        self.assertTrue(any("fragile" in str(x.message) for x in w))
        self.assertIn("WHOLE MODEL", rep["row"].tolist())

    def test_absolute_budget_survives_sampling_when_written_per_account(self):
        frame = make_frame(5, n=60)
        frame["SEG"] = ["a", "b"] * 30
        frame["BAND"] = [0, 1, 2] * 20
        n_full = len(frame)
        c = do.Constraints().set("budget", f"AVG(v_amount) <= {20000 / n_full}")
        full = self.project(frame).scenario("f", flow="constrained", constraints=c).build().optimize(gap=0.0)
        part = self.project(frame, sampling=0.5).scenario("p", flow="constrained", constraints=c).build().optimize(gap=0.0)
        for sc in (full, part):
            self.assertTrue(sc.constraint_report()["satisfied"].all())
        self.assertEqual(len(part.selected), 30)
        self.assertAlmostEqual(part.total_portfolio / part.total, 2.0)               # the sample is scaled back to 100%


class TestWithConstraints(Base):
    def test_reuses_pairs_and_matches_a_fresh_scenario(self):
        frame = make_frame(4)
        pr = self.project(frame)
        loose = do.Constraints().set("approval", "AVG(v_approved) >= 0.5")
        tight = do.Constraints().set("approval", "AVG(v_approved) >= 0.75")
        base = pr.scenario("base", flow="constrained", constraints=loose).build()
        again = base.with_constraints(tight)
        self.assertIs(again.pairs, base.pairs)                                       # nothing was recomputed
        again.optimize(gap=0.0)
        fresh = pr.scenario("fresh", flow="constrained", constraints=tight).build().optimize(gap=0.0)
        self.assertAlmostEqual(again.total, fresh.total, places=6)
        self.assertTrue(again.constraint_report()["satisfied"].all())
        base.optimize(gap=0.0)
        self.assertGreaterEqual(base.total, again.total - 1e-9)                       # a tighter rule cannot help

    def test_rejects_unevaluated_variables_and_simulations(self):
        pr = self.project(make_frame(4))
        base = pr.scenario("base", flow="constrained", constraints=do.Constraints().set("a", "AVG(v_approved) >= 0.5")).build()
        with self.assertRaises(do.ProjectError):
            base.with_constraints(do.Constraints().set("x", "AVG(v_pd) <= 0.5"))     # v_pd was never evaluated here
        sim = pr.scenario("s", flow="simulation", decisions={"APPROVE": 1}).build()
        with self.assertRaises(do.ProjectError):
            sim.with_constraints(do.Constraints().set("a", "AVG(v_approved) >= 0.5"))


if __name__ == "__main__":
    unittest.main()
