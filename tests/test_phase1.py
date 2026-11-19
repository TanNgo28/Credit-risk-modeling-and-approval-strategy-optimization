"""Phase 1 tests: Portfolio, Facts, Variables, Functions, Project, simulation and unconstrained flows.

Hand-computed economics: expected profit of an approved account = amount * (margin - (margin + lgd) * pd)
with margin = 0.2 and lgd = 0.5, i.e.  amount * (0.2 - 0.7 * pd).  Run with:  python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pydo as do                                              # noqa: E402

FLATTEN = textwrap.dedent("""
    def flatten(obj):
        obj.p_amount = obj.account["AMOUNT"]
        obj.p_pd = obj.account["PD"]
        obj.a_approve = obj.decision["APPROVE"]
        obj.f_lgd = obj.facts.get("lgd")
        obj.f_margin = obj.facts.get("margin")
""")
ECON = textwrap.dedent("""
    def econ(obj):
        good = obj.p_amount * obj.f_margin
        bad = -obj.p_amount * obj.f_lgd
        obj.v_exp_profit = obj.a_approve * ((1 - obj.p_pd) * good + obj.p_pd * bad)
        obj.v_approved = obj.a_approve
        obj.v_amount = obj.a_approve * obj.p_amount
        obj.v_label = "not a number"
""")

FRAME = pd.DataFrame({
    "ID":  list("ABCDEF"),
    "SEG": ["x", "x", "x", "y", "y", "y"],
    "AMOUNT": [100, 200, 300, 400, 500, 600],
    "PD":  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "BAU": [1] * 6,
})
EXPECTED = {"A": 13.0, "B": 12.0, "C": -3.0, "D": -32.0, "E": -75.0, "F": -132.0}   # hand calculation


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f1, self.f2 = Path(self.tmp.name, "flatten.py"), Path(self.tmp.name, "econ.py")
        self.f1.write_text(FLATTEN), self.f2.write_text(ECON)

    def make_project(self, frame=FRAME, sampling=1.0, lgd=0.5):
        facts = do.Facts().set("lgd", lgd, "loss given default").set("margin", 0.2)
        variables = do.Variables().set("APPROVE", [0, 1])
        project = do.Project("test")
        project.facts, project.variables = facts, variables
        project.portfolio = do.Portfolio(frame, key_column="ID", random_sampling=sampling, stratify_columns=["SEG"])
        project.functions = do.Functions([self.f1, self.f2])
        project.objective, project.direction = "v_exp_profit", "maximize"
        project.output_variables = ["v_approved", "v_amount", "v_exp_profit"]
        return project


class TestBuildingBlocks(Base):
    def test_portfolio_validation(self):
        with self.assertRaises(ValueError):
            do.Portfolio(pd.concat([FRAME, FRAME.iloc[:1]]), "ID")                 # duplicate key
        with self.assertRaises(ValueError):
            do.Portfolio(FRAME.assign(ID=[None] + list("BCDEF")), "ID")            # null key
        with self.assertRaises(ValueError):
            do.Portfolio(FRAME, "ID", random_sampling=0)
        with self.assertRaises(ValueError):
            do.Portfolio(FRAME, "ID", stratify_columns=["NOPE"])

    def test_stratified_sampling(self):
        big = pd.DataFrame({"ID": range(1000), "SEG": ["a"] * 900 + ["b"] * 90 + ["c"] * 10})
        p = do.Portfolio(big, "ID", random_sampling=0.2, stratify_columns=["SEG"])
        counts = p.data["SEG"].value_counts().to_dict()
        self.assertEqual(counts, {"a": 180, "b": 18, "c": 2})                     # proportional, every stratum kept
        self.assertAlmostEqual(p.scale, 1000 / 200)
        p2 = do.Portfolio(big, "ID", random_sampling=0.2, stratify_columns=["SEG"])
        self.assertTrue(p.data.equals(p2.data))                                   # deterministic
        tiny = do.Portfolio(big, "ID", random_sampling=0.001, stratify_columns=["SEG"])
        self.assertEqual(set(tiny.data["SEG"]), {"a", "b", "c"})                  # a stratum never vanishes

    def test_facts(self):
        facts = do.Facts().set("a", 1)
        self.assertEqual(facts.get("a"), 1)
        self.assertEqual(facts.get("zzz", 7), 7)
        with self.assertRaises(do.UnknownFactError) as cm:
            facts.get("zzz")
        self.assertIn("zzz", str(cm.exception))

    def test_variables_treatments(self):
        v = do.Variables().set("A", [0, 1]).set("B", ["x", "y", "z"])
        self.assertEqual(len(v.treatments()), 6)
        self.assertEqual(v.treatments()[0], {"A": 0, "B": "x"})
        with self.assertRaises(ValueError):
            do.Variables().set("A", [1, 1])

    def test_functions_load_in_file_order_and_report_failures(self):
        fn = do.Functions([self.f1, self.f2])
        self.assertEqual(fn.labels(), ["flatten.py:flatten", "econ.py:econ"])
        bad = do.Functions([lambda obj: obj.nothing_here])
        obj = type("O", (), {})()
        with self.assertRaises(do.PyDOError) as cm:
            bad.run(obj, key="ACC1")
        self.assertIn("ACC1", str(cm.exception))


class TestProject(Base):
    def test_assignment_order(self):
        p = do.Project("x")
        with self.assertRaises(do.ProjectError):
            p.functions = do.Functions([self.f1, self.f2])                          # nothing assigned yet
        with self.assertRaises(do.ProjectError):
            p.objective = "v_exp_profit"                                            # functions not assigned
        project = self.make_project()
        self.assertIn("v_exp_profit", project.calculated_variables)
        self.assertNotIn("v_label", project.calculated_variables)                   # non numeric is not a variable
        with self.assertRaises(do.ProjectError):
            project.objective = "v_label"
        with self.assertRaises(do.ProjectError):
            project.objective = "v_does_not_exist"
        with self.assertRaises(do.ProjectError):
            project.output_variables = ["v_nope"]

    def test_scenario_needs_objective(self):
        p = do.Project("x")
        with self.assertRaises(do.ProjectError):
            p.scenario("s", flow="unconstrained")


class TestFlows(Base):
    def test_simulation_matches_hand_calculation(self):
        project = self.make_project()
        bau = project.scenario("BAU", flow="simulation", variable_to_decision={"APPROVE": "BAU"})
        bau.build().optimize()
        out = bau.export().set_index("ID")
        for acc, expected in EXPECTED.items():
            self.assertAlmostEqual(out.loc[acc, "v_exp_profit"], expected, places=9)
        self.assertAlmostEqual(bau.total, sum(EXPECTED.values()))                   # -217
        self.assertAlmostEqual(bau.score, sum(EXPECTED.values()) / 6)

    def test_simulation_constant_decisions_and_reject_all(self):
        project = self.make_project()
        none = project.scenario("none", flow="simulation", decisions={"APPROVE": 0})
        none.build().optimize()
        self.assertEqual(none.total, 0.0)
        everyone = project.scenario("all", flow="simulation", decisions={"APPROVE": 1})
        everyone.build().optimize()
        self.assertAlmostEqual(everyone.total, -217.0)

    def test_simulation_argument_rules(self):
        project = self.make_project()
        with self.assertRaises(do.ProjectError):                                    # both
            project.scenario("s", flow="simulation", decisions={"APPROVE": 1}, variable_to_decision={"APPROVE": "BAU"})
        with self.assertRaises(do.ProjectError):                                    # neither
            project.scenario("s", flow="simulation")
        with self.assertRaises(do.ProjectError):                                    # column absent
            project.scenario("s", flow="simulation", variable_to_decision={"APPROVE": "MISSING"})
        with self.assertRaises(do.ProjectError):                                    # outside the domain
            project.scenario("s", flow="simulation", decisions={"APPROVE": 2})
        with self.assertRaises(do.ProjectError):                                    # simulation-only argument
            project.scenario("s", flow="unconstrained", decisions={"APPROVE": 1})
        bad = self.make_project(FRAME.assign(BAU=[1, 1, 1, 1, 1, 7]))
        sim = bad.scenario("s", flow="simulation", variable_to_decision={"APPROVE": "BAU"})
        with self.assertRaises(do.ProjectError):                                    # a value outside the domain
            sim.build()
        with self.assertRaises(do.ProjectError):                                    # constrained needs constraints
            project.scenario("s", flow="constrained")

    def test_unconstrained_approves_exactly_the_positive_expected_profits(self):
        project = self.make_project()
        sc = project.scenario("ceiling", flow="unconstrained")
        sc.build().optimize()
        out = sc.export().set_index("ID")
        self.assertEqual(out["APPROVE"].to_dict(), {"A": 1, "B": 1, "C": 0, "D": 0, "E": 0, "F": 0})
        self.assertAlmostEqual(sc.total, 25.0)
        self.assertEqual(len(sc.pairs), 12)                                          # 6 accounts x 2 treatments
        self.assertEqual(len(out), 6)                                                # one selected treatment each

    def test_ordering_simulation_le_unconstrained_and_minimize(self):
        project = self.make_project()
        sim = project.scenario("bau", flow="simulation", decisions={"APPROVE": 1}).build().optimize()
        top = project.scenario("top", flow="unconstrained").build().optimize()
        self.assertLessEqual(sim.total, top.total)
        project.direction = "minimize"                                               # picks the worst treatment per account
        worst = project.scenario("worst", flow="unconstrained").build().optimize()
        # per account min(0, v): A->0, B->0, C->-3, D->-32, E->-75, F->-132
        self.assertAlmostEqual(worst.total, -242.0)
        self.assertLessEqual(worst.total, sim.total)

    def test_tie_goes_to_the_first_treatment(self):
        frame = FRAME.copy()
        frame.loc[0, "AMOUNT"] = 0                                                   # expected profit exactly 0 either way
        sc = self.make_project(frame).scenario("t", flow="unconstrained").build().optimize()
        self.assertEqual(sc.export().set_index("ID").loc["A", "APPROVE"], 0)          # reject, the conservative choice

    def test_facts_drive_the_result_and_export_writes_csv(self):
        project = self.make_project()
        project.facts.set("lgd", 0.0)                                                # no loss at all: everything is profitable
        sc = project.scenario("u", flow="unconstrained").build().optimize()
        self.assertTrue((sc.export()["APPROVE"] == 1).all())
        path = os.path.join(self.tmp.name, "out.csv")
        sc.export(path)
        back = pd.read_csv(path)
        self.assertEqual(list(back.columns), ["ID", "APPROVE", "v_exp_profit", "v_approved", "v_amount"])

    def test_sampling_scales_the_total(self):
        big = pd.DataFrame({"ID": [f"I{i}" for i in range(400)], "SEG": ["x"] * 400,
                            "AMOUNT": 100, "PD": 0.1, "BAU": 1})                    # every account worth exactly 13
        project = self.make_project(big, sampling=0.25)
        sc = project.scenario("s", flow="simulation", decisions={"APPROVE": 1}).build().optimize()
        self.assertEqual(len(sc.selected), 100)
        self.assertAlmostEqual(sc.score, 13.0)
        self.assertAlmostEqual(sc.total_portfolio, 13.0 * 400)                       # 25% sample scaled back to 100%

    def test_optimize_requires_build(self):
        sc = self.make_project().scenario("u", flow="unconstrained")
        with self.assertRaises(do.ProjectError):
            sc.optimize()


if __name__ == "__main__":
    unittest.main()
