"""Phase 5 tests: DecisionTree distillation, apply_policy and project.report.

The leaf-level optimum is checked against an INDEPENDENT enumeration of every assignment of the leaves, with the
constraints recomputed from plain numpy formulas.  Run with:  python -m unittest discover -s tests -v
"""
import itertools
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pydo as do                                                                    # noqa: E402

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
""")


def make_frame(seed=0, n=600):
    rng = np.random.default_rng(seed)
    band = rng.integers(0, 6, n)
    seg = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    pdv = np.clip(0.05 + 0.12 * band + rng.normal(0, 0.03, n) + 0.04 * (seg == "c"), 0.01, 0.95).round(4)
    return pd.DataFrame({"ID": [f"I{i:04d}" for i in range(n)], "BAND": band, "SEG": seg,
                         "AMOUNT": rng.integers(100, 1000, n), "PD": pdv})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = [Path(self.tmp.name, "flatten.py"), Path(self.tmp.name, "econ.py")]
        self.f[0].write_text(FLATTEN), self.f[1].write_text(ECON)

    def project(self, frame, treatments=(0, 1)):
        pr = do.Project("p")
        pr.facts = do.Facts()
        pr.variables = do.Variables().set("APPROVE", list(treatments))
        pr.portfolio = do.Portfolio(frame, "ID")
        pr.functions = do.Functions(self.f)
        pr.objective, pr.direction = "v_exp_profit", "maximize"
        pr.output_variables = ["v_exp_profit", "v_approved", "v_amount"]
        return pr

    def solved(self, frame=None, risk=0.35, approval=0.35, seg_c=0.10):
        frame = make_frame() if frame is None else frame
        pr = self.project(frame)
        cons = (do.Constraints().set("risk", f"WEIGHTED_AVG(v_pd, v_amount) <= {risk}")
                .set("approval", f"AVG(v_approved) >= {approval}")
                .set("seg_c", f"SUM(v_approved) WHERE (SEG = 'c') >= SUM(v_approved) * {seg_c}"))
        sc = pr.scenario("Constrained policy", flow="constrained", constraints=cons).build().optimize(gap=0.0)
        return pr, sc, frame, cons


def independent_feasible(x, frame, risk, approval, seg_c):
    amount, pdv = frame["AMOUNT"].to_numpy(float), frame["PD"].to_numpy(float)
    vol = float((amount * x).sum())
    if vol <= 0 or x.mean() < approval - 1e-12:
        return False
    if float((pdv * amount * x).sum()) / vol > risk + 1e-9:
        return False
    c = (frame["SEG"] == "c").to_numpy()
    return float(x[c].sum()) >= seg_c * float(x.sum()) - 1e-9


class TestTree(Base):
    def test_leaf_optimum_matches_enumeration(self):
        pr, sc, frame, _ = self.solved()
        tree = do.DecisionTree(max_depth=3, max_splits=4, min_weight=20).fit(sc, ["BAND", "SEG", "AMOUNT"], gap=0.0)
        L = tree.n_leaves
        self.assertLessEqual(L, 5)
        obj = (frame["AMOUNT"] * (0.2 - 0.7 * frame["PD"])).to_numpy()
        best = None
        for acts in itertools.product([0, 1], repeat=L):
            x = np.array(acts)[tree.leaf_of_].astype(float)
            if independent_feasible(x, frame, 0.35, 0.35, 0.10):
                v = float((obj * x).sum())
                best = v if best is None or v > best else best
        self.assertIsNotNone(best)
        self.assertAlmostEqual(tree.info["objective_total"], best, places=6)
        self.assertAlmostEqual(tree.policy_.total, best, places=6)
        self.assertTrue(tree.policy_.constraint_report()["satisfied"].all())            # same evaluator, rules hold

    def test_objective_report_and_loss_by_leaf(self):
        pr, sc, frame, _ = self.solved()
        tree = do.DecisionTree(max_depth=3, min_weight=20).fit(sc, ["BAND", "SEG", "AMOUNT"], gap=0.0)
        rep = tree.objective_report()
        self.assertLessEqual(rep.iloc[2, 0], rep.iloc[0, 0] + 1e-9)                      # never beats the account-level optimum
        self.assertEqual(int(rep.iloc[2]["rules violated"]), 0)
        self.assertAlmostEqual(rep.iloc[2]["% of account-level optimum"], rep.iloc[2, 0] / rep.iloc[0, 0])
        loss = tree.leaves_["loss_vs_source_bn"].sum() * 1e9
        self.assertAlmostEqual(loss, (rep.iloc[0, 0] - rep.iloc[2, 0]) * 1e9, places=3)  # leaf losses add up to the total loss
        self.assertEqual(len(tree.performance()), 2)

    def test_rules_partition_and_json_round_trip(self):
        pr, sc, frame, _ = self.solved()
        tree = do.DecisionTree(max_depth=3, min_weight=20).fit(sc, ["BAND", "SEG", "AMOUNT"], gap=0.0)
        self.assertEqual(set(tree.selected_features) - set(tree.attempted_features), set())
        self.assertTrue(all(isinstance(str(r), str) and "THEN" in str(r) for r in tree.rules_))
        leaf = tree._leaf_index(frame)
        self.assertEqual(sorted(set(leaf)), list(range(tree.n_leaves)))                  # a partition: every account in one leaf
        self.assertTrue((leaf == tree.leaf_of_).all())
        path = os.path.join(self.tmp.name, "tree.json")
        tree.save(path)
        with self.assertRaises(FileExistsError):
            tree.save(path)
        tree.save(path, overwrite=True)
        again = do.DecisionTree.load(path)
        self.assertTrue((again.predict(frame) == tree.predict(frame)).all())             # rules alone reproduce the policy
        self.assertEqual([r.description for r in again.rules_], [r.description for r in tree.rules_])
        self.assertIn("CASE", tree.to_sql())

    def test_integer_rules_read_naturally(self):
        pr, sc, frame, _ = self.solved()
        tree = do.DecisionTree(max_depth=2, min_weight=20).fit(sc, ["BAND"], gap=0.0)
        text = " ".join(r.description for r in tree.rules_)
        self.assertRegex(text, r"BAND (<=|>=) \d+|\d+ <= BAND <= \d+")                  # no "BAND <= 2.5"
        self.assertNotRegex(text, r"\d\.5")

    def test_infeasible_when_the_partition_is_too_coarse(self):
        pr, sc, frame, _ = self.solved()
        with self.assertRaises(do.InfeasibleModelError) as cm:
            do.DecisionTree(max_depth=3, max_splits=0, min_weight=20).fit(sc, ["BAND"], gap=0.0)    # a single leaf
        self.assertIn("deeper tree", str(cm.exception))

    def test_input_validation(self):
        pr, sc, frame, cons = self.solved()
        with self.assertRaises(do.ProjectError):
            do.DecisionTree().fit(sc, ["v_amount"])                                       # a calculated variable
        with self.assertRaises(do.ProjectError):
            do.DecisionTree().fit(sc, ["NOPE"])
        with self.assertRaises(NotImplementedError):
            do.DecisionTree().fit(sc, ["BAND"], constrained_growth="other")
        unc = pr.scenario("u", flow="unconstrained").build().optimize()
        with self.assertRaises(do.ProjectError):
            do.DecisionTree().fit(unc, ["BAND"])                                          # needs a solved constrained scenario
        with self.assertRaises(do.ProjectError):
            do.DecisionTree().predict(frame)                                              # not fitted


class TestPolicyAndReport(Base):
    def test_apply_policy_uses_the_same_evaluator(self):
        pr, sc, frame, cons = self.solved()
        pol = sc.apply_policy((frame["PD"] < 0.2).astype(int).to_numpy(), name="cutoff")
        x = (frame["PD"] < 0.2).to_numpy().astype(float)
        expected = float((frame["AMOUNT"] * (0.2 - 0.7 * frame["PD"]) * x).sum())
        self.assertAlmostEqual(pol.total, expected, places=6)
        rep = pol.constraint_report()
        self.assertEqual(len(rep), len(cons))
        self.assertAlmostEqual(rep.set_index("constraint").loc["approval", "lhs"], x.mean())
        with self.assertRaises(do.ProjectError):
            sc.apply_policy(np.full(len(frame), 7))                                       # outside the domain
        sim = pr.scenario("s", flow="simulation", decisions={"APPROVE": 1}).build().optimize()
        with self.assertRaises(do.ProjectError):
            sim.apply_policy(np.ones(len(frame)))

    def test_attach_tree_and_report(self):
        pr, sc, frame, cons = self.solved()
        bau = pr.scenario("BAU", flow="simulation", decisions={"APPROVE": 1}, constraints=cons).build().optimize()
        tree = do.DecisionTree(max_depth=3, min_weight=20).fit(sc, ["BAND", "SEG"], gap=0.0)
        pr.attach_tree(name="Distilled", tree=tree, description="d", source_scenario="Constrained policy")
        rep = pr.report(scenarios=["BAU", "Constrained policy", "Distilled"], reference="BAU").scenario_summary()
        self.assertEqual(list(rep.index), ["BAU", "Constrained policy", "Distilled"])
        self.assertAlmostEqual(rep.loc["BAU", "vs reference (bn VND)"], 0.0)
        self.assertAlmostEqual(rep.loc["Distilled", "expected profit, full portfolio (bn VND)"], tree.policy_.total / 1e9)
        self.assertLessEqual(rep.loc["Distilled", "expected profit, full portfolio (bn VND)"],
                             rep.loc["Constrained policy", "expected profit, full portfolio (bn VND)"] + 1e-12)
        with self.assertRaises(do.ProjectError):
            pr.report(scenarios=["BAU", "nope"])
        with self.assertRaises(do.ProjectError):
            pr.attach_tree(name="x", tree=tree, source_scenario="another scenario")

    def test_three_treatments_are_not_distillable(self):
        frame = make_frame(n=60)
        pr = self.project(frame, treatments=(0, 1, 2))
        sc = pr.scenario("c", flow="constrained", constraints=do.Constraints().set("a", "AVG(v_approved) >= 0.1")).build()
        sc.optimize(gap=0.0)
        with self.assertRaises(NotImplementedError):
            do.DecisionTree().fit(sc, ["BAND"])


if __name__ == "__main__":
    unittest.main()
