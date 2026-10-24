import unittest

from resumos import FOLD_METRICS, fold_summary


class TestFoldSummary(unittest.TestCase):
    def test_multiclass_uses_macro_fields_and_sample_std(self):
        rows = [{"accuracy": .5, "precision_macro": .2, "recall_macro": .4, "f1_macro": .3},
                {"accuracy": .7, "precision_macro": .6, "recall_macro": .8, "f1_macro": .5}]
        summary = fold_summary(rows, classes=['CYT', 'MIT', 'NUC'])
        self.assertEqual(summary['average'], 'macro')
        self.assertIsNone(summary['positive_class'])
        self.assertEqual(summary['classes'], ['CYT', 'MIT', 'NUC'])
        self.assertEqual(summary['metric_keys'], ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
        self.assertAlmostEqual(summary['metrics']['precision_macro']['mean'], .4)
        self.assertAlmostEqual(summary['metrics']['precision_macro']['std'], .4 / 2**.5)
        self.assertNotIn('precision_M', summary['metrics'])
        self.assertEqual(fold_summary(rows)['metrics'], summary['metrics'])

    def test_explicit_multiclass_handles_missing_metrics_without_binary_fallback(self):
        result = fold_summary([{'accuracy': .5}], classes=['a', 'b', 'c'])
        self.assertEqual(result['metrics']['f1_macro'], {'mean': None, 'std': None, 'n': 0})
        with self.assertRaisesRegex(ValueError, 'recall_macro'):
            fold_summary([{'recall_macro': float('nan')}], classes=['a', 'b', 'c'])

    def test_sample_statistics_computed_by_hand(self):
        rows = [
            {"accuracy": .5, "precision_M": .2, "sensitivity_M": .4, "f1_M": .1},
            {"accuracy": .75, "precision_M": .4, "sensitivity_M": .4, "f1_M": .3},
            {"accuracy": 1., "precision_M": .6, "sensitivity_M": .4, "f1_M": .5},
        ]
        result = fold_summary(rows)
        self.assertEqual(result["n_folds"], 3)
        self.assertEqual(result["ddof"], 1)
        self.assertEqual(result["positive_class"], "M")
        expected = {"accuracy": (.75, .25), "precision_M": (.4, .2),
                    "sensitivity_M": (.4, 0.), "f1_M": (.3, .2)}
        for metric, (mean, std) in expected.items():
            with self.subTest(metric=metric):
                self.assertAlmostEqual(result["metrics"][metric]["mean"], mean)
                self.assertAlmostEqual(result["metrics"][metric]["std"], std)
                self.assertEqual(result["metrics"][metric]["n"], 3)

    def test_single_fold_has_no_sample_std(self):
        result = fold_summary([{metric: .8 for metric in FOLD_METRICS}])
        self.assertEqual(result["n_folds"], 1)
        for metric in FOLD_METRICS:
            self.assertEqual(result["metrics"][metric], {"mean": .8, "std": None, "n": 1})

    def test_empty_rows_have_no_statistics(self):
        result = fold_summary([])
        self.assertEqual(result["n_folds"], 0)
        for metric in FOLD_METRICS:
            self.assertEqual(result["metrics"][metric], {"mean": None, "std": None, "n": 0})

    def test_missing_values_report_available_count_per_metric(self):
        rows = [
            {"accuracy": .5, "precision_M": .4, "sensitivity_M": None},
            {"accuracy": None, "precision_M": .8},
            {"accuracy": 1., "precision_M": 0., "sensitivity_M": .7},
        ]
        result = fold_summary(rows)
        self.assertEqual(result["n_folds"], 3)
        metrics = result["metrics"]
        self.assertEqual(metrics["accuracy"]["n"], 2)
        self.assertAlmostEqual(metrics["accuracy"]["mean"], .75)
        self.assertAlmostEqual(metrics["accuracy"]["std"], .5 / (2 ** .5))
        self.assertEqual(metrics["precision_M"]["n"], 3)
        self.assertAlmostEqual(metrics["precision_M"]["mean"], .4)
        self.assertAlmostEqual(metrics["precision_M"]["std"], .4)
        self.assertEqual(metrics["sensitivity_M"], {"mean": .7, "std": None, "n": 1})
        self.assertEqual(metrics["f1_M"], {"mean": None, "std": None, "n": 0})

    def test_invalid_values_raise_instead_of_disappearing(self):
        invalid = [float("nan"), float("inf"), -float("inf"), -.01, 1.01, True, "0.8"]
        for metric in FOLD_METRICS:
            for value in invalid:
                with self.subTest(metric=metric, value=value):
                    with self.assertRaisesRegex(ValueError, metric):
                        fold_summary([{metric: .5}, {metric: value}])

    def test_unequal_fold_sizes_do_not_weight_the_mean(self):
        # 10/10 e 0/90 acertos: média dos folds = 50%; acurácia OOF = 10%.
        rows = [{"fold": 1, "n_test": 10, "accuracy": 1.},
                {"fold": 2, "n_test": 90, "accuracy": 0.}]
        accuracy = fold_summary(rows)["metrics"]["accuracy"]
        self.assertEqual(accuracy["mean"], .5)
        self.assertAlmostEqual(accuracy["std"], 1 / (2 ** .5))
        self.assertNotEqual(accuracy["mean"], 10 / 100)


if __name__ == "__main__":
    unittest.main()
