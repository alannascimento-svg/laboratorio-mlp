"""Integração HTTP somente leitura, em servidor efêmero independente do painel."""
import csv
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import math
from statistics import mean, stdev
import threading
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app import Handler, ROOT, result_summary


class QuietHandler(Handler):
    def log_message(self, format, *args):
        pass


class TestAppResultsHTTP(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": .01},
            daemon=True,
        )
        self.thread.start()
        self.folder = ROOT / "resultados_exemplo"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive(), "Servidor HTTP de teste não encerrou.")

    def get_json(self, path, host=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            headers = {"Host": host} if host is not None else {}
            connection.request("GET", path, headers=headers)
            response = connection.getresponse()
            payload = json.loads(response.read())
            return response.status, dict(response.getheaders()), payload
        finally:
            connection.close()

    def test_result_summarizes_saved_folds_and_identifies_best_accuracy(self):
        with (self.folder / "rodadas.csv").open(encoding="utf8", newline="") as source:
            saved_rows = list(csv.DictReader(source))
        status, _, result = self.get_json("/api/result?run=exemplo")
        self.assertEqual(status, 200)
        summary = result["fold_summary"]
        self.assertEqual(summary["positive_class"], "M")
        self.assertEqual(summary["ddof"], 1)
        self.assertEqual(summary["n_folds"], len(saved_rows))
        self.assertEqual(len(result["folds"]), len(saved_rows))
        for metric in ("accuracy", "precision_M", "sensitivity_M", "f1_M"):
            with self.subTest(metric=metric):
                values = [float(row[metric]) for row in saved_rows]
                self.assertEqual(summary["metrics"][metric]["n"], len(values))
                self.assertAlmostEqual(summary["metrics"][metric]["mean"], mean(values), places=8)
                self.assertAlmostEqual(summary["metrics"][metric]["std"], stdev(values), places=8)
        best = max(float(row["accuracy"]) for row in saved_rows)
        expected_ids = [int(row["fold"]) for row in saved_rows
                        if math.isclose(float(row["accuracy"]), best, rel_tol=1e-12, abs_tol=1e-12)]
        self.assertAlmostEqual(result["highlights"]["accuracy"]["value"], best)
        self.assertEqual(result["highlights"]["accuracy"]["ids"], expected_ids)
        saved_oof = json.loads((self.folder / "metricas_oof.json").read_text(encoding="utf8"))
        self.assertEqual(result["metrics"], saved_oof)

    def test_summary_download_matches_api_without_writing_historical_file(self):
        target = self.folder / "resumo_rodadas.json"
        existed_before = target.exists()
        contents_before = target.read_bytes() if existed_before else None
        try:
            api_status, _, result = self.get_json("/api/result?run=exemplo")
            status, headers, downloaded = self.get_json(
                "/download?run=exemplo&file=resumo_rodadas.json"
            )
            self.assertEqual(api_status, 200)
            self.assertEqual(status, 200)
            self.assertIn("application/json", headers["Content-Type"])
            self.assertIn('filename="resumo_rodadas.json"', headers["Content-Disposition"])
            self.assertEqual(downloaded["run"], "exemplo")
            self.assertEqual(downloaded["source"], "rodadas.csv")
            for key in ("folds", "fold_summary", "highlights"):
                self.assertEqual(downloaded[key], result[key])
        finally:
            self.assertEqual(target.exists(), existed_before)
            if existed_before:
                self.assertEqual(target.read_bytes(), contents_before)

    def test_network_preserves_legacy_architecture_and_saved_parameters(self):
        status, _, network = self.get_json("/api/network?run=exemplo&model=final")
        self.assertEqual(status, 200)
        self.assertEqual(network["architecture"], [30, 16, 2])
        self.assertEqual(network["n_parameters"], 530)
        with np.load(self.folder / "modelo_final.npz", allow_pickle=False) as saved:
            for key in ("W1", "b1", "W2", "b2"):
                with self.subTest(parameter=key):
                    np.testing.assert_array_equal(network["params"][key], saved[key])
            self.assertEqual(network["features"], saved["features"].tolist())
            np.testing.assert_array_equal(network["mean"], saved["mean"])
            np.testing.assert_array_equal(network["scale"], saved["scale"])
            self.assertEqual(saved["classes"].tolist(), ["M", "B"])
            self.assertEqual(saved["codes"].tolist(), ["10", "01"])

    def test_unauthorized_host_is_rejected(self):
        status, _, payload = self.get_json("/api/result?run=exemplo", host="example.invalid")
        self.assertEqual(status, 403)
        self.assertIn("Host não autorizado", payload["error"])

    def test_invalid_run_is_rejected(self):
        for path in (
            "/api/result?run=..%2Fresultados_exemplo",
            "/api/network?run=..%2Fresultados_exemplo&model=final",
            "/download?run=..%2Fresultados_exemplo&file=resumo_rodadas.json",
        ):
            with self.subTest(path=path):
                status, _, payload = self.get_json(path)
                self.assertEqual(status, 400)
                self.assertIn("Identificador de execução inválido", payload["error"])

    def test_nonfinite_csv_metric_is_rejected_but_empty_cell_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for value in ('inf', '-inf', 'NaN', 'nan'):
                with self.subTest(value=value):
                    (folder / 'rodadas.csv').write_text(f'fold,accuracy\n1,{value}\n', encoding='utf8')
                    with self.assertRaisesRegex(ValueError, 'accuracy'):
                        result_summary(folder)
            (folder / 'rodadas.csv').write_text('fold,accuracy\n1,\n', encoding='utf8')
            self.assertEqual(result_summary(folder)['fold_summary']['metrics']['accuracy'],
                             {'mean': None, 'std': None, 'n': 0})


if __name__ == "__main__":
    unittest.main()
