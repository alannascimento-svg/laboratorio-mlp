"""Integração da seleção de colunas; arquivos e treino curto só em temporários."""
import base64
import copy
import csv
import hashlib
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

import app
from entrada import inspect_source
from rede import (Config, FEATURES, MLP, experiment_signature, load_dataset,
                  load_model, load_selected_dataset, model_details, run_experiment)
from public_fixtures import digest_files, legacy_example


ROOT = Path(__file__).resolve().parents[1]
WDBC_PATH = ROOT / "dados" / "cancear.txt"


def schema_for(table):
    return {"delimiter": table["delimiter"],
            "header": "yes" if table["has_header"] else "no",
            "columns": [{key: column[key] for key in ("index", "description", "role")}
                        for column in table["columns"]]}


def small_real_source():
    """60 registros reais WDBC (30 M, 30 B), quatro medidas e ordem diferente."""
    source = inspect_source(WDBC_PATH.read_text(encoding="utf8"))
    chosen = []
    for label in ("M", "B"):
        chosen.extend([row for row in source["rows"] if row[1] == label][:30])
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=";", lineterminator="\r\n")
    writer.writerow(["radius_mean", "diagnosis", "id", "texture_mean", "perimeter_mean", "area_mean"])
    for row in chosen:
        writer.writerow([row[2], row[1], row[0], row[3], row[4], row[5]])
    return "\ufeff" + stream.getvalue()


class TestInputSelectionIntegration(unittest.TestCase):
    def setUp(self):
        self.raw = WDBC_PATH.read_bytes()
        self.table = inspect_source(self.raw.decode("utf8"))
        self.schema = schema_for(self.table)

    def test_original_32_columns_select_30_without_changing_legacy_data(self):
        legacy = load_dataset(WDBC_PATH)
        selected = load_selected_dataset(self.raw, self.schema)
        self.assertEqual(len(self.table["columns"]), 32)
        self.assertEqual(selected.X.shape, (569, 30))
        self.assertEqual(selected.features, FEATURES)
        self.assertEqual(selected.input_schema, self.schema)
        self.assertEqual(MLP(0, selected.X.shape[1], selected.features).architecture, [30, 16, 2])
        for attribute in ("X", "y", "ids", "source_lines"):
            np.testing.assert_array_equal(getattr(selected, attribute), getattr(legacy, attribute))
        self.assertEqual(selected.sha256, legacy.sha256)
        self.assertEqual(selected.sha256, hashlib.sha256(self.raw).hexdigest())
        np.testing.assert_array_equal(load_dataset(WDBC_PATH, self.schema).X, legacy.X)

    def test_ignoring_columns_updates_input_count_and_arithmetic_mean_rounding(self):
        for ignored, architecture in (([2], [29, 16, 2]), ([2, 3], [28, 15, 2])):
            with self.subTest(ignored=ignored):
                schema = copy.deepcopy(self.schema)
                for index in ignored:
                    schema["columns"][index]["role"] = "ignore"
                selected = load_selected_dataset(self.raw, schema)
                result = app.inspect_selection({"input_schema": schema})
                self.assertEqual(result["architecture"], architecture)
                self.assertEqual(selected.X.shape[1], architecture[0])
                expected = [name for index, name in enumerate(FEATURES, 2) if index not in ignored]
                self.assertEqual(selected.features, expected)

    def test_descriptions_are_annotations_and_not_model_feature_names(self):
        before = load_selected_dataset(self.raw, self.schema)
        self.schema["columns"][2]["description"] = "Raio médio — descrição editada"
        after = load_selected_dataset(self.raw, self.schema)
        np.testing.assert_array_equal(before.X, after.X)
        np.testing.assert_array_equal(before.y, after.y)
        self.assertEqual(before.features, after.features)
        self.assertEqual(before.sha256, after.sha256)
        self.assertEqual(after.feature_descriptions[0], "Raio médio — descrição editada")
        self.assertEqual(after.features[0], "radius_mean")

    def test_readonly_inspection_handles_uploads_without_importing_or_writing(self):
        text = small_real_source()
        schema = schema_for(inspect_source(text))
        with tempfile.TemporaryDirectory() as tmp, patch.object(app, "ROOT", Path(tmp)), \
                patch.object(app, "imported_dataset", side_effect=AssertionError("Inspeção tentou importar arquivo")):
            columns = app.inspect_columns({"data_text": text})
            inspected = app.inspect_selection({"data_text": text, "input_schema": schema})
            self.assertEqual((columns["n_rows"], columns["n_columns"]), (60, 6))
            self.assertNotIn("rows", columns)
            self.assertEqual(len(columns["source_lines"]), 8)
            self.assertEqual(inspected["architecture"], [4, 3, 2])
            self.assertEqual((inspected["M"], inspected["B"]), (30, 30))
            self.assertEqual(list(Path(tmp).rglob("*")), [])

    def test_motor_keeps_minimum_sample_class_and_duplicate_checks_after_selection(self):
        text = small_real_source()
        table = inspect_source(text)
        schema = schema_for(table)
        header = text.splitlines()[0]
        # 40 registros não satisfazem o protocolo, mesmo que a tabela seja válida.
        short = "\n".join([header, *text.splitlines()[1:41]])
        with self.assertRaisesRegex(ValueError, ">=50"):
            app.inspect_selection({"data_text": short, "input_schema": schema})
        # Conservar somente a classe M falha a exigência de duas classes.
        one_class = text.replace(";B;", ";M;")
        with self.assertRaisesRegex(ValueError, ">=20"):
            load_selected_dataset(one_class.encode("utf8"), schema)
        # ID diferente não torna duas medições iguais independentes.
        lines = text.splitlines()
        duplicate = lines[1].split(";")
        duplicate[2] = "id_adicional_para_teste"
        duplicated = "\n".join([*lines, ";".join(duplicate)])
        with self.assertRaisesRegex(ValueError, "Vetores de medidas repetidos"):
            load_selected_dataset(duplicated.encode("utf8"), schema)

    def test_short_temporary_run_persists_selection_descriptions_checkpoints_and_raw_bytes(self):
        text = small_real_source()
        raw = text.encode("utf8")
        schema = schema_for(inspect_source(text))
        schema["columns"][5]["role"] = "ignore"
        schema["columns"][0]["description"] = "Raio médio da base original"
        expected_features = ["radius_mean", "texture_mean", "perimeter_mean"]
        expected_descriptions = ["Raio médio da base original", "texture_mean", "perimeter_mean"]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source_path, output = folder / "selecao.csv", folder / "execucao_temporaria"
            source_path.write_bytes(raw)
            cfg = Config(mode="kfold5", max_epochs=1, patience=1, snapshot_interval=1,
                         record_calculations=False)
            result = run_experiment(source_path, output, cfg, input_schema=schema)
            self.assertEqual(result["metrics"]["architecture"], [3, 3, 2])
            self.assertEqual(source_path.read_bytes(), raw)
            self.assertEqual((output / "dados_utilizados.txt").read_bytes(), raw)
            config = json.loads((output / "configuracao.json").read_text(encoding="utf8"))
            self.assertEqual(config["features"], expected_features)
            self.assertEqual(config["feature_descriptions"], expected_descriptions)
            self.assertEqual(config["input_schema"], schema)
            self.assertEqual(config["data_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(json.loads((output / "esquema_entradas.json").read_text(encoding="utf8")), schema)
            checkpoints = sorted(output.rglob("epoca_00000.npz"))
            self.assertEqual(len(checkpoints), 12)  # 5 folds + final, seleção e refit.
            paths = [output / "modelo_final.npz", *sorted((output / "folds").glob("*/modelo.npz")), *checkpoints]
            for path in paths:
                with self.subTest(model=str(path.relative_to(output))):
                    net, mean, scale, metadata = load_model(path)
                    self.assertEqual(net.architecture, [3, 3, 2])
                    self.assertEqual(net.features, expected_features)
                    self.assertEqual(net.feature_descriptions, expected_descriptions)
                    self.assertEqual(net.input_schema, schema)
                    self.assertEqual(metadata["input_schema"], schema)
                    self.assertEqual(metadata["feature_descriptions"], expected_descriptions)
                    self.assertEqual(mean.shape, (3,))
                    self.assertEqual(scale.shape, (3,))
                    details = model_details(path)
                    self.assertEqual(details["feature_keys"], ["x1", "x4", "x5"])
            before = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in output.rglob("*") if path.is_file()}
            with self.assertRaises(FileExistsError):
                run_experiment(source_path, output, cfg, input_schema=schema)
            after = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in output.rglob("*") if path.is_file()}
            self.assertEqual(after, before)

    def test_comparison_separates_selection_and_ignores_description_changes(self):
        reference = ROOT / "resultados_exemplo"
        original_config = json.loads((reference / "configuracao.json").read_text(encoding="utf8"))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for indices in sorted((reference / "folds").glob("*/indices.json")):
                destination = folder / "folds" / indices.parent.name / "indices.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(indices.read_bytes())
            config_path = folder / "configuracao.json"
            def signature(schema):
                config_path.write_text(json.dumps({**original_config, "input_schema": schema}), encoding="utf8")
                return experiment_signature(folder)
            selected_signature = signature(self.schema)
            descriptions = copy.deepcopy(self.schema)
            descriptions["columns"][2]["description"] = "Outra descrição sem mudar os dados"
            self.assertEqual(signature(descriptions), selected_signature)
            subset = copy.deepcopy(self.schema)
            subset["columns"][2]["role"] = "ignore"
            self.assertNotEqual(signature(subset), selected_signature)
            self.assertNotEqual(signature(None), selected_signature)
            self.assertEqual(signature(None), experiment_signature(reference))

    def test_legacy_saved_signatures_and_models_remain_compatible(self):
        reference = ROOT / "resultados_exemplo"
        before = digest_files(reference)
        comparison = json.loads((reference / "comparacao.json").read_text(encoding="utf8"))
        self.assertEqual(experiment_signature(reference), comparison["signature"])
        with tempfile.TemporaryDirectory() as tmp:
            folder = legacy_example(Path(tmp))
            # Arquivo sem os campos introduzidos depois da v1.
            original, original_mean, original_scale, _ = load_model(reference / "modelo_final.npz")
            net, mean, scale, metadata = load_model(folder / "modelo_final.npz")
            self.assertIsNone(net.input_schema)
            self.assertEqual(net.feature_descriptions, net.features)
            self.assertEqual(net.architecture, [30, 16, 2])
            self.assertEqual(net.classes, ["M", "B"])
            self.assertNotIn("input_schema", metadata)
            for key in original.params:
                np.testing.assert_array_equal(net.params[key], original.params[key])
            np.testing.assert_array_equal(mean, original_mean)
            np.testing.assert_array_equal(scale, original_scale)
            X = load_dataset(WDBC_PATH).X
            np.testing.assert_array_equal(net.predict_proba((X - mean) / scale),
                                          original.predict_proba((X - original_mean) / original_scale))
        self.assertEqual(digest_files(reference), before)

    def test_legacy_signature_matches_frozen_fixture_without_recent_fields(self):
        # Hash congelado do formato v1: base, modo, fração e índices. O pequeno
        # documento abaixo testa serialização; não representa um treino válido.
        config = {"data_sha256": "0" * 64, "mode": "kfold5", "validation_fraction": .15}
        splits = [
            {"fold": 1, "train_0based": [2, 3], "test_0based": [0, 1]},
            {"fold": 2, "train_0based": [0, 1], "test_0based": [2, 3]},
        ]
        expected = "399438461de1c3244de09203be480b2dd402dc410eb4c479214c6da0abfaaee0"
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for index, split in enumerate(splits, 1):
                path = folder / "folds" / f"fold_{index:02d}" / "indices.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(split), encoding="utf8")
            for schema_field in ({}, {"input_schema": None}):
                with self.subTest(schema_field=schema_field):
                    (folder / "configuracao.json").write_text(
                        json.dumps({**config, **schema_field}), encoding="utf8")
                    before = digest_files(folder)
                    self.assertEqual(experiment_signature(folder), expected)
                    self.assertEqual(digest_files(folder), before)


class QuietInputHandler(app.Handler):
    def log_message(self, format, *args):
        pass


class TestInputBinaryReading(unittest.TestCase):
    def test_latin1_and_utf8_bom_keep_accents_hash_and_original_imported_bytes(self):
        text = small_real_source().lstrip("\ufeff").replace("diagnosis", "diagnóstico").replace("perimeter_mean", "perímetro")
        for encoding in ("latin-1", "utf-8-sig"):
            with self.subTest(encoding=encoding):
                raw = text.encode(encoding)
                payload = {"data_base64": base64.b64encode(raw).decode("ascii")}
                with tempfile.TemporaryDirectory() as tmp, patch.object(app, "ROOT", Path(tmp)):
                    self.assertEqual(app.input_bytes(payload), raw)
                    columns = app.inspect_columns(payload)
                    self.assertEqual(columns["source_sha256"], hashlib.sha256(raw).hexdigest())
                    self.assertEqual(columns["columns"][1]["name"], "diagnóstico")
                    self.assertEqual(columns["columns"][1]["role"], "target")
                    self.assertEqual(columns["columns"][4]["name"], "perímetro")
                    self.assertNotIn("\ufffd", json.dumps(columns, ensure_ascii=False))
                    selected = app.inspect_selection({**payload, "input_schema": schema_for(columns)})
                    self.assertEqual(selected["n"], 60)
                    self.assertIn("perímetro", selected["features"])
                    self.assertEqual(list(Path(tmp).rglob("*")), [])
                    imported = app.imported_dataset(raw=app.input_bytes(payload))
                    self.assertEqual(imported.parent, Path(tmp) / "dados" / "importados")
                    self.assertEqual(imported.name, hashlib.sha256(raw).hexdigest() + ".txt")
                    self.assertEqual(imported.read_bytes(), raw)
                    self.assertEqual(app.imported_dataset(raw=raw), imported)
                    self.assertEqual(imported.read_bytes(), raw)
                    self.assertEqual(len(list(imported.parent.iterdir())), 1)

    def test_malformed_or_ambiguous_binary_payload_is_rejected_without_writes(self):
        invalid = [{"data_base64": "!!!!"}, {"data_base64": "YQ"},
                   {"data_base64": "YQ==\n"}, {"data_base64": []},
                   {"data_base64": {}}, {"data_base64": "YQ==", "data_text": "a"},
                   {"data_base64": "YQ==", "data_text": ""}]
        with tempfile.TemporaryDirectory() as tmp, patch.object(app, "ROOT", Path(tmp)):
            for payload in invalid:
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    app.input_bytes(payload)
            self.assertEqual(list(Path(tmp).rglob("*")), [])


class TestInputSelectionHTTP(unittest.TestCase):
    def test_multiclass_inspection_and_invalid_protocol_do_not_write_or_train(self):
        # Seis casos por classe. Um ID repetido deixa cinco grupos na classe C0.
        lines = ['id,classe,a,b']
        for c in range(10):
            for j in range(6):
                identifier = f'C{c}_{0 if c == 0 and j == 1 else j}'
                lines.append(f'{identifier},C{c},{c*100+j},{j*j+c/10}')
        raw = ('\n'.join(lines)+'\n').encode('utf8')
        payload = {'data_base64': base64.b64encode(raw).decode('ascii')}
        with tempfile.TemporaryDirectory() as tmp, patch.object(app, 'ROOT', Path(tmp)), \
                patch.object(app, 'run_experiment', side_effect=AssertionError('Treino não deve iniciar')):
            status, columns = self.post('/api/columns', payload)
            self.assertEqual(status, 200)
            self.assertEqual(len(columns['target_summary']['classes']), 10)
            self.assertEqual(columns['source_sha256'], hashlib.sha256(raw).hexdigest())
            schema = schema_for(columns)
            status, selected = self.post('/api/inspect', {**payload, 'input_schema': schema})
            self.assertEqual(status, 200, selected)
            self.assertEqual(selected['architecture'], [2, 6, 10])
            self.assertEqual(selected['classes'], [f'C{i}' for i in range(10)])
            self.assertEqual(selected['class_counts'], {f'C{i}': 6 for i in range(10)})
            self.assertTrue(selected['protocol']['grouped'])
            self.assertEqual(selected['protocol']['n_groups'], 59)
            self.assertEqual(selected['protocol']['supported_modes'], ['kfold5'])
            self.assertNotIn('M', selected)
            self.assertNotIn('B', selected)
            status, rejected = self.post('/api/train', {**payload, 'input_schema': schema,
                                                       'config': {'mode': 'kfold10', 'max_epochs': 1}})
            self.assertEqual(status, 400)
            self.assertIn('kfold5', rejected['error'])
            self.assertEqual(list(Path(tmp).rglob('*')), [])

    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietInputHandler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def post(self, path, payload, authorized=True):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        headers = {"Content-Type": "application/json"}
        if authorized:
            headers["X-Lab-Token"] = app.TOKEN
        try:
            connection.request("POST", path, body=json.dumps(payload).encode("utf8"), headers=headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_columns_requires_token_and_returns_original_32_positions(self):
        status, payload = self.post("/api/columns", {}, authorized=False)
        self.assertEqual(status, 403)
        self.assertIn("não autorizada", payload["error"])
        status, payload = self.post("/api/columns", {})
        self.assertEqual(status, 200)
        self.assertEqual((payload["n_rows"], payload["n_columns"]), (569, 32))
        self.assertEqual([column["key"] for column in payload["columns"]], [f"x{i}" for i in range(1, 33)])
        self.assertEqual(payload["columns"][0]["role"], "id")
        self.assertEqual(payload["columns"][1]["role"], "target")

    def test_uploaded_inspection_and_tampering_never_write_imported_files(self):
        text = small_real_source()
        schema = schema_for(inspect_source(text))
        with tempfile.TemporaryDirectory() as tmp, patch.object(app, "ROOT", Path(tmp)):
            status, inspected = self.post("/api/columns", {"data_text": text})
            self.assertEqual(status, 200)
            self.assertEqual(inspected["n_columns"], 6)
            status, inspected = self.post("/api/inspect", {"data_text": text, "input_schema": schema})
            self.assertEqual(status, 200)
            self.assertEqual(inspected["architecture"], [4, 3, 2])
            cases = []
            target_as_feature = copy.deepcopy(schema)
            target_as_feature["columns"][1]["role"] = "feature"
            target_as_feature["columns"][1]["description"] = "medida"
            cases.append(target_as_feature)
            repeated = copy.deepcopy(schema)
            repeated["columns"][2]["index"] = 0
            cases.append(repeated)
            cases.append({**schema, "delimiter": ","})
            for tampered in cases:
                with self.subTest(schema=tampered):
                    status, error = self.post("/api/inspect", {"data_text": text, "input_schema": tampered})
                    self.assertEqual(status, 400)
                    self.assertIn("error", error)
            self.assertEqual(list(Path(tmp).rglob("*")), [])

    def test_binary_upload_uses_original_encoding_and_rejects_bad_base64(self):
        text = small_real_source().lstrip("\ufeff").replace("perimeter_mean", "perímetro")
        raw = text.encode("latin-1")
        payload = {"data_base64": base64.b64encode(raw).decode("ascii")}
        with tempfile.TemporaryDirectory() as tmp, patch.object(app, "ROOT", Path(tmp)):
            status, columns = self.post("/api/columns", payload)
            self.assertEqual(status, 200)
            self.assertEqual(columns["source_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(columns["columns"][4]["name"], "perímetro")
            status, selected = self.post("/api/inspect", {**payload, "input_schema": schema_for(columns)})
            self.assertEqual(status, 200)
            self.assertIn("perímetro", selected["features"])
            for invalid in ({"data_base64": "*bad*"}, {**payload, "data_text": text}):
                with self.subTest(payload=invalid):
                    status, error = self.post("/api/columns", invalid)
                    self.assertEqual(status, 400)
                    self.assertIn("error", error)
            self.assertEqual(list(Path(tmp).rglob("*")), [])


if __name__ == "__main__":
    unittest.main()
