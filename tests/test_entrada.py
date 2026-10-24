"""Casos pequenos de parsing; não iniciam treino nem gravam dados/resultados."""
import copy
from pathlib import Path
import unittest

from entrada import inspect_source, selected_data, WDBC_FEATURES


ROOT = Path(__file__).resolve().parents[1]


def schema_for(table):
    return {"delimiter": table["delimiter"],
            "header": "yes" if table["has_header"] else "no",
            "columns": [{"index": column["index"], "description": column["description"],
                         "role": column["role"]} for column in table["columns"]]}


class InspectSourceTests(unittest.TestCase):
    def test_wdbc_original_has_32_columns_and_30_real_inputs(self):
        from rede import FEATURES
        table = inspect_source((ROOT / "dados" / "cancear.txt").read_text())
        self.assertEqual(WDBC_FEATURES, FEATURES)
        self.assertEqual((table["n_rows"], table["n_columns"]), (569, 32))
        self.assertFalse(table["has_header"])
        self.assertEqual([column["key"] for column in table["columns"]],
                         [f"x{i}" for i in range(1, 33)])
        self.assertEqual([column["name"] for column in table["columns"]], ["id", "diagnosis", *FEATURES])
        self.assertEqual([column["role"] for column in table["columns"]], ["id", "target"] + ["feature"] * 30)
        result = selected_data(table, schema_for(table))
        self.assertEqual(result["features"], FEATURES)
        self.assertEqual((len(result["X"]), len(result["X"][0])), (569, 30))
        self.assertEqual(result["ids"][0], "842302")
        self.assertEqual(result["source_lines"][0], 142)
        self.assertEqual(result["X"][0][0], 17.99)

    def test_wdbc_truncation_and_bad_preamble_are_not_silently_generic(self):
        source = (ROOT / "dados" / "cancear.txt").read_text()
        lines = source.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("842302,"))
        short = "\n".join(lines[:start] + [",".join(line.split(",")[:-1]) for line in lines[start:] if line])
        with self.assertRaisesRegex(ValueError, "32 colunas"):
            inspect_source(short)
        with self.assertRaisesRegex(ValueError, "Preâmbulo"):
            inspect_source(source.replace("6. Number of attributes: 32", "6. Number of attributes: 31"))
        with self.assertRaisesRegex(ValueError, "sem cabeçalho"):
            inspect_source(source, header="yes")

    def test_arbitrary_preamble_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "colunas"):
            inspect_source("Um relatório qualquer\nid,diagnosis,medida\n01,M,1\n02,B,2")

    def test_generic_headers_order_and_id_leading_zeros(self):
        table = inspect_source("medida,diagnóstico,codigo_id,outra\n1.25,M,0007,3\n2.5,B,0008,4\n")
        result = selected_data(table, schema_for(table))
        self.assertTrue(table["has_header"])
        self.assertEqual([column["role"] for column in table["columns"]], ["feature", "target", "id", "feature"])
        self.assertEqual(result["features"], ["medida", "outra"])
        self.assertEqual(result["X"], [[1.25, 3.0], [2.5, 4.0]])
        self.assertEqual(result["ids"], ["0007", "0008"])
        self.assertEqual(result["y"], [0, 1])

    def test_no_header_keeps_all_rows_and_stable_positions(self):
        table = inspect_source("001,M,2,3\n002,B,4,5\n")
        self.assertFalse(table["has_header"])
        self.assertEqual(table["n_rows"], 2)
        self.assertEqual([column["name"] for column in table["columns"]], ["x1", "x2", "x3", "x4"])
        schema = schema_for(table)
        schema["columns"][0]["role"] = "id"
        result = selected_data(table, schema)
        self.assertEqual(result["features"], ["x3", "x4"])
        self.assertEqual(result["ids"], ["001", "002"])
        self.assertEqual(result["source_lines"], [1, 2])

    def test_no_header_reserved_word_as_id_does_not_discard_first_sample(self):
        for identifier in ("id", "identificador", "diagnosis", "label"):
            for delimiter in (",", ";", "\t"):
                with self.subTest(identifier=identifier, delimiter=delimiter):
                    text = delimiter.join([identifier, "M", "1", "2"]) + "\n"
                    text += delimiter.join(["amostra_b", "B", "3", "4"])
                    table = inspect_source(text)
                    self.assertFalse(table["has_header"])
                    self.assertEqual(table["n_rows"], 2)
                    self.assertEqual(table["source_lines"], [1, 2])
                    self.assertEqual([column["name"] for column in table["columns"]], ["x1", "x2", "x3", "x4"])
                    schema = schema_for(table)
                    schema["columns"][0]["role"] = "id"
                    selected = selected_data(table, schema)
                    self.assertEqual(selected["ids"], [identifier, "amostra_b"])
                    self.assertEqual(selected["X"], [[1.0, 2.0], [3.0, 4.0]])
                    self.assertEqual(selected["y"], [0, 1])
        # Cabeçalho explícito continua permitindo nomes numéricos ou "M".
        named = inspect_source("id,diagnosis,M,2\n001,M,1,2\n002,B,3,4", header="yes")
        self.assertTrue(named["has_header"])
        self.assertEqual(named["n_rows"], 2)

    def test_explicit_header_modes_and_delimiters(self):
        for separator in (",", ";", "\t"):
            with self.subTest(separator=separator):
                text = separator.join(["amostra", "classe", "medida"]) + "\n" + separator.join(["A", "M", "1"]) + "\n"
                table = inspect_source(text, delimiter=separator, header="yes")
                self.assertEqual(table["delimiter"], separator)
                self.assertEqual(table["n_rows"], 1)
                self.assertEqual(inspect_source(text)["delimiter"], separator)
        table = inspect_source("measure,other\n1,2\n3,4", header="no")
        self.assertFalse(table["has_header"])
        self.assertEqual(table["rows"][0], ["measure", "other"])

    def test_decimal_comma_only_with_different_separator(self):
        table = inspect_source('id;classe;a;b\n01;M;1,25;"2,5"\n02;B;3,5;4,75')
        self.assertEqual(selected_data(table, schema_for(table))["X"], [[1.25, 2.5], [3.5, 4.75]])
        table = inspect_source('id,classe,a\n01,M,"1,25"\n02,B,"2,5"')
        schema = schema_for(table)
        schema["columns"][2]["role"] = "feature"
        with self.assertRaisesRegex(ValueError, "não numérica"):
            selected_data(table, schema)

    def test_quoted_multiline_preview_and_physical_source_lines(self):
        text = '\ufeffid,classe,medida,nota\n\n0001,M,1,"texto, com vírgula\ne duas linhas"\n0002,B,2,"  espaços  "\n'
        table = inspect_source(text)
        self.assertEqual(table["source_lines"], [3, 5])
        self.assertEqual(table["rows"][0][3], "texto, com vírgula\ne duas linhas")
        self.assertEqual(table["columns"][3]["preview"], ["texto, com vírgula\ne duas linhas", "  espaços  "])
        self.assertEqual(table["columns"][3]["role"], "ignore")
        self.assertTrue(table["warnings"])
        self.assertEqual(selected_data(table, schema_for(table))["features"], ["medida"])

    def test_preview_limit_does_not_limit_data(self):
        table = inspect_source("classe,medida\n" + "\n".join(f"{'M' if i % 2 else 'B'},{i}" for i in range(21)))
        self.assertEqual(table["n_rows"], 21)
        self.assertEqual(len(table["rows"]), 21)
        self.assertEqual(len(table["columns"][1]["preview"]), 8)
        self.assertEqual(len(selected_data(table, schema_for(table))["X"]), 21)

    def test_empty_cells_not_shifted_and_missing_count(self):
        table = inspect_source("id,diagnosis,a,b\n01,M,,2\n02,B,3,4\n")
        self.assertEqual(table["rows"][0], ["01", "M", "", "2"])
        self.assertEqual(table["columns"][2]["missing"], 1)
        self.assertTrue(table["columns"][2]["numeric"])
        with self.assertRaisesRegex(ValueError, "Linha 2, x3"):
            selected_data(table, schema_for(table))

    def test_duplicate_empty_headers_get_unique_position_names(self):
        table = inspect_source("id,classe,a,a,,x3\n01,M,1,2,3,4\n02,B,5,6,7,8")
        self.assertEqual([column["name"] for column in table["columns"]],
                         ["id", "classe", "x3", "x4", "x5", "x6"])
        self.assertTrue(table["warnings"])
        result = selected_data(table, schema_for(table))
        self.assertEqual(len(set(result["features"])), 4)

    def test_multiple_labels_need_selection_and_reserved_id_names(self):
        table = inspect_source("identificador,a,b,medida\n01,M,B,1\n02,B,M,2")
        self.assertEqual(table["columns"][0]["role"], "id")
        self.assertFalse(any(column["role"] == "target" for column in table["columns"]))
        self.assertTrue(any("inequívoca" in warning for warning in table["warnings"]))
        with self.assertRaisesRegex(ValueError, "exatamente uma"):
            selected_data(table, schema_for(table))
        schema = schema_for(table)
        schema["columns"][1]["role"] = "target"
        self.assertEqual(selected_data(table, schema)["features"], ["medida"])

    def test_empty_ragged_unclosed_quote_and_options_fail(self):
        for text in ("", " \n", "\ufeff", "id,diagnosis,a\n", 'id,diagnosis,a\n01,M,"1\n',
                     '\n\nid,diagnosis,a\n01,M,"1\n', '01,M,"1\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                inspect_source(text)
        for text in ("id,diagnosis,a\n01,M,1\n02,B\n", "id;diagnosis;a\n01;M;1\n02;B\n"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "colunas"):
                inspect_source(text)
        for option in ("perhaps", [], True):
            with self.subTest(option=option), self.assertRaises(ValueError):
                inspect_source("classe,a\nM,1", header=option)
            with self.subTest(option=option), self.assertRaises(ValueError):
                inspect_source("classe,a\nM,1", delimiter=option)

    def test_column_count_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "512"):
            inspect_source(",".join(str(i) for i in range(513)), header="no")


class SelectedDataTests(unittest.TestCase):
    def setUp(self):
        self.table = inspect_source("id,diagnosis,a,b,nota\n001,M,1,2,texto\n002,B,3,4,outro")
        self.schema = schema_for(self.table)

    def test_description_edit_ignoring_feature_and_reordering_payload(self):
        original = copy.deepcopy(self.table)
        self.schema["columns"][2]["description"] = "Raio da célula <anotação>"
        self.schema["columns"][3]["role"] = "ignore"
        self.schema["columns"].reverse()
        result = selected_data(self.table, self.schema)
        self.assertEqual(result["features"], ["a"])
        self.assertEqual(result["descriptions"], ["Raio da célula <anotação>"])
        self.assertEqual(result["X"], [[1.0], [3.0]])
        self.assertEqual([column["index"] for column in result["schema"]["columns"]], list(range(5)))
        self.assertEqual(self.table, original)

    def test_no_id_generates_explicit_source_line_ids(self):
        self.schema["columns"][0]["role"] = "ignore"
        result = selected_data(self.table, self.schema)
        self.assertEqual(result["ids"], ["linha_2", "linha_3"])
        table = inspect_source("\nclasse,a\n\nM,1\nB,2")
        self.assertEqual(selected_data(table, schema_for(table))["ids"], ["linha_4", "linha_5"])

    def test_duplicate_and_missing_ids_rejected(self):
        for text in ("id,classe,a\n001,M,1\n001,B,2", "id,classe,a\n,M,1\n001,B,2"):
            with self.subTest(text=text):
                table = inspect_source(text)
                with self.assertRaisesRegex(ValueError, "ID vazio ou repetido"):
                    selected_data(table, schema_for(table))

    def test_exactly_one_target_and_at_most_one_id(self):
        for index, role, message in ((1, "ignore", "exatamente uma"), (2, "target", "exatamente uma"), (2, "id", "no máximo uma")):
            with self.subTest(role=role):
                schema = copy.deepcopy(self.schema)
                schema["columns"][index]["role"] = role
                with self.assertRaisesRegex(ValueError, message):
                    selected_data(self.table, schema)

    def test_all_features_ignored_or_too_many_features_fail(self):
        for column in self.schema["columns"]:
            if column["role"] == "feature":
                column["role"] = "ignore"
        with self.assertRaisesRegex(ValueError, "1 a 256"):
            selected_data(self.table, self.schema)
        table = inspect_source("classe," + ",".join(f"f{i}" for i in range(257)) + "\nM," + ",".join("1" for _ in range(257)))
        with self.assertRaisesRegex(ValueError, "1 a 256"):
            selected_data(table, schema_for(table))

    def test_labels_and_selected_values_must_be_valid(self):
        for label in ("", "x" * 101):
            with self.subTest(label=label):
                table = inspect_source(f"id,classe,a\n001,{label},1\n002,B,2")
                with self.assertRaisesRegex(ValueError, "rótulo não vazio"):
                    selected_data(table, schema_for(table))
        for label in ("maligno", "m", "10", "0"):
            table = inspect_source(f"id,classe,a\n001,{label},1\n002,B,2")
            self.assertEqual(selected_data(table, schema_for(table))["classes"], sorted([label,"B"]))
        for value in ("", "NaN", "inf", "-Infinity", "1e500", "texto"):
            with self.subTest(value=value):
                table = inspect_source(f"id,classe,a,b\n001,M,{value},5\n002,B,2,6")
                schema = schema_for(table)
                schema["columns"][2]["role"] = "feature"
                with self.assertRaisesRegex(ValueError, "não finita"):
                    selected_data(table, schema)

    def test_original_metadata_cannot_leak_after_payload_changes(self):
        for index in (0, 1):
            schema = copy.deepcopy(self.schema)
            schema["columns"][index]["role"] = "feature"
            schema["columns"][index]["description"] = "Medida numérica"
            schema["columns"][index]["name"] = "medida"
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "não pode ser entrada"):
                selected_data(self.table, schema)
        table = inspect_source("id,id,diagnosis,a\n1,10,M,2\n2,20,B,3")
        schema = schema_for(table)
        schema["columns"][1]["role"] = "feature"
        with self.assertRaisesRegex(ValueError, "não pode ser entrada"):
            selected_data(table, schema)
        table = inspect_source("1,M,2\n2,B,3", header="no")
        schema = schema_for(table)
        schema["columns"][1]["role"] = "feature"
        with self.assertRaisesRegex(ValueError, "não pode ser entrada"):
            selected_data(table, schema)

    def test_payload_indices_roles_and_descriptions_rejected(self):
        cases = []
        for index in (True, -1, 5, "0", 1):
            schema = copy.deepcopy(self.schema)
            schema["columns"][0]["index"] = index
            cases.append(schema)
        for field, value in (("role", "entrada"), ("role", []), ("description", 7), ("description", "a" * 1001)):
            schema = copy.deepcopy(self.schema)
            schema["columns"][0][field] = value
            cases.append(schema)
        cases.extend([{}, {**self.schema, "columns": self.schema["columns"][:-1]},
                      {**self.schema, "delimiter": ";"}, {**self.schema, "delimiter": []},
                      {**self.schema, "header": "no"}, {**self.schema, "columns": [None] * 5}])
        for schema in cases:
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                selected_data(self.table, schema)


if __name__ == "__main__":
    unittest.main()
