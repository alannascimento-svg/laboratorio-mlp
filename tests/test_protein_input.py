"""Inspeção de proteínas: preserva dados e identifica classes dinamicamente."""
from collections import Counter
import copy
import unittest

from entrada import inspect_source, selected_data, PROTEIN_FEATURES


# Pequena documentação sintética no formato reconhecido; nenhum resultado
# histórico ou treinamento é usado como fixture.
PROTEIN_PREAMBLE = """1. Title: Protein Localization Sites

6. Number of Attributes.
         for Yeast dataset:   9 ( 8 predictive, 1 name )
7. Attribute Information.
  1.  Sequence Name: Accession number for the SWISS-PROT database
  2.  mcg: McGeoch's method for signal sequence recognition.
  3.  gvh: von Heijne's method for signal sequence recognition.
  4.  alm: Score of the ALOM prediction program.
  5.  mit: Score of mitochondrial protein analysis.
  6.  erl: Presence of HDEL substring.
  7.  pox: Peroxisomal targeting signal.
  8.  vac: Score of vacuolar protein analysis.
  9.  nuc: Score of nuclear localization signals.

9. Class Distribution. The class is the localization site. Please see Nakai &
                       Kanehisa referenced above for more details.
  CYT (cytosolic or cytoskeletal)                    463
  NUC (nuclear)                                      429
  MIT (mitochondrial)                                244
  ME3 (membrane protein, no N-terminal signal)       163
  ME2 (membrane protein, uncleaved signal)            51
  ME1 (membrane protein, cleaved signal)              44
  EXC (extracellular)                                 37
  VAC (vacuolar)                                      30
  POX (peroxisomal)                                   20
  ERL (endoplasmic reticulum lumen)                    5

"""
PROTEIN_ROWS = [
    "ADT1_YEAST  0.58  0.61  0.47  0.13  0.50  0.00  0.48  0.22  MIT",
    "ADT2_YEAST  0.43  0.67  0.48  0.27  0.50  0.00  0.53  0.22  MIT",
    "AAR2_YEAST  0.58  0.44  0.57  0.13  0.50  0.00  0.54  0.22  NUC",
]
PROTEIN_SAMPLE = PROTEIN_PREAMBLE + "\n".join(PROTEIN_ROWS) + "\n"


def schema_for(table):
    return {"delimiter": table["delimiter"],
            "header": "yes" if table["has_header"] else "no",
            "columns": [{"index": column["index"], "description": column["description"],
                         "role": column["role"]} for column in table["columns"]]}


class ProteinInputTests(unittest.TestCase):
    def test_preamble_detects_columns_and_preserves_first_sample(self):
        table = inspect_source("\ufeff" + PROTEIN_SAMPLE)
        self.assertEqual(table["source_format"], "protein_localization")
        self.assertEqual(table["delimiter"], "whitespace")
        self.assertFalse(table["has_header"])
        self.assertEqual((table["n_rows"], table["n_columns"]), (3, 10))
        self.assertEqual(table["source_lines"], list(range(PROTEIN_PREAMBLE.count("\n") + 1,
                                                          PROTEIN_PREAMBLE.count("\n") + 4)))
        self.assertEqual(table["rows"], [row.split() for row in PROTEIN_ROWS])
        self.assertEqual(table["rows"][0][0], "ADT1_YEAST")
        self.assertEqual([column["key"] for column in table["columns"]],
                         [f"x{i}" for i in range(1, 11)])
        self.assertEqual([column["name"] for column in table["columns"]],
                         ["id", *PROTEIN_FEATURES, "classe"])
        self.assertEqual([column["role"] for column in table["columns"]],
                         ["id"] + ["feature"] * 8 + ["target"])
        self.assertTrue(all(len(column["description"]) > len(column["name"])
                            for column in table["columns"]))

    def test_summary_counts_actual_labels_not_documentation_legend(self):
        table = inspect_source(PROTEIN_SAMPLE)
        self.assertEqual(table["target_summary"], {
            "index": 9, "key": "x10", "supported": True,
            "classes": [{"label": "MIT", "count": 2}, {"label": "NUC", "count": 1}]})
        self.assertFalse(any("somente M/B" in warning for warning in table["warnings"]))
        # As dez classes e suas contagens são derivadas de todas as linhas,
        # inclusive das que ficam além da prévia de oito valores.
        labels = ["CYT", "NUC", "MIT", "ME3", "ME2", "ME1", "EXC", "VAC", "POX", "ERL", "ERL"]
        rows = [f"sample_{i} 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 {label}"
                for i, label in enumerate(labels)]
        result = inspect_source(PROTEIN_PREAMBLE + "\n".join(rows))
        counts = {item["label"]: item["count"] for item in result["target_summary"]["classes"]}
        self.assertEqual(counts, Counter(labels))
        self.assertEqual(len(result["columns"][-1]["preview"]), 8)
        self.assertEqual(sum(counts.values()), 11)

    def test_inspection_preserves_duplicate_rows_ids_and_non_numeric_measure(self):
        rows = [PROTEIN_ROWS[0], PROTEIN_ROWS[0], PROTEIN_ROWS[1].replace("0.43", "inválida")]
        table = inspect_source(PROTEIN_PREAMBLE + "\n".join(rows))
        self.assertEqual(table["rows"], [row.split() for row in rows])
        self.assertEqual(table["n_rows"], 3)
        self.assertEqual(table["columns"][1]["role"], "ignore")
        original = copy.deepcopy(table)
        with self.assertRaisesRegex(ValueError, "pelo menos duas classes"):
            selected_data(table, schema_for(table))
        self.assertEqual(table, original)

    def test_id_and_target_cannot_be_forced_into_inputs(self):
        table = inspect_source(PROTEIN_SAMPLE)
        for index in (0, 9):
            schema = schema_for(table)
            schema["columns"][index]["role"] = "feature"
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "não pode ser entrada"):
                selected_data(table, schema)

    def test_ragged_sample_is_an_error_including_first_sample(self):
        first_line = PROTEIN_PREAMBLE.count("\n") + 1
        for position in (0, 1):
            rows = PROTEIN_ROWS.copy()
            rows[position] = rows[position].rsplit("  ", 1)[0]
            with self.subTest(position=position), self.assertRaisesRegex(ValueError, "colunas"):
                inspect_source(PROTEIN_PREAMBLE + "\n".join(rows))
        with self.assertRaisesRegex(ValueError, f"Linha {first_line}:.*exige 10 colunas"):
            inspect_source(PROTEIN_PREAMBLE + " ".join(PROTEIN_ROWS[0].split()[:-1])
                           + "\n" + PROTEIN_ROWS[1])
        # Uma linha textual após a legenda também é dado inválido; não é
        # descartada por procurar a primeira linha que parece uma amostra.
        with self.assertRaisesRegex(ValueError, "colunas"):
            inspect_source(PROTEIN_PREAMBLE + "AMOSTRA_TRUNCADA\n" + PROTEIN_ROWS[0])
        broken_second = PROTEIN_PREAMBLE + PROTEIN_ROWS[0] + "\nAMOSTRA_TRUNCADA\n"
        with self.assertRaisesRegex(ValueError, f"Linha {first_line + 1}: esperadas 10 colunas"):
            inspect_source(broken_second)

    def test_recognized_preamble_cannot_override_required_layout(self):
        for kwargs in ({"header": "yes"}, {"delimiter": ","}, {"delimiter": "tab"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "whitespace.*sem cabeçalho"):
                inspect_source(PROTEIN_SAMPLE, **kwargs)
        for changed in (
                PROTEIN_SAMPLE.replace("8 predictive", "7 predictive"),
                PROTEIN_SAMPLE.replace("mcg:", "other:"),
                PROTEIN_SAMPLE.replace("  ERL (endoplasmic reticulum lumen)                    5\n", "")):
            with self.subTest(changed=changed[:60]), self.assertRaisesRegex(ValueError, "Preâmbulo Protein"):
                inspect_source(changed)
        with self.assertRaisesRegex(ValueError, "colunas"):
            inspect_source("Relatório genérico\n" + "\n".join(PROTEIN_ROWS))
        for annotation in ('37S', '37*'):
            changed = PROTEIN_SAMPLE.replace('37\n', annotation + '\n')
            self.assertEqual(inspect_source(changed)['rows'], inspect_source(PROTEIN_SAMPLE)['rows'])

    def test_whitespace_auto_explicit_and_decimal_comma_binary(self):
        plain = "id classe medida outra\n001   M\t1.25  2.5\n002\t B 3.5  4.75\n"
        for delimiter in ("auto", "whitespace"):
            table = inspect_source(plain, delimiter=delimiter)
            self.assertEqual(table["delimiter"], "whitespace")
            self.assertEqual(table["source_format"], "delimited")
            selected = selected_data(table, schema_for(table))
            self.assertEqual(selected["ids"], ["001", "002"])
            self.assertEqual(selected["X"], [[1.25, 2.5], [3.5, 4.75]])
            self.assertEqual(table["target_summary"]["supported"], True)
        decimal = inspect_source(plain.replace(".", ","), delimiter="whitespace")
        self.assertEqual(selected_data(decimal, schema_for(decimal))["X"], [[1.25, 2.5], [3.5, 4.75]])

    def test_whitespace_blank_lines_and_no_header_preserve_sample_positions(self):
        table = inspect_source("\n  id  M  1 2\n\n001\tB 3 4\n", delimiter="whitespace")
        self.assertFalse(table["has_header"])
        self.assertEqual(table["source_lines"], [2, 4])
        self.assertEqual(table["n_rows"], 2)
        with self.assertRaisesRegex(ValueError, "Linha 3: esperadas 3 colunas; encontradas 2"):
            inspect_source("id classe medida\n001 M 1\n002 B", delimiter="whitespace")

    def test_summary_is_not_a_promise_of_valid_protocol(self):
        table = inspect_source("id,classe,medida\n1,M,2")
        self.assertFalse(table["target_summary"]["supported"])
        table = inspect_source("id,classe,medida\n1,,2")
        self.assertFalse(table["target_summary"]["supported"])
        self.assertEqual(table["target_summary"]["classes"], [{"label": "", "count": 1}])
        self.assertIsNone(inspect_source("id,medida\n1,2")["target_summary"])


if __name__ == "__main__":
    unittest.main()
