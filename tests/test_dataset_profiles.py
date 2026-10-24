"""Perfis locais conhecidos: posições reais, sem alterar ou treinar as bases."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import unittest

from entrada import (ABALONE_FEATURES, IRIS_FEATURES, WINE_FEATURES,
                     inspect_source, selected_data)


ABALONE_PREAMBLE = """1. Title of Database: Abalone data
6. Number of Attributes: 8
7. Attribute information:
Sex nominal M, F, and I (infant)
Length continuous mm Longest shell measurement
Diameter continuous mm perpendicular to length
Height continuous mm with meat in shell
Whole weight continuous grams whole abalone
Shucked weight continuous grams weight of meat
Viscera weight continuous grams gut weight
Shell weight continuous grams after drying
Rings integer +1.5 gives age in years
9. Class Distribution:
Class Examples
----- --------
1 1
2 1
----- ----
Total 4177

"""
IRIS_PREAMBLE = """1. Title: Iris Plants Database
6. Number of Attributes: 4 numeric, predictive attributes and the class
7. Attribute Information:
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris Setosa
-- Iris Versicolour
-- Iris Virginica
9. Class Distribution: 33.3% for each of 3 classes.

"""
WINE_PREAMBLE = """1. Title of Database: Wine recognition data
1) Alcohol
2) Malic acid
3) Ash
4) Alcalinity of ash
5) Magnesium
6) Total phenols
7) Flavanoids
8) Nonflavanoid phenols
9) Proanthocyanins
10)Color intensity
11)Hue
12)OD280/OD315 of diluted wines
13)Proline
6. Number of Attributes
13
NOTE: 1st attribute is class identifier (1-3)
9. Class Distribution: number of instances per class
class 1 59
class 2 71
class 3 48

"""
ABALONE_ROWS = ["M,0.455,0.365,0.095,0.514,0.2245,0.101,0.15,15",
                "F,0.53,0.42,0.135,0.677,0.2565,0.1415,0.21,9",
                "I,0.35,0.265,0.09,0.2255,0.0995,0.0485,0.07,7"]
IRIS_ROWS = ["5.1,3.5,1.4,0.2,Iris-setosa", "7.0,3.2,4.7,1.4,Iris-versicolor",
             "6.3,3.3,6.0,2.5,Iris-virginica"]
WINE_ROWS = ["1,14.23,1.71,2.43,15.6,127,2.8,3.06,.28,2.29,5.64,1.04,3.92,1065",
             "2,12.37,.94,1.36,10.6,88,1.98,.57,.28,.42,1.95,1.05,1.82,520",
             "3,12.86,1.35,2.32,18,122,1.51,1.25,.21,.94,4.1,.76,1.29,630"]
CASES = [('abalone_gender', ABALONE_PREAMBLE, ABALONE_ROWS, 0, ABALONE_FEATURES),
         ('iris', IRIS_PREAMBLE, IRIS_ROWS, 4, IRIS_FEATURES),
         ('wine', WINE_PREAMBLE, WINE_ROWS, 0, WINE_FEATURES)]


def schema_for(table):
    return {'delimiter': table['delimiter'], 'header': 'yes' if table['has_header'] else 'no',
            'columns': [{key: column[key] for key in ('index', 'role', 'description')}
                        for column in table['columns']]}


class DatasetProfileTests(unittest.TestCase):
    def test_all_profiles_keep_positions_and_first_sample(self):
        for name, preamble, rows, target, features in CASES:
            with self.subTest(profile=name):
                table = inspect_source('\ufeff' + preamble + '\n'.join(rows))
                self.assertEqual(table['source_format'], name)
                self.assertEqual(table['delimiter'], ',')
                self.assertFalse(table['has_header'])
                self.assertEqual(table['n_rows'], 3)
                self.assertEqual(table['n_columns'], len(features) + 1)
                self.assertEqual(table['source_lines'], list(range(preamble.count('\n') + 1, preamble.count('\n') + 4)))
                self.assertEqual(table['rows'], [row.split(',') for row in rows])
                self.assertEqual([column['key'] for column in table['columns']],
                                 [f'x{i+1}' for i in range(len(features) + 1)])
                self.assertEqual([column['role'] for column in table['columns']],
                                 ['target' if i == target else 'feature' for i in range(len(features) + 1)])
                selected = selected_data(table, schema_for(table))
                self.assertEqual(selected['features'], features)
                self.assertEqual(selected['ids'], [f'linha_{line}' for line in table['source_lines']])
                self.assertEqual(selected['X'][0], [float(value) for i, value in enumerate(rows[0].split(',')) if i != target])

    def test_actual_classes_come_from_rows_not_preamble_counts(self):
        for name, preamble, rows, target, _ in CASES:
            with self.subTest(profile=name):
                table = inspect_source(preamble + '\n'.join(rows + [rows[0]]))
                counts = {item['label']: item['count'] for item in table['target_summary']['classes']}
                self.assertEqual(counts, Counter(row.split(',')[target] for row in rows + [rows[0]]))
                self.assertEqual(sum(counts.values()), 4)

    def test_abalone_explicit_gender_target_keeps_rings_and_explains_infant(self):
        table = inspect_source(ABALONE_PREAMBLE + '\n'.join(ABALONE_ROWS))
        selected = selected_data(table, schema_for(table))
        self.assertEqual(selected['classes'], ['F', 'I', 'M'])
        self.assertEqual(selected['y'], [2, 0, 1])
        self.assertEqual(selected['features'][-1], 'rings')
        self.assertEqual([row[-1] for row in selected['X']], [15.0, 9.0, 7.0])
        self.assertIn('I = infantil', table['columns'][0]['description'])
        self.assertTrue(any('outro objetivo' in warning for warning in table['warnings']))

    def test_wine_numeric_class_is_target_not_id_or_feature(self):
        table = inspect_source(WINE_PREAMBLE + '\n'.join(WINE_ROWS))
        self.assertTrue(table['columns'][0]['numeric'])
        self.assertEqual(table['columns'][0]['role'], 'target')
        selected = selected_data(table, schema_for(table))
        self.assertEqual(selected['classes'], ['1', '2', '3'])
        self.assertEqual(selected['X'][0][0], 14.23)

    def test_known_target_cannot_be_forced_into_features(self):
        for name, preamble, rows, target, _ in CASES:
            with self.subTest(profile=name):
                table = inspect_source(preamble + '\n'.join(rows))
                schema = schema_for(table)
                schema['columns'][target]['role'] = 'feature'
                schema['columns'][target]['description'] = 'Mudar descrição não muda o alvo'
                with self.assertRaisesRegex(ValueError, 'não pode ser entrada'):
                    selected_data(table, schema)

    def test_duplicate_rows_are_preserved_and_source_ids_distinct(self):
        table = inspect_source(IRIS_PREAMBLE + '\n'.join([IRIS_ROWS[0], IRIS_ROWS[0], IRIS_ROWS[1]]))
        selected = selected_data(table, schema_for(table))
        self.assertEqual(len(selected['X']), 3)
        self.assertEqual(selected['X'][0], selected['X'][1])
        self.assertNotEqual(selected['ids'][0], selected['ids'][1])

    def test_bad_first_or_later_width_is_not_skipped(self):
        for name, preamble, rows, _, _ in CASES:
            for bad_index in (0, 1):
                for malformed in (','.join(rows[bad_index].split(',')[:-1]), rows[bad_index] + ',extra', 'linha inválida'):
                    with self.subTest(profile=name, bad_index=bad_index, malformed=malformed):
                        bad_rows = rows.copy()
                        bad_rows[bad_index] = malformed
                        line = preamble.count('\n') + bad_index + 1
                        with self.assertRaisesRegex(ValueError, f'Linha {line}:'):
                            inspect_source(preamble + '\n'.join(bad_rows))

    def test_invalid_numeric_measure_stays_required_until_explicitly_ignored(self):
        for name, preamble, rows, target, _ in CASES:
            with self.subTest(profile=name):
                index = 1 if target == 0 else 0
                first = rows[0].split(',')
                first[index] = 'inválida'
                table = inspect_source(preamble + '\n'.join([','.join(first), *rows[1:]]))
                self.assertEqual(table['columns'][index]['role'], 'feature')
                self.assertFalse(table['columns'][index]['numeric'])
                with self.assertRaisesRegex(ValueError, f"Linha {preamble.count(chr(10)) + 1}, x{index+1}"):
                    selected_data(table, schema_for(table))
                schema = schema_for(table)
                schema['columns'][index]['role'] = 'ignore'
                self.assertEqual(len(selected_data(table, schema)['X']), 3)

    def test_bad_label_is_preserved_and_rejected_not_discarded(self):
        for name, preamble, rows, target, _ in CASES:
            with self.subTest(profile=name):
                first = rows[0].split(',')
                first[target] = ''
                table = inspect_source(preamble + '\n'.join([','.join(first), *rows[1:]]))
                self.assertEqual(table['n_rows'], 3)
                self.assertFalse(table['target_summary']['supported'])
                with self.assertRaisesRegex(ValueError, 'rótulo não vazio'):
                    selected_data(table, schema_for(table))

    def test_fixed_layout_rejects_header_and_wrong_separator(self):
        for name, preamble, rows, _, _ in CASES:
            for options in ({'header': 'yes'}, {'delimiter': 'whitespace'}, {'delimiter': ';'}):
                with self.subTest(profile=name, options=options):
                    with self.assertRaisesRegex(ValueError, 'usa vírgula, sem cabeçalho'):
                        inspect_source(preamble + '\n'.join(rows), **options)

    def test_missing_preamble_markers_fail_instead_of_falling_back(self):
        mutations = [(ABALONE_PREAMBLE, ABALONE_ROWS, 'Rings integer +1.5 gives age in years'),
                     (ABALONE_PREAMBLE, ABALONE_ROWS, 'Total 4177'),
                     (IRIS_PREAMBLE, IRIS_ROWS, '4. petal width in cm'),
                     (IRIS_PREAMBLE, IRIS_ROWS, '9. Class Distribution: 33.3% for each of 3 classes.'),
                     (WINE_PREAMBLE, WINE_ROWS, '13)Proline'),
                     (WINE_PREAMBLE, WINE_ROWS, 'NOTE: 1st attribute is class identifier (1-3)'),
                     (WINE_PREAMBLE, WINE_ROWS, 'class 3 48')]
        for preamble, rows, marker in mutations:
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(ValueError, 'Preâmbulo .* incompleto'):
                    inspect_source(preamble.replace(marker, '') + '\n'.join(rows))

    def test_incomplete_distribution_cannot_search_past_malformed_data(self):
        for preamble, rows, marker in [(WINE_PREAMBLE, WINE_ROWS, 'class 2 71'),
                                       (ABALONE_PREAMBLE, ABALONE_ROWS, '2 1')]:
            with self.subTest(marker=marker):
                source = preamble.replace(marker, 'documentação inesperada') + '\n'.join(rows)
                with self.assertRaisesRegex(ValueError, 'Preâmbulo .* incompleto'):
                    inspect_source(source)

    def test_physical_lines_preserved_with_blank_lines_and_crlf(self):
        for name, preamble, rows, _, _ in CASES:
            with self.subTest(profile=name):
                text = ('\n\n' + preamble + rows[0] + '\n\n' + rows[1] + '\n').replace('\n', '\r\n')
                table = inspect_source(text)
                first_line = preamble.count('\n') + 3
                self.assertEqual(table['source_lines'], [first_line, first_line + 2])

    def test_generic_named_targets_work_in_any_position(self):
        for header in ['sex', 'sexo', 'gender', 'gênero', 'species', 'espécie', 'cultivar']:
            with self.subTest(header=header):
                table = inspect_source(f'medida,{header},id\n1,A,001\n2,B,002')
                self.assertTrue(table['has_header'])
                self.assertEqual([column['role'] for column in table['columns']], ['feature', 'target', 'id'])
                self.assertEqual(selected_data(table, schema_for(table))['X'], [[1.0], [2.0]])

    def test_generic_ambiguous_target_is_not_guessed_from_numeric_cardinality(self):
        table = inspect_source('1,0,3.2\n2,1,4.4\n1,0,5.5', header='no')
        self.assertIsNone(table['target_summary'])
        self.assertTrue(all(column['role'] == 'feature' for column in table['columns']))

    def test_descriptions_are_annotations_and_selection_preserves_order(self):
        table = inspect_source(WINE_PREAMBLE + '\n'.join(WINE_ROWS))
        original = deepcopy(table)
        schema = schema_for(table)
        schema['columns'][1]['description'] = 'Descrição editada'
        schema['columns'][2]['role'] = 'ignore'
        selected = selected_data(table, schema)
        self.assertEqual(selected['features'], [WINE_FEATURES[0], *WINE_FEATURES[2:]])
        self.assertEqual(selected['descriptions'][0], 'Descrição editada')
        self.assertEqual(table, original)


DATA_FOLDER = Path(__file__).resolve().parents[3] / 'base_dados'


@unittest.skipUnless(all((DATA_FOLDER / name).is_file() for name in ['genero.txt', 'iris.txt', 'wine.txt']),
                     'Arquivos reais locais opcionais não disponíveis.')
class LocalDatasetProfileTests(unittest.TestCase):
    def test_real_files_are_read_only_and_counts_come_from_records(self):
        expected = [('genero.txt', 'abalone_gender', 4177, 9, 147, {'F': 1307, 'I': 1342, 'M': 1528}),
                    ('iris.txt', 'iris', 150, 5, 71, {'Iris-setosa': 50, 'Iris-versicolor': 50, 'Iris-virginica': 50}),
                    ('wine.txt', 'wine', 178, 14, 102, {'1': 59, '2': 71, '3': 48})]
        for name, profile, rows, columns, first_line, counts in expected:
            with self.subTest(file=name):
                path = DATA_FOLDER / name
                before = path.read_bytes()
                table = inspect_source(before.decode('utf8'))
                selected = selected_data(table, schema_for(table))
                self.assertEqual(table['source_format'], profile)
                self.assertEqual((table['n_rows'], table['n_columns']), (rows, columns))
                self.assertEqual(table['source_lines'][0], first_line)
                self.assertEqual({item['label']: item['count'] for item in table['target_summary']['classes']}, counts)
                self.assertEqual(len(selected['X']), rows)
                self.assertEqual(len(selected['X'][0]), columns - 1)
                self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
