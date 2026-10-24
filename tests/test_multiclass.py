"""Multiclasse e separação de grupos; treinos só sintéticos de uma época."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from entrada import inspect_source
from memoria import read_update
from rede import (Config, MLP, class_codes, connected_groups, dataset_protocol,
                  fit_network, grouped_holdout, load_dataset, load_model,
                  load_selected_dataset, make_folds, metrics, one_hot,
                  predict_dataframe, run_experiment, save_model, wta)


def multiclass_fixture(n_per_class=6):
    rows = ['id,classe,a,b,c']
    for label in range(10):
        for sample in range(n_per_class):
            rows.append(f'id_{label}_{sample},C{label},{label + sample / 17},{sample - label / 11},{label * 10 + sample}')
    text = '\n'.join(rows)
    table = inspect_source(text)
    schema = {'delimiter':table['delimiter'],'header':'yes',
              'columns':[{k:c[k] for k in ('index','description','role')} for c in table['columns']]}
    return text.encode(), schema


class MulticlassTests(unittest.TestCase):
    def test_ten_outputs_encoding_architecture_and_all_gradients(self):
        classes = [f'C{i}' for i in range(10)]
        net = MLP(31, 8, classes=classes)
        self.assertEqual(net.architecture, [8, 9, 10])
        X = np.random.default_rng(2026).normal(size=(10,8))
        targets = one_hot(np.arange(10), 10)
        np.testing.assert_array_equal(targets, np.eye(10))
        np.testing.assert_array_equal(wta(targets), np.arange(10))
        self.assertEqual(class_codes(classes)[0], '1000000000')
        self.assertEqual(class_codes(classes)[-1], '0000000001')
        np.testing.assert_allclose(net.predict_proba(X).sum(axis=1), 1)
        _, gradients = net.loss_gradients(X, targets, .001)
        for key, values in net.params.items():
            for index in np.ndindex(values.shape):
                original = values[index]
                values[index] = original + 1e-6
                plus = net.loss_gradients(X, targets, .001)[0]
                values[index] = original - 1e-6
                minus = net.loss_gradients(X, targets, .001)[0]
                values[index] = original
                self.assertAlmostEqual((plus-minus)/2e-6, gradients[key][index], places=6)

    def test_connected_components_join_id_and_features_transitively(self):
        groups = connected_groups(['A','A','B','C','D'],
                                  np.array([[1.,2.],[3.,4.],[3.,4.],[5.,6.],[0.,-0.]]))
        self.assertEqual(groups.tolist(), [0,0,0,1,2])
        self.assertEqual(connected_groups(['A','B'],np.array([[0.,-0.],[-0.,0.]])).tolist(), [0,0])

    def test_dataset_preserves_duplicates_and_reports_modes(self):
        raw, schema = multiclass_fixture()
        # Repeated ID links different measurements without removing either row.
        raw = raw.replace(b'id_0_1', b'id_0_0')
        data = load_selected_dataset(raw, schema)
        self.assertEqual(data.X.shape, (60,3))
        self.assertEqual(data.classes, [f'C{i}' for i in range(10)])
        self.assertEqual(data.ids[0], data.ids[1])
        self.assertEqual(data.groups[0], data.groups[1])
        protocol = dataset_protocol(data)
        self.assertEqual(protocol['supported_modes'], ['kfold5'])
        self.assertEqual(protocol['n_groups'], 59)
        self.assertEqual(protocol['class_group_counts']['C0'], 5)
        too_few = raw.replace(b'id_0_2', b'id_0_0')
        with self.assertRaisesRegex(ValueError, '5 grupos'):
            load_selected_dataset(too_few, schema)

    def test_outer_inner_groups_all_classes_coverage_and_determinism(self):
        raw, schema = multiclass_fixture()
        data = load_selected_dataset(raw.replace(b'id_0_1',b'id_0_0'), schema)
        cfg = Config(mode='kfold5')
        splits = make_folds(data.y,cfg,data.groups)
        repeated = make_folds(data.y,cfg,data.groups)
        coverage = np.zeros(len(data.y),int)
        for fold, ((train,test),(train2,test2)) in enumerate(zip(splits,repeated),1):
            np.testing.assert_array_equal(train,train2)
            np.testing.assert_array_equal(test,test2)
            coverage[test] += 1
            self.assertFalse(set(data.groups[train]) & set(data.groups[test]))
            self.assertEqual(len(np.unique(data.y[test])), 10)
            a,b = grouped_holdout(data.y[train],data.groups[train],.15,42+1000*fold)
            self.assertLessEqual(len(b)/len(train),.30)
            self.assertEqual(len(np.unique(data.y[train][a])),10)
            self.assertEqual(len(np.unique(data.y[train][b])),10)
            self.assertFalse(set(data.groups[train][a]) & set(data.groups[train][b]))
        np.testing.assert_array_equal(coverage,1)
        for mode in ('variavel','kfold10'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError,'K-fold 5'):
                make_folds(data.y,Config(mode=mode),data.groups)

    def test_macro_metrics_use_every_class_and_stable_confusion_order(self):
        classes = ['A','B','C']
        y = np.array([0,0,1,1,2,2])
        scores = one_hot(np.array([0,1,1,1,2,0]),3)
        result = metrics(y,scores,classes)
        self.assertEqual(result['confusion_matrix'],[[1,1,0],[0,2,0],[1,0,1]])
        self.assertEqual(result['confusion_matrix_order'],classes)
        self.assertAlmostEqual(result['accuracy'],4/6)
        self.assertAlmostEqual(result['precision_macro'],(.5+2/3+1)/3)
        self.assertAlmostEqual(result['recall_macro'],(.5+1+.5)/3)
        self.assertEqual(result['per_class']['C']['support'],2)
        with self.assertRaisesRegex(ValueError,'Classe C ausente'):
            metrics(y[:4],scores[:4],classes)

    def test_save_load_predict_and_legacy_missing_class_metadata(self):
        X = np.arange(24,dtype=float).reshape(8,3)
        scaler = StandardScaler().fit(X)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.npz'
            net = MLP(1,3,classes=['A','B','C'])
            save_model(path,net,scaler,{})
            loaded,mean,scale,meta = load_model(path)
            self.assertEqual(loaded.classes,['A','B','C'])
            self.assertEqual(meta['classes'],loaded.classes)
            np.testing.assert_array_equal(net.predict_proba(scaler.transform(X)),loaded.predict_proba((X-mean)/scale))
            predicted = predict_dataframe(path,pd.DataFrame(X,columns=net.features))
            self.assertTrue({'score_A','score_B','score_C'} <= set(predicted.columns))
            self.assertTrue(predicted.codigo_WTA.str.len().eq(3).all())
            legacy = MLP(1,3)
            save_model(path,legacy,scaler,{})
            with np.load(path,allow_pickle=False) as saved:
                arrays = {key:saved[key].copy() for key in saved.files if key not in {'classes','codes'}}
            metadata = json.loads(str(arrays['metadata'].item()))
            for key in ('classes','output_order','codes'): metadata.pop(key,None)
            arrays['metadata']=np.array(json.dumps(metadata))
            fallback=Path(temporary)/'legacy.npz'
            np.savez_compressed(fallback,**arrays)
            self.assertEqual(load_model(fallback)[0].classes,['M','B'])

    def test_one_epoch_synthetic_full_workflow_and_saved_group_audit(self):
        raw,schema = multiclass_fixture()
        raw = raw.replace(b'id_0_1',b'id_0_0')
        data = load_selected_dataset(raw,schema)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder/'synthetic.csv'; source.write_bytes(raw)
            output = folder/'run'
            result = run_experiment(source,output,Config(mode='kfold5',max_epochs=1,batch_size=32),input_schema=schema)
            self.assertEqual(result['metrics']['classes'],data.classes)
            self.assertEqual(result['metrics']['architecture'],[3,7,10])
            self.assertIn('f1_macro',result['fold_summary']['metrics'])
            oof = pd.read_csv(output/'previsoes_oof.csv')
            self.assertEqual(len(oof),60)
            self.assertTrue(oof.vezes_em_teste.eq(1).all())
            self.assertEqual(len([name for name in oof if name.startswith('score_')]),10)
            self.assertEqual((output/'dados_utilizados.txt').read_bytes(),raw)
            self.assertEqual(len(pd.read_csv(output/'grupos_auditoria.csv')),60)
            for fold in (output/'folds').iterdir():
                ix=json.loads((fold/'indices.json').read_text())
                roles=[ix['inner_fit_0based'],ix['inner_validation_0based'],ix['test_0based']]
                for a in range(3):
                    for b in range(a): self.assertFalse(set(data.groups[roles[a]]) & set(data.groups[roles[b]]))
                norm=pd.read_csv(fold/'normalizacao.csv')
                self.assertAlmostEqual(ix['validation_fraction_effective'],len(ix['inner_validation_0based'])/len(ix['train_0based']))
                self.assertEqual(ix['validation_fraction_requested'],.15)
                np.testing.assert_allclose(norm.mean_fit_interno,data.X[ix['inner_fit_0based']].mean(axis=0))
                net=load_model(fold/'modelo.npz')[0]
                self.assertEqual(net.classes,data.classes)
                self.assertEqual(read_update(fold/'checkpoints'/'retreino'/'memoria_calculo.sqlite3',1)['classes'],data.classes)
            with self.assertRaises(FileExistsError):
                run_experiment(source,output,Config(mode='kfold5',max_epochs=1),input_schema=schema)

    def test_both_optimizers_keep_multiclass_shapes_and_reproducibility(self):
        X=np.random.default_rng(8).normal(size=(30,3)); y=np.arange(30)%3
        for optimizer in ('sgd_momentum','adam'):
            cfg=Config(optimizer=optimizer,max_epochs=1,record_calculations=False)
            first=fit_network(X,y,cfg,31,1,classes=['A','B','C'])[0]
            second=fit_network(X,y,cfg,31,1,classes=['A','B','C'])[0]
            self.assertEqual(first.architecture,[3,3,3])
            np.testing.assert_array_equal(first.predict_proba(X),second.predict_proba(X))

    def test_inner_validation_never_exceeds_thirty_percent(self):
        y=np.repeat(np.arange(10),2)
        with self.assertRaisesRegex(ValueError,'no máximo 30%'):
            grouped_holdout(y,np.arange(len(y)),.15,42)


if __name__ == '__main__': unittest.main()
