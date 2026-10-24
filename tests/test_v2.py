import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rede import (Config, MLP, SGDMomentum, hidden_size, load_dataset, load_model,
    save_model, run_experiment, predict_dataframe, momentum_next, fit_network,
    metric_highlights, model_details, save_prediction, experiment_signature)

ROOT=Path(__file__).resolve().parents[1]

class TestV2(unittest.TestCase):
    def test_13_formula_even_odd(self):
        for n,h in [(1,2),(19,11),(20,11),(30,16),(31,17),(32,17),(40,21),(64,33)]:
            self.assertEqual(hidden_size(n),h)
            net=MLP(1,n)
            self.assertEqual(net.architecture,[n,h,2])
            self.assertEqual(net.n_parameters,(n+1)*h+(h+1)*2)
        for n in [True,0,-1,1.5]:
            with self.assertRaises(ValueError):hidden_size(n)

    def test_14_momentum_analytical_negative_gradient(self):
        params={'w':np.array([1.,-1.])};opt=SGDMomentum(params)
        # g<0 deve AUMENTAR o peso, não trocar artificialmente o sinal de g.
        opt.step(params,{'w':np.array([-2.,3.])},.1,.1)
        np.testing.assert_allclose(params['w'],[1.2,-1.3])
        opt.step(params,{'w':np.array([-2.,3.])},.1,.1)
        np.testing.assert_allclose(params['w'],[1.42,-1.63])
        opt.reset();np.testing.assert_array_equal(opt.velocity['w'],[0.,0.])

    def test_15_adaptation_conditions(self):
        cfg=Config()
        self.assertEqual(momentum_next(.1,1.,.9,.8,cfg),(.11,'aumentar_ate_limite',False))
        self.assertEqual(momentum_next(.9,1.,.9,.8,cfg)[0],.9)
        self.assertEqual(momentum_next(.1,1.,1.1,.8,cfg),(.05,'reduzir_e_zerar_velocidade',True))
        self.assertEqual(momentum_next(.1,1.,.9,-.1,cfg),(.05,'reduzir_e_zerar_velocidade',True))
        self.assertEqual(momentum_next(.1,1.,1.,0.,cfg),(.1,'manter',False))
        self.assertEqual(momentum_next(.1,1.,1.1,-.8,replace(cfg,adaptive_momentum=False)),(.1,'fixo',False))

    def test_16_invalid_parameters(self):
        invalid=[{'momentum_initial':-.1},{'momentum_max':1.0},{'momentum_max':.05},
                 {'momentum_decrease':1.},{'momentum_increase':0.},{'adaptive_momentum':'true'},
                 {'snapshot_interval':0},{'optimizer':'invalid'},{'momentum_initial':float('nan')}]
        for args in invalid:
            with self.assertRaises(ValueError):Config(**args).validate()

    def test_17_dynamic_csv_training_full_workflow(self):
        rng=np.random.default_rng(24)
        X=rng.normal(size=(100,5)); y=np.array(['M','B']*50)
        f=pd.DataFrame(X,columns=['a','b','c','d','e']);f.insert(0,'diagnosis',y);f.insert(0,'id',[f'x{i}' for i in range(100)])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'dados.csv';f.to_csv(path,index=False)
            d=load_dataset(path);self.assertEqual(d.X.shape,(100,5))
            result=run_experiment(path,Path(tmp)/'run',Config(max_epochs=3,snapshot_interval=2))
            self.assertEqual(result['metrics']['architecture'],[5,4,2])
            net,mean,scale,meta=load_model(Path(tmp)/'run/modelo_final.npz')
            self.assertEqual(net.features,['a','b','c','d','e'])
            pred=predict_dataframe(Path(tmp)/'run/modelo_final.npz',f.drop(columns='diagnosis').head())
            self.assertEqual(len(pred),5)
            self.assertEqual(model_details(Path(tmp)/'run/modelo_final.npz')['n_parameters'],34)
            paths=list((Path(tmp)/'run').rglob('epoca_00000.npz'));self.assertTrue(len(paths)>=12)
            self.assertTrue(list((Path(tmp)/'run').rglob('historico.jsonl')))

    def test_18_feature_order_save_load_dynamic(self):
        rng=np.random.default_rng(28);X=rng.normal(size=(40,31));net=MLP(55,31)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';scaler=StandardScaler().fit(X);save_model(path,net,scaler,{})
            n2,mean,scale,meta=load_model(path)
            np.testing.assert_array_equal(n2.params['W1'],net.params['W1'])
            frame=pd.DataFrame(X,columns=net.features)
            a=predict_dataframe(path,frame);b=predict_dataframe(path,frame[frame.columns[::-1]])
            np.testing.assert_array_equal(a.score_M,b.score_M)

    def test_19_highlights_ties(self):
        highlights=metric_highlights([{'fold':1,'accuracy':1.,'sensitivity_M':.9},
                                     {'fold':2,'accuracy':1.,'sensitivity_M':.8},
                                     {'fold':3,'accuracy':.9,'sensitivity_M':1.}])
        self.assertEqual(highlights['accuracy']['ids'],[1,2])
        self.assertEqual(highlights['sensitivity_M']['ids'],[3])

    def test_20_epoch_history_fixed_and_adaptive(self):
        rng=np.random.default_rng(97);X=rng.normal(size=(100,4));y=(X[:,0]>0).astype(int)
        cfg=Config(max_epochs=8,momentum_initial=.1)
        net,h,_=fit_network(X,y,cfg,5,8)
        self.assertEqual(h[0]['momentum_used'],.1)
        self.assertTrue(all(0<=e['momentum_used']<=.9 for e in h))
        self.assertTrue(all(e['gradient_norm']>=0 for e in h))
        self.assertTrue(all(-1<=e['gradient_alignment']<=1 for e in h))
        self.assertTrue(all(0<=e['gradient_negative_fraction']<=1 for e in h))
        net2,h2,_=fit_network(X,y,cfg,5,8);np.testing.assert_array_equal(net.params['W1'],net2.params['W1'])
        _,hf,_=fit_network(X,y,replace(cfg,adaptive_momentum=False),5,8)
        self.assertTrue(all(e['momentum_used']==.1 for e in hf))

    def test_21_predictions_persist_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=ROOT/'resultados_exemplo/modelo_final.npz'
            text=(ROOT/'dados/exemplo_previsao.csv').read_text()
            r,p=save_prediction(model,text,Path(tmp));r2,p2=save_prediction(model,text,Path(tmp))
            self.assertNotEqual(p,p2);self.assertTrue((p/'previsoes.csv').is_file())
            self.assertTrue((p/'entrada.csv').is_file());self.assertTrue((p2/'metadados.json').is_file())
            pd.testing.assert_frame_equal(r,r2)

    def test_22_baseline_compatibility(self):
        net,mean,scale,meta=load_model(ROOT/'resultados_exemplo/modelo_final.npz')
        self.assertEqual(net.architecture,[30,16,2])
        self.assertEqual(len(experiment_signature(ROOT/'resultados_exemplo')),64)

    def test_23_backprop_dynamic_dimension(self):
        rng=np.random.default_rng(45);X=rng.normal(size=(5,5));Y=np.eye(2)[[0,1,1,0,1]];net=MLP(17,5)
        _,g=net.loss_gradients(X,Y,.01)
        for k,a in net.params.items():
            for ix in np.ndindex(a.shape):
                v=a[ix];a[ix]=v+1e-6;lp,_=net.loss_gradients(X,Y,.01)
                a[ix]=v-1e-6;lm,_=net.loss_gradients(X,Y,.01);a[ix]=v
                self.assertAlmostEqual((lp-lm)/2e-6,g[k][ix],places=6)

if __name__=='__main__':unittest.main()

class TestPersistenceFailure(unittest.TestCase):
    def test_24_preserves_failed_run_and_prevents_overwrite(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'failed'
            with patch('rede.select_epochs_and_refit',side_effect=FloatingPointError('falha controlada')):
                with self.assertRaises(FloatingPointError):
                    run_experiment(ROOT/'dados/cancear.txt',folder,Config(max_epochs=1))
            self.assertTrue((folder/'configuracao.json').is_file())
            self.assertTrue((folder/'dados_utilizados.txt').is_file())
            self.assertFalse((folder/'CONCLUIDO.txt').exists())
            self.assertEqual(json.loads((folder/'status.json').read_text())['state'],'interrompido_ou_falhou')
            with self.assertRaises(FileExistsError):run_experiment(ROOT/'dados/cancear.txt',folder,Config(max_epochs=1))
