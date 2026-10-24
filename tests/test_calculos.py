"""Conferência analítica da memória; não escreve modelos ou resultados."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from calculos import snapshot_report, update_report
from memoria import read_update, update_index
from rede import Config, MLP, fit_network, one_hot, recorder_for


class TestCalculos(unittest.TestCase):
    def setUp(self):
        self.net = MLP(2, 2, ['largura', 'altura'], ['Largura medida', 'Altura medida'])
        self.net.params = {
            'W1':np.array([[.5,-.2],[.3,.4]]), 'b1':np.array([.2,-.3]),
            'W2':np.array([[.8,-.6],[.4,.7]]), 'b2':np.array([-.1,.1]),
        }
        self.mean, self.scale = np.array([1.,10.]), np.array([2.,2.])
        self.raw = np.array([[3.,12.],[1.,8.],[5.,10.]])
        self.X = (self.raw-self.mean)/self.scale
        self.y = np.array([0,1,1])

    def snapshot(self, **kwargs):
        return snapshot_report(self.net,self.mean,self.scale,self.raw,self.y,**kwargs)

    def test_forward_real_values_and_no_parameter_mutation(self):
        before = {k:v.copy() for k,v in self.net.params.items()}
        report = self.snapshot(neuron=0,l2=.03)
        f = report['forward']
        np.testing.assert_allclose(f['standardized'],[1,1])
        np.testing.assert_allclose(f['z1'],[1,-.1],atol=1e-15)
        np.testing.assert_allclose(f['activation'],[1,0])
        np.testing.assert_allclose(f['logits'],[.7,-.5])
        np.testing.assert_allclose(f['scores'],[1/(1+np.exp(-1.2)),1/(1+np.exp(1.2))])
        self.assertEqual(f['target'],[1,0])
        self.assertEqual(report['architecture'],[2,2,2])
        self.assertTrue(report['verification']['ok'])
        self.assertIn('não reconstrução',report['notice'])
        self.assertEqual(report['steps'][0]['tables'][0]['rows'][0]['description'],'Largura medida')
        for k in before:
            np.testing.assert_array_equal(before[k],self.net.params[k])
        json.dumps(report,allow_nan=False)

    def test_selected_gradients_by_finite_differences(self):
        report = self.snapshot(neuron=0,l2=.02)
        selected = [('W1',(0,0),report['gradients']['W1'][0]),('W1',(1,0),report['gradients']['W1'][1]),
                    ('b1',(0,),report['gradients']['b1']),('W2',(0,0),report['gradients']['W2'][0]),
                    ('W2',(0,1),report['gradients']['W2'][1]),('b2',(0,),report['gradients']['b2'][0]),
                    ('b2',(1,),report['gradients']['b2'][1])]
        epsilon = 1e-6
        for key, index, analytic in selected:
            original = self.net.params[key][index]
            try:
                self.net.params[key][index]=original+epsilon
                plus,_=self.net.loss_gradients(self.X[:1],one_hot(self.y[:1]),.02)
                self.net.params[key][index]=original-epsilon
                minus,_=self.net.loss_gradients(self.X[:1],one_hot(self.y[:1]),.02)
            finally:
                self.net.params[key][index]=original
            self.assertAlmostEqual((plus-minus)/(2*epsilon),analytic,places=8)

    def test_dead_relu_blocks_data_gradient_but_not_l2(self):
        report = self.snapshot(neuron=1,l2=.1)
        self.assertEqual(report['forward']['relu_derivative'],0.)
        self.assertEqual(report['forward']['delta1_selected'],0.)
        np.testing.assert_allclose(report['gradients']['W1'],.1*self.net.params['W1'][:,1])
        np.testing.assert_allclose(report['gradients']['W2'],.1*self.net.params['W2'][1])
        self.assertEqual(report['gradients']['b1'],0.)
        self.net.params['W1'][:,1]=0
        self.net.params['b1'][1]=0
        at_zero = self.snapshot(neuron=1)
        self.assertEqual(at_zero['forward']['relu_derivative'],0.)
        self.assertTrue(any('derivada clássica não existe' in row[1] for row in at_zero['legend']))

    def test_stable_softmax_with_underflow_has_finite_cross_entropy(self):
        self.net.params['W2'][:]=0
        self.net.params['b2'][:]=[-1000,1000]
        report=self.snapshot(neuron=0)
        self.assertEqual(report['forward']['scores'],[0.,1.])
        self.assertEqual(report['forward']['cross_entropy'],2000.)
        self.assertEqual(report['forward']['delta2'],[-1.,1.])
        json.dumps(report,allow_nan=False)

    def test_invalid_indices_classes_and_scales(self):
        for arguments in [{'neuron':-1},{'neuron':2},{'neuron':True},{'neuron':0,'sample':3},{'neuron':0,'l2':-.1}]:
            with self.assertRaises(ValueError):self.snapshot(**arguments)
        with self.assertRaises(ValueError):
            snapshot_report(self.net,self.mean,[0,1],self.raw,self.y,0)
        with self.assertRaises(ValueError):
            snapshot_report(self.net,self.mean,self.scale,self.raw,[0,1,2],0)

    def payload(self, optimizer):
        before={k:v.copy() for k,v in self.net.params.items()}
        _,gradients=self.net.loss_gradients(self.X,one_hot(self.y,self.net.n_outputs),.03)
        payload={'step':3,'epoch':2,'batch':1,'optimizer':optimizer,'learning_rate':.02,'momentum':.4,'l2':.03,
                 'features':self.net.features,'feature_descriptions':self.net.feature_descriptions,
                 'mean':self.mean,'scale':self.scale,'X':self.X,'y':self.y,
                 'before':before,'gradients':gradients}
        if optimizer=='sgd_momentum':
            vb={k:np.full_like(v,.006) for k,v in before.items()}
            va={k:.4*vb[k]-.02*gradients[k] for k in before}
            payload.update(velocity_before=vb,velocity_after=va,after={k:before[k]+va[k] for k in before})
        else:
            mb={k:np.full_like(v,-.007) for k,v in before.items()}
            vb={k:np.full_like(v,.002) for k,v in before.items()}
            ma={k:.9*mb[k]+.1*gradients[k] for k in before}
            va={k:.999*vb[k]+.001*gradients[k]**2 for k in before}
            after={k:before[k]-.02*(ma[k]/(1-.9**3))/(np.sqrt(va[k]/(1-.999**3))+1e-8) for k in before}
            payload.update(adam_m_before=mb,adam_v_before=vb,adam_m_after=ma,adam_v_after=va,after=after)
        return payload

    def test_recorded_sgd_batch_average_negative_gradient_and_velocity(self):
        payload=self.payload('sgd_momentum')
        report=update_report(payload,0,sample=1)
        self.assertTrue(report['verification']['ok'])
        self.assertEqual(report['mode'],'update')
        self.assertEqual(report['sample'],1)
        self.assertEqual(report['batch']['size'],3)
        self.assertEqual(report['update']['step'],3)
        self.assertNotAlmostEqual(report['gradients']['W1'][0],report['batch']['gradients']['W1'][0],places=5)
        for row in report['batch']['gradient_rows']:
            self.assertAlmostEqual(sum(row['terms']['values'])+row['l2'],row['gradient'],places=13)
        negative=[row for row in report['update']['rows'] if row['gradient']<0]
        self.assertTrue(negative)
        for row in negative:
            self.assertGreater(row['gradient_part'],0)
            self.assertAlmostEqual(row['before']+.4*.006-.02*row['gradient'],row['after'],places=14)
        before_copy=copy.deepcopy(payload['before'])
        update_report(payload,1)
        for k in before_copy:np.testing.assert_array_equal(before_copy[k],payload['before'][k])
        json.dumps(report,allow_nan=False)

    def test_recorded_adam_uses_global_step_bias_correction(self):
        payload=self.payload('adam')
        report=update_report(payload,0)
        self.assertTrue(report['verification']['ok'])
        for row in report['update']['rows']:
            self.assertAlmostEqual(row['mhat'],row['m_after']/(1-.9**3),places=13)
            self.assertAlmostEqual(row['vhat'],row['v_after']/(1-.999**3),places=13)
            self.assertAlmostEqual(row['before']-.02*row['mhat']/(np.sqrt(row['vhat'])+1e-8),row['after'],places=14)
        self.assertIn('não é o μ adaptativo',report['steps'][-1]['note'])
        json.dumps(report,allow_nan=False)

    def test_momentum_origin_initial_and_epoch_alignment_are_checked(self):
        initial=self.payload('sgd_momentum')
        initial['epoch']=1
        initial['batch']=3
        initial['momentum_origin']={'kind':'initial','seed':22,'momentum_initial':.4}
        report=update_report(initial,0)
        self.assertTrue(report['verification']['ok'])
        self.assertTrue(any('Semente efetiva registrada: 22' in text for text in report['steps'][-2]['substitutions']))
        previous=np.arange(1,self.net.n_parameters+1,dtype=float)
        origin={'kind':'epoch_decision','epoch':1,'seed':22,'previous_loss':1.,'objective':.9,
                'previous_gradient':previous,'current_gradient':previous*.5,'alignment':1.,
                'momentum_used':.3,'momentum_next':.4,'action':'aumentar_ate_limite','reset':False,
                'adaptive_momentum':True,'momentum_max':.9,'momentum_increase':.1,
                'momentum_decrease':.5,'adaptation_tolerance':1e-8}
        payload=self.payload('sgd_momentum')
        payload['momentum_origin']=origin
        report=update_report(payload,0)
        self.assertTrue(report['verification']['ok'])
        check=report['momentum_origin_verification']
        self.assertAlmostEqual(check['expected_momentum'],.4)
        dotrow=report['steps'][-2]['tables'][0]['rows'][2]
        self.assertAlmostEqual(sum(dotrow['terms']['values']),np.dot(previous,previous*.5))
        origin['alignment']=.2
        self.assertFalse(update_report(payload,0)['verification']['ok'])

    def test_momentum_zero_norm_reduction_and_actual_velocity_reset(self):
        payload=self.payload('sgd_momentum')
        previous=np.arange(1,self.net.n_parameters+1,dtype=float)
        origin={'kind':'epoch_decision','epoch':1,'seed':22,'previous_loss':1.,'objective':.9,
                'previous_gradient':previous,'current_gradient':np.zeros_like(previous),'alignment':0.,
                'momentum_used':.4,'momentum_next':.4,'action':'manter','reset':False,
                'adaptive_momentum':True,'momentum_max':.9,'momentum_increase':.1,
                'momentum_decrease':.5,'adaptation_tolerance':1e-8}
        payload['momentum_origin']=origin
        self.assertTrue(update_report(payload,0)['verification']['ok'])
        origin.update(current_gradient=-previous,alignment=-1.,momentum_used=.8,action='reduzir_e_zerar_velocidade',reset=True)
        # Primeiro lote após zerar a velocidade não pode usar v anterior não zero.
        self.assertFalse(update_report(payload,0)['verification']['ok'])
        for k in payload['before']:
            payload['velocity_before'][k][:]=0
            payload['velocity_after'][k]=-.02*payload['gradients'][k]
            payload['after'][k]=payload['before'][k]+payload['velocity_after'][k]
        report=update_report(payload,0)
        self.assertTrue(report['verification']['ok'])
        self.assertEqual(report['momentum_origin_verification']['expected_action'],'reduzir_e_zerar_velocidade')

    def test_batch_gradients_by_finite_differences(self):
        payload=self.payload('sgd_momentum')
        report=update_report(payload,0)
        epsilon=1e-6
        for index in [(0,0),(1,0)]:
            original=self.net.params['W1'][index]
            try:
                self.net.params['W1'][index]=original+epsilon
                plus,_=self.net.loss_gradients(self.X,one_hot(self.y),.03)
                self.net.params['W1'][index]=original-epsilon
                minus,_=self.net.loss_gradients(self.X,one_hot(self.y),.03)
            finally:self.net.params['W1'][index]=original
            self.assertAlmostEqual((plus-minus)/(2*epsilon),report['batch']['gradients']['W1'][index[0]],places=8)

    def test_inconsistent_or_missing_history_is_never_marked_verified(self):
        payload=self.payload('sgd_momentum')
        payload['after']['W1'][1,1]+=.001  # Também confere parâmetros fora do neurônio selecionado.
        report=update_report(payload,0)
        self.assertFalse(report['verification']['ok'])
        self.assertAlmostEqual(report['verification']['update_max_abs_error'],.001)
        payload=self.payload('sgd_momentum')
        del payload['velocity_before']
        with self.assertRaisesRegex(ValueError,'incompleto'):update_report(payload,0)
        payload=self.payload('adam')
        payload['adam_v_before']['W1'][0,0]=-.01
        with self.assertRaisesRegex(ValueError,'negativo'):update_report(payload,0)


class TestCalculosMulticlass(unittest.TestCase):
    def setUp(self):
        # Contrato do motor: classes multiclasse na ordem lexical salva no modelo.
        self.classes = ['CYT', 'ERL', 'EXC', 'ME1', 'ME2', 'ME3', 'MIT', 'NUC', 'POX', 'VAC']
        self.net = MLP(21, 8, classes=self.classes)
        self.net.params['W1'] *= .1
        self.net.params['b1'][:] = 2.0  # Evita a quina da ReLU nas diferenças finitas.
        self.mean = np.arange(8, dtype=float)
        self.scale = np.arange(1, 9, dtype=float)
        self.X = np.random.default_rng(123).normal(size=(20, 8))
        self.raw = self.X * self.scale + self.mean
        self.y = np.tile(np.arange(10), 2)

    def snapshot(self, **kwargs):
        return snapshot_report(self.net, self.mean, self.scale, self.raw, self.y, **kwargs)

    def test_snapshot_explains_all_ten_outputs_without_binary_labels(self):
        before = {key:value.copy() for key,value in self.net.params.items()}
        report = self.snapshot(neuron=8, sample=9, l2=.03)
        self.assertEqual(report['architecture'], [8, 9, 10])
        self.assertEqual(report['classes'], self.classes)
        self.assertEqual(report['codes'][9], '0000000001')
        self.assertEqual(report['forward']['target'], [0.] * 9 + [1.])
        self.assertEqual(len(report['forward']['scores']), 10)
        self.assertEqual(len(report['gradients']['W2']), 10)
        self.assertEqual(len(report['gradients']['b2']), 10)
        self.assertTrue(report['verification']['ok'])
        self.assertEqual(len(report['steps'][3]['substitutions']), 10)
        self.assertEqual(len(report['steps'][3]['tables'][0]['rows']), 90)
        self.assertEqual(len(report['steps'][7]['tables'][0]['rows']), 10)
        self.assertIn('VAC = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]', report['steps'][0]['note'])
        self.assertIn('teto((8 + 10)/2) = 9', report['steps'][0]['substitutions'][0])
        serialized = json.dumps(report, ensure_ascii=False, allow_nan=False)
        for misleading in ('S = 2 saídas', 'para M e B', 'Os dois caminhos', 'Jacobiana 2 × 2'):
            self.assertNotIn(misleading, serialized)
        for key in before:
            np.testing.assert_array_equal(before[key], self.net.params[key])

    def test_all_selected_gradients_and_softmax_jacobian_by_finite_differences(self):
        j, sample, penalty = 4, 9, .02
        report = self.snapshot(neuron=j, sample=sample, l2=penalty)
        selected = [('W1', (i,j), report['gradients']['W1'][i]) for i in range(8)]
        selected += [('b1', (j,), report['gradients']['b1'])]
        selected += [('W2', (j,k), report['gradients']['W2'][k]) for k in range(10)]
        selected += [('b2', (k,), report['gradients']['b2'][k]) for k in range(10)]
        epsilon = 1e-6
        for key, index, analytic in selected:
            original = self.net.params[key][index]
            with self.subTest(parameter=key, index=index):
                try:
                    self.net.params[key][index] = original + epsilon
                    plus,_ = self.net.loss_gradients(self.X[sample:sample+1], one_hot(self.y[sample:sample+1],10), penalty)
                    self.net.params[key][index] = original - epsilon
                    minus,_ = self.net.loss_gradients(self.X[sample:sample+1], one_hot(self.y[sample:sample+1],10), penalty)
                finally:
                    self.net.params[key][index] = original
                self.assertAlmostEqual((plus-minus)/(2*epsilon), analytic, places=8)
        jacobian_table = report['steps'][6]['tables'][0]
        self.assertIn('10 × 10', jacobian_table['title'])
        actual = np.array([[row[f'output_{k}'] for k in range(10)] for row in jacobian_table['rows']])
        logits = np.array(report['forward']['logits'])
        numeric = np.zeros((10,10))
        def softmax(values):
            weights = np.exp(values-values.max())
            return weights/weights.sum()
        for k in range(10):
            shift = np.zeros(10)
            shift[k] = epsilon
            numeric[:,k] = (softmax(logits+shift)-softmax(logits-shift))/(2*epsilon)
        np.testing.assert_allclose(actual, numeric, atol=1e-10, rtol=1e-8)

    def test_sgd_and_adam_update_all_outputs_in_saved_order(self):
        for optimizer in ('sgd_momentum','adam'):
            with self.subTest(optimizer=optimizer):
                payload = TestCalculos.payload(self, optimizer)
                payload['classes'] = self.classes
                report = update_report(payload, 8, sample=9)
                self.assertEqual(report['classes'], self.classes)
                self.assertTrue(report['verification']['ok'])
                self.assertEqual(len(report['update']['rows']), 8 + 1 + 10 + 10)
                self.assertEqual(np.shape(report['batch']['delta2']), (20,10))
                batch_table = report['steps'][9]['tables'][0]
                self.assertEqual(batch_table['rows'][9]['class'], 'VAC')
                self.assertEqual(len([col for col in batch_table['columns'] if col['key'].startswith('delta_')]),10)
                for row in report['batch']['gradient_rows']:
                    self.assertAlmostEqual(sum(row['terms']['values']) + row['l2'], row['gradient'], places=13)
                # Não reinterpretar um registro multiclasse sem sua ordem de classes.
                del payload['classes']
                with self.assertRaisesRegex(ValueError, 'Formas ou valores inválidos'):
                    update_report(payload, 0)

    def test_recorded_real_ten_class_steps_match_engine_and_preserve_training(self):
        scaler = StandardScaler().fit(self.raw)
        X = scaler.transform(self.raw)
        for optimizer in ('sgd_momentum', 'adam'):
            with self.subTest(optimizer=optimizer), tempfile.TemporaryDirectory() as temporary:
                cfg = Config(max_epochs=1, batch_size=7, learning_rate=.01, optimizer=optimizer,
                             l2=.001, record_calculations=True, snapshot_interval=1)
                folder = Path(temporary)
                recorder = recorder_for(folder, 'fase_sintetica', scaler, cfg, {'test_only':True})
                recorded, history, epoch = fit_network(X, self.y, cfg, 21, 1,
                                                       classes=self.classes, recorder=recorder)
                plain, plain_history, plain_epoch = fit_network(X, self.y, cfg, 21, 1,
                                                               classes=self.classes)
                self.assertEqual(epoch, plain_epoch)
                self.assertEqual(history, plain_history)
                for key in recorded.params:
                    np.testing.assert_array_equal(recorded.params[key], plain.params[key])
                path = folder/'fase_sintetica'/'memoria_calculo.sqlite3'
                self.assertEqual(update_index(path), {'count':3, 'last_step':3})
                for step in range(1,4):
                    payload = read_update(path, step)
                    self.assertEqual(payload['classes'], self.classes)
                    report = update_report(payload, recorded.n_hidden-1)
                    self.assertTrue(report['verification']['ok'], report['verification'])
                    self.assertEqual(report['architecture'], [8,9,10])
                    self.assertEqual(report['classes'], self.classes)
                for key in recorded.params:
                    np.testing.assert_array_equal(payload['after'][key], recorded.params[key])


if __name__=='__main__':
    unittest.main()
