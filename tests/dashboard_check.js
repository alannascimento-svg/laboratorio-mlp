'use strict';

// State/markup regression checks. Browser layout and user interaction need a browser check.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const base = path.join(__dirname, '..');
const esc = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
const classes = ['CYT', 'ERL', 'EXC', 'ME1', 'ME2', 'ME3', 'MIT', 'NUC', 'POX', 'VAC'];

function node() {
  return {innerHTML: '', textContent: '', value: '', disabled: false, hidden: false,
    dataset: {}, style: {}, attributes: {}, listeners: {}, options: [], selectedOptions: [{text:'Run test'}],
    addEventListener(name, callback) { this.listeners[name] = callback; },
    setAttribute(name, value) { this.attributes[name] = value; },
    querySelectorAll() { return []; }, contains() { return false; }};
}

function checkNetwork() {
  const window = {}, context = vm.createContext({window, document:{activeElement:null}, setTimeout, CustomEvent:class {}});
  vm.runInContext(fs.readFileSync(path.join(base, 'network.js'), 'utf8'), context, {filename:'network.js'});
  function graph() {
    const element = node(), stage = node(), zoom = node();
    const buttons = ['in','out','fit','center'].map(action => ({...node(), dataset:{networkAction:action}}));
    element.querySelector = selector => selector === '[data-network-stage]' ? stage : selector === '[data-network-zoom]' ? zoom : null;
    element.querySelectorAll = selector => selector === '[data-network-action]' ? buttons : [];
    return element;
  }
  function data(labels) {
    const d=8,h=Math.ceil((d+labels.length)/2),o=labels.length;
    return {classes:labels, architecture:[d,h,o], features:Array.from({length:d},(_,i)=>'f'+i), mean:Array(d).fill(0), scale:Array(d).fill(1),
      params:{W1:Array.from({length:d},()=>Array(h).fill(.1)), W2:Array.from({length:h},()=>Array(o).fill(-.2)), b1:Array(h).fill(0), b2:Array(o).fill(.01)}};
  }
  const multi=graph();window.LabNetwork.render(multi,data(classes),0);
  assert.match(multi.innerHTML,/Rede real 8-9-10/);
  assert.match(multi.innerHTML,/10 SAÍDAS/);
  assert.equal((multi.innerHTML.match(/<circle cx="1025"/g)||[]).length,10);
  for(const label of classes)assert.ok(multi.innerHTML.includes(label),label+' is visible');
  assert.match(multi.innerHTML,/1000000000/);
  assert.doesNotMatch(multi.innerHTML,/Maligno|Benigno|undefined/);
  const binary=graph(),legacy=data(['M','B']);delete legacy.classes;
  window.LabNetwork.render(binary,legacy,0);
  assert.match(binary.innerHTML,/Rede real 8-5-2/);
  assert.match(binary.innerHTML,/10 · Maligno/);
  assert.match(binary.innerHTML,/01 · Benigno/);
}

function dashboard() {
  const elements = new Map(), get = id => { if(!elements.has(id))elements.set(id,node());return elements.get(id); };
  get('mode').options=['variavel','kfold5','kfold10'].map(value=>({value,disabled:false}));
  get('mode').value='kfold10';get('network').hidden=true;get('run').value='test';
  let responses = {};
  const context=vm.createContext({document:{getElementById:get,querySelectorAll:()=>[]},
    LabNetwork:{esc,render(){}}, URLSearchParams, setTimeout, clearTimeout, setInterval, clearInterval,
    fetch:async url=>{if(!(url in responses))throw Error('Unexpected API: '+url);return {ok:true,json:async()=>responses[url]};},
  });
  const html=fs.readFileSync(path.join(base,'interface.html'),'utf8');
  const script=html.slice(html.indexOf("'use strict';"),html.lastIndexOf('</script>')).replace('initSettings().catch(e=>notify(e.message));','');
  vm.runInContext(script,context,{filename:'interface.html:inline'});
  return {context,get,setResponses(value){responses=value;}};
}

async function checkDashboard() {
  const f=dashboard();
  const inspected={n:1484,classes,class_counts:Object.fromEntries(classes.map(label=>[label,20])),architecture:[8,9,10],n_parameters:181,
    protocol:{supported_modes:['kfold5'],grouped:true,n_groups:1450}};
  f.context.inputValidated(inspected);
  assert.equal(f.get('train').disabled,false);
  assert.equal(f.get('mode').value,'kfold5');
  assert.deepEqual(f.get('mode').options.map(option=>option.disabled),[true,false,true]);
  assert.match(f.get('protocolNote').textContent,/selecionado no lugar de/);
  assert.match(f.get('protocolNote').textContent,/1450 grupos/);
  assert.match(f.get('dataInfo').textContent,/10 saídas/);
  assert.match(f.get('formula').textContent,/8 → 9 → 10/);
  assert.doesNotMatch(f.get('dataInfo').textContent,/undefined|undefined M/);
  f.get('mode').value='kfold10';f.get('mode').onchange();
  assert.equal(f.get('train').disabled,true,'An incompatible manual mode does not enable training.');
  f.context.inputInvalidated('Corrija a leitura.');
  assert.equal(f.get('train').disabled,true);
  assert.match(f.get('status').textContent,/Corrija a leitura/);

  const metrics={classes,architecture:[8,9,10],n:100,n_folds:5,accuracy:.8,correct:80,precision_macro:.7,recall_macro:.6,f1_macro:.65,
    confusion_matrix_order:classes,confusion_matrix:classes.map((_,i)=>classes.map((__,j)=>i===j?8:0)),
    per_class:Object.fromEntries(classes.map(label=>[label,{precision:.7,recall:.6,f1:.65,support:10}]))};
  const metricKeys=['accuracy','precision_macro','recall_macro','f1_macro'];
  const result={metrics,config:{mode:'kfold5',seed:42},folds:[{fold:1,n_train:80,n_test:20,test_percent:20,chosen_epochs:2,correct:16,...Object.fromEntries(metricKeys.map(key=>[key,.8]))}],
    highlights:{accuracy:{value:.8,ids:[1]}},fold_summary:{n_folds:5,metric_keys:metricKeys,metrics:Object.fromEntries(metricKeys.map(key=>[key,{mean:.8,std:.02,n:5}]))},
    errors:[{id_auditoria:'protein_1',fold_teste:1,classe_real:'CYT',classe_prevista:'NUC',...Object.fromEntries(classes.map(label=>['score_'+label,.1]))}]};
  f.setResponses({'/api/result?run=test':result,'/api/ranking?run=test':{rows:[{id:'test',label:'Protein test',optimizer:'sgd_momentum',...metrics}],highlights:{accuracy:{value:.8,ids:['test']}}},'/api/history?run=test&model=fold_01&phase=':[]});
  await f.context.loadResult();
  assert.match(f.get('summaryNote').textContent,/média macro/);
  assert.match(f.get('foldMetrics').innerHTML,/Precisão · macro/);
  assert.match(f.get('metrics').innerHTML,/F1-score · macro OOF/);
  assert.doesNotMatch(f.get('metrics').innerHTML,/maligno|benigno|undefined/i);
  assert.match(f.get('confusion').innerHTML,/Métricas por classe/);
  for(const label of classes){assert.ok(f.get('confusion').innerHTML.includes(label));assert.ok(f.get('errors').innerHTML.includes('Score '+label));}
  assert.match(f.get('folds').innerHTML,/Precisão macro/);
  assert.match(f.get('rankTable').innerHTML,/Recall macro/);
  assert.match(f.get('bestFold').innerHTML,/Explorar rede da rodada 1/);
  f.setResponses({'/api/predict':{classes,rows:[{linha_csv:2,classe_prevista:'NUC',codigo_WTA:'0000000100',scores:Object.fromEntries(classes.map(label=>[label,.1]))}],csv:'prediction',saved:'temporary-test'}});
  await f.context.predict();
  for(const label of classes)assert.ok(f.get('predictionResults').innerHTML.includes('Score '+label));
  assert.doesNotMatch(f.get('predictionResults').innerHTML,/undefined/);

  // Legacy results retain the stored B/M confusion-matrix order and M-positive summaries.
  const binary={...result,metrics:{architecture:[30,16,2],n:100,n_folds:5,correct:90,accuracy:.9,sensitivity_M:.8,specificity_B:1,f1_M:.88,
    true_positive_M:40,false_negative_M_as_B:10,true_negative_B:50,false_positive_B_as_M:0,roc_auc_M:.94,
    confusion_matrix_order:['B','M'],confusion_matrix:[[50,0],[10,40]]},errors:[],
    fold_summary:{n_folds:5,metrics:Object.fromEntries(['accuracy','precision_M','sensitivity_M','f1_M'].map(key=>[key,{mean:.9,std:.01,n:5}]))}};
  f.setResponses({'/api/result?run=test':binary,'/api/ranking?run=test':{rows:[],highlights:{}},'/api/history?run=test&model=fold_01&phase=':[]});
  await f.context.loadResult();
  assert.match(f.get('summaryNote').textContent,/M como classe positiva/);
  assert.match(f.get('confusion').innerHTML,/<th>Real \/ previsto<\/th><th>B<\/th><th>M<\/th>/);
  assert.match(f.get('metrics').innerHTML,/Sensibilidade · malignos/);
}

(async()=>{checkNetwork();await checkDashboard();console.log('PASS: dynamic network, protocol selection, macro results, class scores, and legacy M/B dashboard.');})().catch(error=>{console.error(error);process.exitCode=1;});
