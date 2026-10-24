'use strict';
// Executa com Node, sem DOM externo e sem treinar uma base.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const base = path.join(__dirname, '..');
const moduleCode = fs.readFileSync(path.join(base, 'training-progress.js'), 'utf8');
const context = vm.createContext({setTimeout, clearTimeout});
vm.runInContext(moduleCode, context);
const {render, createPoller, duration} = context.TrainingProgress;
const tick = () => new Promise(resolve => setImmediate(resolve));
const node = () => ({hidden:false, disabled:false, dataset:{}, textContent:'', value:'', style:{},
  options:[], addEventListener(){}, querySelectorAll(){return []}, classList:{toggle(){}}});
function nodes() { const map = new Map(); return id => {if (!map.has(id)) map.set(id, node()); return map.get(id);}; }
const detail = {stage:'fold', fold:6, folds_total:10, phase:'selecao_interna', epoch:120, epochs_total:800,
  phase_complete:false, phase_elapsed_seconds:125, fold_elapsed_seconds:125, elapsed_seconds:1899,
  updated_utc:'2026-09-19T20:00:00+00:00', event:'epoch'};
const active = {busy:true,status:'running',done:5,total:11,message:'Rodada 6/10',detail,run_id:'new-run'};

function checkRendering() {
  const get=nodes();render(active,get);
  assert.equal(get('progress').value,5);
  assert.equal(get('epochProgress').value,120);
  assert.match(get('epochProgressLabel').textContent,/120 de até 800/);
  assert.match(get('epochProgressNote').textContent,/parada antecipada/);
  assert.match(get('trainingPhase').textContent,/Rodada 6 de 10.*validação interna/);
  assert.match(get('trainingTimes').textContent,/2min 5s.*31min 39s/);
  assert.match(get('trainingUpdated').textContent,/Última atualização:/);
  assert.match(get('progressNote').textContent,/não estima o tempo restante/);
  render({...active,detail:{...detail,epoch:276,phase_complete:true,chosen_epochs:216,event:'complete'}},get);
  assert.equal(get('epochProgress').max,800,'Early stop does not imply reaching the configured maximum.');
  assert.equal(get('epochProgress').value,276);
  assert.match(get('epochProgressNote').textContent,/276 épocas; 216 escolhidas/);
  render({...active,done:10,detail:{...detail,stage:'final',fold:null,phase:'retreino',epoch:0,epochs_total:216}},get);
  assert.match(get('trainingPhase').textContent,/Modelo final.*retreino/);
  assert.equal(get('epochProgress').value,0);
  assert.equal(get('epochProgress').max,216);
  assert.doesNotMatch(get('epochProgressLabel').textContent,/até/);
  render({...active,done:11,message:'Concluído e salvo'},get);
  assert.match(get('status').textContent,/Finalizando/,'Busy=true never claims the execution is finished.');
  render({...active,busy:false,status:'error',detail:null,message:'Falha: disco'},get);
  assert.equal(get('epochProgressBlock').hidden,true);
  assert.equal(get('status').dataset.kind,'blocked');
  assert.equal(get('trainingTimes').textContent,'');
  render({...active,busy:false,status:'completed',done:11,message:'Concluído e salvo'},get);
  assert.equal(get('status').textContent,'Concluído e salvo');
  assert.equal(get('progress').value,11);
  assert.equal(duration(3601),'1h 0min 1s');
}

async function checkSerialPolling() {
  let resolve, reject, calls=0;const timers=new Map(),seen=[],errors=[];let timerID=0;
  const poller=createPoller({fetchStatus:()=>{calls++;return new Promise((yes,no)=>{resolve=yes;reject=no;});},
    onState:state=>seen.push(state),onError:error=>errors.push(error),
    schedule:callback=>{timers.set(++timerID,callback);return timerID;},cancel:id=>timers.delete(id)});
  const first=poller.start();assert.equal(first,poller.refresh());await tick();assert.equal(calls,1);
  assert.equal(timers.size,0,'No timer is queued while a request is pending.');
  resolve(active);await first;assert.equal(seen.length,1);assert.equal(timers.size,1);
  const second=poller.refresh();await tick();assert.equal(calls,2);assert.equal(timers.size,0);
  reject(Error('offline'));await second;assert.equal(errors.length,1);assert.equal(timers.size,1);
  const third=poller.refresh();await tick();poller.stop();resolve(active);await third;
  assert.equal(seen.length,1,'A stopped poller discards late responses.');assert.equal(timers.size,0);
}

async function checkReloadAndFailure() {
  const get=nodes(),requests=[],busy=[];
  get('mode').value='kfold5';get('mode').options=['variavel','kfold5','kfold10'].map(value=>({value}));
  const inspected={n:50,M:25,B:25,architecture:[2,2,2],n_parameters:12,
    protocol:{supported_modes:['kfold5']}};
  const c=vm.createContext({setTimeout,clearTimeout,URLSearchParams,
    document:{getElementById:get,querySelectorAll:()=>[]},LabNetwork:{esc:String},
    InputEditor:{mount:()=>({load:async()=>c.inputValidated(inspected),setBusy:value=>busy.push(value),validate:async()=>inspected,payload:()=>({})})}});
  vm.runInContext(moduleCode,c);
  const html=fs.readFileSync(path.join(base,'interface.html'),'utf8');
  vm.runInContext(html.slice(html.indexOf("'use strict';"),html.lastIndexOf('</script>'))
    .replace('initSettings().catch(e=>notify(e.message));',''),c);
  c.testActive=active;
  c.testRequests=requests;
  vm.runInContext("api=async url=>{testRequests.push(url);return url==='/api/info'?{config:{}}:testActive};refresh=async()=>{};go=()=>{};",c);
  await c.initSettings();
  assert.equal(get('train').disabled,true,'Reload of an active run cannot submit another training.');
  assert.equal(get('datafile').disabled,true);
  assert.equal(get('max_epochs').disabled,true);
  assert.match(get('trainingPhase').textContent,/Rodada 6 de 10/);
  assert.equal(busy.at(-1),true);
  await c.train();assert.equal(requests.filter(url=>url==='/api/train').length,0);
  vm.runInContext('progressWatcher.stop()',c);
  c.receiveTrainingStatus({...active,busy:false,status:'error',message:'Falha no disco',detail:null});
  assert.equal(get('train').disabled,false);
  assert.equal(get('datafile').disabled,false);
  assert.equal(get('status').textContent,'Falha no disco');
  c.receiveTrainingStatus({...active,busy:false,status:'error',message:'Falha no disco',detail:null});
  assert.equal(get('status').textContent,'Falha no disco','Idle polling does not erase a training failure.');
  // Successful training followed by a display failure must retain its completion.
  vm.runInContext("api=async()=>({run_id:'another-run'});refresh=async()=>{throw Error('display offline')};notify=message=>{globalThis.lastNotice=message};",c);
  await c.train();
  assert.match(get('status').textContent,/Concluído.*another-run/);
  assert.match(c.lastNotice,/Treinamento concluído e salvo.*carregar/);
}

(async()=>{checkRendering();await checkSerialPolling();await checkReloadAndFailure();
  console.log('PASS: phase/epoch rendering, early stop, genuine completion, serialized polling, reload locking, and preserved failures.');
})().catch(error=>{console.error(error);process.exitCode=1;});
