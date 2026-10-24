'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const base=path.join(__dirname,'..');
const esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function node(){return {innerHTML:'',textContent:'',value:'',checked:false,disabled:false,hidden:false,
  dataset:{},style:{},options:[],selectedOptions:[{text:'Modelo teste'}],attributes:{},classList:{toggle(){}},
  addEventListener(){},setAttribute(name,value){this.attributes[name]=value;},querySelectorAll(){return [];},contains(){return false;}};}
const inspected={n:60,classes:['M','B'],class_counts:{M:30,B:30},architecture:[3,3,2],n_parameters:20,
  protocol:{supported_modes:['kfold5']}};

async function dashboard(){
  const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
  get('mode').value='kfold5';get('mode').options=[{value:'kfold5'}];
  const c=vm.createContext({document:{getElementById:get,querySelectorAll:()=>[]},LabNetwork:{esc},
    setTimeout,clearTimeout,URLSearchParams,TrainingProgress:{render(){},createPoller:({onState})=>({start:async()=>onState({busy:false,status:'idle'}),refresh:async()=>{}})},
    InputEditor:{mount:()=>({load:async()=>c.inputValidated(inspected),setBusy(){},validate:async()=>inspected,payload:()=>({})})}});
  const html=fs.readFileSync(path.join(base,'interface.html'),'utf8');
  vm.runInContext(html.slice(html.indexOf("'use strict';"),html.lastIndexOf('</script>')).replace('initSettings().catch(e=>notify(e.message));',''),c);
  vm.runInContext("api=async (url,payload)=>{if(url==='/api/train'){globalThis.sent=payload;return {run_id:'temporary'}};return {config:{mode:'kfold5',pca_enabled:false,pca_variance:.95}}};refresh=async()=>{};go=()=>{};",c);
  await c.initSettings();
  assert.equal(get('pca_enabled').checked,false);
  assert.equal(get('pca_variance').value,95,'API fraction appears as user percentage.');
  assert.equal(get('pca_variance').disabled,true);
  assert.match(get('formula').textContent,/rede 3 → 3 → 2/);
  get('pca_enabled').checked=true;get('pca_enabled').onchange();
  assert.equal(get('pca_variance').disabled,false);
  assert.match(get('formula').textContent,/3 medidas selecionadas → K componentes PCA/);
  assert.doesNotMatch(get('formula').textContent,/rede 3 → 3 → 2/);
  assert.equal(get('inputEditor').dataset.pcaEnabled,'true');
  assert.match(get('pcaPreviewNote').textContent,/somente no treino/);
  assert.match(get('dataInfo').textContent,/parâmetros definidos após ajustar o PCA/);
  get('pca_variance').value=101;get('pca_variance').oninput();
  assert.equal(get('train').disabled,true);
  assert.match(get('status').textContent,/no máximo 100%/);
  get('pca_enabled').checked=false;get('pca_enabled').onchange();
  assert.equal(get('pca_variance').value,95,'An invalid disabled cutoff cannot block a non-PCA training.');
  assert.equal(get('train').disabled,false);
  get('pca_enabled').checked=true;get('pca_enabled').onchange();
  get('pca_variance').value=97.5;get('pca_variance').oninput();
  assert.equal(get('train').disabled,false);
  await c.train();
  assert.equal(c.sent.config.pca_enabled,true);
  assert.equal(c.sent.config.pca_variance,.975,'Request converts percentage back to fraction.');
  vm.runInContext("current={metrics:{architecture:[2,2,2]},pca:{enabled:true,n_original_features:3,n_components:2,variance_target:.975,retained_variance:.985},timing:{source:'timing.json',total_seconds:101.5,folds:[{fold:1,selection_seconds:20,refit_seconds:10,total_seconds:31}],final:{selection_seconds:30,refit_seconds:25,total_seconds:60}}};renderRunDetails();",c);
  assert.equal(get('runDuration').textContent,'1min 41s');
  assert.match(get('timingTable').innerHTML,/Rodada 1/);
  assert.match(get('timingTable').innerHTML,/Modelo final/);
  assert.match(get('pcaResult').textContent,/3 medidas → 2 componentes/);
  assert.match(get('pcaResult').textContent,/97,50%.*98,50%/);
  assert.match(get('pcaResultNote').textContent,/podem variar/);
  assert.doesNotMatch(get('pcaResult').textContent,/undefined|NaN/);
  vm.runInContext("current={metrics:{architecture:[30,16,2]},timing:{total_seconds:90,source:'status.json'}};renderRunDetails();",c);
  assert.match(get('runDurationNote').textContent,/status.json/);
  assert.match(get('pcaResult').textContent,/PCA desligado/);
  assert.equal(get('timingDetails').hidden,true);
  vm.runInContext("delete current.timing;renderRunDetails();",c);
  assert.equal(get('runDuration').textContent,'Não registrado');
  assert.doesNotMatch(get('runDuration').textContent,/0 s/);
}

function network(){
  const c=vm.createContext({window:{},document:{activeElement:null},setTimeout,CustomEvent:class{}});
  vm.runInContext(fs.readFileSync(path.join(base,'network.js'),'utf8'),c);
  const root=node(),svg=node(),zoom=node(),buttons=['in','out','fit','center'].map(action=>({...node(),dataset:{networkAction:action}}));
  root.querySelector=selector=>selector==='[data-network-stage]'?svg:selector==='[data-network-zoom]'?zoom:null;
  root.querySelectorAll=selector=>selector==='[data-network-action]'?buttons:[];
  c.window.LabNetwork.render(root,{architecture:[2,2,2],features:['PC1','PC2'],feature_keys:['PC1','PC2'],
    feature_descriptions:['Componente principal 1','Componente principal 2'],mean:[999,888,777],scale:[333,222,111],
    params:{W1:[[.1,.2],[.3,.4]],b1:[0,0],W2:[[.2,.3],[.4,.5]],b2:[0,0]},pca:{n_original_features:3}},0);
  assert.match(root.innerHTML,/2 COMPONENTES · PCA/);
  assert.match(root.innerHTML,/PC1: combinação das 3 medidas padronizadas/);
  assert.match(root.innerHTML,/médias e desvios pertencem às medidas originais/);
  assert.doesNotMatch(root.innerHTML,/média 999|desvio 333|undefined/);
}

(async()=>{await dashboard();network();console.log('PASS: optional PCA defaults/percentage/preview, duration provenance, final PCA summary, and correct component tooltips.');})()
  .catch(error=>{console.error(error);process.exitCode=1;});
