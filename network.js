/* Visualização determinística dos pesos REAIS lidos do NPZ. Sem dados inventados. */
(function(){
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const views=new WeakMap();
let nextId=0;
function render(element,data,selected=-1){
  const [d,h,o]=data.architecture,p=data.params,H=Math.max(780,d*25+140,h*45+175,o*75+175),W=o>2?1450:1220;
  const pca=data.pca||null;
  const classes=data.classes||['M','B'],binary=classes.length===2&&classes[0]==='M'&&classes[1]==='B';
  const codes=data.codes||classes.map((_,index)=>classes.map((__,k)=>k===index?'1':'0').join(''));
  let view=views.get(element);
  // A seleção reutiliza o objeto recebido. Um novo modelo/checkpoint inicia outra vista.
  if(!view||view.data!==data){
    view={data,zoom:1,x:W/2,y:H/2,id:++nextId};
    views.set(element,view);
  }
  const previousFocus=element.contains(document.activeElement)?document.activeElement:null;
  const focusNeuron=previousFocus?.getAttribute('data-neuron');
  const focusAction=previousFocus?.getAttribute('data-network-action');
  const focusStage=previousFocus?.hasAttribute('data-network-stage');
  const activation=(value,fallback)=>({relu:'ReLU',sigmoid:'Sigmoide',softmax:'Softmax'}[String(value).toLowerCase()]||value||fallback);
  const hiddenActivation=activation(data.hidden_activation,'ReLU'),outputActivation=activation(data.output_activation,'Softmax');
  const ys=(i,n)=>n===1?H/2:95+i*(H-175)/(n-1),xin=295,xh=670,xout=1025;
  const outputY=k=>o<=2?ys(k,o)*.50+H*.25:ys(k,o);
  const maxWeight=Math.max(1e-12,...p.W1.flat().map(Math.abs),...p.W2.flat().map(Math.abs));
  const lineColor=v=>v>=0?'#267cbb':'#bd613f';
  let s=`<div class="lab-network"><style>
    .lab-network{font-family:system-ui,sans-serif;background:#fbfcfd;color:#23394a}
    .lab-network-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:12px;border-bottom:1px solid #dce5eb;background:#f3f7fa}
    .lab-network-toolbar button{font:600 13px system-ui,sans-serif;color:#24495d;background:white;border:1px solid #b7cbd7;border-radius:6px;padding:8px 12px;cursor:pointer;min-height:36px}
    .lab-network-toolbar button:disabled{opacity:.45;cursor:default}
    .lab-network-toolbar output{min-width:48px;text-align:center;font-size:12px;font-variant-numeric:tabular-nums}
    .lab-network-help{margin:0;padding:10px 12px;font-size:12px;line-height:1.6;color:#536777;border-top:1px solid #dce5eb}
    .lab-network svg{width:100%;height:min(70vh,720px);min-height:420px;display:block;touch-action:none;user-select:none;cursor:grab}
    .lab-network svg[data-dragging]{cursor:grabbing}
    .lab-network button:focus-visible,.lab-network svg:focus-visible{outline:3px solid #267cbb;outline-offset:-3px}
    .lab-network [data-neuron]:focus-visible{outline:none}
    .lab-network [data-neuron]:focus-visible circle{stroke:#a54715;stroke-width:4}
  </style><div class="lab-network-toolbar" role="group" aria-label="Navegação da rede neural">
    <button type="button" data-network-action="out" aria-label="Diminuir zoom">−</button>
    <output data-network-zoom aria-label="Zoom da rede" aria-live="polite"></output>
    <button type="button" data-network-action="in" aria-label="Aumentar zoom">+</button>
    <button type="button" data-network-action="fit">Ajustar à tela</button>
    <button type="button" data-network-action="center" ${selected<0?'disabled':''}>Centralizar ${selected>=0?'H'+(selected+1):'neurônio'}</button>
  </div><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" tabindex="0" data-network-stage role="group" aria-label="Rede real ${d}-${h}-${o}, pesos e vieses; área navegável" aria-describedby="lab-network-help-${view.id}">`;
  s+=`<text x="24" y="31" font-size="16" fill="#23394a" font-weight="700">${d} ${pca?'COMPONENTES · PCA':'ENTRADAS · Z-SCORE'}</text><text x="570" y="31" font-size="16" fill="#23394a" font-weight="700">${h} OCULTOS · ${esc(hiddenActivation)}</text><text x="990" y="31" font-size="16" fill="#23394a" font-weight="700">${o} SAÍDAS</text><text x="990" y="53" font-size="13" fill="#536777">${esc(outputActivation)} · código one-hot</text>`;
  const edge=(x1,y1,x2,y2,v,title,active)=>`<path d="M${x1} ${y1} C${x1+130} ${y1},${x2-140} ${y2},${x2} ${y2}" fill="none" stroke="${lineColor(v)}" stroke-width="${active?(1+3*Math.abs(v)/maxWeight):.6}" opacity="${active?.76:(selected<0?.13:.045)}"><title>${esc(title)} = ${v.toPrecision(12)}</title></path>`;
  for(let i=0;i<d;i++)for(let j=0;j<h;j++){
    if(d>64&&selected>=0&&selected!==j)continue;
    s+=edge(xin+12,ys(i,d),xh-16,ys(j,h),p.W1[i][j],`${data.features[i]} → H${j+1} (W1[${i},${j}])`,selected===j);
    if(selected===j)s+=`<text x="324" y="${ys(i,d)-5}" font-size="11" fill="${lineColor(p.W1[i][j])}">${p.W1[i][j].toFixed(5)}</text>`;
  }
  for(let j=0;j<h;j++)for(let k=0;k<o;k++){
    s+=edge(xh+16,ys(j,h),xout-24,outputY(k),p.W2[j][k],`H${j+1} → ${classes[k]} (W2[${j},${k}])`,selected===j);
    if(selected===j)s+=`<text x="900" y="${outputY(k)-8}" font-size="12" fill="${lineColor(p.W2[j][k])}" font-weight="700">${p.W2[j][k].toFixed(6)}</text>`;
  }
  for(let i=0;i<d;i++){
    const description=data.feature_descriptions?.[i]||data.features[i];
    const detail=pca?`${description} · ${data.features[i]}: combinação das ${pca.n_original_features} medidas padronizadas. Coeficientes PCA ajustados somente no treino deste estado; médias e desvios pertencem às medidas originais.`:
      `${description} · coluna ${data.features[i]} · média ${data.mean[i]} · desvio ${data.scale[i]}`;
    s+=`<text x="24" y="${ys(i,d)+4}" font-size="12" fill="#415361">${esc(data.feature_keys?.[i]||String(i+1).padStart(2,'0'))} · ${esc(description.slice(0,31))}<title>${esc(detail)}</title></text><circle cx="${xin}" cy="${ys(i,d)}" r="9" fill="#fff" stroke="#8dabbc" stroke-width="1.5"/>`;
  }
  for(let j=0;j<h;j++)s+=`<g data-neuron="${j}" tabindex="${j===(selected>=0?selected:0)?0:-1}" role="button" aria-pressed="${selected===j}" aria-label="Selecionar H${j+1}; bias ${p.b1[j].toPrecision(12)}" style="cursor:pointer"><circle cx="${xh}" cy="${ys(j,h)}" r="${selected===j?21:16}" fill="${selected===j?'#145778':'#e9f0f5'}" stroke="#367b99"/><text x="${xh}" y="${ys(j,h)+4}" text-anchor="middle" font-size="10" fill="${selected===j?'white':'#24495d'}">H${j+1}</text><title>H${j+1} · bias = ${p.b1[j].toPrecision(12)}. Clique, Enter ou Espaço para inspecionar.</title></g>`;
  for(let k=0;k<o;k++)s+=`<circle cx="${xout}" cy="${outputY(k)}" r="28" fill="#183d56"/><text x="${xout}" y="${outputY(k)+5}" font-size="${String(classes[k]).length>2?12:19}" text-anchor="middle" fill="white">${esc(String(classes[k]).slice(0,5))}<title>${esc(classes[k])}</title></text><text x="1067" y="${outputY(k)-5}" font-size="${o>2?12:15}" fill="#23394a" font-weight="700">${esc(codes[k])} · ${esc(binary?(k===0?'Maligno':'Benigno'):String(classes[k]).slice(0,24))}<title>${esc(classes[k])} · ${esc(codes[k])}</title></text><text x="1067" y="${outputY(k)+17}" font-size="11" fill="#536777">bias ${p.b2[k].toFixed(6)}</text>`;
  s+=`<text x="24" y="${H-23}" font-size="12" fill="#536777">Espessura = |peso| · azul = peso positivo · terracota = peso negativo · sinais não indicam diagnóstico.</text></svg><p class="lab-network-help" id="lab-network-help-${view.id}">Amplie e arraste para mover a rede. Use +/− ou Ctrl + rolagem para ampliar. Ajustar à tela restaura o panorama. Teclado: Tab entra na área; setas movem a vista, +/− ampliam e 0 ajusta. Nos neurônios, setas percorrem H1–H${h}; Enter ou Espaço selecionam. Pesos e vieses são os valores reais do modelo salvo.${pca?' As entradas PC são componentes, não colunas originais; a memória de cálculo explica a transformação das medidas até cada componente.':''}</p></div>`;
  element.innerHTML=s;
  const svg=element.querySelector('[data-network-stage]');
  const buttons=Object.fromEntries(Array.from(element.querySelectorAll('[data-network-action]'),b=>[b.dataset.networkAction,b]));
  const maxZoom=Math.max(8,H/400);
  const applyView=()=>{
    const width=W/view.zoom,height=H/view.zoom;
    view.x=Math.max(width/2,Math.min(W-width/2,view.x));
    view.y=Math.max(height/2,Math.min(H-height/2,view.y));
    svg.setAttribute('viewBox',`${view.x-width/2} ${view.y-height/2} ${width} ${height}`);
    element.querySelector('[data-network-zoom]').textContent=`${Math.round(view.zoom*100)}%`;
    buttons.out.disabled=view.zoom<=1;buttons.in.disabled=view.zoom>=maxZoom;
  };
  const zoomBy=factor=>{view.zoom=Math.max(1,Math.min(maxZoom,view.zoom*factor));applyView();};
  const fit=()=>{view.zoom=1;view.x=W/2;view.y=H/2;applyView();};
  const centerNeuron=j=>{view.zoom=Math.min(maxZoom,Math.max(view.zoom,2,H/650));view.x=xh;view.y=ys(j,h);applyView();};
  buttons.in.addEventListener('click',()=>zoomBy(1.25));
  buttons.out.addEventListener('click',()=>zoomBy(1/1.25));
  buttons.fit.addEventListener('click',fit);
  buttons.center.addEventListener('click',()=>{if(selected>=0)centerNeuron(selected);});
  svg.addEventListener('wheel',e=>{if(e.ctrlKey){e.preventDefault();zoomBy(e.deltaY<0?1.15:1/1.15);}},{passive:false});
  svg.addEventListener('keydown',e=>{
    if(e.target!==svg)return;
    const stepX=W/view.zoom*.12,stepY=H/view.zoom*.12;
    if(e.key==='+'||e.key==='=')zoomBy(1.25);
    else if(e.key==='-')zoomBy(1/1.25);
    else if(e.key==='0'||e.key==='Home')fit();
    else if(e.key==='ArrowLeft'){view.x-=stepX;applyView();}
    else if(e.key==='ArrowRight'){view.x+=stepX;applyView();}
    else if(e.key==='ArrowUp'){view.y-=stepY;applyView();}
    else if(e.key==='ArrowDown'){view.y+=stepY;applyView();}
    else return;
    e.preventDefault();
  });
  let drag=null,suppressClick=false;
  svg.addEventListener('pointerdown',e=>{
    if(e.button!==0||!e.isPrimary)return;
    drag={id:e.pointerId,x:e.clientX,y:e.clientY,cx:view.x,cy:view.y,moved:false};
    if(!e.target.closest('[data-neuron]'))svg.focus({preventScroll:true});
  });
  svg.addEventListener('pointermove',e=>{
    if(!drag||drag.id!==e.pointerId)return;
    const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
    if(!drag.moved&&Math.hypot(dx,dy)<5)return;
    if(!drag.moved){drag.moved=true;svg.setPointerCapture(e.pointerId);svg.setAttribute('data-dragging','');}
    const rect=svg.getBoundingClientRect(),scale=Math.min(rect.width/(W/view.zoom),rect.height/(H/view.zoom));
    if(scale>0){view.x=drag.cx-dx/scale;view.y=drag.cy-dy/scale;applyView();}
  });
  const endDrag=e=>{
    if(!drag||drag.id!==e.pointerId)return;
    suppressClick=drag.moved;
    drag=null;svg.removeAttribute('data-dragging');
    if(svg.hasPointerCapture(e.pointerId))svg.releasePointerCapture(e.pointerId);
    // O clique gerado pelo fim de um arraste não troca o neurônio.
    setTimeout(()=>{suppressClick=false;},0);
  };
  svg.addEventListener('pointerup',endDrag);
  svg.addEventListener('pointercancel',endDrag);
  svg.addEventListener('lostpointercapture',endDrag);
  svg.addEventListener('pointerleave',()=>{if(drag&&!drag.moved)drag=null;});
  svg.addEventListener('click',e=>{if(suppressClick){e.preventDefault();e.stopImmediatePropagation();}},true);
  const choose=j=>element.dispatchEvent(new CustomEvent('neuronselect',{detail:j}));
  element.querySelectorAll('[data-neuron]').forEach(n=>{
    const j=Number(n.dataset.neuron);
    n.addEventListener('click',()=>choose(j));
    n.addEventListener('keydown',e=>{
      if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(j);return;}
      let next;
      if(e.key==='ArrowUp'||e.key==='ArrowLeft')next=(j+h-1)%h;
      else if(e.key==='ArrowDown'||e.key==='ArrowRight')next=(j+1)%h;
      else if(e.key==='Home')next=0;
      else if(e.key==='End')next=h-1;
      else return;
      e.preventDefault();
      element.querySelectorAll('[data-neuron]').forEach(node=>node.setAttribute('tabindex','-1'));
      const target=element.querySelector(`[data-neuron="${next}"]`);
      target.setAttribute('tabindex','0');target.focus({preventScroll:true});
      // Mantém o neurônio focado visível, sem alterar seleção ou nível de zoom.
      const margin=30,y=ys(next,h),halfHeight=H/(2*view.zoom),halfWidth=W/(2*view.zoom);
      if(y<view.y-halfHeight+margin)view.y=y+halfHeight-margin;
      else if(y>view.y+halfHeight-margin)view.y=y-halfHeight+margin;
      if(xh<view.x-halfWidth+margin)view.x=xh+halfWidth-margin;
      else if(xh>view.x+halfWidth-margin)view.x=xh-halfWidth+margin;
      applyView();
    });
  });
  applyView();
  if(focusNeuron!==null&&focusNeuron!==undefined)element.querySelector(`[data-neuron="${focusNeuron}"]`)?.focus({preventScroll:true});
  else if(focusAction)buttons[focusAction]?.focus({preventScroll:true});
  else if(focusStage)svg.focus({preventScroll:true});
}
window.LabNetwork={render,esc};
})();
