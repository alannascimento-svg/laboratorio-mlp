"""Painel local sem Streamlit, nuvem, API paga ou dependências de frontend.

Servidor restrito ao loopback. O código pode ser compartilhado; o servidor é local.
Ctrl+C encerra o servidor; fechamento do navegador não encerra o processo.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import math
import mimetypes
import re
import secrets
import threading
import traceback
import webbrowser
import zipfile
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
from memoria import update_index, read_update
from resumos import fold_summary
from entrada import inspect_source
from preprocessamento import source_features
from rede import (Config, FEATURES, WARNING, VERSION, MLP, load_dataset, load_model,
    model_details, metric_highlights, experiment_signature, save_prediction, run_experiment,
    decode_source, load_selected_dataset, dataset_protocol)

ROOT = Path(__file__).resolve().parent
TOKEN = secrets.token_urlsafe(32)
TRAIN_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
STATE = {'busy': False, 'status': 'idle', 'done': 0, 'total': 1,
         'message': 'Pronto para treinar', 'detail': None, 'run_id': None}
DOWNLOADS = {'metricas_oof.json', 'rodadas.csv', 'previsoes_oof.csv', 'erros_oof.csv',
             'particoes_auditoria.csv', 'historico_treinamento.csv', 'modelo_final.npz',
             'modelo_final_metadados.json', 'configuracao.json', 'pesos_final.csv', 'rede_final.json',
             'destaques_rodadas.json', 'modelo_final_selecao.csv', 'modelo_final_treinamento.csv',
             'normalizacao_final.csv', 'modelo_previsao.csv', 'resumo_rodadas.json', 'esquema_entradas.json', 'timing.json'}


def run_folder(run_id: str, completed: bool = True) -> Path:
    if run_id == 'exemplo':
        path = ROOT / 'resultados_exemplo'
    elif re.fullmatch(r'r\d{8}_\d{6}_\d{6}', run_id):
        path = ROOT / 'resultados' / run_id
    else:
        raise ValueError('Identificador de execução inválido.')
    if not path.is_dir() or (completed and not (path / 'CONCLUIDO.txt').exists()):
        raise ValueError('Execução não encontrada ou ainda não concluída.')
    return path


def available_runs() -> list[dict]:
    found = []
    base = ROOT / 'resultados'
    if base.exists():
        for path in sorted(base.iterdir(), reverse=True):
            if path.is_dir() and re.fullmatch(r'r\d{8}_\d{6}_\d{6}', path.name) and (path/'CONCLUIDO.txt').exists():
                cfg = json.loads((path/'configuracao.json').read_text(encoding='utf8'))
                found.append({'id':path.name, 'label':path.name[1:]+' · '+cfg.get('optimizer','adam'),
                              'architecture':cfg.get('architecture')})
    if (ROOT/'resultados_exemplo'/'CONCLUIDO.txt').exists():
        cfg = json.loads((ROOT/'resultados_exemplo'/'configuracao.json').read_text(encoding='utf8'))
        found.append({'id':'exemplo',
                      'label':f"Demonstração pública · {cfg['optimizer']} · até {cfg['max_epochs']} épocas",
                      'architecture':cfg.get('architecture')})
    return found


def incomplete_runs():
    base=ROOT/'resultados'
    if not base.exists(): return []
    return [{'id':p.name,'status':'Incompleta — artefatos preservados'} for p in sorted(base.iterdir(),reverse=True)
            if p.is_dir() and re.fullmatch(r'r\d{8}_\d{6}_\d{6}',p.name) and not (p/'CONCLUIDO.txt').exists()]


def model_base(folder, model):
    if model=='final': return folder/'modelo_final.npz', folder/'checkpoints_final'
    if not re.fullmatch(r'fold_\d{2}',model): raise ValueError('Modelo inválido.')
    base=folder/'folds'/model
    if not base.is_dir(): raise ValueError('Rodada inexistente.')
    return base/'modelo.npz', base/'checkpoints'


def resolve_model(folder, model, checkpoint='trained'):
    primary,base=model_base(folder,model)
    if checkpoint=='trained': return primary
    if not re.fullmatch(r'(selecao_interna|retreino)/epoca_\d{5}\.npz',checkpoint):
        raise ValueError('Checkpoint inválido.')
    target=base/checkpoint
    if not target.is_file(): raise ValueError('Checkpoint inexistente.')
    return target


def checkpoints(folder, model):
    _,base=model_base(folder,model)
    found=[{'id':'trained','label':'Modelo concluído / selecionado'}]
    for phase,label in [('retreino','Retreino'),('selecao_interna','Seleção interna')]:
        for p in sorted((base/phase).glob('epoca_*.npz')):
            found.append({'id':phase+'/'+p.name,'label':label+' · época '+str(int(p.stem.split('_')[1]))})
    return found


def ranking(run):
    reference=experiment_signature(run_folder(run));rows=[]
    for item in available_runs():
        folder=run_folder(item['id'])
        if experiment_signature(folder)!=reference: continue
        met=json.loads((folder/'metricas_oof.json').read_text(encoding='utf8'))
        cfg=json.loads((folder/'configuracao.json').read_text(encoding='utf8'))
        rows.append({**item,**met,'optimizer':cfg.get('optimizer','adam')})
    return {'rows':rows,'highlights':metric_highlights(rows,'id'),
            'warning':'Comparação exploratória: mesma base e mesmas partições. Não prova superioridade clínica.'}


def imported_dataset(text=None, raw=None):
    path=ROOT/'dados'/'cancear.txt'
    if text is not None or raw is not None:
        if raw is None:
            if not isinstance(text,str): raise ValueError('Dados devem ser texto.')
            raw=text.encode('utf8')
        base=ROOT/'dados'/'importados';base.mkdir(parents=True,exist_ok=True)
        path=base/(hashlib.sha256(raw).hexdigest()+'.txt')
        if not path.exists():path.write_bytes(raw)
    return path


def input_bytes(payload: dict) -> bytes:
    text = payload.get('data_text')
    encoded = payload.get('data_base64')
    if encoded is not None:
        if text is not None or not isinstance(encoded, str):
            raise ValueError('Envie apenas um conteúdo de arquivo.')
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError('Conteúdo binário do arquivo inválido.') from exc
    if text is None:
        return (ROOT / 'dados' / 'cancear.txt').read_bytes()
    if not isinstance(text, str):
        raise ValueError('Dados devem ser texto CSV, TSV ou TXT.')
    return text.encode('utf8')


def inspect_columns(payload: dict) -> dict:
    raw = input_bytes(payload)
    table = inspect_source(decode_source(raw), delimiter=payload.get('delimiter', 'auto'),
                           header=payload.get('header', 'auto'))
    return {key: value for key, value in table.items() if key not in {'rows','source_lines'}} | {
        'source_lines': table['source_lines'][:8],
        'source_sha256': hashlib.sha256(raw).hexdigest()}


def inspect_selection(payload: dict) -> dict:
    raw = input_bytes(payload)
    schema = payload.get('input_schema')
    if schema is None:
        table = inspect_source(decode_source(raw))
        schema = {'delimiter': table['delimiter'], 'header': 'yes' if table['has_header'] else 'no',
                  'columns': [{key: col[key] for key in ('index','description','role')} for col in table['columns']]}
    d = load_selected_dataset(raw, schema)
    net = MLP(0, d.X.shape[1], d.features, d.feature_descriptions, classes=d.classes)
    protocol = dataset_protocol(d)
    result = {'n': len(d.y), 'classes': d.classes, 'class_counts': protocol['class_counts'],
            'protocol': protocol,
            'features': d.features, 'feature_descriptions': d.feature_descriptions,
            'architecture': net.architecture, 'n_parameters': net.n_parameters,
            'input_schema': d.input_schema}
    if d.classes == ['M', 'B']:
        result.update(B=int(sum(d.y == 1)), M=int(sum(d.y == 0)))
    return result


def calculation_result(query: dict) -> dict:
    from calculos import snapshot_report, update_report
    run = query.get('run', ['exemplo'])[0]
    model = query.get('model', ['final'])[0]
    checkpoint = query.get('checkpoint', ['trained'])[0]
    neuron = int(query.get('neuron', ['0'])[0])
    sample = int(query.get('sample', ['0'])[0])
    folder = run_folder(run)
    target = resolve_model(folder, model, checkpoint)
    net, mean, scale, meta = load_model(target)
    if not 0 <= neuron < net.n_hidden:
        raise ValueError('Neurônio oculto inválido.')
    config = json.loads((folder/'configuracao.json').read_text(encoding='utf8'))
    source = folder/'dados_utilizados.txt'
    if not source.is_file():
        source = ROOT/'dados'/'cancear.txt'
    if hashlib.sha256(source.read_bytes()).hexdigest() != config['data_sha256']:
        raise ValueError('A base original deste modelo não está disponível com o mesmo SHA-256.')
    dataset = load_dataset(source, config.get('input_schema'))
    if dataset.features != source_features(net):
        raise ValueError('As entradas da base não correspondem às entradas deste modelo.')
    if dataset.classes != net.classes:
        raise ValueError('As classes da base não correspondem às saídas deste modelo.')
    phase = checkpoint.split('/')[0] if checkpoint != 'trained' else 'retreino'
    epoch = int(meta.get('epoch', meta.get('epochs', meta.get('final_epochs', 0))))
    _, checkpoint_root = model_base(folder, model)
    trace_path = checkpoint_root/phase/'memoria_calculo.sqlite3'
    index = update_index(trace_path, epoch)
    mode = query.get('mode', ['update' if 'step' in query else 'snapshot'])[0]
    if mode not in {'snapshot', 'update'}:
        raise ValueError('Fonte de cálculo inválida.')
    context = {'run':run, 'model':model, 'checkpoint':checkpoint, 'neuron':neuron,
        'mode':mode, 'feature_count':net.n_inputs, 'architecture':net.architecture, 'classes':net.classes,
        'trace_count':index['count'], 'last_step':index['last_step'], 'phase':phase, 'epoch':epoch}
    if mode == 'update':
        step = int(query.get('step', [str(index['last_step'])])[0])
        payload = read_update(trace_path, step, epoch)
        if payload.get('classes', ['M', 'B']) != net.classes:
            raise ValueError('As classes da memória de cálculo não correspondem às saídas deste modelo.')
        count = len(payload['y'])
        if not 0 <= sample < count:
            raise ValueError('Amostra fora do lote desta atualização.')
        if model == 'final':
            indices = list(range(len(dataset.y))) if phase == 'retreino' else json.loads(
                (folder/'modelo_final_indices.json').read_text())['inner_fit_0based']
        else:
            audit = json.loads((folder/'folds'/model/'indices.json').read_text())
            indices = audit['train_0based' if phase == 'retreino' else 'inner_fit_0based']
        original_indices = [indices[i] for i in payload['batch_indices_in_phase']]
        samples = [{'index':j, 'id':str(dataset.ids[i]),
                    'label':f"ID {dataset.ids[i]} · {dataset.classes[int(dataset.y[i])]} · linha {dataset.source_lines[i]}"}
                   for j, i in enumerate(original_indices)]
        report = update_report(payload, neuron, sample)
        context.update(step=step, update_epoch=payload['epoch'], batch=payload['batch'])
    else:
        count = len(dataset.y)
        if not 0 <= sample < count:
            raise ValueError('Amostra inválida.')
        report = snapshot_report(net, mean, scale, dataset.X, dataset.y, neuron, sample,
                                 l2=config.get('l2', 0.0))
        samples = [{'index':i, 'id':str(dataset.ids[i]),
                    'label':f"ID {dataset.ids[i]} · {dataset.classes[int(dataset.y[i])]} · linha {dataset.source_lines[i]}"}
                   for i in range(count)]
    context['sample_count'] = count
    return {'context':context, 'samples':samples, 'report':report, 'updates':[]}


def records(frame: pd.DataFrame) -> list[dict]:
    # O JSON do Pandas representa campos ausentes como null, não NaN inválido.
    return json.loads(frame.to_json(orient='records'))


def result_summary(folder: Path) -> dict:
    """Deriva a apresentação dos folds salvos, sem escrever nos históricos."""
    frame = pd.read_csv(folder / 'rodadas.csv', keep_default_na=False, na_values=[''])
    # Preserva a precisão do CSV e distingue campo ausente de infinito inválido.
    rows = frame.astype(object).where(frame.notna(), None).to_dict(orient='records')
    config_path = folder / 'configuracao.json'
    classes = json.loads(config_path.read_text(encoding='utf8')).get('classes') if config_path.exists() else None
    return {'folds': rows, 'fold_summary': fold_summary(rows, classes=classes),
            'highlights': metric_highlights(rows)}


def result_timing(folder: Path) -> dict:
    """Consulta tempos registrados, sem inferir duração por mtime ou regravar históricos."""
    unavailable = {'total_seconds': None, 'source': None, 'folds': [], 'final': None}
    timing_path = folder / 'timing.json'
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding='utf8'))
        def seconds(value):
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Duração inválida no arquivo timing.json.')
            return value
        def phase(value):
            if value is None:
                return None
            return {key: seconds(value.get(key)) for key in ('selection_seconds', 'refit_seconds', 'total_seconds')}
        return {'total_seconds': seconds(timing.get('total_seconds')), 'source': 'timing.json',
                'folds': [{'fold': row['fold'], **phase(row)} for row in timing.get('folds', [])],
                'final': phase(timing.get('final'))}
    status_path = folder / 'status.json'
    if not status_path.is_file():
        return unavailable
    status = json.loads(status_path.read_text(encoding='utf8'))
    try:
        created = datetime.fromisoformat(status['created_utc'])
        completed = datetime.fromisoformat(status['completed_utc'])
        elapsed = (completed - created).total_seconds()
    except (KeyError, TypeError, ValueError):
        return unavailable
    if not math.isfinite(elapsed) or elapsed < 0:
        return unavailable
    return {**unavailable, 'total_seconds': elapsed, 'source': 'status.json'}


def result_pca(folder: Path) -> dict:
    net, _, _, _ = load_model(folder / 'modelo_final.npz')
    pca = getattr(net, 'pca', None)
    return {'enabled': pca is not None, 'n_original_features': len(source_features(net)),
            'n_components': net.n_inputs,
            **({key: pca.get(key) for key in ('variance_target', 'retained_variance', 'fit_samples')} if pca else {})}


class Handler(BaseHTTPRequestHandler):
    server_version = 'LaboratorioMLP/' + VERSION

    def allowed_host(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get('Host', '') in {f'127.0.0.1:{port}', f'localhost:{port}'}

    def send_bytes(self, status: int, payload: bytes, content_type: str, filename: str | None = None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; form-action 'self'")
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, value, status: int = 200):
        self.send_bytes(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8')

    def do_GET(self):
        if not self.allowed_host():
            self.send_json({'error': 'Host não autorizado. Use 127.0.0.1 ou localhost.'}, 403)
            return
        request = urlparse(self.path)
        query = parse_qs(request.query)
        try:
            if request.path == '/':
                page = (ROOT / 'interface.html').read_text(encoding='utf-8').replace('__CSRF_TOKEN__', TOKEN)
                self.send_bytes(200, page.encode(), 'text/html; charset=utf-8')
            elif request.path in {'/network.js', '/input-editor.js', '/training-progress.js'}:
                self.send_bytes(200,(ROOT/request.path[1:]).read_bytes(),'text/javascript; charset=utf-8')
            elif request.path == '/calculos':
                self.send_bytes(200, (ROOT/'calculos.html').read_bytes(), 'text/html; charset=utf-8')
            elif request.path == '/api/calculation':
                self.send_json(calculation_result(query))
            elif request.path == '/api/info':
                d = load_dataset(ROOT / 'dados' / 'cancear.txt')
                self.send_json({'n': len(d.y), 'B': int(sum(d.y == 1)), 'M': int(sum(d.y == 0)),
                    'classes':d.classes, 'class_counts':dataset_protocol(d)['class_counts'],
                    'features': d.features, 'config': asdict(Config()), 'runs': available_runs(), 'incomplete': incomplete_runs(), 'version': VERSION, 'warning': WARNING})
            elif request.path == '/api/status':
                with STATE_LOCK:
                    state = STATE.copy()
                self.send_json(state)
            elif request.path == '/api/result':
                folder = run_folder(query.get('run', ['exemplo'])[0])
                self.send_json({'metrics': json.loads((folder / 'metricas_oof.json').read_text(encoding="utf-8")),
                    **result_summary(folder),
                    'errors': records(pd.read_csv(folder / 'erros_oof.csv', keep_default_na=False, dtype={'id_auditoria': str, 'codigo_real': str, 'codigo_WTA': str})),
                    'final': json.loads((folder / 'modelo_final_metadados.json').read_text(encoding="utf-8")),
                    'timing': result_timing(folder), 'pca': result_pca(folder),
                    'config': json.loads((folder / 'configuracao.json').read_text(encoding="utf-8"))})
            elif request.path == '/api/history':
                folder=run_folder(query.get('run',['exemplo'])[0]); model=query.get('model',['fold_01'])[0]
                phase=query.get('phase',['selecao_interna'])[0]
                if phase not in {'selecao_interna','retreino_externo'}: raise ValueError('Fase inválida.')
                if model=='final':
                    name='modelo_final_selecao.csv' if phase=='selecao_interna' else 'modelo_final_treinamento.csv'
                    hist=pd.read_csv(folder/name)
                else:
                    model_base(folder,model);fold=int(model.split('_')[1]);hist=pd.read_csv(folder/'historico_treinamento.csv')
                    hist=hist[(hist.fold==fold)&(hist.phase==phase)]
                self.send_json(records(hist))
            elif request.path == '/api/checkpoints':
                self.send_json(checkpoints(run_folder(query.get('run',['exemplo'])[0]),query.get('model',['final'])[0]))
            elif request.path == '/api/network':
                target=resolve_model(run_folder(query.get('run',['exemplo'])[0]),query.get('model',['final'])[0],
                                     query.get('checkpoint',['trained'])[0])
                self.send_json(model_details(target))
            elif request.path == '/api/ranking':
                self.send_json(ranking(query.get('run',['exemplo'])[0]))
            elif request.path == '/api/bundle':
                run=query.get('run',['exemplo'])[0];folder=run_folder(run,False);buf=io.BytesIO()
                with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as zipout:
                    for p in sorted(folder.rglob('*')):
                        if p.is_file() and not p.is_symlink() and not p.name.endswith('.tmp'):
                            zipout.write(p,str(Path(run)/p.relative_to(folder)))
                self.send_bytes(200,buf.getvalue(),'application/zip','execucao_'+run+'.zip')
            elif request.path == '/api/template':
                net,_,_,_=load_model(run_folder(query.get('run',['exemplo'])[0])/'modelo_final.npz')
                text=pd.DataFrame(columns=['id']+source_features(net)).to_csv(index=False)
                self.send_bytes(200,text.encode(),'text/csv; charset=utf-8','modelo_previsao.csv')
            elif request.path == '/download':
                name = query.get('file', [''])[0]
                if name not in DOWNLOADS:
                    raise ValueError('Arquivo não permitido.')
                folder = run_folder(query.get('run', ['exemplo'])[0])
                if name == 'resumo_rodadas.json':
                    summary = result_summary(folder)
                    payload = {'run': query.get('run', ['exemplo'])[0],
                               'source': 'rodadas.csv', **summary}
                    self.send_bytes(200, json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2).encode(),
                                    'application/json; charset=utf-8', name)
                else:
                    self.send_bytes(200, (folder / name).read_bytes(), 'application/octet-stream', name)
            elif request.path == '/exemplo.csv':
                self.send_bytes(200, (ROOT / 'dados' / 'exemplo_previsao.csv').read_bytes(), 'text/csv; charset=utf-8', 'exemplo_previsao.csv')
            elif request.path == '/guia':
                file = ROOT / 'GUIA.html'
                self.send_bytes(200, file.read_bytes(), 'text/html; charset=utf-8')
            elif request.path == '/favicon.ico':
                self.send_bytes(204, b'', 'image/x-icon')
            else:
                self.send_json({'error': 'Página não encontrada.'}, 404)
        except (ValueError, OSError) as exc:
            self.send_json({'error': str(exc)}, 400)

    def do_POST(self):
        port = self.server.server_address[1]
        origin = self.headers.get('Origin')
        if (not self.allowed_host() or self.headers.get('X-Lab-Token') != TOKEN or
                (origin is not None and origin not in {f'http://127.0.0.1:{port}', f'http://localhost:{port}'})):
            self.send_json({'error': 'Requisição não autorizada.'}, 403)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 3_000_000:
                raise ValueError('Envie um JSON com até 3 MB.')
            if 'application/json' not in self.headers.get('Content-Type', ''):
                raise ValueError('Content-Type deve ser application/json.')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Objeto JSON esperado.')
            if self.path == '/api/train':
                self.train(payload)
            elif self.path == '/api/columns':
                self.send_json(inspect_columns(payload))
            elif self.path == '/api/inspect':
                self.send_json(inspect_selection(payload))
            elif self.path == '/api/predict':
                folder = run_folder(str(payload.get('run', 'exemplo')))
                text = payload.get('csv', '')
                if not isinstance(text, str):
                    raise ValueError('CSV deve ser texto.')
                result,saved=save_prediction(folder/'modelo_final.npz',text,folder/'inferencias')
                net, _, _, _ = load_model(folder/'modelo_final.npz')
                rows = records(result)
                for row in rows:
                    row['scores'] = {label:row['score_'+label] for label in net.classes}
                self.send_json({'classes':net.classes, 'rows':rows,'csv':result.to_csv(index=False),
                    'saved':str(saved.relative_to(ROOT)),'warning':WARNING})
            else:
                self.send_json({'error': 'Rota não encontrada.'}, 404)
        except (ValueError, TypeError, OSError, FloatingPointError) as exc:
            self.send_json({'error': str(exc)}, 400)
        except Exception:
            traceback.print_exc()
            self.send_json({'error': 'Erro interno. Consulte o terminal local para os detalhes.'}, 500)

    def train(self, payload: dict):
        if not TRAIN_LOCK.acquire(blocking=False):
            self.send_json({'error': 'Já existe um treinamento em execução nesta aplicação.'}, 409)
            return
        try:
            with STATE_LOCK:
                STATE.update(busy=True, status='running', done=0, total=1,
                             message='Conferindo a base e preparando partições',
                             detail=None, run_id=None)
            settings = payload.get('config', {})
            cfg = Config(**settings)
            cfg.validate()
            if cfg.max_epochs > 5000:
                raise ValueError('O painel limita o máximo a 5.000 épocas.')
            schema = payload.get('input_schema')
            if schema is not None:
                inspected = inspect_selection(payload)  # Não grava arquivos antes de validar a seleção.
                if cfg.mode not in inspected['protocol']['supported_modes']:
                    modes = ', '.join(inspected['protocol']['supported_modes'])
                    raise ValueError(f'O protocolo {cfg.mode} não atende às classes/grupos desta base. Use: {modes}.')
            data_path = imported_dataset(payload.get('data_text'),
                raw=input_bytes(payload) if payload.get('data_base64') is not None else None)
            load_dataset(data_path, schema)
            run_id = 'r' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output = ROOT / 'resultados' / run_id
            with STATE_LOCK:
                STATE.update(run_id=run_id, message='Preparando partições')
            def progress(done, total, message):
                with STATE_LOCK:
                    STATE.update(done=done, total=total, message=message)
                    if done < total:
                        STATE['detail'] = None
            def training_progress(detail):
                with STATE_LOCK:
                    STATE['detail'] = dict(detail)
            result = run_experiment(data_path, output, cfg, progress, input_schema=schema,
                                    training_progress=training_progress)
            with STATE_LOCK:
                STATE.update(busy=False, status='completed', done=STATE['total'],
                             message='Concluído. Todos os resultados foram salvos em ' + run_id)
        except Exception as exc:
            message = (str(exc) if isinstance(exc, (ValueError, TypeError, OSError, FloatingPointError))
                       else 'Erro interno. Consulte o terminal local para os detalhes.')
            with STATE_LOCK:
                STATE.update(busy=False, status='error', detail=None,
                             message='Não foi possível concluir o treinamento: ' + message)
            raise
        finally:
            with STATE_LOCK:
                STATE['busy'] = False
            TRAIN_LOCK.release()
        # A resposta pode falhar se a página foi fechada. Isso não desfaz o treino
        # concluído nem deve trocar o estado salvo por erro de treinamento.
        self.send_json({'run_id': run_id, 'metrics': result['metrics']})


def main():
    parser = argparse.ArgumentParser(description=WARNING)
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Use uma porta entre 1024 e 65535.')
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError as exc:
        parser.exit(1, f'Não foi possível abrir a porta {args.port}: {exc}. Tente --port 8502.\n')
    address = f'http://127.0.0.1:{args.port}'
    print(WARNING, flush=True)
    print(f'Painel local: {address}\nCtrl+C para encerrar. Não exponha esta porta na internet.', flush=True)
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nEncerrando o painel.')
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
