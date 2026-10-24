(function (global) {
  'use strict';

  const ROLES = [['feature', 'Entrada'], ['id', 'ID'], ['target', 'Diagnóstico / classe'], ['ignore', 'Ignorar']];
  const SEPARATORS = [['auto', 'Detectar automaticamente'], [',', 'Vírgula (,)'], [';', 'Ponto e vírgula (;)'], ['\t', 'Tabulação'], ['whitespace', 'Espaços múltiplos']];
  const STORAGE_PREFIX = 'mlp-input-editor-v1:';
  let nextId = 0;

  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  }

  function mount(element, options) {
    if (!element || typeof options?.api !== 'function') throw Error('Editor de entradas: elemento e API são necessários.');
    const api = options.api;
    const onValidated = options.onValidated || function () {};
    const onInvalid = options.onInvalid || function () {};
    const uid = 'input-editor-' + (++nextId);
    let sourceText = null, sourceBase64 = null, sourceName = 'Base original', metadata = null, columns = [];
    let revision = 0, timer = null, loading = false, busy = false, pendingReading = false;
    let rowControls = [], storageUnavailable = false, readingError = null;

    element.replaceChildren();
    element.classList.add('input-editor');
    const style = node('style', `
.input-editor{margin-top:18px;min-width:0}
.input-editor *{box-sizing:border-box}
.input-editor [hidden]{display:none!important}
.input-editor .ie-read-controls{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:end}
.input-editor .ie-read-controls .ie-apply{grid-column:1/-1;justify-self:start}
.input-editor .ie-button{font:inherit;font-size:12px;padding:8px 10px;border:1px solid #bfced9;border-radius:7px;background:#fff;color:var(--navy,#173d56);cursor:pointer;white-space:normal}
.input-editor .ie-button:disabled{opacity:.55;cursor:default}
.input-editor .ie-button[aria-pressed=true]{background:#e5f3ed;border-color:#9ecab7}
.input-editor .ie-summary{display:flex;flex-wrap:wrap;gap:8px;margin-top:17px}
.input-editor .ie-count{padding:7px 10px;border-radius:6px;background:#edf5fa;font-size:12px}
.input-editor .ie-formula{font-size:17px;font-weight:700;color:var(--teal,#12786d);margin:10px 0}
.input-editor .ie-note{font-size:12px;color:var(--muted,#5c7283);overflow-wrap:anywhere}
.input-editor .ie-status{font-size:13px;padding:10px 12px;background:#edf5fa;border-radius:7px;margin:12px 0}
.input-editor .ie-status[data-kind=error]{background:#fff1eb;color:#943f27}
.input-editor .ie-status[data-kind=valid]{background:#edf8f2;color:#21694c}
.input-editor .ie-warning{margin:10px 0;padding:9px 12px;border-left:3px solid #c49142;background:#fff8ec;font-size:12px;overflow-wrap:anywhere}
.input-editor .ie-table-wrap{overflow:auto;max-height:640px;border:1px solid #dbe5ec;border-radius:8px;margin-top:14px}
.input-editor .ie-table{width:100%;min-width:660px;border-collapse:collapse;white-space:normal;text-align:left;font-size:12px}
.input-editor .ie-table th{position:sticky;top:0;z-index:1;background:#f3f7fa;padding:11px 9px;border-bottom:1px solid #dbe5ec}
.input-editor .ie-table td{padding:10px 9px;vertical-align:top;border-bottom:1px solid #e9eef2;white-space:normal}
.input-editor .ie-table .ie-col-key{min-width:120px;max-width:165px;overflow-wrap:anywhere}
.input-editor .ie-table .ie-col-description{min-width:185px}
.input-editor .ie-table .ie-col-role{min-width:135px}
.input-editor .ie-table input,.input-editor .ie-table select{padding:8px;font-size:12px;width:100%}
.input-editor .ie-key{display:block;font-weight:750;font-size:14px;color:var(--navy,#173d56)}
.input-editor .ie-column-type{margin-top:5px;font-size:11px;color:var(--muted,#5c7283)}
.input-editor .ie-column-type[data-warning=true]{color:#943f27}
.input-editor .ie-data-row[data-role=ignore]{background:#f8fafb}
.input-editor .ie-data-row[data-role=id],.input-editor .ie-data-row[data-role=target]{background:#f2f7fb}
.input-editor .ie-preview-cell{background:#f6f9fb}
.input-editor .ie-preview-table{min-width:0;width:100%;font-size:12px;white-space:normal;margin-top:8px}
.input-editor .ie-preview-table th{position:static;background:#edf2f6;padding:6px 8px}
.input-editor .ie-preview-table td{padding:6px 8px;white-space:pre-wrap;overflow-wrap:anywhere;max-width:430px}
.input-editor .ie-visually-hidden{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(max-width:550px){.input-editor .ie-read-controls{grid-template-columns:1fr}.input-editor .ie-table-wrap{max-height:540px}.input-editor .ie-formula{font-size:15px}}
`);
    const reader = node('div', undefined, 'ie-read-controls');
    function addSelect(labelText, suffix, choices) {
      const wrap = node('div'), label = node('label', labelText), select = node('select');
      select.id = uid + '-' + suffix;
      label.htmlFor = select.id;
      choices.forEach(([value, text]) => {
        const option = node('option', text); option.value = value; select.append(option);
      });
      wrap.append(label, select); reader.append(wrap); return select;
    }
    const separator = addSelect('Separador das colunas', 'delimiter', SEPARATORS);
    const header = addSelect('Primeira linha contém nomes?', 'header', [['auto', 'Detectar automaticamente'], ['yes', 'Sim, é cabeçalho'], ['no', 'Não, já contém valores']]);
    const applyReading = node('button', 'Aplicar leitura e redefinir colunas', 'ie-button ie-apply');
    applyReading.type = 'button';
    reader.append(applyReading);
    const readingNote = node('p', 'Mudar a leitura redefine descrições e seleção de colunas somente ao aplicar.', 'ie-note');
    const sourceNote = node('p', 'Nenhuma base inspecionada.', 'ie-note');
    const summary = node('div', undefined, 'ie-summary');
    const formula = node('div', 'H = teto((E + S) / 2), com S igual ao número de classes.', 'ie-formula');
    formula.setAttribute('aria-live', 'polite');
    const explanation = node('p', 'E conta apenas as colunas marcadas como Entrada. S conta as classes do alvo. ID fica para auditoria; Diagnóstico / classe fornece a resposta esperada. Ambos ficam fora da entrada da rede. Os nomes x1, x2… preservam a posição de cada coluna no arquivo.', 'ie-note');
    const status = node('div', 'Selecione uma base para conferir as colunas.', 'ie-status');
    status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
    const warnings = node('div');
    const classSummary = node('details'); classSummary.hidden = true;
    const storageNote = node('p', '', 'ie-note'); storageNote.hidden = true;
    const tableWrap = node('div', undefined, 'ie-table-wrap'); tableWrap.hidden = true;
    tableWrap.tabIndex = 0; tableWrap.setAttribute('role', 'region'); tableWrap.setAttribute('aria-label', 'Todas as colunas do arquivo; role para ver mais colunas');
    element.append(style, reader, readingNote, sourceNote, summary, formula, explanation, status, warnings, classSummary, storageNote, tableWrap);

    function setStatus(message, kind) {
      status.textContent = message;
      status.dataset.kind = kind || 'pending';
    }
    function invalidate(message) {
      revision++;
      clearTimeout(timer); timer = null;
      setStatus(message, 'pending');
      onInvalid(message);
      return revision;
    }
    function storageFailure() {
      storageUnavailable = true;
      storageNote.hidden = false;
      storageNote.textContent = 'O navegador não permitiu guardar suas escolhas. O editor continua funcionando nesta página.';
    }
    function readSaved(sha) {
      if (!sha || storageUnavailable) return null;
      try {
        const saved = JSON.parse(global.localStorage.getItem(STORAGE_PREFIX + sha) || 'null');
        if (saved?.version !== 1 || !Array.isArray(saved.columns) || !SEPARATORS.some(([value]) => value !== 'auto' && value === saved.delimiter) || !['yes', 'no'].includes(saved.header)) return null;
        return saved;
      } catch (_) { storageFailure(); return null; }
    }
    function saveChoices() {
      if (!metadata?.source_sha256 || storageUnavailable || pendingReading) return;
      const saved = {version: 1, delimiter: metadata.delimiter, header: metadata.has_header ? 'yes' : 'no', columns: columns.map(column => ({index: column.index, description: column.description, role: column.role}))};
      try { global.localStorage.setItem(STORAGE_PREFIX + metadata.source_sha256, JSON.stringify(saved)); }
      catch (_) { storageFailure(); }
    }
    function updateControls() {
      const locked = busy || loading;
      separator.disabled = header.disabled = locked;
      applyReading.disabled = locked || (!pendingReading && !readingError);
      for (const control of rowControls) {
        control.description.disabled = control.role.disabled = control.previewButton.disabled = locked || pendingReading;
        const isMetadata = ['id', 'target'].includes(control.column.role);
        control.toggle.disabled = locked || pendingReading || isMetadata;
        control.toggle.textContent = isMetadata ? 'Fora da rede' : control.column.role === 'feature' ? 'Ignorar' : 'Usar';
        control.toggle.setAttribute('aria-label', isMetadata ? `${control.column.key}: fora da rede, usado como ${control.column.role === 'id' ? 'ID' : 'Diagnóstico'}` : `${control.column.role === 'feature' ? 'Ignorar' : 'Usar'} ${control.column.key} como entrada`);
        control.toggle.setAttribute('aria-pressed', String(control.column.role === 'feature'));
        control.row.dataset.role = control.column.role;
        control.role.value = control.column.role;
      }
      element.setAttribute('aria-busy', String(locked));
    }
    function updateSummary() {
      summary.replaceChildren();
      if (!metadata) {
        formula.textContent = 'H = teto((E + S) / 2), com S igual ao número de classes.';
        return;
      }
      const inputs = columns.filter(column => column.role === 'feature').length;
      [`${columns.length} colunas no arquivo`, `${inputs} entradas usadas (E)`, `${columns.length - inputs} fora da rede`].forEach(text => summary.append(node('span', text, 'ie-count')));
      const target = metadata.target_summary;
      const targetSelected = target && columns[target.index]?.role === 'target';
      const outputs = targetSelected ? target.classes.length : null;
      if (outputs !== null) summary.append(node('span', `${outputs} classes identificadas (S)`, 'ie-count'));
      if (outputs === null || outputs < 2) {
        formula.textContent = `${inputs} entradas identificadas. Confira uma coluna alvo com pelo menos duas classes para calcular S e H.`;
        return;
      }
      const hidden = Math.ceil((inputs + outputs) / 2);
      formula.textContent = inputs ? `H = teto((E + S) / 2) = teto((${inputs} + ${outputs}) / 2) = ${hidden} · Rede ${inputs} → ${hidden} → ${outputs}` : 'Selecione ao menos uma Entrada para montar a rede. H = teto((E + S) / 2).';
    }
    function edited() {
      invalidate('Seleção alterada. Conferindo a base com estas colunas…');
      updateSummary(); updateControls(); saveChoices();
      timer = setTimeout(() => { validate(); }, 300);
    }
    function renderColumns() {
      rowControls = [];
      tableWrap.replaceChildren();
      const table = node('table', undefined, 'ie-table');
      const caption = node('caption', 'Descrição, uso na rede e prévia dos valores de cada coluna', 'ie-visually-hidden');
      const head = node('thead'), headings = node('tr'), body = node('tbody');
      ['Coluna do arquivo', 'Descrição editável', 'Papel da coluna', 'Usar na rede', 'Valores'].forEach(text => { const cell = node('th', text); cell.scope = 'col'; headings.append(cell); });
      head.append(headings); table.append(caption, head, body);
      for (const column of columns) {
        const row = node('tr', undefined, 'ie-data-row');
        const keyCell = node('td', undefined, 'ie-col-key');
        keyCell.append(node('span', column.key, 'ie-key'), node('span', column.name || column.key));
        const type = node('div', `${column.numeric ? 'Numérica' : 'Texto ou valores mistos'} · ${column.missing || 0} vazio(s)`, 'ie-column-type');
        type.dataset.warning = String(!column.numeric || column.missing > 0);
        type.title = 'Tipo e campos vazios conferidos em toda a coluna.';
        keyCell.append(type);
        const descriptionCell = node('td', undefined, 'ie-col-description');
        const description = node('input'); description.type = 'text'; description.maxLength = 1000;
        description.id = uid + '-' + column.key + '-description'; description.value = column.description;
        const descriptionLabel = node('label', `Descrição de ${column.key}`, 'ie-visually-hidden'); descriptionLabel.htmlFor = description.id;
        descriptionCell.append(descriptionLabel, description);
        description.addEventListener('input', () => { column.description = description.value; edited(); });
        const roleCell = node('td', undefined, 'ie-col-role');
        const role = node('select'); role.id = uid + '-' + column.key + '-role';
        const roleLabel = node('label', `Papel de ${column.key}`, 'ie-visually-hidden'); roleLabel.htmlFor = role.id;
        ROLES.forEach(([value, text]) => { const option = node('option', text); option.value = value; role.append(option); });
        role.value = column.role;
        role.addEventListener('change', () => { column.role = role.value; edited(); });
        roleCell.append(roleLabel, role);
        const toggleCell = node('td'), toggle = node('button', '', 'ie-button'); toggle.type = 'button';
        toggle.addEventListener('click', () => { if (['id', 'target'].includes(column.role)) return; column.role = column.role === 'feature' ? 'ignore' : 'feature'; edited(); });
        toggleCell.append(toggle);
        const previewCell = node('td'), previewButton = node('button', 'Mostrar valores', 'ie-button'); previewButton.type = 'button';
        previewButton.setAttribute('aria-label', `Mostrar valores de ${column.key}`);
        const previewRow = node('tr'); previewRow.hidden = true; previewRow.id = uid + '-' + column.key + '-preview';
        previewButton.setAttribute('aria-controls', previewRow.id); previewButton.setAttribute('aria-expanded', 'false');
        previewButton.addEventListener('click', () => {
          const open = previewRow.hidden; previewRow.hidden = !open;
          previewButton.textContent = open ? 'Ocultar valores' : 'Mostrar valores';
          previewButton.setAttribute('aria-label', `${open ? 'Ocultar' : 'Mostrar'} valores de ${column.key}`);
          previewButton.setAttribute('aria-expanded', String(open));
          if (open && !previewRow.childElementCount) renderPreview(previewRow, column);
        });
        previewCell.append(previewButton); row.append(keyCell, descriptionCell, roleCell, toggleCell, previewCell);
        body.append(row, previewRow); rowControls.push({column, row, description, role, toggle, previewButton});
      }
      tableWrap.append(table); tableWrap.hidden = !columns.length;
      updateSummary(); updateControls();
    }
    function renderPreview(row, column) {
      const cell = node('td', undefined, 'ie-preview-cell'); cell.colSpan = 5;
      const values = (column.preview || []).slice(0, 8), lines = metadata.source_lines || [];
      cell.append(node('strong', `${column.key} · prévia dos primeiros ${values.length} de ${metadata.n_rows} registros`));
      cell.append(node('p', 'Tipo e quantidade de campos vazios foram conferidos em toda a coluna. Esta prévia mostra até 8 valores, sem alterar o arquivo.', 'ie-note'));
      const table = node('table', undefined, 'ie-preview-table'), head = node('thead'), headings = node('tr'), body = node('tbody');
      [lines.length ? 'Linha no arquivo' : 'Registro da prévia', 'Valor original'].forEach(text => { const th = node('th', text); th.scope = 'col'; headings.append(th); });
      head.append(headings);
      values.forEach((value, index) => { const tr = node('tr'); tr.append(node('td', lines[index] ?? (index + 1)), node('td', value === '' ? '(vazio)' : value)); body.append(tr); });
      table.append(head, body); cell.append(table); row.append(cell);
    }
    function payload() {
      if (!metadata || loading) throw Error('Aguarde a leitura da base antes de continuar.');
      if (pendingReading) throw Error('Aplique a nova leitura antes de continuar.');
      const result = {input_schema: {delimiter: metadata.delimiter, header: metadata.has_header ? 'yes' : 'no', columns: columns.map(column => ({index: column.index, description: column.description, role: column.role}))}};
      if (sourceBase64 !== null) result.data_base64 = sourceBase64;
      else if (sourceText !== null) result.data_text = sourceText;
      return result;
    }
    async function validate() {
      clearTimeout(timer); timer = null;
      if (!metadata || loading || pendingReading) {
        const message = pendingReading ? 'Aplique a leitura para conferir estas colunas.' : readingError || 'Aguarde a leitura de uma base válida.';
        setStatus(message, !pendingReading && readingError ? 'error' : 'pending'); onInvalid(message); return null;
      }
      const token = invalidate('Conferindo as entradas selecionadas…');
      try {
        const result = await api('/api/inspect', payload());
        if (token !== revision) return null;
        const classes = result.classes || ['M', 'B'];
        const counts = result.class_counts || {M: result.M, B: result.B};
        setStatus(`Base conferida: ${result.n} registros · ${classes.map(label => `${counts[label]} ${label}`).join(' / ')} · ${result.architecture[0]} entradas · ${result.architecture[2]} saídas. Pronta para configurar o treinamento.`, 'valid');
        saveChoices(); onValidated(result); return result;
      } catch (error) {
        if (token !== revision) return null;
        const message = error?.message || 'Não foi possível conferir a seleção.';
        setStatus(message, 'error'); onInvalid(message); return null;
      }
    }
    async function detect(text, filename, delimiter, headerChoice, restore) {
      const token = invalidate('Lendo o arquivo e identificando todas as colunas…');
      sourceText = text; sourceName = filename; metadata = null; columns = []; rowControls = [];
      pendingReading = false; loading = true; readingError = null;
      warnings.replaceChildren(); classSummary.replaceChildren(); classSummary.hidden = true;
      tableWrap.replaceChildren(); tableWrap.hidden = true;
      sourceNote.textContent = 'Fonte: ' + sourceName; updateSummary(); updateControls();
      const request = {delimiter, header: headerChoice};
      if (sourceBase64 !== null) request.data_base64 = sourceBase64;
      else if (text !== null) request.data_text = text;
      try {
        let detected = await api('/api/columns', request);
        if (token !== revision) return null;
        let saved = restore ? readSaved(detected.source_sha256) : null;
        if (saved && (saved.delimiter !== detected.delimiter || saved.header !== (detected.has_header ? 'yes' : 'no'))) {
          detected = await api('/api/columns', {...request, delimiter: saved.delimiter, header: saved.header});
          if (token !== revision) return null;
        }
        if (!Array.isArray(detected.columns) || !detected.columns.length) throw Error('O arquivo não contém colunas para inspecionar.');
        metadata = detected;
        columns = detected.columns.map(column => ({...column, description: String(column.description ?? column.name ?? column.key)}));
        if (saved && saved.columns.length === columns.length && saved.columns.every((column, index) => column.index === columns[index].index && typeof column.description === 'string' && column.description.length <= 1000 && ROLES.some(([role]) => role === column.role))) {
          saved.columns.forEach((column, index) => { columns[index].description = column.description; columns[index].role = column.role; });
          separator.value = saved.delimiter; header.value = saved.header;
          readingNote.textContent = 'Descrições e seleção recuperadas para este mesmo arquivo. Mudar e aplicar a leitura redefine essas escolhas.';
        } else {
          saved = null;
          readingNote.textContent = 'Mudar a leitura redefine descrições e seleção de colunas somente ao aplicar.';
        }
        const separatorName = SEPARATORS.find(([value]) => value === metadata.delimiter)?.[1] || metadata.delimiter;
        sourceNote.textContent = `Fonte: ${sourceName} · ${metadata.n_rows} registros · ${metadata.n_columns} colunas · ${separatorName} · ${metadata.has_header ? 'com cabeçalho' : 'sem cabeçalho'}`;
        (metadata.warnings || []).forEach(message => warnings.append(node('div', message, 'ie-warning')));
        const target = metadata.target_summary;
        if (target?.classes?.length) {
          classSummary.append(node('summary', `${target.classes.length} classes identificadas em ${target.key} · ver contagens`));
          classSummary.append(node('p', 'Contagem dos valores reais do alvo identificado no arquivo, sem usar os totais do texto introdutório.', 'ie-note'));
          const classTable = node('table', undefined, 'ie-preview-table'), head = node('thead'), heading = node('tr'), body = node('tbody');
          ['Classe', 'Registros'].forEach(text => { const th = node('th', text); th.scope = 'col'; heading.append(th); });
          target.classes.forEach(item => { const row = node('tr'); row.append(node('td', item.label || '(vazio)'), node('td', item.count)); body.append(row); });
          head.append(heading); classTable.append(head, body); classSummary.append(classTable); classSummary.hidden = false;
        }
        loading = false; renderColumns();
        return await validate();
      } catch (error) {
        if (token !== revision) return null;
        metadata = null; columns = []; rowControls = []; loading = false;
        tableWrap.replaceChildren(); tableWrap.hidden = true;
        const message = error?.message || 'Não foi possível ler o arquivo.';
        readingError = message;
        updateSummary(); updateControls();
        setStatus(message, 'error'); onInvalid(message); throw error;
      }
    }
    function load(text = null, filename = 'Base original', rawBase64 = null) {
      if (text !== null && typeof text !== 'string') return Promise.reject(Error('O conteúdo do arquivo deve ser texto.'));
      sourceBase64 = rawBase64;
      separator.value = 'auto'; header.value = 'auto';
      return detect(text, String(filename), 'auto', 'auto', true);
    }
    function readingChanged() {
      pendingReading = true;
      invalidate('Leitura alterada. Clique em “Aplicar leitura e redefinir colunas” para conferir novamente.');
      readingNote.textContent = 'Ao aplicar, as descrições e a seleção atuais serão redefinidas a partir do arquivo.';
      updateControls();
    }
    separator.addEventListener('change', readingChanged); header.addEventListener('change', readingChanged);
    applyReading.addEventListener('click', () => { detect(sourceText, sourceName, separator.value, header.value, false).catch(() => {}); });
    updateControls();
    return {load, payload, validate, setBusy(value) { busy = Boolean(value); updateControls(); }};
  }
  global.InputEditor = {mount};
})(window);
