'use strict';

// Run with Node. This checks editor state and callbacks without a browser or DOM library.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    this.listeners = {};
    this.className = '';
    this.hidden = false;
    this.disabled = false;
    this.value = '';
    this._text = '';
    this.classList = {add: name => { this.className += ' ' + name; }};
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  get childElementCount() { return this.children.length; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text = ''; this.children = children; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  dispatch(name) { for (const callback of this.listeners[name] || []) callback(); }
}

function all(element, predicate) {
  return [element, ...element.children.flatMap(child => all(child, () => true))].filter(predicate);
}
const byClass = (element, name) => all(element, child => child.className.split(/\s+/).includes(name));
const clone = value => JSON.parse(JSON.stringify(value));

function fixture(api) {
  const storage = new Map();
  const window = {localStorage: {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
  }};
  const context = vm.createContext({window, document: {createElement: tag => new Element(tag)}, setTimeout, clearTimeout});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'input-editor.js'), 'utf8'), context, {filename: 'input-editor.js'});
  const root = new Element('div'), invalid = [], validated = [], requests = [];
  const editor = window.InputEditor.mount(root, {
    api: async (url, payload) => { requests.push({url, payload: clone(payload)}); return api(url, payload); },
    onInvalid: message => invalid.push(message),
    onValidated: result => validated.push(result),
  });
  return {root, editor, invalid, validated, requests};
}

function column(index, role, preview, name = 'x' + (index + 1)) {
  return {index, key: 'x' + (index + 1), name, description: name, role,
    numeric: role === 'feature', missing: 0, preview};
}

const proteinClasses = ['CYT', 'ERL', 'EXC', 'ME1', 'ME2', 'ME3', 'MIT', 'NUC', 'POX', 'VAC'];
const protein = {
  delimiter: 'whitespace', source_format: 'protein_localization', has_header: false,
  source_sha256: 'protein-fixture', n_rows: 12, n_columns: 10, source_lines: [79, 80, 81],
  warnings: ['Texto introdutório reconhecido; dados começam na linha 79.'],
  target_summary: {index: 9, key: 'x10', classes: proteinClasses.map(label => ({label, count: ['MIT', 'NUC'].includes(label) ? 2 : 1})), supported: true},
  columns: [column(0, 'id', ['ADT1_YEAST', 'ADT2_YEAST', 'AAR2_YEAST']),
    ...Array.from({length: 8}, (_, index) => column(index + 1, 'feature', ['0.58', '0.43', '0.58'])),
    column(9, 'target', ['MIT', 'MIT', 'NUC'])],
};
const csv = {
  delimiter: ',', source_format: 'delimited', has_header: true,
  source_sha256: 'csv-fixture', n_rows: 50, n_columns: 4, source_lines: [2, 3], warnings: [],
  target_summary: {index: 1, key: 'x2', classes: [{label: 'M', count: 22}, {label: 'B', count: 28}], supported: true},
  columns: [column(0, 'id', ['001', '002'], 'id'), column(1, 'target', ['M', 'B'], 'diagnosis'),
    column(2, 'feature', ['1.2', '2.1'], 'measure_a'), column(3, 'feature', ['3.4', '4.3'], 'measure_b')],
};
const valid = {n: 50, B: 28, M: 22, architecture: [2, 2, 2]};
const proteinValid = {n: 12, classes: proteinClasses, class_counts: Object.fromEntries(protein.target_summary.classes.map(item => [item.label, item.count])), architecture: [8, 9, 10], n_parameters: 181};
const selectionError = 'A seleção precisa de ao menos uma entrada numérica.';

async function checkProteinInspection() {
  const f = fixture(async url => {
    if (url === '/api/columns') return clone(protein);
    if (url === '/api/inspect') return clone(proteinValid);
    assert.fail('Unexpected API: ' + url);
  });
  assert.deepEqual(await f.editor.load('protein source bytes as text', 'proteina.txt'), proteinValid);
  assert.equal(f.validated.length, 1, 'Valid multiclass inspection reports readiness.');
  assert.equal(byClass(f.root, 'ie-status')[0].dataset.kind, 'valid');
  assert.equal(byClass(f.root, 'ie-table-wrap')[0].hidden, false, 'The column inspection remains available.');
  assert.equal(byClass(f.root, 'ie-data-row').length, 10);
  const payload = clone(f.editor.payload());
  assert.equal(payload.input_schema.delimiter, 'whitespace');
  assert.deepEqual(payload.input_schema.columns.map(item => item.role), ['id', ...Array(8).fill('feature'), 'target']);
  assert.equal(payload.data_text, 'protein source bytes as text');
  assert.match(byClass(f.root, 'ie-summary')[0].textContent, /8 entradas usadas/);
  assert.match(byClass(f.root, 'ie-formula')[0].textContent, /Rede 8 → 9 → 10/);
  assert.doesNotMatch(byClass(f.root, 'ie-formula')[0].textContent, /Rede 8 → 5 → 2/);
  assert.ok(all(f.root, el => el.tagName === 'OPTION').some(el => el.value === 'whitespace'));
  const classDetails = all(f.root, el => el.tagName === 'DETAILS')[0];
  assert.equal(classDetails.hidden, false);
  const counts = all(classDetails, el => el.tagName === 'TBODY')[0];
  assert.deepEqual(counts.children.map(row => row.children.map(cell => cell.textContent)), protein.target_summary.classes.map(item => [item.label, String(item.count)]));
  const preview = all(f.root, el => el.attributes['aria-label'] === 'Mostrar valores de x1')[0];
  preview.dispatch('click');
  const firstPreview = all(f.root, el => el.id?.endsWith('-x1-preview'))[0];
  assert.equal(firstPreview.hidden, false);
  const firstValue = all(firstPreview, el => el.tagName === 'TBODY')[0].children[0];
  assert.deepEqual(firstValue.children.map(el => el.textContent), ['79', 'ADT1_YEAST']);
  assert.deepEqual(await f.editor.validate(), proteinValid);
  assert.equal(byClass(f.root, 'ie-data-row').length, 10, 'Revalidation does not erase recognized columns.');
  assert.equal(f.requests.filter(request => request.url === '/api/train').length, 0);
}

async function checkValidationFailureKeepsColumns() {
  const f = fixture(async url => {
    if (url === '/api/columns') return clone(protein);
    throw Error(selectionError);
  });
  assert.equal(await f.editor.load('protein data', 'proteina.txt'), null);
  assert.equal(f.validated.length, 0);
  assert.equal(byClass(f.root, 'ie-status')[0].dataset.kind, 'error');
  assert.equal(byClass(f.root, 'ie-data-row').length, 10);
  assert.equal(f.invalid.at(-1), selectionError);
}

async function checkReadingRetry() {
  const parseError = 'Linha 7: esperadas 1 colunas; encontradas 2.';
  let fail = true;
  const f = fixture(async url => {
    if (url === '/api/columns') { if (fail) throw Error(parseError); return clone(csv); }
    if (url === '/api/inspect') return clone(valid);
    assert.fail('Unexpected API: ' + url);
  });
  await assert.rejects(f.editor.load('csv data', 'base.csv'), new RegExp('Linha 7'));
  const apply = byClass(f.root, 'ie-apply')[0];
  assert.equal(apply.disabled, false, 'Failed reading must allow retry with current controls.');
  assert.equal(await f.editor.validate(), null);
  assert.equal(f.invalid.at(-1), parseError, 'Validation must preserve the precise parsing error.');
  assert.equal(byClass(f.root, 'ie-status')[0].textContent, parseError);
  assert.equal(byClass(f.root, 'ie-status')[0].dataset.kind, 'error');
  assert.throws(() => f.editor.payload(), /Aguarde a leitura/);
  fail = false;
  apply.dispatch('click');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.validated.length, 1, 'Retry reaches successful inspection without changing a control.');
  assert.equal(byClass(f.root, 'ie-status')[0].dataset.kind, 'valid');
  assert.equal(apply.disabled, true);
}

async function checkCsvRegression() {
  const f = fixture(async url => url === '/api/columns' ? clone(csv) : clone(valid));
  assert.deepEqual(await f.editor.load('id,diagnosis,measure_a,measure_b\n001,M,1.2,3.4', 'base.csv'), valid);
  assert.equal(f.validated.length, 1);
  assert.equal(byClass(f.root, 'ie-data-row').length, 4);
  assert.match(byClass(f.root, 'ie-formula')[0].textContent, /Rede 2 → 2 → 2/);
  assert.match(byClass(f.root, 'ie-status')[0].textContent, /Pronta para configurar o treinamento/);
  assert.equal(byClass(f.root, 'ie-status')[0].dataset.kind, 'valid');
  const payload = clone(f.editor.payload());
  assert.equal(payload.input_schema.header, 'yes');
  assert.equal(payload.input_schema.delimiter, ',');
  assert.deepEqual(payload.input_schema.columns.map(item => item.role), ['id', 'target', 'feature', 'feature']);
  const rows = byClass(f.root, 'ie-data-row');
  assert.equal(rows[0].children[3].children[0].disabled, true, 'ID stays outside the network.');
  assert.equal(rows[1].children[3].children[0].disabled, true, 'Target stays outside the network.');
  assert.equal(rows[2].children[3].children[0].disabled, false);
  f.editor.setBusy(true);
  assert.equal(rows[2].children[3].children[0].disabled, true);
  f.editor.setBusy(false);
  assert.equal(rows[2].children[3].children[0].disabled, false);
}

(async () => {
  await checkProteinInspection();
  await checkValidationFailureKeepsColumns();
  await checkReadingRetry();
  await checkCsvRegression();
  console.log('PASS: protein 8→9→10 columns/classes/preview, rejected selection, reading retry, and valid M/B CSV editor regression.');
})().catch(error => { console.error(error); process.exitCode = 1; });
