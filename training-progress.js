(function (root) {
  'use strict';

  function duration(seconds) {
    const whole = Math.max(0, Math.floor(Number(seconds) || 0));
    const hours = Math.floor(whole / 3600), minutes = Math.floor(whole % 3600 / 60);
    return (hours ? hours + 'h ' : '') + (hours || minutes ? minutes + 'min ' : '') + whole % 60 + 's';
  }

  function render(state, get) {
    const detail = state.detail, total = Math.max(1, Number(state.total) || 1);
    const done = Math.min(total, Math.max(0, Number(state.done) || 0));
    get('trainingProgress').hidden = state.status === 'idle';
    get('progress').max = total;
    get('progress').value = done;
    get('overallProgressLabel').textContent = `Etapas concluídas: ${done} de ${total}`;
    get('progressNote').textContent = 'Cada rodada e o modelo final contam como uma etapa. Esta barra não estima o tempo restante.';
    get('status').dataset.kind = state.status === 'error' ? 'blocked' : state.busy ? 'pending' : 'ready';
    get('status').textContent = state.busy && done === total ? 'Finalizando e salvando os resultados…' : state.message;
    get('epochProgressBlock').hidden = !detail;
    get('trainingRun').textContent = state.run_id ? 'Execução: ' + state.run_id : '';
    if (!detail) {
      get('trainingPhase').textContent = '';
      get('trainingTimes').textContent = '';
      get('trainingUpdated').textContent = '';
      get('epochProgress').value = 0;
      return;
    }
    const selection = detail.phase === 'selecao_interna';
    const stage = detail.stage === 'final' ? 'Modelo final' : `Rodada ${detail.fold} de ${detail.folds_total}`;
    const phase = selection ? 'seleção com validação interna' : 'retreino com as épocas escolhidas';
    const epochs = Math.max(1, Number(detail.epochs_total) || 1);
    get('trainingPhase').textContent = `${stage} · ${phase}${detail.phase_complete ? ' · fase encerrada' : ''}`;
    get('epochProgress').max = epochs;
    get('epochProgress').value = Math.min(epochs, Math.max(0, Number(detail.epoch) || 0));
    get('epochProgressLabel').textContent = `Épocas concluídas nesta fase: ${detail.epoch} de ${selection ? 'até ' : ''}${epochs}`;
    get('epochProgressNote').textContent = selection
      ? (detail.phase_complete ? `Seleção encerrada após ${detail.epoch} épocas${detail.chosen_epochs ? '; ' + detail.chosen_epochs + ' escolhidas para o retreino' : ''}.`
        : 'O limite é um máximo: a parada antecipada pode encerrar esta fase antes. Uma época percorre todos os lotes de treino da fase.')
      : 'O retreino começa com novos pesos e usa a quantidade de épocas escolhida na validação interna.';
    get('trainingTimes').textContent = `Tempo da fase: ${duration(detail.phase_elapsed_seconds)} · ${detail.stage === 'final' ? 'modelo final' : 'rodada'}: ${duration(detail.fold_elapsed_seconds)} · execução: ${duration(detail.elapsed_seconds)}`;
    const updated = new Date(detail.updated_utc);
    get('trainingUpdated').textContent = Number.isNaN(updated.getTime()) ? '' : 'Última atualização: ' + updated.toLocaleTimeString('pt-BR');
  }

  // Uma nova consulta só começa depois da anterior. refresh() compartilha a
  // consulta pendente, e stop() invalida respostas que ainda estejam a caminho.
  function createPoller({fetchStatus, onState, onError, interval = 1000,
                         schedule = setTimeout, cancel = clearTimeout}) {
    let enabled = false, timer = null, pending = null, generation = 0;
    function refresh() {
      if (!enabled) return Promise.resolve();
      if (pending) return pending;
      if (timer !== null) { cancel(timer); timer = null; }
      const current = generation;
      pending = Promise.resolve().then(fetchStatus).then(state => {
        if (enabled && current === generation) onState(state);
        return state;
      }).catch(error => {
        if (enabled && current === generation) onError(error);
      }).finally(() => {
        pending = null;
        if (enabled) timer = schedule(refresh, interval);
      });
      return pending;
    }
    return {
      start() { if (!enabled) { enabled = true; generation++; } return refresh(); },
      refresh,
      stop() { enabled = false; generation++; if (timer !== null) cancel(timer); timer = null; },
    };
  }

  root.TrainingProgress = {duration, render, createPoller};
})(typeof window === 'undefined' ? globalThis : window);
