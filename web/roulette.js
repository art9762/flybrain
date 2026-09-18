'use strict';

const $ = id => document.getElementById(id);
const format = value => Number.isFinite(Number(value)) ? Number(value).toLocaleString('ru-RU') : '—';
const percent = value => Number.isFinite(value) ? `${Math.round(value * 100)}%` : '—';
const ITEMS = [
  {id: 'magnifier', action: 2, name: 'Лупа', description: 'Показывает вам тип следующего патрона.'},
  {id: 'beer', action: 3, name: 'Пиво', description: 'Извлекает следующий патрон и раскрывает его тип. Сбрасывает эффект пилы.'},
  {id: 'cigarette', action: 4, name: 'Сигарета', description: 'Восстанавливает один заряд здоровья, если здоровье ниже максимума.'},
  {id: 'saw', action: 5, name: 'Пила', description: 'Удваивает урон следующего выстрела. Холостой тоже расходует эффект.'},
  {id: 'cuffs', action: 6, name: 'Наручники', description: 'Пропускают следующую передачу хода сопернику. Один раз за зарядку.'},
  {id: 'inverter', action: 7, name: 'Инвертор', description: 'Меняет следующий патрон: боевой на холостой или наоборот. Тип остаётся неизвестным.'},
];

let state = null;
let connected = false;
let polling = false;
let mutationPending = false;
let mutationVersion = 0;
let flyTimer = null;
let statusTimer = null;
let lastFingerprint = null;
let stopRequested = false;
let dismissedRunError = null;
let flashTimer = null;
let localMessage = null;
let flyFailure = false;

function svgUse(symbol, className = '') {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  if (className) svg.setAttribute('class', className);
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', `#${symbol}`);
  svg.append(use);
  return svg;
}

function initializeItems() {
  for (const item of ITEMS) {
    const button = document.createElement('button');
    button.className = 'item-button';
    button.dataset.action = String(item.action);
    button.disabled = true;
    button.title = `${item.name}. ${item.description}`;
    button.append(svgUse(`icon-${item.id}`));
    const name = document.createElement('span');
    name.className = 'item-name';
    name.textContent = item.name;
    const count = document.createElement('span');
    count.className = 'item-count';
    count.id = `item-count-${item.id}`;
    count.textContent = '0';
    button.append(name, count);
    $('inventory').append(button);
    const term = document.createElement('dt');
    term.textContent = item.name;
    const detail = document.createElement('dd');
    detail.textContent = item.description;
    $('rules-items').append(term, detail);
  }
}

function showError(message) {
  $('error-text').textContent = message;
  $('error-banner').hidden = false;
}

function clearError() {
  $('error-banner').hidden = true;
  dismissedRunError = state?.run?.error || null;
}

async function jsonRequest(path, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 90000);
  try {
    const response = await fetch(path, body === undefined
      ? {cache: 'no-store', signal: controller.signal}
      : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body), signal: controller.signal});
    let payload;
    try { payload = await response.json(); }
    catch { throw new Error(`Сервер вернул неожиданный ответ (${response.status}).`); }
    if (!response.ok) {
      const detail = typeof payload.detail === 'string' ? payload.detail : null;
      throw new Error(detail || payload.error || `Не удалось выполнить действие (${response.status}).`);
    }
    return payload;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('Сервер не ответил вовремя. Проверяем состояние перед следующим действием.');
    if (error instanceof TypeError) throw new Error('Нет связи с игровым сервером. Проверьте, что он запущен.');
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function acceptStatus(next) {
  if (!next || !next.game || !next.run) return;
  state = next;
  connected = true;
  render();
}

async function refreshStatus() {
  if (polling || mutationPending) return;
  polling = true;
  const version = mutationVersion;
  try {
    const next = await jsonRequest('/api/status');
    if (version === mutationVersion && !mutationPending) acceptStatus(next);
  } catch (error) {
    if (version === mutationVersion) {
      connected = false;
      cancelFly();
      render();
      if (!state) showError(error.message);
    }
  } finally {
    polling = false;
  }
}

function cancelFly() {
  if (flyTimer) clearTimeout(flyTimer);
  flyTimer = null;
}

function brainReady() {
  return !!state?.brain && !state?.run?.loading;
}

function serverWorking() {
  return !!(state?.run?.loading || state?.run?.busy || state?.run?.training);
}

function canPlay() {
  return connected && brainReady() && !mutationPending && !serverWorking() && state.game.winner === null;
}

function planFlyTurn() {
  if (canPlay() && state.game.turn === 1 && !document.hidden && !flyFailure) {
    if (!flyTimer) flyTimer = setTimeout(async () => {
      flyTimer = null;
      if (canPlay() && state.game.turn === 1) await mutate('/api/fly', {});
    }, 950);
  } else cancelFly();
}

async function mutate(path, body) {
  if (mutationPending || !connected) return;
  mutationPending = true;
  mutationVersion += 1;
  cancelFly();
  clearError();
  render();
  let failed = false;
  try {
    const result = await jsonRequest(path, body);
    acceptStatus(result);
    if (path === '/api/new') {
      flyFailure = false;
      localMessage = 'Новая партия. Патроны перемешаны, предметы выданы.';
      lastFingerprint = null;
    } else if (path === '/api/checkpoint') {
      localMessage = body.action === 'save' ? 'Мозг сохранён. Обученные веса доступны для следующего запуска.' : 'Сохранённые веса загружены.';
    } else localMessage = null;
  } catch (error) {
    failed = true;
    if (path === '/api/fly') flyFailure = true;
    showError(error.message);
  } finally {
    // A timed-out request may still have completed on the server. Reconcile
    // before allowing another move, rather than retrying the same action.
    if (failed) {
      try { acceptStatus(await jsonRequest('/api/status')); }
      catch { connected = false; }
    }
    mutationPending = false;
    render();
  }
}

function renderHealth(element, hp, maximum) {
  element.replaceChildren();
  const value = Math.max(0, Number(hp) || 0);
  const max = Math.max(1, Number(maximum) || 4);
  for (let i = 0; i < max; i++) element.append(svgUse('icon-bolt', i >= value ? 'empty' : ''));
  const label = document.createElement('span');
  label.className = 'health-number';
  label.textContent = `${value}/${max}`;
  element.append(label);
  element.setAttribute('aria-label', `${element.id === 'human-health' ? 'Ваше здоровье' : 'Здоровье мухи'}: ${value} из ${max}`);
}

function renderShells(game) {
  const exact = game.counts_exact !== false && game.live !== null && game.blank !== null;
  const live = Number(game.live) || 0;
  const blank = Number(game.blank) || 0;
  const total = Number(game.shells_remaining) || live + blank;
  $('live-count').textContent = exact ? live : '?';
  $('blank-count').textContent = exact ? blank : '?';
  const key = `${exact}:${live}:${blank}:${total}`;
  if ($('shells').dataset.key !== key) {
    $('shells').dataset.key = key;
    $('shells').replaceChildren();
    for (let i = 0; i < total; i++) {
      const shell = document.createElement('span');
      shell.className = exact ? `shell${i >= live ? ' blank' : ''}` : 'shell unknown';
      shell.setAttribute('aria-hidden', 'true');
      $('shells').append(shell);
    }
  }
  $('shells').setAttribute('aria-label', exact ? `${live} боевых и ${blank} холостых. Порядок неизвестен.` : `${total} патронов. После инверсии точный состав неизвестен.`);
  document.querySelector('.shell-order').textContent = exact ? 'ПОРЯДОК НЕИЗВЕСТЕН' : `ИНВЕРСИЯ / ОСТАЛОСЬ ${total}`;
  const known = game.known_shell;
  $('known-shell').hidden = known === null || known === undefined;
  if (known !== null && known !== undefined) {
    const isLive = known === 'live' || known === true;
    $('known-shell').textContent = `ЛУПА → СЛЕДУЮЩИЙ ${isLive ? 'БОЕВОЙ' : 'ХОЛОСТОЙ'}`;
  }
}

function renderInventory(game) {
  const legal = new Set(game.legal_actions || []);
  const enabled = canPlay() && game.turn === 0;
  document.querySelectorAll('[data-action]').forEach(button => {
    button.disabled = !enabled || !legal.has(Number(button.dataset.action));
  });
  for (const item of ITEMS) {
    const count = Number(game.inventories?.[0]?.[item.id]) || 0;
    $(`item-count-${item.id}`).textContent = count > 0 ? `×${count}` : '—';
    document.querySelector(`[data-action="${item.action}"]`).setAttribute('aria-label', `${item.name}: ${count}. ${item.description}`);
  }
  const dealerKey = JSON.stringify(game.inventories?.[1] || {});
  if ($('dealer-items').dataset.key !== dealerKey) {
    $('dealer-items').dataset.key = dealerKey;
    $('dealer-items').replaceChildren();
    for (const item of ITEMS) {
      const count = Number(game.inventories?.[1]?.[item.id]) || 0;
      if (!count) continue;
      const cell = document.createElement('span');
      cell.className = 'dealer-item';
      cell.title = `${item.name}: ${count}`;
      cell.setAttribute('aria-label', `${item.name}: ${count}`);
      cell.append(svgUse(`icon-${item.id}`));
      const number = document.createElement('span');
      number.textContent = count;
      cell.append(number);
      $('dealer-items').append(cell);
    }
  }
}

function renderLog(game) {
  const log = Array.isArray(game.log) ? game.log : [];
  const fingerprint = JSON.stringify(log);
  if (lastFingerprint !== fingerprint) {
    if (lastFingerprint !== null && log.length) {
      const lastShot = [...log].reverse().find(event => Number(event.action_index) < 2 && event.action_index !== undefined);
      const previousLog = JSON.parse(lastFingerprint);
      const oldShot = [...previousLog].reverse().find(event => Number(event.action_index) < 2 && event.action_index !== undefined);
      if (lastShot && JSON.stringify(lastShot) !== JSON.stringify(oldShot)) {
        $('game-table').classList.remove('fired');
        void $('game-table').offsetWidth;
        $('game-table').classList.add('fired');
        clearTimeout(flashTimer);
        flashTimer = setTimeout(() => $('game-table').classList.remove('fired'), 400);
      }
    }
    lastFingerprint = fingerprint;
    $('event-log').replaceChildren();
    for (const event of log) {
      const row = document.createElement('li');
      row.textContent = typeof event === 'string' ? event : event.text || event.message || 'Действие завершено.';
      $('event-log').append(row);
    }
    $('event-log').scrollTop = $('event-log').scrollHeight;
  }
  const event = log.at(-1);
  $('last-event').textContent = localMessage || (typeof event === 'string' ? event : event?.text || event?.message) || 'Стол готов. Осталось разбудить соперника.';
}

function renderTurn(game, run) {
  let status, hint;
  if (!connected) {
    status = 'НЕТ СОЕДИНЕНИЯ'; hint = 'Ждём игровой сервер…';
  } else if (run.loading) {
    status = 'МУХА ПРОСЫПАЕТСЯ'; hint = 'Загружается весь коннектом. Это может занять несколько минут.';
  } else if (run.training) {
    status = 'ИДЁТ ОБУЧЕНИЕ'; hint = 'Стол на паузе. После обучения можно продолжить партию.';
  } else if (!brainReady()) {
    status = 'ВАШ СОПЕРНИК ЕЩЁ СПИТ'; hint = 'Нажмите «Разбудить муху», чтобы начать.';
  } else if (game.winner !== null) {
    status = 'ПАРТИЯ ЗАВЕРШЕНА'; hint = 'Мозг сохранит опыт обучения для следующей партии.';
  } else if (flyFailure && game.turn === 1) {
    status = 'ХОД МУХИ ПРИОСТАНОВЛЕН'; hint = 'Не удалось завершить решение.';
  } else if (game.turn === 1 || run.busy) {
    status = 'МУХА ПРИНИМАЕТ РЕШЕНИЕ'; hint = game.saw_active?.[1] ? 'У мухи активна пила: следующий боевой нанесёт двойной урон.' : 'Её ход. Следите за патронами и предметами.';
  } else if (mutationPending) {
    status = 'ДЕЙСТВИЕ ВЫПОЛНЯЕТСЯ'; hint = 'Обновляем состояние стола…';
  } else {
    status = 'ВАШ ХОД';
    hint = game.saw_active?.[0] ? 'Пила активна. Следующий боевой нанесёт двойной урон.' : game.cuffed?.[1] ? 'Муха в наручниках. Следующая передача хода будет пропущена.' : game.inverted ? 'Следующий патрон изменён инвертором. Точный состав неизвестен.' : 'Используйте предмет или выберите, в кого стрелять.';
  }
  $('turn-status').lastChild.textContent = ` ${status}`;
  $('turn-status').classList.toggle('waiting', !connected || run.loading || run.training || run.busy || game.turn === 1);
  $('turn-hint').textContent = hint;
  $('retry-fly').hidden = !flyFailure || game.turn !== 1 || game.winner !== null;
  $('retry-fly').disabled = !canPlay();
  $('fly-portrait').classList.toggle('thinking', connected && brainReady() && (run.training || run.busy || (game.turn === 1 && game.winner === null)));
  const finished = game.winner !== null;
  $('round-overlay').hidden = !finished;
  if (finished) {
    $('round-result').textContent = game.winner === 0 ? 'Человек выжил.' : game.winner === 1 ? 'Муха победила.' : 'Равновесие.';
    $('round-copy').textContent = game.winner === 0 ? 'На этот раз ваших решений оказалось достаточно. Муха ждёт реванша.' : game.winner === 1 ? 'У маленького мозга оказались большие шансы. Попробуем ещё раз?' : 'Достигнут лимит действий. Эта партия заканчивается ничьей.';
  }
  const idle = connected && !mutationPending && !serverWorking();
  $('new-game-button').disabled = !idle;
  $('play-again').disabled = !idle;
}

function renderBrain(brain, run, training) {
  const ready = brainReady();
  const idle = connected && !mutationPending && !serverWorking();
  $('brain-state').className = `brain-state${run.loading || run.training || run.busy ? ' working' : ready ? ' ready' : ''}`;
  $('brain-state-text').textContent = !connected ? 'НЕТ СВЯЗИ' : run.loading ? 'ЗАГРУЖАЕТСЯ КОННЕКТОМ' : run.training ? 'ИДЁТ ОБУЧЕНИЕ' : ready ? 'КОННЕКТОМ АКТИВЕН' : 'ОЖИДАЕТ ЗАПУСКА';
  const backend = typeof brain?.backend === 'object' ? brain.backend.backend : brain?.backend;
  $('backend-label').textContent = ready ? backend || (run.backend !== 'auto' ? run.backend : '') : '';
  $('neuron-count').textContent = brain ? format(brain.neurons) : '—';
  $('edge-count').textContent = brain ? format(brain.edges) : '—';
  $('wake-button').disabled = !idle || ready;
  $('wake-label').textContent = run.loading ? 'Муха просыпается…' : ready ? 'Муха за столом' : 'Разбудить муху';
  $('wake-note').textContent = run.loading ? 'Первый запуск: загрузка графа и подготовка модели.' : ready ? 'В игре участвует весь граф. Веса во время партии заморожены.' : state?.checkpoint_available ? 'При запуске загрузится сохранённый мозг.' : 'При первом запуске загружается весь граф.';
  $('train-button').disabled = !ready || !idle;
  $('training-steps').disabled = !ready || !idle;
  $('save-button').disabled = !ready || !idle;
  $('restore-button').disabled = !ready || !idle || !state?.checkpoint_available;
  $('training-progress').hidden = !run.training;
  $('stop-button').disabled = !run.training || stopRequested || mutationPending;
  const completed = Number(training.steps) || 0;
  const target = Math.max(1, Number(training.target_steps) || 2000);
  $('progress-bar').max = target;
  $('progress-bar').value = Math.min(completed, target);
  $('progress-text').textContent = stopRequested ? 'Завершаем и сохраняем…' : training.state === 'evaluating' ? completed ? 'Проверяем результат…' : 'Проверяем исходный мозг…' : `${format(completed)} / ${format(target)}`;
  $('trained-steps').textContent = format(completed);
  $('training-episodes').textContent = completed || training.episodes ? format(training.episodes || 0) : '—';
  const before = training.teacher_agreement_before;
  const after = training.teacher_agreement_after;
  $('teacher-score').textContent = Number.isFinite(after) ? `${percent(before)} → ${percent(after)}` : Number.isFinite(before) ? `${percent(before)} → …` : '—';
  const evidence = Number.isFinite(after) && (training.state === 'complete' || training.state === 'stopped');
  const changed = Number(brain?.learningUpdates) > 0 || Number(brain?.changedWeights) > 0;
  $('training-badge').classList.toggle('trained', evidence || changed);
  $('training-badge').textContent = !ready ? 'НЕ ЗАГРУЖЕНА' : run.training ? 'ОБУЧЕНИЕ' : evidence ? 'ЕСТЬ ПРОВЕРКА' : changed ? 'ВЕСА ОБНОВЛЕНЫ' : 'НЕ ОБУЧЕНА';
  $('fly-kind').textContent = evidence || changed ? 'ОБУЧЕННЫЕ ВЕСА' : 'КОННЕКТОМ';
  if (run.training) {
    $('training-result-note').textContent = 'В конце — проверка на отдельных ситуациях и автоматическое сохранение.';
  } else if (evidence) {
    const random = training.evaluation?.random;
    const heuristic = training.evaluation?.heuristic;
    const samples = training.heldout_after?.samples;
    const parts = [];
    if (samples) parts.push(`Совпадение на ${format(samples)} отдельных ситуациях.`);
    if (random?.games) parts.push(`Победы против случайного: ${random.wins}/${random.games}.`);
    if (heuristic?.games) parts.push(`Против учителя: ${heuristic.wins}/${heuristic.games}.`);
    parts.push('Это небольшой тест, не гарантия силы игры.');
    $('training-result-note').textContent = parts.join(' ');
  } else if (training.state === 'error') {
    $('training-result-note').textContent = 'Обучение завершилось с ошибкой. Результат не подтверждён.';
  } else if (changed) {
    $('training-result-note').textContent = 'Загружены изменённые веса. Метрики этого обучения отсутствуют.';
  } else $('training-result-note').textContent = 'Метрики появятся после обучения.';
  $('graph-detail').textContent = ready ? `В полном графе: ${format(brain.neurons)} нейронов, ${format(brain.edges)} связей. Пластичных связей: ${format(brain.plasticEdges)}. На входе — 48 числовых признаков видимого состояния.` : '';
  if (!run.training) stopRequested = false;
}

function render() {
  $('connection').className = `connection ${connected ? 'online' : 'offline'}`;
  $('connection').lastChild.textContent = connected ? ' ЛОКАЛЬНО / ONLINE' : ' НЕТ СОЕДИНЕНИЯ';
  if (!state) return;
  const {game, run, brain, training = {}} = state;
  renderHealth($('human-health'), game.hp?.[0], game.max_hp);
  renderHealth($('fly-health'), game.hp?.[1], game.max_hp);
  $('load-label').textContent = `ЗАРЯДКА ${String(game.load_number || 1).padStart(2, '0')}`;
  renderShells(game);
  renderInventory(game);
  renderLog(game);
  renderTurn(game, run);
  renderBrain(brain, run, training);
  if (run.error && run.error !== dismissedRunError) showError(run.error);
  if (!run.error) dismissedRunError = null;
  planFlyTurn();
}

initializeItems();
document.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', () => {
  const action = Number(button.dataset.action);
  if (!button.disabled && canPlay() && state.game.turn === 0 && state.game.legal_actions.includes(action)) {
    localMessage = null;
    void mutate('/api/action', {action});
  }
}));
$('retry-fly').addEventListener('click', () => {
  flyFailure = false;
  clearError();
  render();
});
$('wake-button').addEventListener('click', () => void mutate('/api/load', {backend: 'auto'}));
$('train-button').addEventListener('click', () => {
  stopRequested = false;
  void mutate('/api/train', {steps: Number($('training-steps').value)});
});
$('stop-button').addEventListener('click', () => {
  stopRequested = true;
  void mutate('/api/stop', {});
});
$('save-button').addEventListener('click', () => void mutate('/api/checkpoint', {action: 'save'}));
$('restore-button').addEventListener('click', () => void mutate('/api/checkpoint', {action: 'load'}));
$('new-game-button').addEventListener('click', () => void mutate('/api/new', {starting_player: 0}));
$('play-again').addEventListener('click', () => void mutate('/api/new', {starting_player: 0}));
$('error-dismiss').addEventListener('click', clearError);
$('log-toggle').addEventListener('click', () => {
  const expanded = $('event-log').hidden;
  $('event-log').hidden = !expanded;
  $('log-toggle').setAttribute('aria-expanded', String(expanded));
  $('log-toggle').querySelector('span').textContent = expanded ? '−' : '+';
  if (expanded) $('event-log').scrollTop = $('event-log').scrollHeight;
});
$('rules-open').addEventListener('click', () => $('rules-dialog').showModal());
$('rules-close').addEventListener('click', () => $('rules-dialog').close());
$('rules-dialog').addEventListener('click', event => {
  const rect = $('rules-dialog').getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $('rules-dialog').close();
});
document.addEventListener('keydown', event => {
  if (event.repeat || event.altKey || event.ctrlKey || event.metaKey || $('rules-dialog').open || /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(document.activeElement?.tagName || '')) return;
  const action = event.key === '1' ? 0 : event.key === '2' ? 1 : null;
  if (action !== null && canPlay() && state.game.turn === 0 && state.game.legal_actions.includes(action)) {
    event.preventDefault();
    localMessage = null;
    void mutate('/api/action', {action});
  }
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) cancelFly();
  else void refreshStatus();
});
window.addEventListener('pagehide', () => {
  cancelFly();
  clearInterval(statusTimer);
});
window.addEventListener('pageshow', () => {
  clearInterval(statusTimer);
  statusTimer = setInterval(() => { if (!document.hidden) void refreshStatus(); }, 1000);
  void refreshStatus();
});
void refreshStatus();
