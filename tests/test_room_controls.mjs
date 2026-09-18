import assert from 'node:assert/strict';
import { test } from 'node:test';
import { controlSettings, syncRunControls } from '../web/room-controls.mjs';

function controls() {
  return {
    backend: { value: 'auto', disabled: false },
    learning: { checked: true },
    speed: { value: '1' },
  };
}

function state(overrides = {}, brain = null) {
  return {
    brain,
    run: { backend_requested: 'mps', loading: false, learning: false, speed: 4, ...overrides },
  };
}

test('reload restores the loaded GPU selection and preserves run settings', () => {
  const inputs = controls();
  const loaded = state({}, { backend: { backend: 'mps' } });
  syncRunControls(inputs, loaded, true);
  assert.equal(inputs.backend.value, 'mps');
  assert.equal(inputs.backend.disabled, true);
  assert.equal(inputs.learning.checked, false);
  assert.equal(inputs.speed.value, '4');
  assert.deepEqual(controlSettings(inputs, loaded), { learning: false, speed: 4 });
});

test('loaded brain never receives a device change even from a stale selector', () => {
  for (const selected of ['cpu', 'mps']) {
    const loaded = state({ backend_requested: selected }, { backend: { backend: selected } });
    assert.equal('backend' in controlSettings(controls(), loaded), false);
  }
});

test('reload during loading locks and restores the requested GPU', () => {
  const inputs = controls();
  const loading = state({ loading: true });
  syncRunControls(inputs, loading, true);
  assert.equal(inputs.backend.value, 'mps');
  assert.equal(inputs.backend.disabled, true);
  assert.equal('backend' in controlSettings(inputs, loading), false);
});

test('auto stays auto even when the runtime chose Metal', () => {
  const inputs = controls();
  syncRunControls(inputs, state({ backend_requested: 'auto' }, { backend: { backend: 'mps' } }), true);
  assert.equal(inputs.backend.value, 'auto');
});

test('before loading, polling preserves the user device choice and start sends it', () => {
  const inputs = controls();
  const unloaded = state({ backend_requested: 'auto' });
  syncRunControls(inputs, unloaded, true);
  inputs.backend.value = 'mps';
  inputs.learning.checked = true;
  inputs.speed.value = '16';
  syncRunControls(inputs, unloaded);
  assert.equal(inputs.backend.disabled, false);
  assert.deepEqual(controlSettings(inputs, unloaded), { learning: true, speed: 16, backend: 'mps' });
});

test('a later loaded status replaces an obsolete device selection', () => {
  const inputs = controls();
  syncRunControls(inputs, state({ backend_requested: 'auto' }), true);
  syncRunControls(inputs, state({}, {}));
  assert.equal(inputs.backend.value, 'mps');
  assert.equal(inputs.backend.disabled, true);
});

test('before the first status there is no implicit device change', () => {
  assert.equal('backend' in controlSettings(controls(), null), false);
});
