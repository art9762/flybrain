// The server owns device selection once the brain is loading or loaded.
export function syncRunControls(controls, state, initialize = false) {
  const locked = Boolean(state.brain || state.run.loading);
  controls.backend.disabled = locked;
  if (initialize || locked) {
    controls.backend.value = state.run.backend_requested;
  }
  if (initialize) {
    controls.learning.checked = state.run.learning;
    controls.speed.value = String(state.run.speed);
  }
}

export function controlSettings(controls, state) {
  const settings = {
    learning: controls.learning.checked,
    speed: Number(controls.speed.value),
  };
  // Resume/pause/step must not request a device switch from a stale tab.
  if (state && !state.brain && !state.run.loading) {
    settings.backend = controls.backend.value;
  }
  return settings;
}
