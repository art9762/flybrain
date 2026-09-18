const failedEvents = new Set(['nothing_in_reach', 'not_at_phone', 'not_at_roulette',
  'choose_red_or_black', 'insufficient_credits', 'cooldown', 'request_unavailable']);

export function groupEvents(events) {
  // Movement is summarized separately; do not let it split identical failures
  // in this filtered event feed. This also handles older server snapshots.
  return events.reduce((groups, event) => {
    const previous = groups.at(-1);
    if (failedEvents.has(event.event) && previous?.event === event.event
        && previous.station === event.station) {
      return [...groups.slice(0, -1), { ...previous,
        count: (previous.count || 1) + (event.count || 1),
        last_time_s: event.last_time_s ?? event.time_s,
        reward: (previous.reward || 0) + (event.reward || 0) }];
    }
    return [...groups, event];
  }, []);
}

export function eventText(event, labels, descriptions) {
  if (typeof event === 'string') return event;
  const end = event.last_time_s ?? event.time_s;
  const repeats = event.count > 1 ? ` · ×${event.count}` : '';
  const station = event.station ? ` · ${labels[event.station] || event.station}` : '';
  const reward = event.reward ? ` · ${event.reward > 0 ? '+' : ''}${event.reward.toFixed(2)}` : '';
  return `${Math.floor(end)}с · ${descriptions[event.event] || event.event}${station}${repeats}${reward}`;
}

export function activityText(activity) {
  if (!activity) return '';
  return `Завершено занятий: ${activity.completed_uses}. За последние ${activity.window_steps} шагов: `
    + `движений и поворотов ${activity.movement_steps}, пустых попыток ${activity.invalid_attempts}.`;
}
