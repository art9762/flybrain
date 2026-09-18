import assert from 'node:assert/strict';
import { test } from 'node:test';
import { activityText, eventText, groupEvents } from '../web/room-events.mjs';

test('filtered feed groups old and new failures without modifying source events', () => {
  const events = [{ event: 'nothing_in_reach', station: null, time_s: 1, reward: 0 },
    { event: 'nothing_in_reach', station: null, time_s: 2, count: 3, last_time_s: 4, reward: -0.06 },
    { event: 'hookah_used', station: 'hookah', time_s: 6, reward: 5 },
    { event: 'hookah_used', station: 'hookah', time_s: 9, reward: 5 }];
  const grouped = groupEvents(events);
  assert.equal(grouped.length, 3);
  assert.equal(grouped[0].count, 4);
  assert.equal(grouped[0].last_time_s, 4);
  assert.equal(events[0].count, undefined);
});

test('repeated failures show latest time, count and cumulative penalty', () => {
  const text = eventText({ time_s: 2, last_time_s: 4, count: 3,
    event: 'nothing_in_reach', reward: -0.06 }, {}, { nothing_in_reach: 'Нет предмета рядом в поле зрения' });
  assert.equal(text, '4с · Нет предмета рядом в поле зрения · ×3 · -0.06');
});

test('success labels and reward survive event formatting', () => {
  assert.equal(eventText({ time_s: 8, event: 'drink_finished', station: 'whisky', reward: 2 },
    { whisky: 'Виски' }, { drink_finished: 'Сделан глоток' }), '8с · Сделан глоток · Виски · +2.00');
});

test('activity summary includes movement excluded from the event feed', () => {
  assert.equal(activityText({ window_steps: 20, movement_steps: 10, invalid_attempts: 8, completed_uses: 77 }),
    'Завершено занятий: 77. За последние 20 шагов: движений и поворотов 10, пустых попыток 8.');
});
