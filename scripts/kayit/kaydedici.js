// Injected into every document by kaydedici.py. Reports human actions together with the
// Reflex snapshot taken immediately BEFORE the action, so demonstrations share the exact
// runtime input format of System 1 (same snapshot.js, same viewport-clipped element list).
(() => {
  // Top document only: iframes (ads, embeds, about:blank) inherit the parent's back/forward
  // navigation type and would report phantom history moves and overwrite the pre-state.
  if (window.__reflexKayitKurulu || window.top !== window) return;
  window.__reflexKayitKurulu = true;

  const SCROLL_STEP_PX = 420; // matches Browser.act() scroll delta
  const snapshot = () => (__REFLEX_SNAPSHOT__);
  const send = data => {
    try { window.__reflexKayit(JSON.stringify(data)); } catch (_) {}
  };
  const state = () => {
    try {
      const s = snapshot();
      if (s && typeof s.full_text === 'string') s.full_text = s.full_text.slice(0, 20000);
      return s;
    } catch (e) {
      return {snapshot_error: String(e)};
    }
  };
  window.__reflexKayitDurum = state;

  const isSecret = el => Boolean(el) && (el.type === 'password' ||
    /cc-|one-time-code|current-password|new-password/.test(el.getAttribute?.('autocomplete') || ''));
  const editableOf = t => t?.closest?.('input,textarea,[contenteditable="true"]') || null;
  const valueOf = el => isSecret(el) ? '<gizli>' :
    ('value' in el ? String(el.value) : (el.isContentEditable ? el.innerText.trim() : ''));
  const nodeOf = el => window.__jevFast?.ids?.get(el) ?? null;

  // Innermost observed action element that contains the event target (labels count for inputs).
  const matchAction = (s, target) => {
    const nodes = window.__jevFast?.nodes;
    if (!s || !nodes || !target) return null;
    let best = null;
    let bestArea = Infinity;
    for (const a of s.actions || []) {
      if (typeof a.node !== 'number') continue;
      const el = nodes.get(a.node);
      if (!el) continue;
      const hit = el === target || el.contains(target) ||
        (el.labels && [...el.labels].some(l => l.contains(target)));
      if (!hit) continue;
      const r = el.getBoundingClientRect();
      const area = r.width * r.height;
      if (area < bestArea) { best = a.node; bestArea = area; }
    }
    if (best === null) {
      // Click on the padding of a row that wraps exactly one listed control (e.g. a suggestion <li>).
      const inside = [...new Set((s.actions || []).filter(a => typeof a.node === 'number' &&
        target.contains(nodes.get(a.node) || null) && nodes.get(a.node) !== target).map(a => a.node))];
      if (inside.length !== 1) return null;
      best = inside[0];
    }
    const own = (s.actions || []).filter(a => a.node === best);
    const kinds = [...new Set(own.map(a => a.kind))];
    const label = String(own[0]?.label ?? '');
    return {node: best, kinds, label: kinds.includes('select') ? label.split(' → ')[0] : label};
  };

  let focused = null;

  // Browser back/forward/reload happen in the toolbar, outside the page; detect them from the page side.
  const reportHistory = kind => send({type: 'history', kind, url: location.href, state: state()});
  addEventListener('DOMContentLoaded', () => {
    const nav = performance.getEntriesByType('navigation')[0]?.type;
    if (nav === 'back_forward' || nav === 'reload') reportHistory(nav);
  });
  addEventListener('pageshow', e => { if (e.persisted) reportHistory('back_forward'); });
  // Snapshot every loaded page so a toolbar Back pressed right after arriving gets the true pre-state.
  addEventListener('load', () => setTimeout(() => send({type: 'page_state', state: state()}), 400));
  addEventListener('popstate', () => reportHistory('back_forward'));

  addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    const link = e.target?.closest?.('a[target]');
    if (link) link.removeAttribute('target'); // Reflex opens links in the same tab
    const s = state();
    const m = matchAction(s, e.target);
    const editable = editableOf(e.target);
    if (editable) focused = editable;
    // A click the snapshot does not list (custom div/span widgets). Kept as evidence of a coverage gap.
    let unmapped = null;
    if (!m) {
      let el = e.target;
      for (let i = 0; el && el !== document.body && i < 5; i++, el = el.parentElement) {
        const cur = getComputedStyle(el).cursor;
        // Native controls count even with a default cursor (e.g. a transparent radio laid over its row).
        const control = el.matches('input,button,select,textarea,a[href],label');
        if (cur === 'pointer' || control || el.hasAttribute('onclick') || el.hasAttribute('jsaction') ||
            el.getAttribute('tabindex') === '0') {
          const r = el.getBoundingClientRect();
          const text = (el.innerText || '').trim() || (el.parentElement?.innerText || '').trim();
          unmapped = {tag: el.tagName.toLowerCase(), type: el.getAttribute('type'), name: el.getAttribute('name'),
                      text: text.slice(0, 80),
                      cls: String(el.className || '').slice(0, 80), role: el.getAttribute('role'), cursor: cur,
                      rect: {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)}};
          break;
        }
      }
    }
    send({type: 'pointer', node: m?.node ?? null, kinds: m?.kinds ?? [], label: m?.label ?? null,
          value: editable ? valueOf(editable) : null, unmapped, state: s});
  }, true);

  // Typing into a field that was focused without a click (autofocus, Tab).
  addEventListener('keydown', e => {
    const el = editableOf(e.target);
    if (!el) return;
    if (el !== focused) {
      focused = el;
      const s = state();
      send({type: 'focus', node: nodeOf(el), value: valueOf(el), state: s});
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      send({type: 'enter', node: nodeOf(el), value: valueOf(el)});
    }
  }, true);

  let inputTimer = null;
  addEventListener('input', e => {
    const el = editableOf(e.target);
    if (!el) return;
    clearTimeout(inputTimer);
    inputTimer = setTimeout(() => send({type: 'input', node: nodeOf(el), value: valueOf(el)}), 200);
  }, true);

  addEventListener('change', e => {
    const el = e.target;
    if (el?.tagName !== 'SELECT') return;
    send({type: 'select', node: nodeOf(el), value: el.value, option: el.selectedOptions?.[0]?.label ?? ''});
  }, true);

  // Wheel bursts become SCROLL steps: one at the start of a burst, one more per SCROLL_STEP_PX.
  let burst = null;
  addEventListener('wheel', e => {
    if (Math.abs(e.deltaY) < 1) return;
    const dir = e.deltaY > 0 ? 'down' : 'up';
    if (!burst || burst.dir !== dir) {
      burst = {dir, y: scrollY, moved: false};
      send({type: 'scroll', dir, y: scrollY, state: state()});
    }
    clearTimeout(burst.timer);
    burst.timer = setTimeout(() => {
      send({type: 'scroll_end', y: scrollY, moved: burst.moved});
      burst = null;
    }, 450);
  }, {capture: true, passive: false}); // non-passive: compositor waits, so the snapshot is pre-scroll

  addEventListener('scroll', () => {
    if (!burst) return;
    burst.moved = true; // window or any inner scroller moved
    if (Math.abs(scrollY - burst.y) < SCROLL_STEP_PX) return;
    burst.y = scrollY;
    send({type: 'scroll', dir: burst.dir, y: scrollY, state: state()});
  }, {capture: true, passive: true});
})();
