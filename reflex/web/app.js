"use strict";

const TOKEN = window.__REFLEX_TOKEN__ || "";
let currentState = null;
let runningLoop = false;
let stepInFlight = false;
let showDomBoxes = true;
let activeView = "launch";
let streamConnected = false;

const PRESETS = {
  ozal: {
    url: "https://www.google.com",
    goal: "Search Google for Özal Yıldırım's personal website, navigate to his blog section, and extract all blog post titles he has published.",
  },
  gemini_chat: {
    url: "https://gemini.google.com/app",
    goal: "Have a 2-turn conversation with Gemini: First, ask a short riddle and wait for its answer; second, comment on its answer and ask a brief question about philosophy of mind, then summarize the reply.",
  },
  books: {
    url: "https://books.toscrape.com/",
    goal: "Navigate to the Mystery category, open the book 'Sharp Objects', and verify its exact price and stock availability.",
  },
  wiki: {
    url: "https://www.wikipedia.org/",
    goal: "Search Wikipedia for 'Hagia Sophia', open the article, and summarize its architectural history and dome dimensions.",
  },
  fixture_modal: {
    url: `${window.location.origin}/fixture.html`,
    goal: "Dismiss the promo modal if present, select Destination Tokyo and Stay 2026-05-14, and click Search Stays.",
  },
  fixture_ask: {
    url: `${window.location.origin}/fixture.html`,
    goal: "Enter my passenger email address into the Passenger Email field and click Search Stays.",
  },
  fixture_guard: {
    url: `${window.location.origin}/fixture.html`,
    goal: "Click the 'Pay Now $420.00' button to complete the reservation payment.",
  },
};

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function switchView(viewName) {
  activeView = viewName;
  document.querySelectorAll(".rail-btn[data-view]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === viewName);
  });
  document.querySelectorAll(".view-pane").forEach((pane) => {
    pane.classList.toggle("active", pane.id === `view-${viewName}`);
  });
  if (viewName === "live" && currentState) {
    requestAnimationFrame(() => renderDomOverlay(currentState));
  }
}

async function apiGet(path) {
  const res = await fetch(`/api/${path}`, {
    headers: { "X-Reflex-Token": TOKEN },
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function apiPost(path, body = {}) {
  const res = await fetch(`/api/${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Reflex-Token": TOKEN,
    },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function ensureLiveStream(state) {
  const img = $("viewport-img");
  const empty = $("viewport-empty");
  if (!img || !empty) return;

  if (!state || !state.page || state.status === "idle") {
    img.style.display = "none";
    empty.style.display = "flex";
    streamConnected = false;
    return;
  }

  empty.style.display = "none";
  img.style.display = "block";

  const isActiveSession = ["ready", "waiting_user", "waiting_confirmation"].includes(state.status) || runningLoop || stepInFlight;
  if (isActiveSession && !streamConnected) {
    streamConnected = true;
    img.src = `/api/stream?token=${encodeURIComponent(TOKEN)}&t=${Date.now()}`;
  } else if (!isActiveSession && state.page.screenshot && !streamConnected) {
    img.src = `data:image/jpeg;base64,${state.page.screenshot}`;
  }
}

function renderDomOverlay(state) {
  const overlay = $("dom-overlay");
  const img = $("viewport-img");
  const cursor = $("rx-cursor");
  const callout = $("rx-callout");
  if (!overlay || !img) return;
  overlay.innerHTML = "";

  if (!state || !state.page) {
    if (cursor) cursor.classList.remove("visible");
    return;
  }

  const pageW = state.page.w || 1120;
  const pageH = state.page.h || 780;
  const rect = img.getBoundingClientRect();
  const parentRect = img.parentElement.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const natW = img.naturalWidth || pageW;
  const natH = img.naturalHeight || pageH;
  const imgAspect = natW / natH;
  const boxAspect = rect.width / rect.height;

  let renderW = rect.width;
  let renderH = rect.height;
  let extraX = 0;
  let extraY = 0;
  if (boxAspect > imgAspect) {
    renderW = rect.height * imgAspect;
    extraX = (rect.width - renderW) / 2;
  } else {
    renderH = rect.width / imgAspect;
    extraY = (rect.height - renderH) / 2;
  }

  const offsetLeft = rect.left - parentRect.left + extraX;
  const offsetTop = rect.top - parentRect.top + extraY;
  overlay.style.left = `${offsetLeft}px`;
  overlay.style.top = `${offsetTop}px`;
  overlay.style.width = `${renderW}px`;
  overlay.style.height = `${renderH}px`;

  const scaleX = renderW / pageW;
  const scaleY = renderH / pageH;

  const lastStep = (state.history || []).slice(-1)[0];
  if (showDomBoxes) {
    (state.elements || []).forEach((el) => {
      if (!el.rect) return;
      const box = document.createElement("div");
      const isLastTarget = lastStep && String(lastStep.target) === String(el.index);
      box.className = "dom-box" + (isLastTarget ? (lastStep.actor === "S2" ? " active-s2" : " active-s1") : "");
      box.style.left = `${el.rect.x * scaleX}px`;
      box.style.top = `${el.rect.y * scaleY}px`;
      box.style.width = `${Math.max(8, el.rect.w * scaleX)}px`;
      box.style.height = `${Math.max(8, el.rect.h * scaleY)}px`;

      const tag = document.createElement("span");
      tag.className = "dom-tag";
      tag.textContent = el.index;
      box.appendChild(tag);
      overlay.appendChild(box);
    });
  }

  // Position gliding cursor & ripple at last_action_rect
  const actRect = (lastStep && lastStep.rect) || state.page.last_action_rect;
  if (cursor && actRect && typeof actRect.x === "number") {
    const cx = (actRect.cx ?? (actRect.x + (actRect.w || 24) / 2)) * scaleX + offsetLeft;
    const cy = (actRect.cy ?? (actRect.y + (actRect.h || 20) / 2)) * scaleY + offsetTop;
    cursor.style.left = `${Math.max(12, Math.min(parentRect.width - 28, cx))}px`;
    cursor.style.top = `${Math.max(12, Math.min(parentRect.height - 28, cy))}px`;
    const isS2 = lastStep && lastStep.actor === "S2";
    cursor.classList.toggle("s2", Boolean(isS2));
    cursor.classList.add("visible");
    if (callout && lastStep) {
      callout.textContent = stepVerb(lastStep.operation || lastStep.kind)[0];
    }
  } else if (cursor) {
    cursor.classList.remove("visible");
  }
}

// Minimal, safe Markdown for chat and answers: everything is HTML-escaped first, then a small set of
// constructs is turned into tags (headings, lists, tables, code, quotes, bold/italic, http(s) links).
function mdInline(raw) {
  const codes = [];
  let s = escapeHtml(raw).replace(/`([^`\n]+)`/g, (_, c) => {
    codes.push(c);
    return `\u0000${codes.length - 1}\u0000`;
  });
  const link = (url, text) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>`;
  s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, text, url) => link(url, text));
  s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?])/g, (_, pre, url) => pre + link(url, url));
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>").replace(/__([^_\n]+)__/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>");
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
}

// One-line plain text for previews: Markdown markers and table pipes removed.
function plainPreview(text) {
  return String(text ?? "")
    .replace(/^\s*(#{1,4}|>|[-*•+]|\d+[.)])\s+/gm, "")
    .replace(/^\s*\|?\s*:?-{2,}.*$/gm, "")
    .replace(/[*_`|]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 220);
}

function renderMarkdown(text) {
  const lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let para = [];
  const flush = () => {
    if (para.length) out.push(`<p>${para.map(mdInline).join("<br>")}</p>`);
    para = [];
  };
  const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const isSep = (line) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line);

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      flush();
      const body = [];
      for (i += 1; i < lines.length && !/^\s*```/.test(lines[i]); i += 1) body.push(lines[i]);
      out.push(`<pre><code>${escapeHtml(body.join("\n"))}</code></pre>`);
      continue;
    }
    if (!line.trim()) {
      flush();
      continue;
    }
    const h = line.match(/^\s*(#{1,4})\s+(.*)$/);
    if (h) {
      flush();
      out.push(`<h${h[1].length}>${mdInline(h[2])}</h${h[1].length}>`);
      continue;
    }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(line)) {
      flush();
      out.push("<hr>");
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && isSep(lines[i + 1])) {
      flush();
      const head = cells(line);
      const rows = [];
      for (i += 2; i < lines.length && lines[i].includes("|") && lines[i].trim(); i += 1) rows.push(cells(lines[i]));
      i -= 1;
      out.push(
        `<div class="md-table"><table><thead><tr>${head.map((c) => `<th>${mdInline(c)}</th>`).join("")}</tr></thead>` +
          `<tbody>${rows.map((r) => `<tr>${head.map((_, k) => `<td>${mdInline(r[k] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`
      );
      continue;
    }
    const listRe = /^\s*([-*•+]|\d+[.)])\s+(.*)$/;
    if (listRe.test(line)) {
      flush();
      const ordered = /^\s*\d/.test(line);
      const items = [];
      for (; i < lines.length && listRe.test(lines[i]) && /^\s*\d/.test(lines[i]) === ordered; i += 1) {
        items.push(`<li>${mdInline(lines[i].match(listRe)[2])}</li>`);
      }
      i -= 1;
      out.push(ordered ? `<ol>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`);
      continue;
    }
    if (/^\s*>\s?/.test(line)) {
      flush();
      const quote = [];
      for (; i < lines.length && /^\s*>\s?/.test(lines[i]); i += 1) quote.push(mdInline(lines[i].replace(/^\s*>\s?/, "")));
      i -= 1;
      out.push(`<blockquote>${quote.join("<br>")}</blockquote>`);
      continue;
    }
    para.push(line);
  }
  flush();
  return out.join("");
}

// Plain-language labels for browser operations shown in the step deck.
const STEP_VERBS = {
  CLICK: ["Clicked", "↖"],
  TYPE: ["Typed", "⌨"],
  TYPE_TEXT: ["Typed", "⌨"],
  FILL: ["Typed", "⌨"],
  SELECT: ["Selected", "☰"],
  SELECT_OPTION: ["Selected", "☰"],
  SCROLL_DOWN: ["Scrolled down", "↓"],
  SCROLL_UP: ["Scrolled up", "↑"],
  GOTO_URL: ["Opened", "↗"],
  PRESS_KEY: ["Pressed", "⏎"],
  PRESS_ENTER: ["Pressed Enter", "⏎"],
  GO_BACK: ["Went back", "←"],
  HOVER: ["Hovered", "◎"],
  OBSERVE: ["Looked at the page", "◉"],
  COMPLETE_TASK: ["Finished", "✓"],
};
let lastDeckStep = 0;

function stepVerb(op) {
  const key = String(op || "").toUpperCase();
  if (STEP_VERBS[key]) return STEP_VERBS[key];
  const words = key.toLowerCase().replace(/_/g, " ");
  return [words.charAt(0).toUpperCase() + words.slice(1) || "Step", "•"];
}

// The supervisor's own words for a step: its reasoning, else its next goal; internal notes are skipped.
function stepReason(step, escalations) {
  const esc = escalations.find((e) => e.step === step.step);
  const text = esc?.reasoning || esc?.next_goal || "";
  if (text) return text;
  const note = step.escalation_note || "";
  return /\s/.test(note) ? note : "";
}

function renderDecisionStack(state) {
  const container = $("decision-stack");
  if (!container) return;
  const history = state.history || [];
  const escalations = state.escalations || [];
  if ($("dstack-count")) $("dstack-count").textContent = String(history.length);

  if (history.length === 0) {
    lastDeckStep = 0;
    container.innerHTML = `<div class="lb-muted" style="padding:4px 4px 0">Each step appears here: what Reflex did, and why.</div>`;
    return;
  }

  const last = history[history.length - 1];
  const [verb, icon] = stepVerb(last.operation);
  const v = last.verification || {};
  const why = stepReason(last, escalations);
  const fresh = last.step !== lastDeckStep ? " fresh" : "";
  lastDeckStep = last.step;
  const changed = v.url_changed ? "New page ✓" : v.page_changed || v.element_state_changed ? "Page changed ✓" : "No visible change";
  const loop = v.oscillation_detected || v.loop_detected;

  let html = `
    <div class="st-now${fresh}">
      <div class="st-top">
        <div class="st-icon">${icon}</div>
        <div class="st-title">
          <div class="st-verb">${escapeHtml(verb)}</div>
          <div class="st-target">${escapeHtml(last.text ? `"${last.text}"` : last.target || last.action || "")}</div>
        </div>
        <span class="st-num">#${last.step}</span>
      </div>
      ${why ? `<div class="st-why"><b>Why:</b> ${escapeHtml(why)}</div>` : ""}
      <div class="st-chips">
        <span class="st-chip ${v.effective_progress ? "ok" : ""}">${changed}</span>
        ${loop ? `<span class="st-chip warn">${v.oscillation_detected ? "Going back and forth" : "Repeating itself"}</span>` : ""}
        <span class="st-chip">${((last.step_ms ?? last.latency_ms ?? 0) / 1000).toFixed(1)} s</span>
      </div>
    </div>`;

  history
    .slice(Math.max(0, history.length - 4), history.length - 1)
    .reverse()
    .forEach((ps, i) => {
      const [pVerb] = stepVerb(ps.operation);
      html += `<div class="st-prev" style="--i:${i}"><b>#${ps.step} ${escapeHtml(pVerb)}</b><span>${escapeHtml(ps.text ? `"${ps.text}"` : ps.target || ps.action || "")}</span></div>`;
    });
  container.innerHTML = html;
}

function finalAnswer(state) {
  if (state.status !== "done") return "";
  const log = state.context?.chat_log || [];
  const msg = [...log].reverse().find((m) => m.role === "assistant" && (m.kind || "chat") === "chat");
  return msg?.content || "";
}

function renderPlanCard(state) {
  const body = $("plan-body");
  const head = document.querySelector("#plan-card .lb-label");
  if (!body) return;

  const answer = finalAnswer(state);
  const badgeSlot = $("answer-check-badge");
  if (badgeSlot) badgeSlot.innerHTML = "";
  // In conversation mode the answer is already in the chat, so the card keeps the plan and lists sources.
  const chatMode = $("live-bento")?.dataset.mode === "chat";
  if (answer && chatMode) {
    const checks = state.answer_checks || [];
    const failed = checks.find((c) => c.verdict === "fail");
    if (badgeSlot && checks.length) {
      badgeSlot.innerHTML = failed
        ? `<span class="ans-check sent-back" title="${escapeHtml(failed.unmet.join("; "))}">↺ Muse Spark Open sent it back once</span>`
        : `<span class="ans-check">✓ Checked by Muse Spark Open</span>`;
    }
  }
  if (answer && !chatMode) {
    if (head) head.textContent = "Answer";
    const checks = state.answer_checks || [];
    const failed = checks.find((c) => c.verdict === "fail");
    if (badgeSlot && checks.length) {
      badgeSlot.innerHTML = failed
        ? `<span class="ans-check sent-back" title="${escapeHtml(failed.unmet.join("; "))}">↺ Muse Spark Open sent it back once</span>`
        : `<span class="ans-check">✓ Checked by Muse Spark Open</span>`;
    }
    const why = failed ? `<div class="lb-muted" style="margin-top:8px">Sent back because: ${escapeHtml(failed.unmet.join("; "))}</div>` : "";
    body.innerHTML = `<div class="ans-fan"><div class="ans-card md">${renderMarkdown(answer)}</div></div>${why}`;
    return;
  }

  if (head) head.textContent = "Plan";
  const subgoals = state.context?.plan?.subgoals || [];
  const visited = [...new Set((state.context?.visited_urls || []).filter((u) => /^https?:\/\//.test(u)))].slice(-6).reverse();
  const sources =
    chatMode && visited.length
      ? `<div class="pl-sources"><div class="lb-label" style="margin-bottom:4px">Pages visited</div>${visited
          .map((u) => `<a href="${escapeHtml(u)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(u)}">${escapeHtml(u.replace(/^https?:\/\/(www\.)?/, ""))}</a>`)
          .join("")}</div>`
      : "";
  if (subgoals.length) {
    const mark = { completed: "✓", failed: "✕" };
    body.innerHTML =
      subgoals
        .map((sg) => `<div class="pl-item ${escapeHtml(sg.status)}"><span class="pl-dot">${mark[sg.status] || ""}</span><span>${escapeHtml(sg.text)}</span></div>`)
        .join("") + sources;
    return;
  }
  const active = state.context?.active_subgoal;
  body.innerHTML =
    (active && state.history?.length
      ? `<div class="pl-item active"><span class="pl-dot"></span><span>${escapeHtml(active)}</span></div>`
      : `<div class="lb-muted">Muse Spark writes a short plan when the task starts.</div>`) + sources;
}

function formatElapsed(ms) {
  const s = Math.max(0, Math.round((ms || 0) / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// While Muse Spark decides the next step: a live "thinking" card in the step deck and a scan line over the browser.
let thinkingTimer = null;
function setThinking(on) {
  const steps = document.querySelector(".lb-steps");
  const wrap = document.querySelector(".lb-stage .viewport-wrap");
  const label = $("st-thinking-time");
  clearInterval(thinkingTimer);
  steps?.classList.toggle("thinking", on);
  wrap?.classList.toggle("thinking", on);
  if (!on || !label) return;
  const t0 = performance.now();
  label.textContent = "0.0 s";
  thinkingTimer = setInterval(() => {
    label.textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`;
  }, 100);
}

// Ambient fill for the letterbox around the live frame: a tiny, blurred copy of the current frame.
function initViewportAmbient() {
  const img = $("viewport-img");
  const canvas = $("viewport-ambient");
  if (!img || !canvas) return;
  const ctx = canvas.getContext("2d");
  canvas.width = 48;
  canvas.height = 30;
  setInterval(() => {
    if (img.style.display === "none" || !img.naturalWidth || document.hidden) return;
    try {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.classList.add("on");
    } catch (_) {
      canvas.classList.remove("on");
    }
  }, 500);
}

// Site rules: grouped by website, the user's own rules first, each with a delete button.
function renderSiteRules(mem) {
  const box = $("memory-rules-box");
  const domains = Object.keys(mem).sort();
  const total = domains.reduce((n, d) => n + mem[d].length, 0);
  if ($("memory-count")) $("memory-count").textContent = String(total);
  if (!total) {
    box.innerHTML = `<div class="lb-muted">No rules yet. Add one above, or turn on site memory in Settings to let Reflex record what it learns.</div>`;
    return;
  }
  box.innerHTML = domains
    .map((dom) => {
      const rules = [...mem[dom]].sort((a, b) => (a.source === "user" ? 0 : 1) - (b.source === "user" ? 0 : 1) || a.created_at - b.created_at);
      return `<div class="mem-site"><div class="mem-domain">${escapeHtml(dom)}</div>${rules
        .map(
          (r) => `<div class="mem-rule">
            <span class="mem-src ${r.source === "user" ? "user" : ""}">${r.source === "user" ? "Yours" : "Learned"}</span>
            <span class="mem-text">${escapeHtml(r.rule)}</span>
            <button class="lb-icon-btn small" data-del-domain="${escapeHtml(dom)}" data-del-created="${r.created_at}" title="Delete rule" aria-label="Delete rule">✕</button>
          </div>`
        )
        .join("")}</div>`;
    })
    .join("");
}

function initSiteRules() {
  const form = $("site-rule-form");
  const target = $("site-rule-target");
  const msg = $("site-rule-msg");
  if (!form) return;
  const say = (text, ok) => {
    if (!msg) return;
    msg.textContent = text;
    msg.classList.toggle("ok", Boolean(ok));
  };
  target?.addEventListener("focus", () => {
    const url = currentState?.page?.url;
    if (!target.value && url && /^https?:/.test(url)) target.value = new URL(url).hostname.replace(/^www\./, "");
  });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const s = await apiPost("site_rules", { action: "add", target: target.value, rule: $("site-rule-text").value });
      $("site-rule-text").value = "";
      say("Rule saved. Reflex will use it on this website.", true);
      renderState(s);
    } catch (err) {
      say(err.message.replace(/^Error:\s*/, ""), false);
    }
  });
  $("memory-rules-box")?.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-del-domain]");
    if (!btn) return;
    try {
      renderState(await apiPost("site_rules", { action: "delete", domain: btn.dataset.delDomain, created_at: Number(btn.dataset.delCreated) }));
    } catch (err) {
      say(err.message, false);
    }
  });
}

// Download the current session (task, steps with reasons, plan, answer check, chat, cost) as JSON; no screenshots.
function exportSessionJson() {
  const st = currentState;
  if (!st || !(st.history || []).length) {
    alert("There are no steps to download yet.");
    return;
  }
  const strip = ({ screenshot, ...rest }) => rest;
  const data = {
    exported_at: new Date().toISOString(),
    task: st.goal,
    status: st.status,
    final_url: st.page?.url || null,
    models: { supervisor: st.config?.supervisor_model, mode: st.config?.mode },
    metrics: st.metrics,
    plan: st.context?.plan || null,
    steps: (st.history || []).map(strip),
    supervisor_decisions: (st.escalations || []).map(({ usage, ...rest }) => rest),
    answer_checks: st.answer_checks || [],
    chat: (st.context?.chat_log || []).filter((m) => m.kind !== "escalation_note"),
    pages_visited: st.context?.visited_urls || [],
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `reflex-session-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// Short in-app explanations behind the "?" buttons.
const INFO_TEXT = {
  models: `<b>How Reflex decides</b>
    <p><span class="info-tag s2">S2 · Supervisor</span> Meta <b>Muse Spark 1.3</b> reads the page as a numbered list of elements and decides every step, with a short reason. This is the default mode, <b>S2 Only</b>.</p>
    <p><span class="info-tag s1">S1 · Fast reflex</span> An optional small model (Jev in the cloud or Turnike on your machine) that takes confident clicks and typing so the supervisor is called less often. Experimental; not used in S2 Only.</p>
    <p><span class="info-tag ok">Answer check</span> Meta <b>Muse Spark Open</b> checks the final answer against every requirement of your task. If something is missing, Reflex goes back once.</p>`,
};

function initInfoPopovers() {
  const pop = $("info-pop");
  if (!pop) return;
  const close = () => {
    pop.classList.remove("open");
    pop.setAttribute("aria-hidden", "true");
  };
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".info-btn");
    if (!btn) {
      if (!e.target.closest("#info-pop")) close();
      return;
    }
    e.preventDefault();
    if (pop.classList.contains("open") && pop.dataset.for === btn.dataset.info) return close();
    pop.innerHTML = INFO_TEXT[btn.dataset.info] || "";
    pop.dataset.for = btn.dataset.info;
    pop.classList.add("open");
    pop.setAttribute("aria-hidden", "false");
    const r = btn.getBoundingClientRect();
    const w = Math.min(380, window.innerWidth - 24);
    pop.style.width = `${w}px`;
    pop.style.left = `${Math.max(12, Math.min(window.innerWidth - w - 12, r.left + r.width / 2 - w / 2))}px`;
    const below = r.bottom + 10;
    pop.style.top = `${below + pop.offsetHeight > window.innerHeight - 12 ? Math.max(12, r.top - pop.offsetHeight - 10) : below}px`;
  });
  document.addEventListener("keydown", (e) => e.key === "Escape" && close());
}

function initLiveLayout() {
  const bento = $("live-bento");
  if (!bento) return;
  const layout = localStorage.getItem("reflex_live_layout");
  const mode = localStorage.getItem("reflex_live_mode");
  if (layout === "browser") bento.dataset.layout = "browser";
  if (mode === "chat") bento.dataset.mode = "chat";

  const setMode = (next) => {
    bento.dataset.mode = next;
    localStorage.setItem("reflex_live_mode", next);
    if (currentState) renderPlanCard(currentState);
    const knob = $("btn-live-mode");
    if (knob) knob.title = next === "chat" ? "Show the browser again" : "Hide the browser and open the conversation";
    if (next === "chat") {
      const box = $("chat-messages");
      if (box) box.scrollTop = box.scrollHeight;
      $("input-chat")?.focus();
    }
  };
  $("btn-live-mode")?.addEventListener("click", () => setMode(bento.dataset.mode === "chat" ? "browser" : "chat"));
  $("chat-peek")?.addEventListener("click", () => setMode("chat"));
  $("btn-browser-size")?.addEventListener("click", () => {
    bento.dataset.layout = bento.dataset.layout === "browser" ? "normal" : "browser";
    localStorage.setItem("reflex_live_layout", bento.dataset.layout);
    $("btn-browser-size").title = bento.dataset.layout === "browser" ? "Shrink browser" : "Enlarge browser";
  });
}

function renderScrubberAndActionSpace(state) {
  const track = $("scrubber-track");
  const lbl = $("scrub-label");
  const aspace = $("aspace-row");
  const history = state.history || [];

  if (lbl) lbl.textContent = `STEP ${history.length} / ${state.config?.max_steps || 60}`;
  if (track) {
    track.innerHTML = history
      .map((h) => {
        const cls = h.verification?.loop_detected || h.verification?.oscillation_detected ? "warn" : h.actor === "S2" ? "s2" : "s1";
        const hDur = h.step_ms ?? h.latency_ms ?? 0;
        return `<div class="scrub-tick ${cls}" title="#${h.step} [${h.actor}] ${escapeHtml(h.operation)}: ${escapeHtml(h.action)} (${hDur}ms)"></div>`;
      })
      .join("");
  }

  if (aspace) {
    const decProbs = (state.decisions || []).slice(-1)[0];
    if (decProbs) {
      aspace.innerHTML = `
        <span class="aspace-chip top"><span>S1 CHOICE: ${escapeHtml(decProbs.operation)} ${decProbs.target ? `[${escapeHtml(decProbs.target)}]` : ""}</span><span class="p">${Math.round((decProbs.confidence || 0) * 100)}%</span></span>
        <span class="aspace-chip"><span>CONF</span><span class="p">${Math.round((decProbs.joint_confidence || 0) * 100)}%</span></span>
        <span class="aspace-chip"><span>GATE</span><span class="p">${decProbs.gate?.should_escalate ? `ESCALATED (${escapeHtml(decProbs.gate.trigger_type)})` : "FAST PATH ✓"}</span></span>
        <span class="aspace-chip"><span>MODEL</span><span class="p">${escapeHtml(decProbs.model || "")}</span></span>
      `;
    }
  }
}

// Messages typed by the user show up at once; the server copy replaces them when it arrives.
let pendingChat = [];
let chatWaiting = false;
const AVATAR_REFLEX = `<svg viewBox="0 0 36 22" fill="none" aria-hidden="true"><path d="M2 11h7l3.5-8 5.5 16 4.2-11 2.8 3H34" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const AVATAR_USER = `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><circle cx="10" cy="7" r="3.4" stroke="currentColor" stroke-width="1.7"/><path d="M3.8 17c.8-3.2 3.3-5 6.2-5s5.4 1.8 6.2 5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>`;

function chatRow(m) {
  const user = m.role === "user";
  const cls = m.kind === "ask_user" ? "ask_user" : m.kind === "confirm_action" ? "confirm_action" : user ? "user" : "assistant";
  const roleLbl = user ? "You" : m.kind === "ask_user" ? "Reflex asks" : m.kind === "confirm_action" ? "Safety guard" : "Reflex";
  const body = user ? escapeHtml(m.content) : `<div class="md">${renderMarkdown(m.content)}</div>`;
  return `<div class="crow ${user ? "me" : "them"}${m.pending ? " pending" : ""}">
      <div class="cavatar ${user ? "user" : "reflex"}">${user ? AVATAR_USER : AVATAR_REFLEX}</div>
      <div class="cmsg ${cls}"><div class="cmsg-role">${roleLbl}</div>${body}</div>
    </div>`;
}

function renderChatLog(chatBox, log) {
  const serverUserTexts = log.filter((m) => m.role === "user").map((m) => m.content);
  pendingChat = pendingChat.filter((p) => !serverUserTexts.includes(p.content));
  const rows = [...log, ...pendingChat];
  const atBottom = chatBox.scrollHeight - chatBox.scrollTop - chatBox.clientHeight < 60;
  let html = rows.length
    ? rows.map(chatRow).join("")
    : `<div class="crow them"><div class="cavatar reflex">${AVATAR_REFLEX}</div><div class="cmsg assistant"><div class="cmsg-role">Reflex</div>Ask questions while the task runs, steer the goal, or answer when Reflex asks you something.</div></div>`;
  if (chatWaiting || runningLoop) {
    html += `<div class="crow them typing"><div class="cavatar reflex">${AVATAR_REFLEX}</div><div class="cmsg assistant"><span class="dots"><i></i><i></i><i></i></span>${runningLoop ? " Working on the task…" : ""}</div></div>`;
  }
  chatBox.innerHTML = html;
  if (atBottom || pendingChat.length) chatBox.scrollTop = chatBox.scrollHeight;
}

function renderGuardAndChat(state) {
  const modal = $("guard-modal");
  const railSafety = $("rail-safety");
  if (modal) {
    if (state.status === "waiting_confirmation" && state.pending_confirmation) {
      modal.classList.add("visible");
      modal.classList.remove("ask-mode");
      $("guard-modal-tag").textContent = "🛡️ SAFETY GUARD HOLD · IRREVERSIBLE ACTION";
      $("guard-modal-summary").textContent = state.pending_confirmation.summary || "Confirm this action?";
      $("guard-modal-reason").textContent = state.pending_confirmation.reason || "";
      if (railSafety) railSafety.classList.add("has-alert");
    } else {
      modal.classList.remove("visible");
      if (railSafety) railSafety.classList.remove("has-alert");
    }
  }

  const chatBox = $("chat-messages");
  const log = (state.context?.chat_log || []).filter((m) => m.kind !== "escalation_note");
  if (chatBox) renderChatLog(chatBox, log);

  // One-line preview of the latest message from Reflex, shown under the browser.
  const peek = $("chat-peek");
  if (peek) {
    const last = [...log].reverse().find((m) => m.role !== "user" && (m.kind || "chat") !== "escalation_note");
    const asks = last?.kind === "ask_user";
    peek.classList.toggle("ask", Boolean(asks));
    peek.innerHTML = last ? `<b>${asks ? "Reflex asks:" : "Reflex:"}</b> ${escapeHtml(plainPreview(last.content))}` : "";
  }
}

function renderSecondaryViews(state) {
  // Trajectory view
  const trajList = $("trajectory-full-list");
  const trajLbl = $("traj-summary-lbl");
  const history = state.history || [];
  if (trajLbl) trajLbl.textContent = `${history.length} steps completed`;
  if (trajList) {
    if (history.length === 0) {
      trajList.innerHTML = `<div class="pbox">No steps executed yet.</div>`;
    } else {
      trajList.innerHTML = history
        .slice()
        .reverse()
        .map((h) => {
          const isS2 = h.actor === "S2";
          const hDur = h.step_ms ?? h.latency_ms ?? 0;
          return `
            <div class="dcard ${isS2 ? "s2" : "s1"}">
              <div class="dcard-top">
                <span class="actor-badge ${isS2 ? "s2" : "s1"}">#${h.step} ${h.actor} · ${escapeHtml(h.operation)}</span>
                <span style="font-family:var(--mono);font-size:12px;color:var(--ink-3)">${hDur} ms (S1:${h.s1_ms ?? 0} S2:${h.s2_ms ?? 0} Act:${h.act_ms ?? 0} Obs:${h.observe_ms ?? 0}) · ${escapeHtml(h.url || "")}</span>
              </div>
              <div class="dcard-act">${h.target ? `[${escapeHtml(h.target)}] ` : ""}${escapeHtml(h.action)}</div>
              ${h.text ? `<div class="dcard-reason">Typed: "${escapeHtml(h.text)}"</div>` : ""}
              ${h.escalation_note ? `<div class="dcard-reason">${escapeHtml(h.escalation_note)}</div>` : ""}
            </div>`;
        })
        .join("");
    }
  }

  // Context view
  if ($("ctx-compiled-goal")) $("ctx-compiled-goal").textContent = state.compiled_goal || state.goal || "-";
  if ($("ctx-subgoal-hint")) {
    $("ctx-subgoal-hint").textContent = `Active Subgoal: ${state.context?.active_subgoal || "-"}\nTactical Hint: ${state.context?.tactical_hint || "(none)"}`;
  }
  if ($("ctx-working-memory")) {
    $("ctx-working-memory").textContent = state.context?.working_memory || "(No System 2 working memory recorded yet)";
  }
  if ($("ctx-rejected-links")) {
    const rej = state.context?.rejected_links || [];
    const vis = state.context?.visited_urls || [];
    $("ctx-rejected-links").textContent = `Rejected Links (${rej.length}):\n${rej.join("\n") || "(none)"}\n\nVisited URLs (${vis.length}):\n${vis.join("\n") || "(none)"}`;
  }
  if ($("ctx-facts-block")) {
    $("ctx-facts-block").textContent = JSON.stringify(
      {
        user_facts: state.context?.user_facts || {},
        verified_dom_facts: state.context?.verified_dom_facts || [],
      },
      null,
      2
    );
  }
  if ($("ctx-elem-count")) $("ctx-elem-count").textContent = String((state.elements || []).length);
  if ($("ctx-elements-block")) {
    $("ctx-elements-block").textContent = (state.elements || [])
      .map((e) => `[${e.index}] (${e.role}) ${e.label}${e.href ? ` -> ${e.href}` : ""} | ops: ${(e.operations || []).join(",")}`)
      .join("\n") || "(none)";
  }

  // Safety view
  if ($("safety-active-box")) {
    $("safety-active-box").textContent = state.pending_confirmation
      ? JSON.stringify(state.pending_confirmation, null, 2)
      : "No pending destructive action hold.";
  }
  if ($("safety-approved-box")) {
    const app = state.context?.approved_actions || [];
    $("safety-approved-box").textContent = app.length ? app.join("\n") : "None";
  }

  // Site memory view
  if ($("memory-rules-box")) renderSiteRules(state.site_memory || {});

  // Keys status
  if (state.keys) {
    if ($("meta-key-badge")) {
      $("meta-key-badge").textContent = state.keys.meta_api_configured
        ? `ACTIVE (${state.keys.meta_api_masked})`
        : "NOT SET (OpenRouter Fallback)";
    }
    if ($("or-key-badge")) {
      $("or-key-badge").textContent = state.keys.openrouter_configured
        ? `ACTIVE (${state.keys.openrouter_masked})`
        : "NOT SET";
    }
  }
}

const STATUS_LABELS = {
  idle: "IDLE",
  ready: "READY",
  done: "DONE",
  waiting_user: "AWAITING USER",
  waiting_confirmation: "CONFIRM HOLD",
  blocked: "PAUSED",
};

function renderState(state) {
  currentState = state;
  if (!state) return;

  const st = (state.status || "idle").toLowerCase();
  const badge = $("status-badge");
  const stTxt = $("status-text");
  if (badge && stTxt) {
    if (st === "error") {
      stTxt.textContent = "ERROR: " + (state.last_error || "Session stopped");
      badge.className = "status-pill error";
    } else {
      stTxt.textContent = runningLoop ? "RUNNING" : (STATUS_LABELS[st] || st.toUpperCase());
      badge.className =
        "status-pill " +
        (runningLoop || st === "ready"
          ? "running"
          : st === "done"
          ? "done"
          : st.startsWith("waiting") || st === "blocked"
          ? "waiting"
          : "");
    }
  }

  if ($("top-goal-crumb") && state.goal) {
    $("top-goal-crumb").textContent = state.goal;
  }

  const m = state.metrics || {};
  if ($("m-steps")) $("m-steps").textContent = String(m.steps ?? 0);
  if ($("m-elapsed")) $("m-elapsed").textContent = formatElapsed(m.elapsed_ms);
  if ($("m-s1")) $("m-s1").textContent = String(m.s1_steps ?? 0);
  if ($("m-s2")) $("m-s2").textContent = String(m.s2_escalations ?? 0);
  if ($("m-lat")) $("m-lat").textContent = `${m.median_step_ms ?? m.median_decision_ms ?? 0} ms`;
  if ($("m-cost")) {
    $("m-cost").textContent =
      m.total_cost_usd === "unknown"
        ? "unknown"
        : `$${Number(m.total_cost_usd ?? 0).toFixed(4)}`;
  }

  if (state.config) {
    if ($("top-mode-select") && state.config.mode) {
      $("top-mode-select").value = state.config.mode;
    }
    if ($("top-s1-select")) {
      $("top-s1-select").disabled = state.config.mode === "s2_only";
    }
    if ($("top-s1-select") && state.config.decision_backend) {
      $("top-s1-select").value = state.config.decision_backend;
    }
    if ($("top-s2-select") && state.config.supervisor_model) {
      $("top-s2-select").value = state.config.supervisor_model;
    }
    syncCustomSelects();
  }

  if ($("page-url-badge")) {
    $("page-url-badge").textContent = state.page?.url || "about:blank";
  }
  if ($("dom-count")) {
    $("dom-count").textContent = String((state.elements || []).length);
  }

  if ($("btn-pause")) $("btn-pause").disabled = !runningLoop;
  if ($("btn-continue")) $("btn-continue").disabled = runningLoop || !state.page;
  if ($("btn-step")) $("btn-step").disabled = runningLoop || stepInFlight || !state.page;

  ensureLiveStream(state);
  renderDomOverlay(state);
  renderDecisionStack(state);
  renderPlanCard(state);
  renderScrubberAndActionSpace(state);
  renderGuardAndChat(state);
  renderSecondaryViews(state);
}

async function runContinuousLoop() {
  if (runningLoop) return;
  runningLoop = true;
  renderState(currentState);

  try {
    while (runningLoop && currentState && currentState.status === "ready") {
      stepInFlight = true;
      setThinking(true);
      const next = await apiPost("step");
      stepInFlight = false;
      setThinking(false);
      renderState(next);
      if (next.status !== "ready") {
        break;
      }
      await new Promise((r) => setTimeout(r, 60));
    }
  } catch (err) {
    stepInFlight = false;
    setThinking(false);
    console.error("Step error:", err);
    const errText = err.message || String(err);
    if (currentState) {
      currentState.status = "error";
      currentState.last_error = errText;
    }
    const badge = $("status-badge");
    const stTxt = $("status-text");
    if (badge && stTxt) {
      stTxt.textContent = "ERROR: " + errText;
      badge.className = "status-pill error";
    }
    const chatBox = $("chat-messages");
    if (chatBox) {
      const errMsg = `<div class="cmsg error" style="border-left:3px solid var(--guard,#ff6e6e);background:rgba(255,110,110,0.1);"><div class="cmsg-role" style="color:var(--guard,#ff6e6e)">⚠️ RUNTIME ERROR</div>${escapeHtml(errText)}</div>`;
      chatBox.innerHTML += errMsg;
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  } finally {
    runningLoop = false;
    stepInFlight = false;
    if (currentState) renderState(currentState);
  }
}

async function startSessionAndRun() {
  runningLoop = false;
  const url = $("input-url").value.trim() || "https://www.google.com";
  const goal = $("input-goal").value.trim();
  if (!goal) return;

  switchView("live");
  streamConnected = false;

  try {
    const state = await apiPost("start", {
      url,
      goal,
      mode: $("top-mode-select")?.value || "s2_only",
      decision_backend: $("top-s1-select").value,
      supervisor_model: $("top-s2-select").value,
      require_confirmation_for_destructive: $("chk-safety").checked,
      verify_done_with_supervisor: $("chk-verify-done").checked,
      enable_site_memory: $("chk-memory").checked,
      max_steps: parseInt($("input-max-steps").value || "60", 10),
    });
    renderState(state);
    await runContinuousLoop();
  } catch (err) {
    alert(`Failed to start session: ${err.message}`);
  }
}

function initThemeToggle() {
  const root = document.documentElement;
  const btn = $("btn-theme-toggle");
  const saved = localStorage.getItem("reflex_theme");
  const initialTheme = saved === "dark" ? "dark" : "light";
  root.dataset.theme = initialTheme;

  const syncThemeIcons = () => {
    const isDark = root.dataset.theme === "dark";
    const sun = btn?.querySelector(".icon-sun");
    const moon = btn?.querySelector(".icon-moon");
    if (sun && moon) {
      sun.style.display = isDark ? "block" : "none";
      moon.style.display = isDark ? "none" : "block";
    }
  };

  syncThemeIcons();

  if (btn) {
    btn.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      localStorage.setItem("reflex_theme", next);
      syncThemeIcons();
    });
  }
}

function initSilkCanvas() {
  const canvas = $("silk-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let width = 0;
  let height = 0;
  let dpr = 1;

  const resize = () => {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };

  resize();
  window.addEventListener("resize", resize);

  // Soft volumetric silk ribbons flowing diagonally like smooth white porcelain/silk folds
  const ribbons = [
    { yStart: 0.20, yEnd: 0.04, cp1y: 0.34, cp2y: 0.02, amp: 24, speed: 0.00055, phase: 0.0 },
    { yStart: 0.44, yEnd: 0.18, cp1y: 0.56, cp2y: 0.16, amp: 32, speed: 0.00048, phase: 1.4 },
    { yStart: 0.66, yEnd: 0.38, cp1y: 0.48, cp2y: 0.54, amp: 28, speed: 0.00062, phase: 2.7 },
    { yStart: 0.86, yEnd: 0.58, cp1y: 0.72, cp2y: 0.76, amp: 30, speed: 0.00042, phase: 4.1 },
    { yStart: 1.06, yEnd: 0.78, cp1y: 0.88, cp2y: 0.96, amp: 22, speed: 0.00052, phase: 5.3 },
  ];

  const SEGMENTS = 48;

  const renderFrame = (now) => {
    const isDark = document.documentElement.dataset.theme === "dark";

    // 1. Base atmospheric canvas fill
    const bgGrad = ctx.createLinearGradient(0, 0, width, height);
    if (!isDark) {
      bgGrad.addColorStop(0.0, "#F8FAFC");
      bgGrad.addColorStop(0.5, "#EDF2F8");
      bgGrad.addColorStop(1.0, "#E1E9F2");
    } else {
      bgGrad.addColorStop(0.0, "#1A202C");
      bgGrad.addColorStop(0.5, "#131720");
      bgGrad.addColorStop(1.0, "#0D1017");
    }
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    // 2. Render each volumetric silk fold with soft cast shadow + luminous crest gradient
    // Keep the silk quiet: it is drawn at partial opacity over the base gradient.
    ctx.save();
    ctx.globalAlpha = isDark ? 0.55 : 0.5;
    const t = now * 0.6;
    ribbons.forEach((r, idx) => {
      // The ribbon's resting curve sways slowly; folds travel along its length like silk in a light breeze.
      const sway = Math.sin(t * r.speed * 0.5 + r.phase) * r.amp * 0.5;
      const x0 = -width * 0.12;
      const x3 = width * 1.12;
      const y0 = height * r.yStart + sway;
      const y3 = height * r.yEnd - sway * 0.6;
      const cp1x = width * 0.34;
      const cp2x = width * 0.68;
      const cp1y = height * r.cp1y + Math.cos(t * r.speed * 0.4 + r.phase) * r.amp * 0.4;
      const cp2y = height * r.cp2y - sway * 0.4;
      const travel = t * r.speed * 0.9;
      const pts = [];
      for (let i = 0; i <= SEGMENTS; i += 1) {
        const u = i / SEGMENTS;
        const v = 1 - u;
        const bx = v * v * v * x0 + 3 * v * v * u * cp1x + 3 * v * u * u * cp2x + u * u * u * x3;
        const by = v * v * v * y0 + 3 * v * v * u * cp1y + 3 * v * u * u * cp2y + u * u * u * y3;
        const fold =
          0.65 * Math.sin(u * Math.PI * 2.4 - travel + r.phase) +
          0.35 * Math.sin(u * Math.PI * 4.2 - travel * 1.6 + r.phase * 1.7);
        pts.push([bx, by + fold * r.amp * 0.6 * (0.45 + 0.55 * Math.sin(Math.PI * u))]);
      }
      const trace = (offset, reverse) => {
        const list = reverse ? [...pts].reverse() : pts;
        list.forEach(([px, py], i) => (i === 0 && !reverse ? ctx.moveTo(px, py + offset) : ctx.lineTo(px, py + offset)));
      };

      const midY = (y0 + y3) * 0.5;
      const foldHeight = Math.max(220, height * 0.34);

      // Soft ambient shadow cast just above the silk ridge
      ctx.save();
      ctx.beginPath();
      trace(-55, false);
      trace(60, true);
      ctx.closePath();

      const shadowGrad = ctx.createLinearGradient(width * 0.45, midY - 65, width * 0.55, midY + 45);
      if (!isDark) {
        shadowGrad.addColorStop(0.0, "rgba(184, 198, 218, 0.0)");
        shadowGrad.addColorStop(0.55, "rgba(176, 192, 214, 0.2)");
        shadowGrad.addColorStop(1.0, "rgba(184, 198, 218, 0.0)");
      } else {
        shadowGrad.addColorStop(0.0, "rgba(6, 8, 12, 0.0)");
        shadowGrad.addColorStop(0.55, "rgba(5, 7, 11, 0.22)");
        shadowGrad.addColorStop(1.0, "rgba(6, 8, 12, 0.0)");
      }
      ctx.fillStyle = shadowGrad;
      ctx.fill();
      ctx.restore();

      // Main volumetric silk sheet below the ridge
      ctx.save();
      ctx.beginPath();
      trace(0, false);
      ctx.lineTo(x3, height + 100);
      ctx.lineTo(x0, height + 100);
      ctx.closePath();

      const sheetGrad = ctx.createLinearGradient(
        width * 0.42,
        midY - 20,
        width * 0.58,
        midY + foldHeight
      );
      if (!isDark) {
        sheetGrad.addColorStop(0.0, "rgba(255, 255, 255, 0.96)");
        sheetGrad.addColorStop(0.16, "rgba(250, 252, 255, 0.90)");
        sheetGrad.addColorStop(0.52, "rgba(230, 237, 246, 0.84)");
        sheetGrad.addColorStop(1.0, "rgba(214, 224, 238, 0.90)");
      } else {
        const lift = idx % 2 === 0 ? 3 : 0;
        sheetGrad.addColorStop(0.0, `rgba(${36 + lift}, ${45 + lift}, ${60 + lift}, 0.7)`);
        sheetGrad.addColorStop(0.20, "rgba(27, 33, 44, 0.8)");
        sheetGrad.addColorStop(0.60, "rgba(18, 23, 32, 0.9)");
        sheetGrad.addColorStop(1.0, "rgba(13, 17, 24, 0.95)");
      }
      ctx.fillStyle = sheetGrad;
      ctx.fill();

      // Soft specular rim highlight along the silk fold crest
      ctx.beginPath();
      trace(0, false);
      ctx.strokeStyle = !isDark
        ? "rgba(255, 255, 255, 0.6)"
        : "rgba(165, 190, 225, 0.07)";
      ctx.lineWidth = !isDark ? 1.4 : 1.0;
      ctx.stroke();
      ctx.restore();
    });

    ctx.restore();

    requestAnimationFrame(renderFrame);
  };

  requestAnimationFrame(renderFrame);
}

// Logo intro: a pixel arrow cursor draws the pulse line block by block, clicks its end, and the blocks resolve
// into the vector logo. Plays once per page load and again when the logo is clicked.
const PIXEL_ARROW = [
  "X",
  "XX",
  "X.X",
  "X..X",
  "X...X",
  "X....X",
  "X.....X",
  "X......X",
  "X.......X",
  "X........X",
  "X.....XXXXX",
  "X..X..X",
  "X.X X..X",
  "XX  X..X",
  "X    X..X",
  "     X..X",
  "      X..X",
  "      X..X",
  "       XX",
];

function playLogoIntro() {
  const brand = document.querySelector(".reflex-hero-brand");
  const svg = brand?.querySelector(".hero-wave-icon");
  const path = svg?.querySelector("path");
  if (!brand || !svg || !path || brand.querySelector(".logo-intro-canvas")) return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || !svg.getScreenCTM()) return;

  const BLOCK = 3; // trail block size, px
  const CURSOR_PX = 2; // one cursor bitmap pixel, px
  const PAD = { left: 12, top: 12, right: 44, bottom: 52 };
  const T_APPEAR = 220;
  const T_DRAW = 1300;
  const T_CLICK = 260;
  const T_RESOLVE = 520;
  const total = T_APPEAR + T_DRAW + T_CLICK + T_RESOLVE;

  const brandRect = brand.getBoundingClientRect();
  const svgRect = svg.getBoundingClientRect();
  const canvas = document.createElement("canvas");
  canvas.className = "logo-intro-canvas";
  canvas.setAttribute("aria-hidden", "true");
  const cssW = svgRect.width + PAD.left + PAD.right;
  const cssH = svgRect.height + PAD.top + PAD.bottom;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  canvas.style.left = `${svgRect.left - brandRect.left - PAD.left}px`;
  canvas.style.top = `${svgRect.top - brandRect.top - PAD.top}px`;
  brand.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;

  // Path points in canvas coordinates.
  const ctm = svg.getScreenCTM();
  const length = path.getTotalLength();
  const toCanvas = (len) => {
    const p = path.getPointAtLength(len);
    const s = new DOMPoint(p.x, p.y).matrixTransform(ctm);
    return { x: s.x - svgRect.left + PAD.left, y: s.y - svgRect.top + PAD.top };
  };
  const start = toCanvas(0);
  const end = toCanvas(length);

  const trail = color();
  const cells = [];
  const seen = new Set();
  let drawnTo = 0;
  const lay = (upTo) => {
    for (; drawnTo <= upTo; drawnTo += 0.35) {
      const p = toCanvas(Math.min(drawnTo, length));
      const cx = Math.round(p.x / BLOCK) * BLOCK;
      const cy = Math.round(p.y / BLOCK) * BLOCK;
      const key = `${cx},${cy}`;
      if (!seen.has(key)) {
        seen.add(key);
        cells.push([cx, cy]);
      }
    }
  };

  function color() {
    return getComputedStyle(svg).color || "#0D9488";
  }
  const isDark = () => document.documentElement.dataset.theme === "dark";
  const drawCursor = (x, y, alpha, scale) => {
    const px = CURSOR_PX * scale;
    const ox = Math.round(x);
    const oy = Math.round(y);
    ctx.globalAlpha = alpha;
    PIXEL_ARROW.forEach((row, r) => {
      [...row].forEach((ch, c) => {
        if (ch === " ") return;
        ctx.fillStyle = ch === "X" ? (isDark() ? "#0B0F16" : "#0F172A") : "#FFFFFF";
        ctx.fillRect(ox + c * px, oy + r * px, Math.ceil(px), Math.ceil(px));
      });
    });
    ctx.globalAlpha = 1;
  };
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  svg.style.transition = "none";
  svg.style.opacity = "0";
  const t0 = performance.now();

  const frame = (now) => {
    const t = now - t0;
    ctx.clearRect(0, 0, cssW, cssH);

    const drawT = clamp01((t - T_APPEAR) / T_DRAW);
    const along = ease(drawT) * length;
    lay(along);
    const head = drawT > 0 ? toCanvas(along) : start;
    const resolveT = clamp01((t - T_APPEAR - T_DRAW - T_CLICK) / T_RESOLVE);

    // Trail blocks, fading out as the vector logo fades in.
    ctx.fillStyle = trail;
    ctx.globalAlpha = 1 - resolveT;
    for (const [cx, cy] of cells) ctx.fillRect(cx - BLOCK / 2, cy - BLOCK / 2, BLOCK + 1, BLOCK + 1);
    ctx.globalAlpha = 1;
    if (resolveT > 0) {
      svg.style.opacity = String(resolveT);
    }

    // Click: the cursor presses, and four blocks burst from the line's end.
    const clickT = clamp01((t - T_APPEAR - T_DRAW) / T_CLICK);
    const pressed = clickT > 0 && clickT < 0.45;
    if (clickT > 0) {
      const burst = clamp01((t - T_APPEAR - T_DRAW) / (T_CLICK + 260));
      const r = 4 + burst * 12;
      ctx.fillStyle = trail;
      ctx.globalAlpha = 1 - burst;
      for (const [dx, dy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
        ctx.fillRect(Math.round((end.x + dx * r) / BLOCK) * BLOCK - 1.5, Math.round((end.y + dy * r) / BLOCK) * BLOCK - 1.5, BLOCK, BLOCK);
      }
      ctx.globalAlpha = 1;
    }

    const appear = clamp01(t / T_APPEAR);
    const leave = clamp01((t - T_APPEAR - T_DRAW - T_CLICK) / (T_RESOLVE * 0.8));
    drawCursor(head.x + (pressed ? 1 : 0) + leave * 6, head.y + (pressed ? 1 : 0) + leave * 6, appear * (1 - leave), pressed ? 0.9 : 1);

    if (t < total) {
      requestAnimationFrame(frame);
    } else {
      svg.style.opacity = "";
      svg.style.transition = "";
      canvas.remove();
    }
  };
  requestAnimationFrame(frame);
}

function initLogoIntro() {
  const svg = document.querySelector(".reflex-hero-brand .hero-wave-icon");
  if (!svg) return;
  svg.style.cursor = "pointer";
  svg.addEventListener("click", playLogoIntro);
  const go = () => requestAnimationFrame(() => requestAnimationFrame(playLogoIntro));
  if (document.fonts?.ready) document.fonts.ready.then(go);
  else go();
}

function initUI() {
  initThemeToggle();
  initSilkCanvas();
  initLogoIntro();
  initLiveLayout();
  initViewportAmbient();
  initSiteRules();
  initInfoPopovers();
  $("btn-export-json")?.addEventListener("click", exportSessionJson);
  $("btn-export-steps")?.addEventListener("click", exportSessionJson);

  // Rail navigation
  document.querySelectorAll(".rail-btn[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });
  if ($("brand-home")) {
    $("brand-home").addEventListener("click", () => switchView("launch"));
  }
  if ($("btn-new-task")) {
    $("btn-new-task").addEventListener("click", () => switchView("launch"));
  }

  // Launch & Presets
  if ($("btn-launch")) {
    $("btn-launch").addEventListener("click", startSessionAndRun);
  }
  document.querySelectorAll(".scen-card[data-preset]").forEach((card) => {
    card.addEventListener("click", () => {
      const p = PRESETS[card.dataset.preset];
      if (!p) return;
      $("input-url").value = p.url;
      $("input-goal").value = p.goal;
      startSessionAndRun();
    });
  });

  // Live Mode / S1 / S2 Switchers
  if ($("top-mode-select")) {
    $("top-mode-select").addEventListener("change", async (e) => {
      if ($("top-s1-select")) $("top-s1-select").disabled = e.target.value === "s2_only";
      syncCustomSelects();
      try {
        const s = await apiPost("config", { mode: e.target.value });
        renderState(s);
      } catch (_) {}
    });
  }
  if ($("top-s1-select")) {
    $("top-s1-select").addEventListener("change", async (e) => {
      try {
        const s = await apiPost("config", { decision_backend: e.target.value });
        renderState(s);
      } catch (_) {}
    });
  }
  if ($("top-s2-select")) {
    $("top-s2-select").addEventListener("change", async (e) => {
      try {
        const s = await apiPost("config", { supervisor_model: e.target.value });
        renderState(s);
      } catch (_) {}
    });
  }

  // Run controls
  if ($("btn-continue")) {
    $("btn-continue").addEventListener("click", async () => {
      if (!currentState || !currentState.page) {
        return startSessionAndRun();
      }
      if (currentState.status !== "ready") {
        currentState.status = "ready";
      }
      await runContinuousLoop();
    });
  }
  if ($("btn-step")) {
    $("btn-step").addEventListener("click", async () => {
      if (stepInFlight || runningLoop) return;
      try {
        stepInFlight = true;
        const s = await apiPost("step");
        stepInFlight = false;
        renderState(s);
      } catch (err) {
        stepInFlight = false;
        alert(err.message);
      }
    });
  }
  if ($("btn-pause")) {
    $("btn-pause").addEventListener("click", () => {
      runningLoop = false;
      renderState(currentState);
    });
  }
  if ($("btn-reset")) {
    $("btn-reset").addEventListener("click", async () => {
      runningLoop = false;
      streamConnected = false;
      const s = await apiPost("reset");
      renderState(s);
    });
  }

  // DOM Boxes toggle
  if ($("btn-toggle-boxes")) {
    $("btn-toggle-boxes").addEventListener("click", () => {
      showDomBoxes = !showDomBoxes;
      $("btn-toggle-boxes").classList.toggle("active", showDomBoxes);
      if (currentState) renderDomOverlay(currentState);
    });
  }

  // Interactive viewport click (e.g. Cloudflare / manual click)
  const vpImg = $("viewport-img");
  if (vpImg) {
    vpImg.style.cursor = "crosshair";
    vpImg.addEventListener("load", () => {
      if (currentState) renderDomOverlay(currentState);
    });
    vpImg.addEventListener("click", async (e) => {
      if (!currentState || !currentState.page || runningLoop || stepInFlight) return;
      const rect = vpImg.getBoundingClientRect();
      const pageW = currentState.page.w || 1120;
      const pageH = currentState.page.h || 780;
      const x = Math.round(((e.clientX - rect.left) / rect.width) * pageW);
      const y = Math.round(((e.clientY - rect.top) / rect.height) * pageH);
      try {
        const s = await apiPost("click_xy", { x, y });
        renderState(s);
      } catch (_) {}
    });
  }

  // Safety Guard Approve / Reject
  if ($("btn-approve")) {
    $("btn-approve").addEventListener("click", async () => {
      const s = await apiPost("confirm", { approved: true });
      renderState(s);
      if (s.status === "ready") await runContinuousLoop();
    });
  }
  if ($("btn-reject")) {
    $("btn-reject").addEventListener("click", async () => {
      const alt = ($("input-alt-instruction")?.value || "").trim();
      const s = await apiPost("confirm", { approved: false, alternative_instruction: alt });
      renderState(s);
      if (s.status === "ready" && alt) await runContinuousLoop();
    });
  }

  // Co-Pilot Chat form
  if ($("chat-form")) {
    $("chat-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const inp = $("input-chat");
      const msg = (inp?.value || "").trim();
      if (!msg) return;
      inp.value = "";
      runningLoop = false;
      pendingChat.push({ role: "user", content: msg, kind: "chat", pending: true });
      chatWaiting = true;
      const box = $("chat-messages");
      if (box) renderChatLog(box, (currentState?.context?.chat_log || []).filter((m) => m.kind !== "escalation_note"));
      try {
        const s = await apiPost("reply", {
          message: msg,
          url: $("input-url")?.value || "https://www.google.com",
          mode: $("top-mode-select")?.value || "s2_only",
          decision_backend: $("top-s1-select")?.value || "openrouter_jev",
          supervisor_model: $("top-s2-select")?.value || "muse-spark-1.3-contributor",
        });
        chatWaiting = false;
        renderState(s);
        if (s.status === "ready") {
          await runContinuousLoop();
        }
      } catch (err) {
        chatWaiting = false;
        pendingChat = [];
        if (currentState) renderState(currentState);
        alert(err.message);
      }
    });
  }

  // Save API Keys in Settings
  if ($("btn-save-keys")) {
    $("btn-save-keys").addEventListener("click", async () => {
      const metaKey = ($("input-meta-key")?.value || "").trim();
      const orKey = ($("input-or-key")?.value || "").trim();
      const s = await apiPost("config", {
        meta_api_key: metaKey || undefined,
        openrouter_api_key: orKey || undefined,
      });
      if ($("input-meta-key")) $("input-meta-key").value = "";
      if ($("input-or-key")) $("input-or-key").value = "";
      if ($("save-keys-msg")) {
        $("save-keys-msg").textContent = "✓ Saved to .env & active!";
        setTimeout(() => {
          if ($("save-keys-msg")) $("save-keys-msg").textContent = "";
        }, 3000);
      }
      renderState(s);
    });
  }

  // Clear Site Memory
  if ($("btn-clear-memory")) {
    $("btn-clear-memory").addEventListener("click", async () => {
      if (!confirm("Delete all site rules, including the ones you wrote?")) return;
      const s = await apiPost("config", { clear_site_memory: true });
      renderState(s);
    });
  }

  initCustomSelects();

  window.addEventListener("resize", () => {
    if (currentState) renderDomOverlay(currentState);
  });

  // Initial state load
  apiGet("state")
    .then((s) => {
      renderState(s);
      if (s && s.page && s.status !== "idle") {
        switchView("live");
      }
    })
    .catch(() => {});
}

function syncCustomSelects() {
  document.querySelectorAll(".SysSwitch.enhanced").forEach((wrap) => {
    const sel = wrap.querySelector("select");
    const lbl = wrap.querySelector(".saas-select-label");
    const trig = wrap.querySelector(".saas-select-trigger");
    if (!sel || !lbl || !trig) return;
    const selectedOpt = sel.options[sel.selectedIndex];
    lbl.textContent = selectedOpt ? selectedOpt.textContent : "";
    trig.disabled = sel.disabled;
    wrap.classList.toggle("is-disabled", sel.disabled);
    wrap.querySelectorAll(".saas-select-opt").forEach((btn) => {
      btn.classList.toggle("selected", btn.dataset.val === sel.value);
    });
  });
}

function initCustomSelects() {
  document.querySelectorAll(".SysSwitch").forEach((wrap) => {
    const sel = wrap.querySelector("select");
    if (!sel || wrap.classList.contains("enhanced")) return;
    wrap.classList.add("enhanced");

    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "saas-select-trigger";
    trigger.innerHTML = `
      <span class="saas-select-label"></span>
      <span class="saas-select-chev">
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><path d="M4.5 6.5L8 10l3.5-3.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </span>
    `;

    const menu = document.createElement("div");
    menu.className = "saas-select-menu";

    Array.from(sel.options).forEach((opt) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "saas-select-opt";
      item.dataset.val = opt.value;
      item.innerHTML = `
        <span>${escapeHtml(opt.textContent)}</span>
        <span class="saas-opt-check">✓</span>
      `;
      item.addEventListener("click", (e) => {
        e.stopPropagation();
        if (sel.disabled) return;
        sel.value = opt.value;
        wrap.classList.remove("open");
        syncCustomSelects();
        sel.dispatchEvent(new Event("change", { bubbles: true }));
      });
      menu.appendChild(item);
    });

    trigger.addEventListener("click", (e) => {
      e.stopPropagation();
      if (sel.disabled) return;
      const wasOpen = wrap.classList.contains("open");
      document.querySelectorAll(".SysSwitch.open").forEach((w) => w.classList.remove("open"));
      if (!wasOpen) wrap.classList.add("open");
    });

    wrap.appendChild(trigger);
    wrap.appendChild(menu);
    sel.addEventListener("change", syncCustomSelects);
  });

  document.addEventListener("click", () => {
    document.querySelectorAll(".SysSwitch.open").forEach((w) => w.classList.remove("open"));
  });

  syncCustomSelects();
}

document.addEventListener("DOMContentLoaded", initUI);
