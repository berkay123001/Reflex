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
      callout.textContent = `${lastStep.actor} · ${lastStep.operation || lastStep.kind} · ${lastStep.step_ms ?? lastStep.latency_ms ?? 0}ms`;
    }
  } else if (cursor) {
    cursor.classList.remove("visible");
  }
}

function renderDecisionStack(state) {
  const container = $("decision-stack");
  const countEl = $("dstack-count");
  if (!container) return;

  const history = state.history || [];
  const decisions = state.decisions || [];
  const escalations = state.escalations || [];
  if (countEl) countEl.textContent = `${history.length} steps`;

  if (history.length === 0 && escalations.length === 0) {
    container.innerHTML = `
      <div class="dcard s1">
        <div class="dcard-top">
          <span class="actor-badge s1">READY</span>
          <span style="font-family:var(--mono);font-size:11px;color:var(--ink-3)">3-Layer Verification</span>
        </div>
        <div class="dcard-reason">Session ready. Press <b>▶ Run</b> or <b>⏭ Step</b> to begin the live stream.</div>
      </div>`;
    return;
  }

  const lastStep = history[history.length - 1];
  const lastEsc = escalations[escalations.length - 1];
  const lastDec = decisions[decisions.length - 1];

  let html = "";
  if (lastStep) {
    const isS2 = lastStep.actor === "S2";
    const v = lastStep.verification || {};
    const opPct = Math.round((lastStep.confidence || 0) * 100);
    const tgtPct = lastStep.target_confidence != null ? Math.round(lastStep.target_confidence * 100) : 100;
    const jntPct = Math.round((lastStep.joint_confidence || lastStep.confidence || 0) * 100);
    const marPct = Math.round((lastStep.target_margin ?? 1) * 100);

    const matchingEsc = isS2 ? escalations.find((e) => e.step === lastStep.step) || lastEsc : null;
    const stepDur = lastStep.step_ms ?? lastStep.latency_ms ?? 0;
    const stepBreakdownTitle = `S1: ${lastStep.s1_ms ?? 0}ms | S2: ${lastStep.s2_ms ?? 0}ms | Act: ${lastStep.act_ms ?? 0}ms | Observe: ${lastStep.observe_ms ?? 0}ms`;

    html += `
      <div class="dcard ${isS2 ? "s2" : "s1"}">
        <div class="dcard-top">
          <span class="actor-badge ${isS2 ? "s2" : "s1"}">#${lastStep.step} ${isS2 ? "🧠 S2 SUPERVISOR" : "⚡ S1 REFLEX"} · ${escapeHtml(lastStep.operation)}</span>
          <span style="font-family:var(--mono);font-size:11px;color:var(--ink-3)" title="${stepBreakdownTitle}">${stepDur} ms</span>
        </div>
        <div class="dcard-act">${lastStep.target ? `[${escapeHtml(lastStep.target)}] ` : ""}${escapeHtml(lastStep.action)}</div>
        ${lastStep.text ? `<div class="dcard-reason">✎ Typed: <b>"${escapeHtml(lastStep.text)}"</b></div>` : ""}
        ${lastStep.escalation_note ? `<div class="dcard-reason">🧠 ${escapeHtml(lastStep.escalation_note)}</div>` : ""}
        ${
          matchingEsc && (matchingEsc.evaluation_previous_goal || matchingEsc.memory || matchingEsc.next_goal)
            ? `<div class="s2-state-box">
                ${matchingEsc.evaluation_previous_goal ? `<div><b>Eval:</b> ${escapeHtml(matchingEsc.evaluation_previous_goal)}</div>` : ""}
                ${matchingEsc.memory ? `<div><b>Memory:</b> ${escapeHtml(matchingEsc.memory)}</div>` : ""}
                ${matchingEsc.next_goal ? `<div><b>Next Goal:</b> ${escapeHtml(matchingEsc.next_goal)}</div>` : ""}
              </div>`
            : ""
        }
        <div class="bars-grid">
          <div class="cbar">
            <div class="cbar-top"><span>Action</span><span>${opPct}%</span></div>
            <div class="cbar-track"><div class="cbar-fill ${isS2 ? "s2" : ""}" style="width:${opPct}%"></div></div>
          </div>
          <div class="cbar">
            <div class="cbar-top"><span>Target</span><span>${tgtPct}%</span></div>
            <div class="cbar-track"><div class="cbar-fill ${isS2 ? "s2" : ""}" style="width:${tgtPct}%"></div></div>
          </div>
          <div class="cbar">
            <div class="cbar-top"><span>Joint Conf</span><span>${jntPct}%</span></div>
            <div class="cbar-track"><div class="cbar-fill ${isS2 ? "s2" : ""}" style="width:${jntPct}%"></div></div>
          </div>
          <div class="cbar">
            <div class="cbar-top"><span>Margin</span><span>${marPct}%</span></div>
            <div class="cbar-track"><div class="cbar-fill ${isS2 ? "s2" : ""}" style="width:${marPct}%"></div></div>
          </div>
        </div>
        <div class="ver-row">
          <span class="v-chip ok">Applied ✓</span>
          <span class="v-chip ${v.effective_progress ? "ok" : ""}">${v.url_changed ? "URL Changed ✓" : v.page_changed ? "DOM Updated ✓" : "Same Page"}</span>
          <span class="v-chip ${v.loop_detected || v.oscillation_detected ? "warn" : "ok"}">${v.oscillation_detected ? "Oscillation ⚠" : v.loop_detected ? "Loop ⚠" : "No Loop ✓"}</span>
          <span class="v-chip">${escapeHtml(lastStep.model || "")}</span>
        </div>
      </div>`;

    // Stacked previous 3 steps
    const prevSteps = history.slice(Math.max(0, history.length - 4), history.length - 1).reverse();
    prevSteps.forEach((ps) => {
      const psDur = ps.step_ms ?? ps.latency_ms ?? 0;
      html += `
        <div class="dstack-prev">
          <span>#${ps.step} <b style="color:${ps.actor === "S2" ? "var(--s2)" : "var(--s1)"}">${ps.actor}</b> ${escapeHtml(ps.operation)} · ${escapeHtml((ps.action || "").slice(0, 38))}</span>
          <span>${psDur}ms</span>
        </div>`;
    });
  } else if (lastDec) {
    html += `<div class="dcard s1"><div class="dcard-act">Step 1 Decision</div></div>`;
  }

  container.innerHTML = html;
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
  if (chatBox) {
    const log = state.context?.chat_log || [];
    if (log.length === 0) {
      chatBox.innerHTML = `<div class="cmsg assistant"><div class="cmsg-role">REFLEX CO-PILOT</div>Ask questions during execution, steer the goal, or reply to human-in-the-loop prompts here.</div>`;
    } else {
      chatBox.innerHTML = log
        .map((m) => {
          const cls = m.kind === "ask_user" ? "ask_user" : m.kind === "confirm_action" ? "confirm_action" : m.role === "user" ? "user" : "assistant";
          const roleLbl = m.role === "user" ? "YOU" : m.kind === "ask_user" ? "❓ ASK_USER PROMPT" : m.kind === "confirm_action" ? "🛡️ SAFETY GUARD" : "🤖 REFLEX AGENT";
          return `<div class="cmsg ${cls}"><div class="cmsg-role">${roleLbl}</div>${escapeHtml(m.content)}</div>`;
        })
        .join("");
      chatBox.scrollTop = chatBox.scrollHeight;
    }
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
  if ($("memory-rules-box")) {
    const mem = state.site_memory || {};
    $("memory-rules-box").textContent = Object.keys(mem).length ? JSON.stringify(mem, null, 2) : "No domain rules recorded yet.";
  }

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
      const next = await apiPost("step");
      stepInFlight = false;
      renderState(next);
      if (next.status !== "ready") {
        break;
      }
      await new Promise((r) => setTimeout(r, 140));
    }
  } catch (err) {
    stepInFlight = false;
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
      try {
        const s = await apiPost("reply", {
          message: msg,
          url: $("input-url")?.value || "https://www.google.com",
          mode: $("top-mode-select")?.value || "s2_only",
          decision_backend: $("top-s1-select")?.value || "openrouter_jev",
          supervisor_model: $("top-s2-select")?.value || "muse-spark-1.3-contributor",
        });
        renderState(s);
        if (s.status === "ready") {
          await runContinuousLoop();
        }
      } catch (err) {
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
