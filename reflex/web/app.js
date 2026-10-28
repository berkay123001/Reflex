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
    goal: "Özal Yıldırım'ın kişisel sitesine gir, blog yazılarına bak ve sayfada paylaştığı tüm blog yazılarını bana getir.",
  },
  gemini_chat: {
    url: "https://gemini.google.com/app?hl=tr",
    goal: "Gemini ile Türkçe 2 tur sohbet et: İlk turda kısa ve eğlenceli bir bilmece sor, cevabını bekle; ikinci turda verdiği cevaba yorum yapıp yapay zeka bilinci hakkında kısa bir soru sor ve yanıtını özetle.",
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
          <span style="font-family:var(--mono);font-size:10px;color:var(--ink-3)">3-Layer Verification</span>
        </div>
        <div class="dcard-reason">Oturum hazır. Canlı akışı başlatmak için <b>▶ Run</b> veya <b>⏭ Step</b> tuşuna basın.</div>
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
    const stepBreakdownTitle = `S1: ${lastStep.s1_ms ?? 0}ms | S2: ${lastStep.s2_ms ?? 0}ms | Act: ${lastStep.act_ms ?? 0}ms | Obs: ${lastStep.observe_ms ?? 0}ms`;

    html += `
      <div class="dcard ${isS2 ? "s2" : "s1"}">
        <div class="dcard-top">
          <span class="actor-badge ${isS2 ? "s2" : "s1"}">#${lastStep.step} ${isS2 ? "🧠 S2 SUPERVISOR" : "⚡ S1 REFLEX"} · ${escapeHtml(lastStep.operation)}</span>
          <span style="font-family:var(--mono);font-size:10px;color:var(--ink-3)" title="${stepBreakdownTitle}">${stepDur} ms</span>
        </div>
        <div class="dcard-act">${lastStep.target ? `[${escapeHtml(lastStep.target)}] ` : ""}${escapeHtml(lastStep.action)}</div>
        ${lastStep.text ? `<div class="dcard-reason">✎ Typed: <b>"${escapeHtml(lastStep.text)}"</b></div>` : ""}
        ${lastStep.escalation_note ? `<div class="dcard-reason">🧠 ${escapeHtml(lastStep.escalation_note)}</div>` : ""}
        ${
          matchingEsc && (matchingEsc.evaluation_previous_goal || matchingEsc.memory || matchingEsc.next_goal)
            ? `<div class="s2-state-box">
                ${matchingEsc.evaluation_previous_goal ? `<div><b>Eval:</b> ${escapeHtml(matchingEsc.evaluation_previous_goal)}</div>` : ""}
                ${matchingEsc.memory ? `<div><b>Memory:</b> ${escapeHtml(matchingEsc.memory)}</div>` : ""}
                ${matchingEsc.next_goal ? `<div><b>Next:</b> ${escapeHtml(matchingEsc.next_goal)}</div>` : ""}
              </div>`
            : ""
        }
        <div class="bars-grid">
          <div class="cbar">
            <div class="cbar-top"><span>Operation</span><span>${opPct}%</span></div>
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
          <span class="v-chip ok">Exec ✓</span>
          <span class="v-chip ${v.effective_progress ? "ok" : ""}">${v.url_changed ? "URL Changed ✓" : v.page_changed ? "DOM Diff ✓" : "In-Place"}</span>
          <span class="v-chip ${v.loop_detected || v.oscillation_detected ? "warn" : "ok"}">${v.oscillation_detected ? "Oscillation ⚠" : v.loop_detected ? "Loop ⚠" : "Loop-Free ✓"}</span>
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
    html += `<div class="dcard s1"><div class="dcard-act">Step #1 Decision</div></div>`;
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
    const lastDec = (state.decisions || []).slice(-1)[0];
    const opProbs = lastDec?.gate ? null : null;
    const decProbs = (state.decisions || []).slice(-1)[0];
    if (decProbs) {
      aspace.innerHTML = `
        <span class="aspace-chip top"><span>S1 CHOICE: ${escapeHtml(decProbs.operation)} ${decProbs.target ? `[${escapeHtml(decProbs.target)}]` : ""}</span><span class="p">${Math.round((decProbs.confidence || 0) * 100)}%</span></span>
        <span class="aspace-chip"><span>JOINT</span><span class="p">${Math.round((decProbs.joint_confidence || 0) * 100)}%</span></span>
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
      $("guard-modal-summary").textContent = state.pending_confirmation.summary || "Approve action?";
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
      chatBox.innerHTML = `<div class="cmsg assistant"><div class="cmsg-role">REFLEX CO-PILOT</div>Görev sırasında bana soru sorabilir, hedefi değiştirebilir veya ASK_USER sorularını buradan yanıtlayabilirsin.</div>`;
    } else {
      chatBox.innerHTML = log
        .map((m) => {
          const cls = m.kind === "ask_user" ? "ask_user" : m.kind === "confirm_action" ? "confirm_action" : m.role === "user" ? "user" : "assistant";
          const roleLbl = m.role === "user" ? "USER" : m.kind === "ask_user" ? "❓ ASK_USER" : m.kind === "confirm_action" ? "🛡️ SAFETY GUARD" : "🤖 REFLEX AGENT";
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
  if (trajLbl) trajLbl.textContent = `${history.length} steps executed`;
  if (trajList) {
    if (history.length === 0) {
      trajList.innerHTML = `<div class="pbox">Henüz adım çalıştırılmadı.</div>`;
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
                <span style="font-family:var(--mono);font-size:11px;color:var(--ink-3)">${hDur} ms (S1:${h.s1_ms ?? 0} S2:${h.s2_ms ?? 0} Act:${h.act_ms ?? 0} Obs:${h.observe_ms ?? 0}) · ${escapeHtml(h.url || "")}</span>
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
    $("ctx-working-memory").textContent = state.context?.working_memory || "(No Supervisor escalation memory recorded yet)";
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
        ? `CONFIGURED (${state.keys.meta_api_masked})`
        : "NOT SET (Using OpenRouter Fallback)";
    }
    if ($("or-key-badge")) {
      $("or-key-badge").textContent = state.keys.openrouter_configured
        ? `CONFIGURED (${state.keys.openrouter_masked})`
        : "NOT SET";
    }
    if ($("launch-meta-badge")) {
      $("launch-meta-badge").textContent = state.keys.meta_api_configured
        ? `Meta API Active (${state.keys.meta_api_masked})`
        : "Meta API / OpenRouter Ready";
    }
  }
}

function renderState(state) {
  currentState = state;
  if (!state) return;

  const st = (state.status || "idle").toLowerCase();
  const badge = $("status-badge");
  const stTxt = $("status-text");
  if (badge && stTxt) {
    if (st === "error") {
      stTxt.textContent = "ERROR: " + (state.last_error || "Session halted");
      badge.className = "status-pill error";
    } else {
      stTxt.textContent = runningLoop ? "RUNNING" : st.toUpperCase();
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
      m.total_cost_usd === "bilinmiyor"
        ? "bilinmiyor"
        : `$${Number(m.total_cost_usd ?? 0).toFixed(4)}`;
  }

  if (state.config) {
    if ($("top-s1-select") && state.config.decision_backend) {
      $("top-s1-select").value = state.config.decision_backend;
    }
    if ($("top-s2-select") && state.config.supervisor_model) {
      $("top-s2-select").value = state.config.supervisor_model;
    }
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

function initUI() {
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

  // Live S1 / S2 Switchers
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
          decision_backend: $("top-s1-select")?.value || "local_turnike",
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

document.addEventListener("DOMContentLoaded", initUI);
