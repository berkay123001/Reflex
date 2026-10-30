"use strict";

const TOKEN = window.__REFLEX_TOKEN__ || "";
let currentState = null;
let runningLoop = false;
let stepInFlight = false;
let showDomBoxes = true;
let activeView = "launch";
let streamConnected = false;

// G6: Studio Dashboard State
let activeStageTab = "live"; // "live" | "mapping"
let mapZoom = 1.0;
let selectedStepIndex = null;
let consoleFilter = "all"; // "all" | "s1" | "s2" | "chat"
let consoleCollapsed = false;

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
  if (viewName === "mapping") {
    // Jump straight to live view and open the Execution Mapping tab
    switchView("live");
    switchStageTab("mapping");
    return;
  }

  activeView = viewName;
  document.querySelectorAll(".rail-btn[data-view]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === viewName);
  });
  document.querySelectorAll(".view-pane").forEach((pane) => {
    pane.classList.toggle("active", pane.id === `view-${viewName}`);
  });
  if (viewName === "live" && currentState) {
    requestAnimationFrame(() => {
      renderDomOverlay(currentState);
      if (activeStageTab === "mapping") {
        renderExecutionMapping(currentState);
      }
    });
  }
}

function switchStageTab(tabName) {
  activeStageTab = tabName;
  const btnLive = $("btn-tab-live");
  const btnMap = $("btn-tab-mapping");
  const stageBrowser = $("stage-browser-view");
  const stageMap = $("stage-mapping-view");

  if (btnLive) btnLive.classList.toggle("active", tabName === "live");
  if (btnMap) btnMap.classList.toggle("active", tabName === "mapping");

  if (stageBrowser) {
    stageBrowser.style.display = tabName === "live" ? "flex" : "none";
  }
  if (stageMap) {
    stageMap.style.display = tabName === "mapping" ? "flex" : "none";
  }

  if (tabName === "live") {
    if (currentState) {
      ensureLiveStream(currentState);
      requestAnimationFrame(() => renderDomOverlay(currentState));
    }
  } else if (tabName === "mapping") {
    if (currentState) {
      requestAnimationFrame(() => renderExecutionMapping(currentState));
    }
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
  const parentRect = img.parentElement ? img.parentElement.getBoundingClientRect() : rect;
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

// G6: Bento KPI Cards Renderer
function renderBentoKPIs(state) {
  if (!state) return;
  const history = state.history || [];
  const maxSteps = state.config?.max_steps || 60;
  const m = state.metrics || {};

  // Card 1: Step Budget
  const stepCount = history.length;
  const stepPct = Math.min(100, Math.round((stepCount / maxSteps) * 100));
  if ($("kpi-steps-val")) $("kpi-steps-val").textContent = String(stepCount);
  if ($("kpi-steps-max")) $("kpi-steps-max").textContent = `/ ${maxSteps}`;
  if ($("kpi-steps-ratio")) $("kpi-steps-ratio").textContent = `${stepPct}%`;
  if ($("kpi-steps-bar")) $("kpi-steps-bar").style.width = `${stepPct}%`;
  if ($("kpi-steps-sub")) $("kpi-steps-sub").textContent = `Budget remaining: ${Math.max(0, maxSteps - stepCount)}`;

  // Card 2: Dual-Engine Ratio
  const s1Count = m.s1_steps ?? history.filter((h) => h.actor === "S1").length;
  const s2Count = m.s2_escalations ?? history.filter((h) => h.actor === "S2").length;
  const totalEngineSteps = s1Count + s2Count || 1;
  const s1Pct = Math.round((s1Count / totalEngineSteps) * 100);
  const s2Pct = 100 - s1Pct;

  if ($("kpi-s1-count")) $("kpi-s1-count").textContent = String(s1Count);
  if ($("kpi-s2-count")) $("kpi-s2-count").textContent = String(s2Count);
  if ($("kpi-bar-s1")) $("kpi-bar-s1").style.width = `${s1Pct}%`;
  if ($("kpi-bar-s2")) $("kpi-bar-s2").style.width = `${s2Pct}%`;
  if ($("kpi-s1-pct")) $("kpi-s1-pct").textContent = `${s1Pct}%`;
  if ($("kpi-s2-pct")) $("kpi-s2-pct").textContent = `${s2Pct}%`;
  if ($("kpi-engine-mode")) {
    $("kpi-engine-mode").textContent = (state.config?.mode || "s2_only").toUpperCase();
  }

  // Card 3: P50 Latency
  const medStepMs = m.median_step_ms ?? m.median_decision_ms ?? 0;
  if ($("kpi-lat-val")) $("kpi-lat-val").textContent = `${medStepMs} ms`;
  const elapsedSec = ((m.elapsed_ms || 0) / 1000).toFixed(1);
  if ($("kpi-elapsed-val")) $("kpi-elapsed-val").textContent = `Elapsed: ${elapsedSec}s`;

  // Card 4: Estimated Cost & Tokens
  const costFormatted =
    m.total_cost_usd === "bilinmiyor"
      ? "bilinmiyor"
      : `$${Number(m.total_cost_usd ?? 0).toFixed(4)}`;
  if ($("kpi-cost-val")) $("kpi-cost-val").textContent = costFormatted;
  const usage = m.usage || {};
  if ($("kpi-tokens-val")) {
    $("kpi-tokens-val").textContent = `P:${usage.prompt_tokens ?? 0} · C:${usage.completion_tokens ?? 0} · Tot:${usage.total_tokens ?? 0}`;
  }

  // Card 5: Safety & Governance
  const approvedList = state.context?.approved_actions || [];
  const siteRulesCount = Object.keys(state.site_memory || {}).length;
  if (state.status === "waiting_confirmation") {
    if ($("kpi-guard-val")) $("kpi-guard-val").textContent = "HOLD ACTIVE";
    if ($("kpi-guard-pill")) $("kpi-guard-pill").textContent = "HOLD";
  } else if (approvedList.length > 0) {
    if ($("kpi-guard-val")) $("kpi-guard-val").textContent = `${approvedList.length} APPROVED`;
    if ($("kpi-guard-pill")) $("kpi-guard-pill").textContent = "VERIFIED";
  } else {
    if ($("kpi-guard-val")) $("kpi-guard-val").textContent = "GUARD ACTIVE";
    if ($("kpi-guard-pill")) $("kpi-guard-pill").textContent = "PASSIVE";
  }
  if ($("kpi-guard-sub")) {
    $("kpi-guard-sub").textContent = `${approvedList.length} approved · ${siteRulesCount} rules`;
  }
}

// G6: Execution Mapping (Pure Vanilla HTML + SVG DAG Flow)
function renderExecutionMapping(state) {
  const container = $("mapping-nodes-layer");
  const svg = $("mapping-svg-lines");
  const wrap = $("mapping-transform-layer");
  const countChip = $("map-step-count");
  if (!container || !svg || !wrap) return;

  container.innerHTML = "";
  svg.innerHTML = "";

  const history = (state && state.history) || [];
  if (countChip) countChip.textContent = `${history.length + 1} nodes`;

  const nodeWidth = 215;
  const nodeHeight = 84;
  const stepGapX = 245;

  const nodePositions = [];

  // 1. Root Goal Node
  const rootX = 30;
  const rootY = 110;
  nodePositions.push({ x: rootX, y: rootY, type: "root", stepIndex: 0 });

  const rootNode = document.createElement("div");
  rootNode.className = "map-node root";
  rootNode.style.left = `${rootX}px`;
  rootNode.style.top = `${rootY}px`;
  rootNode.innerHTML = `
    <div class="map-node-head">
      <span class="map-node-badge">🎯 ROOT GOAL</span>
      <span class="map-node-lat">INITIAL</span>
    </div>
    <div class="map-node-act" title="${escapeHtml(state?.goal || "Ready")}">${escapeHtml(state?.goal || "Ready")}</div>
    <div class="map-node-sub" title="${escapeHtml(state?.page?.url || state?.url || "about:blank")}">
      🌐 ${escapeHtml(state?.page?.url || state?.url || "about:blank")}
    </div>
  `;
  container.appendChild(rootNode);

  // 2. Trajectory Step Nodes
  history.forEach((step, idx) => {
    const isS2 = step.actor === "S2";
    const isUnsettled = step.note === "page_unsettled" || step.verification?.loop_detected;
    const isGuard = step.operation === "CONFIRM" || step.action?.includes("Safety Guard");

    const currX = 30 + (idx + 1) * stepGapX;
    // Multi-lane layout: Lane 1 (top y=20 for S2), Lane 0 (middle y=110 for S1), Lane 2 (bottom y=200 for retry/warn)
    let currY = 110;
    if (isS2) {
      currY = 20;
    } else if (isUnsettled) {
      currY = 200;
    } else if (isGuard) {
      currY = 200;
    }

    nodePositions.push({
      x: currX,
      y: currY,
      type: isS2 ? "s2" : isGuard ? "guard" : isUnsettled ? "warn" : "s1",
      stepIndex: step.step,
    });

    const nodeEl = document.createElement("div");
    const nodeTypeClass = isS2 ? "s2" : isGuard ? "guard" : isUnsettled ? "warn" : "s1";
    nodeEl.className = `map-node ${nodeTypeClass}` + (selectedStepIndex === step.step ? " active-selected" : "");
    nodeEl.style.left = `${currX}px`;
    nodeEl.style.top = `${currY}px`;

    const icon = isS2 ? "🧠" : isGuard ? "🛡️" : isUnsettled ? "⚠️" : "⚡";
    const label = isS2 ? "S2 AUDIT" : isGuard ? "SAFETY" : isUnsettled ? "UNSETTLED" : "S1 ACTION";
    const dur = step.step_ms ?? step.latency_ms ?? 0;

    let subTagHtml = "";
    if (step.verification?.page_changed) subTagHtml += '<span class="v-chip ok">DOM Diff ✓</span>';
    if (step.verification?.url_changed) subTagHtml += '<span class="v-chip ok">URL ✓</span>';
    if (step.verification?.loop_detected) subTagHtml += '<span class="v-chip warn">Loop ⚠</span>';

    nodeEl.innerHTML = `
      <div class="map-node-head">
        <span class="map-node-badge">${icon} #${step.step} ${label}</span>
        <span class="map-node-lat">${dur} ms</span>
      </div>
      <div class="map-node-act" title="${escapeHtml(step.operation)}: ${escapeHtml(step.action)}">
        ${escapeHtml(step.operation)} ${step.target ? `[${escapeHtml(step.target)}]` : ""}
      </div>
      <div class="map-node-sub" title="${escapeHtml(step.action || "-")}">
        ${escapeHtml(step.action || "-")}
      </div>
      <div class="map-node-tags">${subTagHtml}</div>
    `;

    nodeEl.addEventListener("click", () => {
      openStepInspector(step.step);
    });

    container.appendChild(nodeEl);
  });

  // 3. Terminal State Node (if done / blocked / error)
  const isTerminal = ["done", "blocked", "error"].includes(state?.status);
  if (isTerminal) {
    const termX = 30 + (history.length + 1) * stepGapX;
    const termY = 110;
    nodePositions.push({ x: termX, y: termY, type: "done", stepIndex: -1 });

    const termEl = document.createElement("div");
    const isSuccess = state.status === "done";
    termEl.className = `map-node ${isSuccess ? "done" : "warn"}`;
    termEl.style.left = `${termX}px`;
    termEl.style.top = `${termY}px`;
    termEl.innerHTML = `
      <div class="map-node-head">
        <span class="map-node-badge">${isSuccess ? "🏁 COMPLETE" : "⛔ HALTED"}</span>
      </div>
      <div class="map-node-act">${isSuccess ? "Goal Achieved ✓" : escapeHtml(state.status.toUpperCase())}</div>
      <div class="map-node-sub">${escapeHtml(state.last_error || "Terminal state")}</div>
    `;
    container.appendChild(termEl);
  }

  // 4. Draw Connecting Bezier Curves on SVG Canvas
  let pathsSvg = "";
  for (let i = 0; i < nodePositions.length - 1; i++) {
    const from = nodePositions[i];
    const to = nodePositions[i + 1];

    const x1 = from.x + nodeWidth;
    const y1 = from.y + nodeHeight / 2;
    const x2 = to.x;
    const y2 = to.y + nodeHeight / 2;

    const midX = (x1 + x2) / 2;
    const pathD = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
    const edgeCls = to.type || "s1";

    pathsSvg += `<path class="map-edge ${edgeCls}" d="${pathD}" />`;
  }
  svg.innerHTML = pathsSvg;

  // 5. Adjust Canvas Geometry & Scale
  const totalNodesCount = nodePositions.length;
  const totalWidth = Math.max(980, 60 + totalNodesCount * stepGapX + 100);
  wrap.style.width = `${totalWidth}px`;
  wrap.style.height = "320px";
  svg.setAttribute("width", String(totalWidth));
  svg.setAttribute("height", "320");
  wrap.style.transform = `scale(${mapZoom})`;
}

// G6: Step Inspector Drawer
function openStepInspector(stepIndex) {
  selectedStepIndex = stepIndex;
  const drawer = $("step-inspector-drawer");
  if (!drawer || !currentState) return;

  const history = currentState.history || [];
  const step = history.find((h) => h.step === stepIndex);
  if (!step) return;

  const isS2 = step.actor === "S2";
  const dur = step.step_ms ?? step.latency_ms ?? 0;

  // Header
  if ($("inspector-title")) {
    $("inspector-title").textContent = `Step #${step.step} · ${isS2 ? "S2 SUPERVISOR" : "S1 REFLEX"}`;
  }
  if ($("inspector-badge")) {
    $("inspector-badge").textContent = step.operation || "ACTION";
    $("inspector-badge").className = `insp-pill ${isS2 ? "s2" : "s1"}`;
  }
  if ($("inspector-latency")) $("inspector-latency").textContent = `${dur} ms`;

  // Timing Breakdown
  if ($("insp-t-s1")) $("insp-t-s1").textContent = `${step.s1_ms ?? 0} ms`;
  if ($("insp-t-s2")) $("insp-t-s2").textContent = `${step.s2_ms ?? 0} ms`;
  if ($("insp-t-act")) $("insp-t-act").textContent = `${step.act_ms ?? 0} ms`;
  if ($("insp-t-obs")) $("insp-t-obs").textContent = `${step.observe_ms ?? 0} ms`;

  // Action Parameters & Target
  if ($("insp-op")) $("insp-op").textContent = step.operation || "-";
  if ($("insp-target")) {
    $("insp-target").textContent = `${step.target ? `[Index ${step.target}] ` : ""}${step.action || "-"}`;
  }
  if ($("insp-text-row") && $("insp-text")) {
    if (step.text) {
      $("insp-text-row").style.display = "flex";
      $("insp-text").textContent = step.text;
    } else {
      $("insp-text-row").style.display = "none";
    }
  }
  if ($("insp-rect")) {
    if (step.rect) {
      $("insp-rect").textContent = `x:${step.rect.x}, y:${step.rect.y}, w:${step.rect.w}, h:${step.rect.h}`;
    } else {
      $("insp-rect").textContent = "none (viewport action)";
    }
  }

  // Confidence & Calibration Distribution
  const opPct = Math.round((step.confidence || 0) * 100);
  const tgtPct = step.target_confidence != null ? Math.round(step.target_confidence * 100) : 100;
  const jntPct = Math.round((step.joint_confidence || step.confidence || 0) * 100);
  const marPct = Math.round((step.target_margin ?? 1) * 100);

  if ($("insp-conf-op")) $("insp-conf-op").textContent = `${opPct}%`;
  if ($("insp-bar-op")) $("insp-bar-op").style.width = `${opPct}%`;
  if ($("insp-conf-tgt")) $("insp-conf-tgt").textContent = `${tgtPct}%`;
  if ($("insp-bar-tgt")) $("insp-bar-tgt").style.width = `${tgtPct}%`;
  if ($("insp-conf-jnt")) $("insp-conf-jnt").textContent = `${jntPct}%`;
  if ($("insp-bar-jnt")) $("insp-bar-jnt").style.width = `${jntPct}%`;
  if ($("insp-conf-mar")) $("insp-conf-mar").textContent = `${marPct}%`;
  if ($("insp-bar-mar")) $("insp-bar-mar").style.width = `${marPct}%`;

  // DOM Verification & State Change
  if ($("insp-url-transition")) {
    $("insp-url-transition").textContent = `${step.before_url || step.url || "-"} → ${step.url || "-"}`;
  }
  const v = step.verification || {};
  if ($("insp-chip-diff")) {
    $("insp-chip-diff").textContent = v.url_changed ? "URL Changed ✓" : v.page_changed ? "DOM Diff ✓" : "In-Place";
    $("insp-chip-diff").className = `v-chip ${v.effective_progress || v.page_changed ? "ok" : ""}`;
  }
  if ($("insp-chip-loop")) {
    $("insp-chip-loop").textContent = v.oscillation_detected ? "Oscillation ⚠" : v.loop_detected ? "Loop ⚠" : "Loop-Free ✓";
    $("insp-chip-loop").className = `v-chip ${v.loop_detected || v.oscillation_detected ? "warn" : "ok"}`;
  }
  if ($("insp-ver-note")) {
    $("insp-ver-note").textContent = step.note || v.reason || (v.page_changed ? "DOM mutated successfully" : "Execution verified");
  }

  // Supervisor Reasoning
  const matchingEsc = isS2
    ? (currentState.escalations || []).find((e) => e.step === step.step) || (currentState.escalations || []).slice(-1)[0]
    : null;
  const s2Sec = $("insp-s2-sec");
  if (s2Sec) {
    if (isS2 || matchingEsc) {
      s2Sec.style.display = "flex";
      if ($("insp-s2-trigger")) $("insp-s2-trigger").textContent = matchingEsc?.trigger_type || step.escalation_note || "-";
      if ($("insp-s2-eval")) $("insp-s2-eval").textContent = matchingEsc?.evaluation_previous_goal || "-";
      if ($("insp-s2-memory")) $("insp-s2-memory").textContent = matchingEsc?.memory || "-";
      if ($("insp-s2-next")) $("insp-s2-next").textContent = matchingEsc?.next_goal || "-";
    } else {
      s2Sec.style.display = "none";
    }
  }

  drawer.classList.add("open");

  // Highlight selected node on Execution Map
  document.querySelectorAll(".map-node").forEach((node, i) => {
    node.classList.toggle("active-selected", i === stepIndex);
  });
}

function closeStepInspector() {
  selectedStepIndex = null;
  const drawer = $("step-inspector-drawer");
  if (drawer) drawer.classList.remove("open");
  document.querySelectorAll(".map-node").forEach((node) => {
    node.classList.remove("active-selected");
  });
}

// G6: Collapsible Live Console
function renderLiveConsole(state) {
  const logScroll = $("console-log-scroll");
  const countAll = $("cnt-log-all");
  const countS1 = $("cnt-log-s1");
  const countS2 = $("cnt-log-s2");
  if (!logScroll || !state) return;

  const history = state.history || [];
  const escalations = state.escalations || [];

  const allEvents = [];

  history.forEach((h) => {
    const isS2 = h.actor === "S2";
    const dur = h.step_ms ?? h.latency_ms ?? 0;
    allEvents.push({
      time: `#${h.step}`,
      type: isS2 ? "s2" : "s1",
      badge: isS2 ? "🧠 S2" : "⚡ S1",
      msg: `[${h.operation}] ${h.target ? `Target #${h.target} · ` : ""}${h.action || ""}${h.text ? ` (typed: "${h.text}")` : ""}`,
      dur: `${dur}ms`,
      stepIndex: h.step,
    });
  });

  escalations.forEach((esc) => {
    allEvents.push({
      time: `#${esc.step || "ESC"}`,
      type: "s2",
      badge: "🧠 S2 ESC",
      msg: `Supervisor Trigger: ${esc.trigger_type || "escalation"} · ${esc.next_goal || esc.decision || ""}`,
      dur: "",
      stepIndex: esc.step,
    });
  });

  if (state.last_error) {
    allEvents.push({
      time: "ERR",
      type: "warn",
      badge: "⚠️ ERROR",
      msg: state.last_error,
      dur: "",
    });
  }

  const s1Count = allEvents.filter((e) => e.type === "s1").length;
  const s2Count = allEvents.filter((e) => e.type === "s2").length;
  if (countAll) countAll.textContent = String(allEvents.length);
  if (countS1) countS1.textContent = String(s1Count);
  if (countS2) countS2.textContent = String(s2Count);

  if (consoleFilter === "chat") {
    logScroll.style.display = "none";
    if ($("console-chat-wrapper")) $("console-chat-wrapper").style.display = "flex";
    return;
  }

  logScroll.style.display = "flex";
  if ($("console-chat-wrapper")) $("console-chat-wrapper").style.display = "none";

  const filteredEvents = allEvents.filter((ev) => {
    if (consoleFilter === "all") return true;
    return ev.type === consoleFilter;
  });

  if (filteredEvents.length === 0) {
    logScroll.innerHTML = `<div class="c-log-empty">No ${consoleFilter.toUpperCase()} events logged yet.</div>`;
    return;
  }

  logScroll.innerHTML = filteredEvents
    .map((ev) => `
      <div class="c-log-row" ${ev.stepIndex ? `onclick="openStepInspector(${ev.stepIndex})"` : ""}>
        <span class="c-log-time">${escapeHtml(ev.time)}</span>
        <span class="c-log-badge ${ev.type}">${escapeHtml(ev.badge)}</span>
        <span class="c-log-msg" title="${escapeHtml(ev.msg)}">${escapeHtml(ev.msg)}</span>
        <span class="c-log-dur">${escapeHtml(ev.dur)}</span>
      </div>
    `)
    .join("");

  logScroll.scrollTop = logScroll.scrollHeight;
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
        <div class="dcard-reason">Henüz adım çalıştırılmadı. Canlı akışı başlatmak için <b>▶ Run</b> veya <b>⏭ Step</b> tuşuna basın.</div>
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
      <div class="dcard ${isS2 ? "s2" : "s1"}" onclick="openStepInspector(${lastStep.step})">
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
        <div class="dstack-prev" onclick="openStepInspector(${ps.step})">
          <span>#${ps.step} <b style="color:${ps.actor === "S2" ? "var(--s2)" : "var(--s1)"}">${ps.actor}</b> ${escapeHtml(ps.operation)} · ${escapeHtml((ps.action || "").slice(0, 36))}</span>
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
        return `<div class="scrub-tick ${cls}" onclick="openStepInspector(${h.step})" title="#${h.step} [${h.actor}] ${escapeHtml(h.operation)}: ${escapeHtml(h.action)} (${hDur}ms)"></div>`;
      })
      .join("");
  }

  if (aspace) {
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
            <div class="dcard ${isS2 ? "s2" : "s1"}" onclick="openStepInspector(${h.step})">
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

  // Render Studio components
  renderBentoKPIs(state);
  ensureLiveStream(state);
  renderDomOverlay(state);
  renderDecisionStack(state);
  renderScrubberAndActionSpace(state);
  renderGuardAndChat(state);
  renderLiveConsole(state);
  renderSecondaryViews(state);

  if (activeStageTab === "mapping") {
    renderExecutionMapping(state);
  }
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
      decision_backend: $("top-s1-select")?.value || "openrouter_jev",
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

  // Stage Switcher (Live Browser vs Execution Mapping)
  if ($("btn-tab-live")) {
    $("btn-tab-live").addEventListener("click", () => switchStageTab("live"));
  }
  if ($("btn-tab-mapping")) {
    $("btn-tab-mapping").addEventListener("click", () => switchStageTab("mapping"));
  }

  // Execution Mapping Toolbar Controls
  if ($("btn-map-zoom-in")) {
    $("btn-map-zoom-in").addEventListener("click", () => {
      mapZoom = Math.min(2.0, mapZoom + 0.15);
      const layer = $("mapping-transform-layer");
      if (layer) layer.style.transform = `scale(${mapZoom})`;
      if ($("btn-map-zoom-reset")) $("btn-map-zoom-reset").textContent = `${Math.round(mapZoom * 100)}%`;
    });
  }
  if ($("btn-map-zoom-out")) {
    $("btn-map-zoom-out").addEventListener("click", () => {
      mapZoom = Math.max(0.4, mapZoom - 0.15);
      const layer = $("mapping-transform-layer");
      if (layer) layer.style.transform = `scale(${mapZoom})`;
      if ($("btn-map-zoom-reset")) $("btn-map-zoom-reset").textContent = `${Math.round(mapZoom * 100)}%`;
    });
  }
  if ($("btn-map-zoom-reset")) {
    $("btn-map-zoom-reset").addEventListener("click", () => {
      mapZoom = 1.0;
      const layer = $("mapping-transform-layer");
      if (layer) layer.style.transform = `scale(${mapZoom})`;
      $("btn-map-zoom-reset").textContent = "100%";
    });
  }
  if ($("btn-map-center-latest")) {
    $("btn-map-center-latest").addEventListener("click", () => {
      const wrap = $("mapping-canvas-wrap");
      if (wrap) wrap.scrollLeft = wrap.scrollWidth;
    });
  }

  // Step Inspector Drawer Close
  if ($("btn-close-inspector")) {
    $("btn-close-inspector").addEventListener("click", closeStepInspector);
  }

  // Collapsible Live Console Controls
  if ($("btn-toggle-console")) {
    $("btn-toggle-console").addEventListener("click", () => {
      consoleCollapsed = !consoleCollapsed;
      const dock = $("live-console-dock");
      if (dock) dock.classList.toggle("collapsed", consoleCollapsed);
      $("btn-toggle-console").textContent = consoleCollapsed ? "▲" : "▼";
      requestAnimationFrame(() => {
        if (currentState) renderDomOverlay(currentState);
      });
    });
  }

  document.querySelectorAll(".c-tab[data-filter]").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".c-tab[data-filter]").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      consoleFilter = tab.dataset.filter;
      if (currentState) renderLiveConsole(currentState);
    });
  });

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

  window.addEventListener("resize", () => {
    if (currentState) {
      renderDomOverlay(currentState);
      if (activeStageTab === "mapping") {
        renderExecutionMapping(currentState);
      }
    }
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

// Global functions for inline DOM event bindings (if any)
window.openStepInspector = openStepInspector;
window.closeStepInspector = closeStepInspector;

document.addEventListener("DOMContentLoaded", initUI);
