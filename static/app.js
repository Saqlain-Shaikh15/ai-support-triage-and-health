/**
 * app.js: Support AI dashboard frontend.
 *
 * Talks to the FastAPI backend:
 *   GET  /api/health
 *   POST /api/triage  {ticket}      -> {result, sources}
 *   POST /api/brief   {account_id}  -> {result, account, meta}
 */

"use strict";

const state = { activeTab: "triage" };

const EXAMPLES = {
  triage:
    "Hey something is wrong with the billing system, billing went from 80 users to 100 users when we had 80 active users",
  brief: "ACC-1899",
};

/* ============================================================
   UTILITIES
============================================================ */

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  let data = {};
  try {
    data = await res.json();
  } catch (_) {
    /* non-JSON error body */
  }

  if (!res.ok) throw new Error(data.error || `Request failed (${res.status}).`);
  return data;
}

/* ============================================================
   MARKDOWN (small, safe renderer for LLM output)
   Input is HTML-escaped first, so nothing from the model can inject markup.
============================================================ */

function renderMarkdown(source) {
  const lines = escapeHtml(source).split(/\r?\n/);
  const out = [];
  let para = [];
  let list = null; // {type: 'ul'|'ol', items: []}
  let table = null; // string[] of raw rows

  const inline = (s) =>
    s
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");

  const flushPara = () => {
    if (para.length) out.push(`<p>${inline(para.join(" "))}</p>`);
    para = [];
  };
  const flushList = () => {
    if (list) {
      const items = list.items.map((i) => `<li>${inline(i)}</li>`).join("");
      out.push(`<${list.type}>${items}</${list.type}>`);
    }
    list = null;
  };
  const flushTable = () => {
    if (!table) return;
    const cells = (row) =>
      row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
    const isSep = (row) => /^\s*\|?[\s:-]+(\|[\s:-]+)+\|?\s*$/.test(row);

    let head = null;
    let rows = table;
    if (table.length > 1 && isSep(table[1])) {
      head = cells(table[0]);
      rows = table.slice(2);
    }
    const thead = head
      ? `<thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead>`
      : "";
    const tbody = rows
      .map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`)
      .join("");
    out.push(`<div class="table-wrap"><table>${thead}<tbody>${tbody}</tbody></table></div>`);
    table = null;
  };
  const flushAll = () => {
    flushPara();
    flushList();
    flushTable();
  };

  for (const line of lines) {
    if (!line.trim()) {
      flushAll();
      continue;
    }

    let m;
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
      flushAll();
      const level = Math.min(m[1].length + 2, 6);
      out.push(`<h${level}>${inline(m[2])}</h${level}>`);
    } else if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
      flushAll();
      out.push("<hr>");
    } else if (/^\s*\|.*\|\s*$/.test(line)) {
      flushPara();
      flushList();
      (table = table || []).push(line);
    } else if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) {
      flushPara();
      flushTable();
      if (!list || list.type !== "ul") {
        flushList();
        list = { type: "ul", items: [] };
      }
      list.items.push(m[1]);
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      flushPara();
      flushTable();
      if (!list || list.type !== "ol") {
        flushList();
        list = { type: "ol", items: [] };
      }
      list.items.push(m[1]);
    } else if ((m = line.match(/^&gt;\s?(.*)$/))) {
      flushAll();
      out.push(`<blockquote>${inline(m[1])}</blockquote>`);
    } else {
      flushList();
      flushTable();
      para.push(line.trim());
    }
  }
  flushAll();
  return out.join("");
}

/* ============================================================
   TAB SWITCHING
============================================================ */

function switchTab(tab) {
  state.activeTab = tab;

  document.querySelectorAll(".tab-panel").forEach((el) => {
    el.classList.toggle("active", el.id === `panel-${tab}`);
  });

  document.querySelectorAll(".nav-btn").forEach((btn) => {
    const isActive = btn.id === `nav-${tab}`;
    btn.classList.toggle("active", isActive);
    if (isActive) btn.setAttribute("aria-current", "page");
    else btn.removeAttribute("aria-current");
  });
}

/* ============================================================
   CHAT LOG
============================================================ */

function removeEmptyState(log) {
  const empty = log.querySelector(".empty-state");
  if (empty) empty.remove();
}

/**
 * Append a message. AI messages are rendered as markdown and can carry
 * a sources row and a short meta line.
 */
function appendMessage(log, role, text, extras = {}) {
  removeEmptyState(log);

  const msg = document.createElement("div");
  msg.className = `msg msg-${role}`;

  const head = document.createElement("div");
  head.className = "msg-head";

  const label = document.createElement("span");
  label.className = "msg-label";
  label.textContent = role === "user" ? "You" : role === "ai" ? "Support AI" : "Something went wrong";
  head.appendChild(label);

  if (role === "ai") {
    const copy = document.createElement("button");
    copy.className = "copy-btn";
    copy.type = "button";
    copy.textContent = "Copy";
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        copy.textContent = "Copied";
      } catch (_) {
        copy.textContent = "Copy failed";
      }
      setTimeout(() => (copy.textContent = "Copy"), 1600);
    });
    head.appendChild(copy);
  }

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  if (role === "ai") {
    bubble.classList.add("md");
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = text;
  }

  msg.appendChild(head);
  msg.appendChild(bubble);

  if (extras.meta) {
    const meta = document.createElement("p");
    meta.className = "msg-meta";
    meta.textContent = extras.meta;
    msg.appendChild(meta);
  }

  if (extras.sources && extras.sources.length) {
    msg.appendChild(buildSources(extras.sources));
  }

  log.appendChild(msg);
  log.scrollTop = log.scrollHeight;
  return msg;
}

function buildSources(sources) {
  const wrap = document.createElement("div");
  wrap.className = "sources";

  const title = document.createElement("span");
  title.className = "sources-title";
  title.textContent = "Drawn from";
  wrap.appendChild(title);

  sources.forEach((s) => {
    const chip = document.createElement("span");
    chip.className = `source-chip source-${s.type}`;
    const id = document.createElement("span");
    id.className = "source-id";
    id.textContent = s.id;
    chip.appendChild(id);
    if (s.detail) {
      const detail = document.createElement("span");
      detail.className = "source-detail";
      detail.textContent = s.detail;
      chip.appendChild(detail);
    }
    chip.title = s.type === "ticket" ? "Similar past ticket" : "Knowledge base";
    wrap.appendChild(chip);
  });

  return wrap;
}

function appendLoader(log) {
  removeEmptyState(log);
  const loader = document.createElement("div");
  loader.className = "loading-indicator";
  loader.setAttribute("role", "status");
  loader.setAttribute("aria-label", "Working on it");
  loader.innerHTML = "<span></span><span></span><span></span>";
  log.appendChild(loader);
  log.scrollTop = log.scrollHeight;
  return loader;
}

/**
 * Empty state with a one-click example so the first run needs no typing.
 */
function showEmptyState(log, message, exampleLabel, onExample) {
  log.innerHTML = "";

  const wrap = document.createElement("div");
  wrap.className = "empty-state";
  wrap.innerHTML = `
    <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
      <rect x="3" y="5" width="18" height="14" rx="2"/>
      <path d="M9 10h6M9 14h4" stroke-linecap="round"/>
    </svg>`;

  const p = document.createElement("p");
  p.textContent = message;
  wrap.appendChild(p);

  if (exampleLabel && onExample) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "example-btn";
    btn.textContent = exampleLabel;
    btn.addEventListener("click", onExample);
    wrap.appendChild(btn);
  }

  log.appendChild(wrap);
}

/* ============================================================
   BUTTON LOADING STATE
============================================================ */

function setButtonLoading(btn, loading, label) {
  btn.disabled = loading;
  btn.classList.toggle("is-loading", loading);
  btn.querySelector(".btn-label").textContent = loading ? "Working" : label;
}

/* ============================================================
   ACCOUNT CARD
============================================================ */

function healthBadgeClass(status) {
  const map = {
    healthy: "badge-healthy",
    "at risk": "badge-at-risk",
    churning: "badge-churning",
    new: "badge-new",
  };
  return map[(status || "").toLowerCase()] || "badge-new";
}

function formatARR(value) {
  if (value === null || value === undefined) return "n/a";
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}m`;
  if (value >= 1_000) return `$${(value / 1_000).toFixed(0)}k`;
  return `$${value}`;
}

function renderAccountCard(account) {
  const card = document.getElementById("account-card");
  const inner = document.getElementById("account-card-inner");

  const health = account.health_status || "n/a";
  const dash = (v) => (v === null || v === undefined || v === "" ? "n/a" : v);

  const chips = [
    { label: "Company", value: dash(account.company) },
    { label: "Plan", value: dash(account.plan_tier) },
    { label: "ARR", value: formatARR(account.arr_usd) },
    { label: "Seats active", value: `${dash(account.seats_active)} of ${dash(account.seats_licensed)}` },
    {
      label: "Health",
      html: `<span class="badge ${healthBadgeClass(health)}">${escapeHtml(health)}</span>`,
    },
    { label: "Usage", value: dash(account.usage_trend) },
    { label: "Open tickets", value: dash(account.open_tickets) },
    { label: "P1, last 30 days", value: dash(account.p1_tickets_last_30d) },
    { label: "TAM", value: dash(account.tam) },
    { label: "Renewal", value: dash(account.renewal_date) },
    { label: "NPS", value: dash(account.nps_score) },
  ];

  inner.innerHTML = chips
    .map(
      (c) => `
      <div class="acc-chip">
        <span class="acc-chip-label">${escapeHtml(c.label)}</span>
        <span class="acc-chip-value">${c.html ?? escapeHtml(String(c.value))}</span>
      </div>`
    )
    .join("");

  card.classList.remove("hidden");
}

function clearAccountCard() {
  document.getElementById("account-card").classList.add("hidden");
  document.getElementById("account-card-inner").innerHTML = "";
}

/* ============================================================
   TRIAGE
============================================================ */

async function submitTriage() {
  const input = document.getElementById("triage-input");
  const log = document.getElementById("triage-log");
  const btn = document.getElementById("triage-submit");
  const ticket = input.value.trim();

  if (!ticket || btn.disabled) return;

  appendMessage(log, "user", ticket);
  input.value = "";

  const loader = appendLoader(log);
  setButtonLoading(btn, true, "Analyse ticket");

  try {
    const data = await postJSON("/api/triage", { ticket });
    loader.remove();
    appendMessage(log, "ai", data.result, { sources: data.sources });
  } catch (err) {
    loader.remove();
    appendMessage(log, "error", err.message);
  } finally {
    setButtonLoading(btn, false, "Analyse ticket");
  }
}

function showTriageEmpty() {
  showEmptyState(
    document.getElementById("triage-log"),
    "Paste a support ticket below to get a classification and a draft reply.",
    "Use an example ticket",
    () => {
      const input = document.getElementById("triage-input");
      input.value = EXAMPLES.triage;
      input.focus();
    }
  );
}

function clearTriage() {
  document.getElementById("triage-input").value = "";
  showTriageEmpty();
}

/* ============================================================
   ACCOUNT BRIEF
============================================================ */

async function submitBrief() {
  const input = document.getElementById("brief-input");
  const log = document.getElementById("brief-log");
  const btn = document.getElementById("brief-submit");
  const accountId = input.value.trim().toUpperCase();

  if (!accountId || btn.disabled) return;

  clearAccountCard();
  appendMessage(log, "user", `Brief for ${accountId}`);
  input.value = "";

  const loader = appendLoader(log);
  setButtonLoading(btn, true, "Generate brief");

  try {
    const data = await postJSON("/api/brief", { account_id: accountId });
    loader.remove();

    if (data.account) renderAccountCard(data.account);

    let meta;
    if (data.meta) {
      const { tickets_reviewed: n, tickets_flagged: f, window_days: d } = data.meta;
      meta = `${n} ticket${n === 1 ? "" : "s"} reviewed from the last ${d} days. ${f} flagged for churn or escalation language.`;
    }
    appendMessage(log, "ai", data.result, { meta });
  } catch (err) {
    loader.remove();
    appendMessage(log, "error", err.message);
  } finally {
    setButtonLoading(btn, false, "Generate brief");
  }
}

function showBriefEmpty() {
  showEmptyState(
    document.getElementById("brief-log"),
    "Enter an account ID below to generate a brief.",
    "Use ACC-1899",
    () => {
      const input = document.getElementById("brief-input");
      input.value = EXAMPLES.brief;
      input.focus();
    }
  );
}

function clearBrief() {
  document.getElementById("brief-input").value = "";
  clearAccountCard();
  showBriefEmpty();
}

/* ============================================================
   KEYBOARD + INIT
============================================================ */

document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    if (state.activeTab === "triage") submitTriage();
    else submitBrief();
  }
});

async function checkHealth() {
  const dot = document.getElementById("status-dot");
  const label = document.getElementById("status-label");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.llm_configured) {
      label.textContent = "Ready";
      dot.className = "footer-dot";
    } else {
      label.textContent = "GROQ_API_KEY missing";
      dot.className = "footer-dot warn";
    }
  } catch (_) {
    label.textContent = "Server offline";
    dot.className = "footer-dot down";
  }
}

(function init() {
  showTriageEmpty();
  showBriefEmpty();
  checkHealth();
})();