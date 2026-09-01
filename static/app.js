/**
 * app.js — Support AI dashboard frontend logic.
 *
 * Responsibilities:
 *   - Tab switching between Ticket Triage and Account Health Brief
 *   - Sending form data to Flask API endpoints
 *   - Rendering user echoes, AI responses, and error messages into chat logs
 *   - Building the account summary card from JSON data
 *   - Managing loading state (button disable + spinner)
 */

"use strict";

/* ============================================================
   STATE
   Simple in-memory state — no localStorage needed.
============================================================ */
const state = {
  /** @type {'triage'|'brief'} */
  activeTab: "triage",
};

/* ============================================================
   TAB SWITCHING
============================================================ */

/**
 * Switch the visible panel and update nav button highlight.
 * @param {'triage'|'brief'} tab
 */
function switchTab(tab) {
  state.activeTab = tab;

  // Panel visibility
  document.querySelectorAll(".tab-panel").forEach((el) => {
    el.classList.toggle("active", el.id === `panel-${tab}`);
  });

  // Nav button active state
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    const isActive = btn.id === `nav-${tab}`;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-current", isActive ? "page" : "false");
  });
}

/* ============================================================
   CHAT LOG HELPERS
============================================================ */

/**
 * Append a message bubble to a chat log element.
 *
 * @param {HTMLElement} log       - The .chat-log container.
 * @param {'user'|'ai'|'error'}  role
 * @param {string}      text      - The content to display.
 */
function appendMessage(log, role, text) {
  // Remove the empty-state placeholder if present
  const empty = log.querySelector(".empty-state");
  if (empty) empty.remove();

  const msg = document.createElement("div");
  msg.className = `msg msg-${role}`;

  // Small label above the bubble so the user knows who said what
  const label = document.createElement("div");
  label.className = "msg-label";
  label.textContent = role === "user" ? "You" : role === "ai" ? "Support AI" : "Error";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  // Use textContent so we don't accidentally execute injected HTML from the LLM
  bubble.textContent = text;

  msg.appendChild(label);
  msg.appendChild(bubble);
  log.appendChild(msg);

  // Auto-scroll to the newest message
  log.scrollTop = log.scrollHeight;

  return msg;
}

/**
 * Insert a three-dot loading spinner into the log.
 * Returns the element so the caller can remove it later.
 *
 * @param {HTMLElement} log
 * @returns {HTMLElement}
 */
function appendLoader(log) {
  const empty = log.querySelector(".empty-state");
  if (empty) empty.remove();

  const loader = document.createElement("div");
  loader.className = "loading-indicator";
  loader.setAttribute("aria-label", "Thinking…");
  loader.innerHTML = "<span></span><span></span><span></span>";

  log.appendChild(loader);
  log.scrollTop = log.scrollHeight;
  return loader;
}

/**
 * Render the empty-state placeholder inside a log.
 * @param {HTMLElement} log
 * @param {string}      message
 */
function showEmptyState(log, message) {
  log.innerHTML = `
    <div class="empty-state">
      <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <rect x="3" y="5" width="18" height="14" rx="2"/>
        <path d="M9 10h6M9 14h4" stroke-linecap="round"/>
      </svg>
      <p>${message}</p>
    </div>
  `;
}

/* ============================================================
   LOADING STATE — button management
============================================================ */

/**
 * Lock or unlock a submit button during a pending request.
 * @param {HTMLButtonElement} btn
 * @param {boolean}           loading
 * @param {string}            defaultLabel  - Text to restore when done.
 */
function setButtonLoading(btn, loading, defaultLabel) {
  btn.disabled = loading;
  btn.querySelector(".btn-arrow").style.display = loading ? "none" : "";
  // Keep the text node; only swap if needed
  if (loading) {
    btn.firstChild.textContent = "Working…";
  } else {
    btn.firstChild.textContent = defaultLabel + "\u00A0"; // NBSP keeps width
  }
}

/* ============================================================
   ACCOUNT SUMMARY CARD
============================================================ */

/**
 * Determine the CSS class for the health status badge.
 * @param {string} status
 * @returns {string}
 */
function healthBadgeClass(status) {
  const map = {
    healthy:  "badge-healthy",
    "at risk": "badge-at-risk",
    churning: "badge-churning",
    new:      "badge-new",
  };
  return map[(status || "").toLowerCase()] || "badge-new";
}

/**
 * Format a USD integer as a compact string, e.g. 240000 → "$240k".
 * @param {number} value
 * @returns {string}
 */
function formatARR(value) {
  if (!value && value !== 0) return "—";
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}m`;
  if (value >= 1_000)     return `$${(value / 1_000).toFixed(0)}k`;
  return `$${value}`;
}

/**
 * Populate and reveal the account summary card.
 * @param {Object} account  - The account object returned by the API.
 */
function renderAccountCard(account) {
  const card  = document.getElementById("account-card");
  const inner = document.getElementById("account-card-inner");

  const health  = account.health_status || "—";
  const badgeClass = healthBadgeClass(health);

  // Build chip HTML — each chip is a label + value pair
  const chips = [
    { label: "Company",    value: account.company || "—" },
    { label: "Plan",       value: account.plan_tier || "—" },
    { label: "ARR",        value: formatARR(account.arr_usd) },
    {
      label: "Seats",
      value: `${account.seats_active ?? "—"} / ${account.seats_licensed ?? "—"} active`,
    },
    {
      label: "Health",
      value: `<span class="badge ${badgeClass}">${health}</span>`,
      raw: true,   // flag: render as innerHTML, not textContent
    },
    { label: "TAM",        value: account.tam || "—" },
    { label: "Renewal",    value: account.renewal_date || "—" },
    { label: "NPS Score",  value: account.nps_score ?? "—" },
  ];

  inner.innerHTML = chips
    .map(
      ({ label, value, raw }) => `
      <div class="acc-chip">
        <span class="acc-chip-label">${label}</span>
        ${
          raw
            ? `<span class="acc-chip-value">${value}</span>`
            : `<span class="acc-chip-value">${escapeHtml(String(value))}</span>`
        }
      </div>`
    )
    .join("");

  card.classList.remove("hidden");
}

/**
 * Hide and clear the account summary card.
 */
function clearAccountCard() {
  document.getElementById("account-card").classList.add("hidden");
  document.getElementById("account-card-inner").innerHTML = "";
}

/* ============================================================
   UTILITY
============================================================ */

/**
 * Escape HTML special characters to prevent XSS from account/ticket data.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/* ============================================================
   TRIAGE — submit + clear
============================================================ */

async function submitTriage() {
  const input  = document.getElementById("triage-input");
  const log    = document.getElementById("triage-log");
  const btn    = document.getElementById("triage-submit");
  const ticket = input.value.trim();

  if (!ticket) return;

  // Echo the user's input immediately for responsiveness
  appendMessage(log, "user", ticket);
  input.value = "";

  const loader = appendLoader(log);
  setButtonLoading(btn, true, "Analyse ticket");

  try {
    const response = await fetch("/api/triage", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ ticket }),
    });

    const data = await response.json();

    loader.remove();
    setButtonLoading(btn, false, "Analyse ticket");

    if (!response.ok) {
      appendMessage(log, "error", data.error || "An unexpected error occurred.");
    } else {
      appendMessage(log, "ai", data.result);
    }
  } catch (err) {
    loader.remove();
    setButtonLoading(btn, false, "Analyse ticket");
    appendMessage(log, "error", `Network error: ${err.message}`);
  }
}

function clearTriage() {
  const log = document.getElementById("triage-log");
  showEmptyState(log, "Paste a support ticket and press Analyse.");
  document.getElementById("triage-input").value = "";
}

/* ============================================================
   ACCOUNT HEALTH BRIEF — submit + clear
============================================================ */

async function submitBrief() {
  const input     = document.getElementById("brief-input");
  const log       = document.getElementById("brief-log");
  const btn       = document.getElementById("brief-submit");
  const accountId = input.value.trim().toUpperCase();

  if (!accountId) return;

  clearAccountCard();
  appendMessage(log, "user", `Brief for ${accountId}`);
  input.value = "";

  const loader = appendLoader(log);
  setButtonLoading(btn, true, "Generate brief");

  try {
    const response = await fetch("/api/brief", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ account_id: accountId }),
    });

    const data = await response.json();

    loader.remove();
    setButtonLoading(btn, false, "Generate brief");

    if (!response.ok) {
      appendMessage(log, "error", data.error || "An unexpected error occurred.");
    } else {
      // Show the compact account summary card above the log
      if (data.account) renderAccountCard(data.account);
      appendMessage(log, "ai", data.result);
    }
  } catch (err) {
    loader.remove();
    setButtonLoading(btn, false, "Generate brief");
    appendMessage(log, "error", `Network error: ${err.message}`);
  }
}

function clearBrief() {
  const log = document.getElementById("brief-log");
  showEmptyState(log, "Enter an account ID and press Generate brief.");
  document.getElementById("brief-input").value = "";
  clearAccountCard();
}

/* ============================================================
   KEYBOARD SHORTCUT
   Ctrl/Cmd + Enter submits the active panel's form.
============================================================ */
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    if (state.activeTab === "triage") submitTriage();
    else                              submitBrief();
  }
});

/* ============================================================
   INIT — render empty states on page load
============================================================ */
(function init() {
  showEmptyState(
    document.getElementById("triage-log"),
    "Paste a support ticket and press Analyse."
  );
  showEmptyState(
    document.getElementById("brief-log"),
    "Enter an account ID and press Generate brief."
  );
})();