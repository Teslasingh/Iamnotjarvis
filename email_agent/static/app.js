const state = {
  emails: [],
  profile: {},
  selectedId: null,
  selectedDraftId: null,
  selectedEmail: null,
  emailViewMode: "html",
  gmailAuthorized: false,
  syncInfo: null,
  syncing: false,
  sortBy: "recent",
  loadingEmail: false,
  busyAction: null,
  toastTimer: null,
};

const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const raw = await response.text();
    let message = raw || response.statusText;
    try {
      message = JSON.parse(raw).detail || message;
    } catch {
      // Non-JSON responses are fine; show the raw text.
    }
    throw new Error(message || response.statusText);
  }
  return response.json();
}

function toast(message, type = "info") {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast toast-${type}`;
  element.classList.remove("hidden");
  if (state.toastTimer) window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => element.classList.add("hidden"), 4200);
}

function setBusy(action, busy) {
  state.busyAction = busy ? action : null;
  const map = {
    reanalyze: "#reanalyze-button",
    draft: "#draft-button",
    archive: "#archive-button",
    saveDraft: "#save-draft-button",
    send: "#send-button",
  };
  Object.entries(map).forEach(([key, selector]) => {
    const button = $(selector);
    if (!button) return;
    const isThis = key === action && busy;
    if (key === "send" || key === "saveDraft") {
      // handled separately for sent drafts
      if (!$("#draft-panel")?.classList.contains("is-sent")) {
        button.disabled = Boolean(state.busyAction);
      }
    } else {
      button.disabled = Boolean(state.busyAction);
    }
    if (isThis) {
      button.dataset.originalLabel = button.dataset.originalLabel || button.textContent;
      button.textContent =
        key === "reanalyze"
          ? "Analyzing..."
          : key === "draft"
            ? "Generating..."
            : key === "archive"
              ? "Archiving..."
              : button.textContent;
    } else if (button.dataset.originalLabel && !state.busyAction) {
      button.textContent = button.dataset.originalLabel;
      delete button.dataset.originalLabel;
    }
  });
}

function renderSyncProgress(progress = {}) {
  const percent = Number(progress.percent || 0);
  const fill = $("#sync-progress-fill");
  const percentLabel = $("#sync-progress-percent");
  const stageLabel = $("#sync-progress-stage");
  const message = $("#sync-progress-message");
  const panel = $("#sync-panel");
  if (!fill || !percentLabel) return;

  fill.style.width = `${percent}%`;
  percentLabel.textContent = `${percent}%`;
  if (stageLabel) stageLabel.textContent = formatSyncStage(progress.stage);
  if (message) message.textContent = progress.message || "Waiting to sync...";

  const checked = Number(progress.checked || 0);
  const newCount = Number(progress.new || 0);
  const analyzed = Number(progress.analyzed || 0);
  const pending = Number(progress.pending_total || 0);
  const setStat = (id, value) => {
    const el = $(id);
    if (el) el.textContent = value;
  };
  setStat("#sync-stat-checked", checked);
  setStat("#sync-stat-new", newCount);
  setStat("#sync-stat-analyzed", analyzed);
  setStat("#sync-stat-pending", pending);

  if (panel && (progress.active || percent > 0)) panel.open = true;
  if (panel && !progress.active && progress.stage === "done") {
    window.setTimeout(() => {
      if (!state.syncing) panel.open = false;
    }, 1800);
  }
}

function formatSyncStage(stage) {
  const labels = {
    idle: "Idle",
    starting: "Starting",
    labels: "Preparing",
    fetch: "Fetching mail",
    cleanup: "Saving",
    analyze: "Analyzing",
    done: "Complete",
    error: "Error",
  };
  return labels[stage] || stage || "Idle";
}

function renderSyncStatus(data) {
  const element = $("#sync-status");
  if (!element) return;

  if (state.syncing) {
    element.textContent = `Syncing Gmail inbox (last ${state.windowDays} days)...`;
    updateSyncButton();
    return;
  }

  const info = data?.sync_info || state.syncInfo || {};
  state.syncInfo = info;
  const windowDays = info.window_days ?? state.windowDays ?? 5;
  state.windowDays = windowDays;
  const total = info.total_in_db ?? 0;
  const pending = info.pending_analysis ?? 0;
  const lastSynced = formatDateTime(info.last_synced_at);
  const latestSubject = info.latest_email_subject || "No emails yet";
  const mode = data?.mode === "incremental" ? "Checked for new mail" : `Loaded last ${windowDays} days`;

  if (!state.gmailAuthorized) {
    element.textContent = "Connect Gmail to sync your inbox.";
    return;
  }

  const newEmails = data?.new_emails ?? 0;
  const analyzed = data?.analyzed ?? 0;
  const changes =
    newEmails || analyzed
      ? `${newEmails} new, ${analyzed} analyzed · `
      : pending
        ? `${pending} pending analysis · `
        : "";

  element.textContent = `${changes}${total} emails saved · ${mode} · Last sync ${lastSynced} · Latest: ${latestSubject}`;
  updateSyncButton();
}

function updateSyncButton() {
  const button = $("#sync-button");
  if (!button) return;
  if (!state.gmailAuthorized) {
    button.disabled = true;
    button.textContent = "Connect Gmail";
    return;
  }
  button.disabled = state.syncing;
  button.textContent = state.syncing ? "Syncing..." : "Sync";
}

function updateConnectButton() {
  const link = $("#connect-gmail");
  if (!link) return;
  if (state.gmailAuthorized) {
    link.textContent = "Gmail Connected";
    link.classList.add("is-connected");
    link.setAttribute("title", "Reconnect or switch Gmail account");
  } else {
    link.textContent = "Connect Gmail";
    link.classList.remove("is-connected");
    link.removeAttribute("title");
  }
}

function formatDateTime(value) {
  if (!value) return "never";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

async function loadStatus() {
  const data = await api("/api/status");
  state.gmailAuthorized = Boolean(data.gmail_authorized);
  state.syncInfo = data.sync_info || null;
  state.windowDays = (data.sync_info && data.sync_info.window_days) || 5;
  const hint = $("#sync-status");
  if (hint && !state.syncing) {
    hint.textContent = `Loading inbox from the last ${state.windowDays} days...`;
  }
  $("#status").innerHTML = [
    ["Gmail", data.gmail_authorized ? `Connected as ${data.gmail_email}` : "Not connected"],
    ["OpenAI", data.openai_configured ? "Configured" : "Heuristic fallback"],
    ["Profile", data.profile_ready ? "Ready" : "Needs details"],
    ["Resume", data.resume_uploaded ? "Uploaded" : "Not uploaded"],
  ]
    .map(([label, value]) => `<div class="status-item"><strong>${label}:</strong> ${escapeHtml(value)}</div>`)
    .join("");
  $("#profile-summary-text").textContent = data.profile_summary || "Upload a resume to build your profile.";
  renderSavedResume(data);
  renderSyncStatus(data);
  updateSyncButton();
  updateConnectButton();
}

async function loadProfile() {
  const profile = await api("/api/profile");
  state.profile = profile;
  const form = $("#profile-form");
  for (const [key, value] of Object.entries(profile)) {
    const field = form.elements[key];
    if (field && field.type !== "file") {
      field.value = typeof value === "object" ? "" : value || "";
    }
  }
  $("#profile-summary-text").textContent = profile.profile_summary || "Upload a resume to build your profile.";
  renderSavedResume(profile);
}

async function loadAgentPrompt() {
  const data = await api("/api/agent-prompt");
  $("#agent-prompt").value = data.agent_prompt || "";
}

async function saveAgentPrompt() {
  try {
    const data = await api("/api/agent-prompt", {
      method: "PUT",
      body: JSON.stringify({ agent_prompt: $("#agent-prompt").value }),
    });
    $("#agent-prompt").value = data.agent_prompt || "";
    const queued = Number(data.pending_reanalysis || 0);
    toast(
      queued
        ? `Prompt saved. ${queued} email(s) queued. Click Sync to re-analyze with the new instructions.`
        : "Prompt saved. Click Sync after new mail arrives to analyze with the new instructions.",
      "success",
    );
  } catch (error) {
    toast(error.message, "error");
  }
}

async function resetAgentPrompt() {
  try {
    const data = await api("/api/agent-prompt/reset", { method: "POST" });
    $("#agent-prompt").value = data.agent_prompt || "";
    const queued = Number(data.pending_reanalysis || 0);
    toast(
      queued
        ? `Agent prompt reset. ${queued} email(s) queued. Click Sync to re-analyze.`
        : "Agent prompt reset to default.",
      "success",
    );
  } catch (error) {
    toast(error.message, "error");
  }
}

async function saveProfile(event) {
  event.preventDefault();
  const formData = new FormData(event.target);
  const preserved = [
    "phone",
    "location",
    "linkedin",
    "portfolio",
    "summary",
    "work_experience",
    "common_application_details",
    "resume_notes",
  ];
  for (const key of preserved) {
    if (state.profile[key]) {
      formData.set(key, state.profile[key]);
    }
  }
  try {
    const profile = await api("/api/profile", { method: "POST", body: formData });
    state.profile = profile;
    renderSavedResume(profile);
    toast("Profile saved and resume parsed.", "success");
    await Promise.all([loadStatus(), loadProfile()]);
  } catch (error) {
    toast(error.message, "error");
  }
}

function openProfileModal() {
  const form = $("#profile-modal-form");
  for (const [key, value] of Object.entries(state.profile)) {
    const field = form.elements[key];
    if (field && field.type !== "file") {
      field.value = typeof value === "object" ? "" : value || "";
    }
  }
  renderSavedResumeInto($("#modal-saved-resume-info"), state.profile);
  $("#profile-modal").classList.remove("hidden");
}

function closeProfileModal() {
  $("#profile-modal").classList.add("hidden");
}

async function saveProfileModal(event) {
  event.preventDefault();
  const form = event.target;
  const formData = new FormData(form);
  const preserved = [
    "phone",
    "location",
    "linkedin",
    "portfolio",
    "summary",
    "work_experience",
    "common_application_details",
    "resume_notes",
  ];
  for (const key of preserved) {
    if (state.profile[key]) {
      formData.set(key, state.profile[key]);
    }
  }
  const saveButton = $("#profile-modal-save");
  saveButton.disabled = true;
  saveButton.textContent = "Saving...";
  try {
    const profile = await api("/api/profile", { method: "POST", body: formData });
    state.profile = profile;
    renderSavedResume(profile);
    toast("Profile saved and resume parsed.", "success");
    await Promise.all([loadStatus(), loadProfile()]);
    closeProfileModal();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    saveButton.disabled = false;
    saveButton.textContent = "Save Profile";
  }
}

function renderSavedResumeInto(element, profile) {
  if (!element) return;
  const fileName = profile.resume_file_name || fileNameFromPath(profile.resume_path);
  if (!profile.resume_uploaded && !fileName) {
    element.innerHTML = `<div class="empty compact">No resume saved yet.</div>`;
    return;
  }
  element.innerHTML = `
    <div class="saved-resume-card">
      <strong>Saved resume</strong>
      <span>${escapeHtml(fileName || "resume")}</span>
      <small>Upload a new file above to replace it.</small>
    </div>`;
}

function renderSavedResume(profile) {
  const element = $("#saved-resume-info");
  if (!element) return;
  const fileName = profile.resume_file_name || fileNameFromPath(profile.resume_path);
  if (!profile.resume_uploaded && !fileName) {
    element.innerHTML = `<div class="empty compact">No resume saved yet.</div>`;
    return;
  }
  element.innerHTML = `
    <div class="saved-resume-card">
      <strong>Saved resume</strong>
      <span>${escapeHtml(fileName || "Resume uploaded")}</span>
      <small>Resume is used for AI profile comparison.</small>
    </div>
  `;
}

function fileNameFromPath(path) {
  if (!path) return "";
  return String(path).split(/[\\/]/).filter(Boolean).pop() || "";
}

async function syncInbox({ silent = false } = {}) {
  if (!state.gmailAuthorized) {
    if (!silent) toast("Connect Gmail first.", "error");
    return;
  }
  if (state.syncing) return;
  state.syncing = true;
  updateSyncButton();
  renderSyncStatus();
  renderSyncProgress({ active: true, percent: 0, stage: "starting", message: "Starting sync..." });

  try {
    const start = await api("/api/sync", { method: "POST", body: JSON.stringify({ analyze_new: true }) });
    let progress = start.progress || {};
    if (!start.started && !progress.active) {
      renderSyncProgress(progress);
      return;
    }

    while (progress.active) {
      await new Promise((resolve) => window.setTimeout(resolve, 350));
      progress = await api("/api/sync/progress");
      renderSyncProgress(progress);
    }

    renderSyncProgress(progress);
    if (progress.error) throw new Error(progress.error);

    const data = progress.result || {};
    renderSyncStatus(data);
    if (data.new_emails > 0 || data.analyzed > 0) {
      const parts = [];
      if (data.new_emails > 0) parts.push(`${data.new_emails} new`);
      if (data.analyzed > 0) parts.push(`${data.analyzed} analyzed`);
      toast(`Synced ${parts.join(", ")} email(s).`, "success");
    } else if (!silent) {
      toast("Inbox is up to date.", "success");
    }
    await Promise.all([loadEmails(), loadStatus()]);
    if (state.selectedId) {
      await selectEmail(state.selectedId, { quiet: true });
    }
  } catch (error) {
    renderSyncProgress({ active: false, percent: 0, stage: "error", message: error.message });
    renderSyncStatus();
    toast(error.message, "error");
  } finally {
    state.syncing = false;
    updateSyncButton();
    renderSyncStatus();
  }
}

async function loadEmails() {
  const data = await api("/api/emails?limit=150&include_non_jobs=true");
  state.emails = data.emails || [];
  renderEmailList();
}

function relevanceRank(email) {
  const order = {
    Relevant: 4,
    Analyzed: 3,
    Analyzing: 2,
    "Not Analyzed": 1,
    "Not Relevant": 0,
  };
  return order[email.status || "Not Analyzed"] ?? 0;
}

function sortEmails(emails) {
  const sorted = [...emails];
  if (state.sortBy === "match") {
    return sorted.sort((a, b) => {
      const scoreDiff = Number(b.match_score || 0) - Number(a.match_score || 0);
      if (scoreDiff !== 0) return scoreDiff;
      return Number(b.internal_date || 0) - Number(a.internal_date || 0);
    });
  }
  if (state.sortBy === "relevance") {
    return sorted.sort((a, b) => {
      const relevanceDiff = relevanceRank(b) - relevanceRank(a);
      if (relevanceDiff !== 0) return relevanceDiff;
      const scoreDiff = Number(b.match_score || 0) - Number(a.match_score || 0);
      if (scoreDiff !== 0) return scoreDiff;
      return Number(b.internal_date || 0) - Number(a.internal_date || 0);
    });
  }
  return sorted.sort((a, b) => Number(b.internal_date || 0) - Number(a.internal_date || 0));
}

function statusBadgeClass(status) {
  const value = String(status || "Not Analyzed");
  if (value === "Relevant") return "status-relevant";
  if (value === "Not Relevant") return "status-irrelevant";
  if (value === "Analyzing") return "status-analyzing";
  if (value === "Analyzed") return "status-analyzed";
  return "status-pending";
}

function truncate(text, max = 140) {
  const value = String(text || "").replace(/\s+/g, " ").trim();
  if (value.length <= max) return value;
  return `${value.slice(0, max - 1)}…`;
}

function renderEmailList() {
  const list = $("#email-list");
  if (state.syncing && !state.emails.length) {
    list.innerHTML = `<div class="empty">Syncing inbox...</div>`;
    return;
  }
  const emails = sortEmails(state.emails);
  if (!emails.length) {
    list.innerHTML = `<div class="empty">No emails in the last 5 days yet.</div>`;
    return;
  }
  list.innerHTML = emails
    .map((email) => {
      const active = email.gmail_id === state.selectedId ? "active" : "";
      const score = Number(email.match_score || 0);
      const isJob = isJobEmail(email);
      const confidenceLabel = isJob ? `${score}%` : "—";
      const title = email.subject || email.job_title || email.role || "(no subject)";
      const summary = email.analysis_json?.summary || email.snippet || "";
      const status = email.status || "Not Analyzed";
      return `
        <article class="email-card ${active}" data-id="${email.gmail_id}" tabindex="0" role="button" aria-pressed="${active ? "true" : "false"}">
          <div class="card-header">
            <h4>${escapeHtml(title)}</h4>
            <span class="score ${scoreClass(score)}">${escapeHtml(confidenceLabel)}</span>
          </div>
          <p class="email-card-meta">${escapeHtml(email.sender || "Unknown sender")} · ${escapeHtml(formatDate(email.received_at))}</p>
          <div class="email-card-badges">
            <span class="badge status ${statusBadgeClass(status)}">${escapeHtml(status)}</span>
            ${email.latest_draft_status === "sent" ? '<span class="badge sent">Reply sent</span>' : ""}
          </div>
          <p class="email-card-summary">${escapeHtml(truncate(summary))}</p>
        </article>
      `;
    })
    .join("");
  list.querySelectorAll(".email-card").forEach((card) => {
    card.addEventListener("click", () => selectEmail(card.dataset.id));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectEmail(card.dataset.id);
      }
    });
  });
}

function showDetailLoading() {
  $("#detail-empty").classList.add("hidden");
  $("#email-detail").classList.remove("hidden");
  $("#detail-subject").textContent = "Loading email…";
  $("#detail-meta").textContent = "Fetching full message";
  $("#match-overview").innerHTML = `<div class="detail-skeleton"></div>`;
  $("#analysis").innerHTML = `<div class="detail-skeleton short"></div>`;
  $("#email-body").innerHTML = `<div class="detail-skeleton tall"></div>`;
  $("#email-view-toggle").hidden = true;
  $("#draft-panel").classList.add("hidden");
  $("#draft-empty").classList.remove("hidden");
  $("#draft-empty").textContent = "Loading…";
}

async function selectEmail(gmailId, { quiet = false } = {}) {
  if (!gmailId) return;
  state.selectedId = gmailId;
  state.loadingEmail = true;
  renderEmailList();
  if (!quiet) showDetailLoading();

  try {
    const email = await api(`/api/emails/${gmailId}`);
    if (state.selectedId !== gmailId) return;
    state.selectedEmail = email;
    state.selectedDraftId = email.draft?.id || null;
    $("#detail-empty").classList.add("hidden");
    $("#email-detail").classList.remove("hidden");
    $("#detail-subject").textContent = email.job_title || email.subject || "(no subject)";
    $("#detail-meta").textContent = `${email.company || email.sender || "Unknown sender"} · ${formatDate(email.received_at)} · ${email.status || "Not Analyzed"}`;
    $("#match-overview").innerHTML = renderMatchOverview(email);
    $("#analysis").innerHTML = formatAnalysisHtml(email);
    renderEmailBody(email);
    renderDraft(email.draft);
    $("#draft-empty").textContent = "No draft yet.";
  } catch (error) {
    if (!quiet) toast(error.message, "error");
  } finally {
    state.loadingEmail = false;
  }
}

function renderEmailBody(email) {
  const container = $("#email-body");
  const toggle = $("#email-view-toggle");
  const hasHtml = Boolean(email.body_html && email.body_html.trim());
  const text = email.body_text || email.snippet || "";

  if (hasHtml) {
    toggle.hidden = false;
    toggle.querySelectorAll(".toggle-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.view === state.emailViewMode);
    });
    if (state.emailViewMode === "html") {
      container.innerHTML = "";
      container.classList.add("is-html");
      container.classList.remove("is-text");
      const frame = document.createElement("iframe");
      frame.className = "email-html-frame";
      frame.title = "Email content";
      frame.setAttribute("sandbox", "");
      frame.setAttribute("referrerpolicy", "no-referrer");
      frame.srcdoc = wrapEmailHtml(email.body_html);
      container.appendChild(frame);
      frame.addEventListener("load", () => resizeEmailFrame(frame));
      return;
    }
  } else {
    toggle.hidden = true;
  }

  container.classList.remove("is-html");
  container.classList.add("is-text");
  container.innerHTML = formatPlainEmailHtml(text);
}

function wrapEmailHtml(html) {
  const safe = String(html || "");
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><base target="_blank" rel="noopener noreferrer">
<style>
  html, body { margin: 0; padding: 0; background: #fff; color: #1f2937; }
  body {
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    font-size: 14px;
    line-height: 1.55;
    padding: 16px 18px;
    word-wrap: break-word;
    overflow-wrap: anywhere;
  }
  img { max-width: 100%; height: auto; }
  a { color: #0369a1; }
  table { max-width: 100%; }
  pre, code { white-space: pre-wrap; word-break: break-word; }
</style></head><body>${safe}</body></html>`;
}

function resizeEmailFrame(frame) {
  try {
    const doc = frame.contentDocument;
    if (!doc) return;
    const height = Math.max(doc.body?.scrollHeight || 0, doc.documentElement?.scrollHeight || 0, 240);
    frame.style.height = `${Math.min(Math.max(height + 8, 240), 900)}px`;
  } catch {
    frame.style.height = "420px";
  }
}

function formatPlainEmailHtml(text) {
  const raw = String(text || "").replace(/\r\n/g, "\n").trim();
  if (!raw) return `<p class="empty compact">No email body available.</p>`;

  const blocks = raw.split(/\n{2,}/);
  return blocks
    .map((block) => {
      const lines = block.split("\n").map((line) => {
        const escaped = escapeHtml(line);
        const linked = linkify(escaped);
        if (/^&gt;/.test(escaped) || /^\|/.test(escaped)) {
          return `<span class="quoted-line">${linked || "&nbsp;"}</span>`;
        }
        return linked || "&nbsp;";
      });
      return `<p>${lines.join("<br>")}</p>`;
    })
    .join("");
}

function linkify(escapedText) {
  return escapedText.replace(
    /(https?:\/\/[^\s<&]+)|(www\.[^\s<&]+)|([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})/g,
    (match) => {
      if (match.includes("@") && !match.startsWith("http") && !match.startsWith("www.")) {
        return `<a href="mailto:${match}">${match}</a>`;
      }
      const href = match.startsWith("http") ? match : `https://${match}`;
      return `<a href="${href}" target="_blank" rel="noopener noreferrer">${match}</a>`;
    },
  );
}

function renderMatchOverview(email) {
  const score = Number(email.match_score || 0);
  const isJob = isJobEmail(email);
  const displayScore = isJob ? `${score}%` : "—";
  const analysis = email.analysis_json || {};
  const jobType = analysis.job_type || email.job_type || "";
  return `
    <div class="score-ring ${scoreClass(score)}">${escapeHtml(displayScore)}</div>
    <div class="match-copy">
      <strong>${escapeHtml(email.company || email.sender || "Unknown sender")}</strong>
      <p>${escapeHtml(email.job_title || email.role || email.subject || "No role detected")}</p>
      <div class="match-tags">
        <span class="badge status ${statusBadgeClass(email.status)}">${escapeHtml(email.status || "Not Analyzed")}</span>
        ${jobType ? `<span class="badge type">${escapeHtml(jobType)}</span>` : ""}
      </div>
    </div>
  `;
}

function formatAnalysisHtml(email) {
  const analysis = email.analysis_json || {};
  if (!analysis || !Object.keys(analysis).length) {
    return `<p class="analysis-empty">No analysis yet. It will run automatically on the next sync.</p>`;
  }
  const confidence = analysis.confidence ?? (Number(analysis.match_score ?? email.match_score ?? 0) / 100);
  const confidencePct = Math.round(Number(confidence) * 100);
  const explanation = analysis.confidence_explanation || analysis.match_explanation || "";
  const summary = analysis.summary || "";
  const action = analysis.recommended_action || analysis.action_needed || "Review";
  const matched = asList(analysis.matched_skills || email.matched_skills);
  const missing = asList(analysis.missing_skills || email.missing_skills);
  const required = asList(analysis.required_skills || email.required_skills);

  return `
    <div class="analysis-grid">
      <div class="analysis-stat">
        <span class="analysis-label">Confidence</span>
        <strong>${confidencePct}%</strong>
      </div>
      <div class="analysis-stat">
        <span class="analysis-label">Action</span>
        <strong>${escapeHtml(action)}</strong>
      </div>
    </div>
    ${summary ? `<p class="analysis-summary">${escapeHtml(summary)}</p>` : ""}
    ${explanation ? `<p class="analysis-explanation">${escapeHtml(explanation)}</p>` : ""}
    ${renderSkillGroup("Matched skills", matched, "matched")}
    ${renderSkillGroup("Missing skills", missing, "missing")}
    ${required.length ? renderSkillGroup("Required skills", required, "required") : ""}
  `;
}

function renderSkillGroup(label, skills, kind) {
  if (!skills.length) {
    return `<div class="skill-group"><span class="analysis-label">${escapeHtml(label)}</span><span class="skill-chip muted">None</span></div>`;
  }
  return `
    <div class="skill-group">
      <span class="analysis-label">${escapeHtml(label)}</span>
      <div class="skill-chips">
        ${skills.map((skill) => `<span class="skill-chip ${kind}">${escapeHtml(skill)}</span>`).join("")}
      </div>
    </div>
  `;
}

function asList(value) {
  if (Array.isArray(value)) return value.map((item) => String(item).trim()).filter(Boolean);
  if (typeof value === "string" && value.trim()) {
    return value.split(",").map((item) => item.trim()).filter(Boolean);
  }
  return [];
}

function renderDraft(draft) {
  const panel = $("#draft-panel");
  const banner = $("#draft-sent-banner");
  const sentTime = $("#draft-sent-time");
  const actions = $("#draft-actions");
  const attachLabel = $("#draft-attach-label");
  const subjectInput = $("#draft-subject");
  const bodyInput = $("#draft-body");
  const attachInput = $("#draft-attach-resume");

  if (!draft) {
    panel.classList.add("hidden");
    panel.classList.remove("is-sent");
    $("#draft-empty").classList.remove("hidden");
    return;
  }

  panel.classList.remove("hidden");
  $("#draft-empty").classList.add("hidden");
  subjectInput.value = draft.subject || "";
  bodyInput.value = draft.body || "";
  attachInput.checked = Boolean(draft.attach_resume);

  const isSent = draft.status === "sent";
  panel.classList.toggle("is-sent", isSent);
  banner.classList.toggle("hidden", !isSent);
  actions.classList.toggle("hidden", isSent);
  attachLabel.classList.toggle("hidden", isSent);
  subjectInput.readOnly = isSent;
  bodyInput.readOnly = isSent;
  attachInput.disabled = isSent;

  if (isSent) {
    sentTime.textContent = formatDateTime(draft.updated_at || draft.created_at);
  } else {
    sentTime.textContent = "";
    resetDraftButtons();
  }
}

function resetDraftButtons() {
  const sendButton = $("#send-button");
  const saveButton = $("#save-draft-button");
  if (sendButton) {
    sendButton.disabled = false;
    sendButton.textContent = "Approve & Send";
  }
  if (saveButton) {
    saveButton.disabled = false;
  }
}

function updateSelectedEmailView(email) {
  if (!email || email.gmail_id !== state.selectedId) return;
  $("#detail-meta").textContent = `${email.company || email.sender || "Unknown sender"} · ${formatDate(email.received_at)} · ${email.status || "Not Analyzed"}`;
  $("#match-overview").innerHTML = renderMatchOverview(email);
}

function clearDetail() {
  state.selectedId = null;
  state.selectedEmail = null;
  state.selectedDraftId = null;
  $("#email-detail").classList.add("hidden");
  $("#detail-empty").classList.remove("hidden");
  renderEmailList();
}

async function reanalyze() {
  if (!state.selectedId || state.busyAction) return;
  setBusy("reanalyze", true);
  try {
    const email = await api(`/api/emails/${state.selectedId}/analyze?force=true`, { method: "POST" });
    state.selectedEmail = { ...state.selectedEmail, ...email };
    $("#analysis").innerHTML = formatAnalysisHtml(email);
    $("#match-overview").innerHTML = renderMatchOverview(email);
    $("#detail-meta").textContent = `${email.company || email.sender || "Unknown sender"} · ${formatDate(email.received_at)} · ${email.status || "Not Analyzed"}`;
    toast("Analysis refreshed.", "success");
    await Promise.all([loadEmails(), loadStatus()]);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy("reanalyze", false);
  }
}

async function generateDraft() {
  if (!state.selectedId || state.busyAction) return;
  setBusy("draft", true);
  try {
    const draft = await api(`/api/emails/${state.selectedId}/draft`, { method: "POST" });
    state.selectedDraftId = draft.id;
    renderDraft(draft);
    toast("Draft generated. Review before sending.", "success");
    $("#draft-body").focus();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy("draft", false);
  }
}

async function saveDraft() {
  if (!state.selectedDraftId || state.busyAction) return;
  setBusy("saveDraft", true);
  try {
    const draft = await api(`/api/drafts/${state.selectedDraftId}`, {
      method: "PUT",
      body: JSON.stringify({
        subject: $("#draft-subject").value,
        body: $("#draft-body").value,
        attach_resume: $("#draft-attach-resume").checked,
      }),
    });
    renderDraft(draft);
    toast("Draft saved.", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy("saveDraft", false);
  }
}

async function sendDraft() {
  if (!state.selectedDraftId || state.busyAction) return;
  if (!confirm("Send this approved reply through Gmail API now?")) return;

  const sendButton = $("#send-button");
  const saveButton = $("#save-draft-button");
  setBusy("send", true);
  try {
    sendButton.disabled = true;
    saveButton.disabled = true;
    sendButton.textContent = "Sending...";
    await api(`/api/drafts/${state.selectedDraftId}`, {
      method: "PUT",
      body: JSON.stringify({
        subject: $("#draft-subject").value,
        body: $("#draft-body").value,
        attach_resume: $("#draft-attach-resume").checked,
      }),
    });
    const result = await api(`/api/drafts/${state.selectedDraftId}/send`, { method: "POST" });
    const draft = result.draft || null;
    renderDraft(draft);

    const email = state.emails.find((item) => item.gmail_id === state.selectedId);
    if (email) {
      email.status = "Relevant";
      email.analysis_status = "Relevant";
      email.latest_draft_status = draft?.status || "sent";
      updateSelectedEmailView(email);
    }

    toast("Reply sent and email kept marked Relevant.", "success");
    renderEmailList();
    await loadEmails();
    if (state.selectedId) {
      await selectEmail(state.selectedId, { quiet: true });
    }
  } catch (error) {
    if (!$("#draft-panel").classList.contains("is-sent")) {
      resetDraftButtons();
    }
    toast(error.message, "error");
  } finally {
    setBusy("send", false);
  }
}

async function archiveEmail() {
  if (!state.selectedId || state.busyAction) return;
  setBusy("archive", true);
  try {
    const archivedId = state.selectedId;
    await api(`/api/emails/${archivedId}/archive`, { method: "POST" });
    state.emails = state.emails.filter((email) => email.gmail_id !== archivedId);
    toast("Email archived in Gmail.", "success");
    clearDetail();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy("archive", false);
  }
}

function isJobEmail(email) {
  const analysis = email.analysis_json || {};
  if (typeof analysis.is_job_email === "boolean") return analysis.is_job_email;
  if (typeof analysis.is_job === "boolean") return analysis.is_job;
  return email.status !== "Not Relevant";
}

function scoreClass(score) {
  if (score >= 80) return "high";
  if (score >= 65) return "good";
  if (score > 0) return "low";
  return "none";
}

function formatDate(value) {
  if (!value) return "No date";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

$("#profile-form").addEventListener("submit", saveProfile);
$("#edit-profile-button").addEventListener("click", openProfileModal);
$("#profile-modal-form").addEventListener("submit", saveProfileModal);
document.querySelectorAll("[data-close='profile-modal']").forEach((el) => {
  el.addEventListener("click", closeProfileModal);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("#profile-modal").classList.contains("hidden")) {
    closeProfileModal();
  }
});
$("#save-agent-prompt").addEventListener("click", saveAgentPrompt);
$("#reset-agent-prompt").addEventListener("click", resetAgentPrompt);
$("#sync-button").addEventListener("click", () => syncInbox());
$("#reset-layout-button").addEventListener("click", resetLayoutWidths);
$("#sort-emails").addEventListener("change", (event) => {
  state.sortBy = event.target.value;
  renderEmailList();
});
$("#reanalyze-button").addEventListener("click", reanalyze);
$("#draft-button").addEventListener("click", generateDraft);
$("#save-draft-button").addEventListener("click", saveDraft);
$("#send-button").addEventListener("click", sendDraft);
$("#archive-button").addEventListener("click", archiveEmail);
$("#email-view-toggle").addEventListener("click", (event) => {
  const button = event.target.closest(".toggle-btn");
  if (!button || !state.selectedEmail) return;
  state.emailViewMode = button.dataset.view || "html";
  renderEmailBody(state.selectedEmail);
});

const LAYOUT_STORAGE_KEY = "email_agent_layout_widths";
const DEFAULT_LAYOUT = { sidebar: 270, inbox: 560 };

function loadLayoutWidths() {
  try {
    const saved = JSON.parse(localStorage.getItem(LAYOUT_STORAGE_KEY) || "{}");
    return {
      sidebar: Number(saved.sidebar) || DEFAULT_LAYOUT.sidebar,
      inbox: Number(saved.inbox) || DEFAULT_LAYOUT.inbox,
    };
  } catch {
    return { ...DEFAULT_LAYOUT };
  }
}

function saveLayoutWidths(widths) {
  localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(widths));
}

function applyLayoutWidths(widths) {
  const sidebarCol = $("#sidebar-col");
  const inboxCol = $("#inbox-col");
  if (sidebarCol) sidebarCol.style.width = `${widths.sidebar}px`;
  if (inboxCol) inboxCol.style.width = `${widths.inbox}px`;
}

function resetLayoutWidths() {
  saveLayoutWidths(DEFAULT_LAYOUT);
  applyLayoutWidths(DEFAULT_LAYOUT);
  toast("Layout reset to default.", "success");
}

function initColumnResize() {
  const widths = loadLayoutWidths();
  applyLayoutWidths(widths);

  let activeTarget = null;
  let startX = 0;
  let startWidth = 0;
  let currentWidths = { ...widths };

  document.querySelectorAll(".resize-handle").forEach((handle) => {
    handle.addEventListener("mousedown", (event) => {
      activeTarget = handle.dataset.target;
      startX = event.clientX;
      startWidth = activeTarget === "sidebar" ? currentWidths.sidebar : currentWidths.inbox;
      handle.classList.add("active");
      document.body.classList.add("is-resizing");
      event.preventDefault();
    });
  });

  window.addEventListener("mousemove", (event) => {
    if (!activeTarget) return;
    const delta = event.clientX - startX;
    const workspace = $("#workspace");
    const workspaceWidth = workspace?.clientWidth || window.innerWidth;
    const handleSpace = 20;
    const detailMin = 280;

    if (activeTarget === "sidebar") {
      const maxSidebar = Math.max(220, workspaceWidth - currentWidths.inbox - detailMin - handleSpace);
      currentWidths.sidebar = Math.max(220, Math.min(maxSidebar, startWidth + delta));
    } else {
      const maxInbox = Math.max(360, workspaceWidth - currentWidths.sidebar - detailMin - handleSpace);
      currentWidths.inbox = Math.max(360, Math.min(maxInbox, startWidth + delta));
    }
    applyLayoutWidths(currentWidths);
  });

  window.addEventListener("mouseup", () => {
    if (!activeTarget) return;
    saveLayoutWidths(currentWidths);
    activeTarget = null;
    document.body.classList.remove("is-resizing");
    document.querySelectorAll(".resize-handle.active").forEach((handle) => handle.classList.remove("active"));
  });
}

async function bootstrap() {
  try {
    initColumnResize();
    await Promise.all([loadStatus(), loadProfile(), loadAgentPrompt(), loadEmails()]);
    const progress = await api("/api/sync/progress").catch(() => null);
    if (progress) renderSyncProgress(progress);
    await syncInbox({ silent: true });
  } catch (error) {
    toast(error.message, "error");
  }
}

bootstrap();
