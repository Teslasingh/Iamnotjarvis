const $ = (sel) => document.querySelector(sel);

const logEl = $("#event-log");
const chatStream = $("#chat-stream");
const jobList = $("#job-list");
const sendBtn = $("#btn-send");
const attachBtn = $("#btn-attach");
const fileInput = $("#file-input");
const attachmentChips = $("#attachment-chips");
const statusEl = $("#status");
const clearLogBtn = $("#btn-clear-log");
const clearJobsBtn = $("#btn-clear-jobs");

const pageClientId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
const jobs = new Map();
const expandedJobIds = new Set();
const pendingFiles = [];
const LARGE_FILE_WARN_BYTES = 100 * 1024 * 1024;

let jobsPollTimer = null;
let liveOutputContainer = null;

chatStream.textContent = "";

function formatBytes(n) {
  if (!Number.isFinite(n)) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function renderAttachmentChips() {
  if (!attachmentChips) return;
  attachmentChips.innerHTML = "";
  if (!pendingFiles.length) {
    attachmentChips.classList.add("hidden");
    return;
  }
  attachmentChips.classList.remove("hidden");
  for (let i = 0; i < pendingFiles.length; i += 1) {
    const file = pendingFiles[i];
    const chip = document.createElement("span");
    chip.className = "file-chip";
    const name = document.createElement("strong");
    name.textContent = file.name;
    name.title = file.name;
    const size = document.createElement("span");
    size.textContent = formatBytes(file.size);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.title = "Remove";
    remove.addEventListener("click", () => {
      pendingFiles.splice(i, 1);
      renderAttachmentChips();
    });
    chip.append(name, size, remove);
    attachmentChips.appendChild(chip);
  }
}

async function uploadPendingFiles() {
  if (!pendingFiles.length) return [];
  const form = new FormData();
  for (const file of pendingFiles) form.append("files", file, file.name);
  const resp = await fetch("/api/upload", {
    method: "POST",
    credentials: "same-origin",
    body: form,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`Upload failed (${resp.status})${text ? `: ${text}` : ""}`);
  }
  const data = await resp.json();
  pendingFiles.length = 0;
  renderAttachmentChips();
  return (data.files || []).map((f) => f.path).filter(Boolean);
}

function attachmentChip(name, sizeText) {
  const chip = document.createElement("span");
  chip.className = "file-chip";
  const label = document.createElement("strong");
  label.textContent = name;
  label.title = name;
  chip.append(label);
  if (sizeText) {
    const size = document.createElement("span");
    size.textContent = sizeText;
    chip.append(size);
  }
  return chip;
}

async function renderOutputItem(container, output) {
  if (!output || (!output.url && !output.path)) return;
  const item = document.createElement("div");
  item.className = "output-item";

  const label = document.createElement("div");
  label.className = "output-label";
  label.textContent = output.name || "Output file";
  item.appendChild(label);

  const kind = output.preview_kind || "download";
  const downloadUrl =
    output.url || (output.path ? `/api/files/by-path?path=${encodeURIComponent(output.path)}` : "");
  const inlineUrl = downloadUrl ? `${downloadUrl}${downloadUrl.includes("?") ? "&" : "?"}inline=true` : "";

  if (kind === "image" && inlineUrl) {
    const img = document.createElement("img");
    img.className = "output-image";
    img.src = inlineUrl;
    img.alt = output.name || "output";
    img.loading = "lazy";
    item.appendChild(img);
  } else if (kind === "text" && inlineUrl) {
    try {
      const resp = await fetch(inlineUrl, { credentials: "same-origin" });
      if (resp.ok) {
        const text = await resp.text();
        const pre = document.createElement("pre");
        pre.className = "output-pre";
        pre.textContent = text.length > 65536 ? `${text.slice(0, 65536)}\n…` : text;
        item.appendChild(pre);
      }
    } catch {
      /* fall through to download link */
    }
  }

  if (downloadUrl) {
    const link = document.createElement("a");
    link.className = "output-download";
    link.href = downloadUrl;
    link.textContent = `Download ${output.name || "file"}`;
    link.download = output.name || "";
    item.appendChild(link);
  }
  container.appendChild(item);
}

async function appendOutputs(container, outputs) {
  if (!outputs || !outputs.length) return;
  const block = document.createElement("div");
  block.className = "output-block";
  for (const output of outputs) {
    await renderOutputItem(block, output);
  }
  container.appendChild(block);
}

function ensureLiveOutputContainer() {
  if (liveOutputContainer && liveOutputContainer.isConnected) {
    return liveOutputContainer.querySelector(".output-block");
  }
  liveOutputContainer = document.createElement("div");
  liveOutputContainer.className = "msg agent msg-live-outputs";
  const hint = document.createElement("div");
  hint.className = "output-label";
  hint.textContent = "Outputs";
  liveOutputContainer.appendChild(hint);
  const block = document.createElement("div");
  block.className = "output-block";
  liveOutputContainer.appendChild(block);
  chatStream.appendChild(liveOutputContainer);
  chatStream.scrollTop = chatStream.scrollHeight;
  return block;
}

function appendLog(obj) {
  const line = JSON.stringify(obj);
  logEl.textContent += line + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(text) {
  if (!statusEl) return;
  statusEl.textContent = text;
}

function wsUrl(path) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}${path}`;
}

async function hydrateJobsFromServer() {
  try {
    const r = await fetch("/api/jobs");
    if (!r.ok) return;
    const j = await r.json();
    for (const job of j.jobs || []) {
      const prev = jobs.get(job.id) || {};
      jobs.set(job.id, { ...prev, ...job });
    }
    renderJobs();
  } catch {
    /* ignore */
  }
}

function startJobsPolling() {
  if (jobsPollTimer) clearInterval(jobsPollTimer);
  jobsPollTimer = setInterval(hydrateJobsFromServer, 8000);
}

function connectEvents() {
  const eventsWs = new WebSocket(wsUrl("/ws/events"));
  eventsWs.onopen = () => {
    setStatus("Connected");
    hydrateJobsFromServer();
    startJobsPolling();
  };
  eventsWs.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      appendLog(data);
      updateJobFromEvent(data);
      if (data.type === "task_analyzed" && data.client_id === pageClientId) {
        setStatus("Analyzing task…");
      }
      if (data.type === "query_expanded" && data.client_id === pageClientId) {
        setStatus("Query expanded…");
      }
      if (data.type === "orchestration_start" && data.client_id === pageClientId) {
        setStatus("Multi-agent workflow…");
      }
      if (data.type === "subagent_start" && data.client_id === pageClientId) {
        setStatus(`Sub-agent: ${data.role || "working"}…`);
      }
      if (data.type === "subagent_complete" && data.client_id === pageClientId) {
        setStatus(data.failed ? "Sub-agent retrying…" : "Sub-agent done…");
      }
      if (data.type === "orchestration_complete" && data.client_id === pageClientId) {
        setStatus("Synthesizing reply…");
      }
      if (data.type === "chat_complete" && data.reply && data.client_id === pageClientId) {
        if (liveOutputContainer && liveOutputContainer.isConnected) {
          liveOutputContainer.remove();
        }
        liveOutputContainer = null;
        appendAgent(data.reply, data.outputs || []);
        setStatus("Idle");
      }
      if (data.type === "output_ready" && data.client_id === pageClientId && data.output) {
        const block = ensureLiveOutputContainer();
        renderOutputItem(block, data.output);
        chatStream.scrollTop = chatStream.scrollHeight;
      }
      if ((data.type === "chat_error" && data.client_id === pageClientId) || data.type === "llm_error") {
        setStatus("Error");
      }
      if (data.type === "assistant_delta" && data.text) {
        setStatus("Thinking…");
      }
    } catch {
      appendLog({ raw: ev.data });
    }
  };
  eventsWs.onclose = () => {
    if (jobsPollTimer) {
      clearInterval(jobsPollTimer);
      jobsPollTimer = null;
    }
    setStatus("Disconnected; reconnecting…");
    setTimeout(connectEvents, 1500);
  };
}

async function appendAgent(text, outputs) {
  const div = document.createElement("div");
  div.className = "msg agent";
  const body = document.createElement("div");
  body.className = "msg-text";
  body.textContent = text;
  div.appendChild(body);
  chatStream.appendChild(div);
  await appendOutputs(div, outputs);
  chatStream.scrollTop = chatStream.scrollHeight;
}

function appendUser(text, attachments) {
  const div = document.createElement("div");
  div.className = "msg user";
  if (attachments && attachments.length) {
    const row = document.createElement("div");
    row.className = "msg-attachments";
    for (const file of attachments) {
      row.appendChild(attachmentChip(file.name || file.path || "file", formatBytes(file.size)));
    }
    div.appendChild(row);
  }
  const body = document.createElement("div");
  body.className = "msg-text";
  body.textContent = "> " + text;
  div.appendChild(body);
  chatStream.appendChild(div);
  chatStream.scrollTop = chatStream.scrollHeight;
}

function updateJobFromEvent(ev) {
  const id = ev.job_id;
  if (!id) return;

  if (ev.type === "job_stop_requested") {
    const prev = jobs.get(id) || {};
    jobs.set(id, {
      ...prev,
      id,
      command: ev.command || prev.command || "",
      stopping: true,
    });
    renderJobs();
    return;
  }

  if (ev.type === "job_finished") {
    const existing = jobs.get(id) || {};
    const out = ev.stdout_tail || existing.stdout_tail || "";
    const err = ev.stderr_tail || existing.stderr_tail || "";
    jobs.set(id, {
      ...existing,
      id,
      command: ev.command || existing.command || "",
      cwd: ev.cwd || existing.cwd || "",
      status: ev.status || existing.status,
      return_code: ev.return_code ?? existing.return_code,
      report: ev.report || existing.report,
      outcome: ev.outcome || existing.outcome,
      stdout_tail: out,
      stderr_tail: err,
      last_output: (out + err).slice(-500),
      stopping: false,
    });
    renderJobs();
    return;
  }

  const existing = jobs.get(id) || { id, command: ev.command || "", cwd: ev.cwd || "", status: "running" };
  if (ev.type === "job_created") {
    existing.command = ev.command || existing.command;
    existing.cwd = ev.cwd || existing.cwd;
    existing.status = "running";
    existing.return_code = null;
    existing.report = undefined;
    if (existing.created_at == null) existing.created_at = Date.now() / 1000;
  } else if (ev.type === "job_output") {
    const stream = ev.stream === "stderr" ? "stderr" : "stdout";
    const key = stream === "stderr" ? "stderr_tail" : "stdout_tail";
    const chunk = ev.text || "";
    existing[key] = ((existing[key] || "") + chunk).slice(-12000);
    existing.last_output = ((existing.last_output || "") + chunk).slice(-800);
  } else if (ev.type === "job_exit") {
    existing.status = ev.return_code === 0 ? "done" : "error";
    existing.return_code = ev.return_code;
  } else if (ev.type === "job_timeout") {
    existing.status = "timeout";
    existing.return_code = -1;
  } else if (ev.type === "job_error") {
    existing.status = "error";
    existing.last_output = ev.error || existing.last_output || "";
    existing.stderr_tail = ((existing.stderr_tail || "") + (ev.error || "")).slice(-12000);
  } else {
    return;
  }
  jobs.set(id, existing);
  renderJobs();
}

function toggleJobExpand(jobId) {
  if (expandedJobIds.has(jobId)) expandedJobIds.delete(jobId);
  else expandedJobIds.add(jobId);
  renderJobs();
}

function renderJobs() {
  jobList.innerHTML = "";
  if (!jobs.size) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No shell jobs yet — start one from chat; background jobs persist across messages.";
    jobList.appendChild(li);
    return;
  }
  const sorted = Array.from(jobs.values()).sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  for (const job of sorted.slice(0, 40)) {
    const li = document.createElement("li");
    li.className = "job-card";
    const rawStatus = job.status || "running";
    const running = rawStatus === "running";
    const displayLabel = running && job.stopping ? "stopping" : rawStatus;
    if (running) li.classList.add("job-card--running");

    const top = document.createElement("div");
    top.className = "job-card-top";

    const status = document.createElement("span");
    status.className = `job-status ${displayLabel}`;
    if (running) status.classList.add("job-status--pulse");
    status.textContent = displayLabel;

    const cmd = document.createElement("span");
    cmd.className = "job-command";
    cmd.title = job.command || "";
    cmd.textContent = job.command || job.id;

    const actions = document.createElement("div");
    actions.className = "job-actions";

    const expandBtn = document.createElement("button");
    expandBtn.type = "button";
    expandBtn.className = "btn ghost small-btn job-expand-btn";
    expandBtn.textContent = expandedJobIds.has(job.id) ? "Hide" : "Details";
    expandBtn.addEventListener("click", () => toggleJobExpand(job.id));

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "btn ghost small-btn";
    copyBtn.textContent = "Copy ID";
    copyBtn.addEventListener("click", () => {
      navigator.clipboard?.writeText(job.id).catch(() => {});
    });

    if (running) {
      const stopBtn = document.createElement("button");
      stopBtn.type = "button";
      stopBtn.className = "btn ghost small-btn btn-stop-job";
      stopBtn.textContent = job.stopping ? "Stop…" : "Stop";
      stopBtn.disabled = !!job.stopping;
      stopBtn.addEventListener("click", async () => {
        const row = jobs.get(job.id);
        if (!row || row.status !== "running") return;
        row.stopping = true;
        jobs.set(job.id, row);
        renderJobs();
        try {
          const r = await fetch(`/api/jobs/${encodeURIComponent(job.id)}/stop`, {
            method: "POST",
            credentials: "same-origin",
          });
          if (!r.ok) {
            row.stopping = false;
            jobs.set(job.id, row);
            renderJobs();
          }
        } catch {
          row.stopping = false;
          jobs.set(job.id, row);
          renderJobs();
        }
      });
      actions.append(expandBtn, stopBtn, copyBtn);
    } else {
      actions.append(expandBtn, copyBtn);
    }
    top.append(status, cmd, actions);

    const meta = document.createElement("div");
    meta.className = "job-meta";
    const rc = job.return_code;
    const rcText = rc === null || rc === undefined ? "—" : String(rc);
    meta.textContent = `cwd: ${job.cwd || "—"} · exit: ${rcText} · ${(job.id || "").slice(0, 8)}…`;

    li.append(top, meta);

    if (job.report) {
      const rep = document.createElement("div");
      rep.className = "job-report";
      rep.textContent = job.report;
      li.appendChild(rep);
    }

    if (job.last_output && !expandedJobIds.has(job.id)) {
      const peek = document.createElement("div");
      peek.className = "job-peek";
      peek.textContent = job.last_output.trim();
      li.appendChild(peek);
    }

    if (expandedJobIds.has(job.id)) {
      const detail = document.createElement("div");
      detail.className = "job-detail";

      if (job.outcome && job.outcome.summary) {
        const sum = document.createElement("div");
        sum.className = "job-detail-label job-outcome";
        sum.textContent = `outcome · ${job.outcome.summary}`;
        detail.appendChild(sum);
      }

      const outLabel = document.createElement("div");
      outLabel.className = "job-detail-label";
      outLabel.textContent = "stdout";
      const outPre = document.createElement("pre");
      outPre.className = "job-output";
      outPre.textContent = (job.stdout_tail || job.stdout_preview || "").trim() || "(empty)";

      const errLabel = document.createElement("div");
      errLabel.className = "job-detail-label";
      errLabel.textContent = "stderr";
      const errPre = document.createElement("pre");
      errPre.className = "job-output job-output--err";
      errPre.textContent = (job.stderr_tail || job.stderr_preview || "").trim() || "(empty)";

      detail.append(outLabel, outPre, errLabel, errPre);
      li.appendChild(detail);
    }

    jobList.appendChild(li);
  }
}

$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#chat-input");
  const msg = input.value.trim();
  if (!msg) return;

  const localAttachments = pendingFiles.map((f) => ({ name: f.name, size: f.size }));
  appendUser(msg, localAttachments);
  input.value = "";
  sendBtn.disabled = true;
  if (attachBtn) attachBtn.disabled = true;
  sendBtn.textContent = "Sending…";
  setStatus("Running…");
  liveOutputContainer = null;

  try {
    for (const file of pendingFiles) {
      if (file.size > LARGE_FILE_WARN_BYTES) {
        appendAgent(`Note: ${file.name} is ${formatBytes(file.size)} — upload may take a while.`);
      }
    }
    if (pendingFiles.length) {
      setStatus("Uploading…");
      sendBtn.textContent = "Uploading…";
    }
    const attachmentPaths = await uploadPendingFiles();
    setStatus("Running…");
    sendBtn.textContent = "Sending…";
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({
        message: msg,
        client_id: pageClientId,
        attachments: attachmentPaths.length ? attachmentPaths : undefined,
      }),
    });
    if (!resp.ok) {
      setStatus("Error");
      appendAgent(`Request failed: ${resp.status}`);
    }
  } catch (exc) {
    setStatus("Error");
    appendAgent(`Request failed: ${exc}`);
  } finally {
    sendBtn.disabled = false;
    if (attachBtn) attachBtn.disabled = false;
    sendBtn.textContent = "Send";
  }
});

if (attachBtn && fileInput) {
  attachBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    for (const file of fileInput.files || []) pendingFiles.push(file);
    fileInput.value = "";
    renderAttachmentChips();
  });
}

$("#chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#chat-form").requestSubmit();
  }
});

if (clearLogBtn) {
  clearLogBtn.addEventListener("click", () => {
    logEl.textContent = "";
  });
}

if (clearJobsBtn) {
  clearJobsBtn.addEventListener("click", () => {
    jobs.clear();
    expandedJobIds.clear();
    renderJobs();
  });
}

connectEvents();
renderJobs();
