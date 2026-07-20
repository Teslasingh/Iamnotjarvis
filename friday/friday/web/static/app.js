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
const clearTasksBtn = $("#btn-clear-tasks");
const jobCountBadge = $("#job-count-badge");
const planCard = $("#plan-card");
const taskList = $("#task-list");
const autonomySummary = $("#autonomy-summary");

const pageClientId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
const jobs = new Map();
const tasks = new Map();
const expandedJobIds = new Set();
const dismissedTaskIds = new Set();
const pendingFiles = [];
const LARGE_FILE_WARN_BYTES = 100 * 1024 * 1024;
const ACTIVE_TASK_STATUSES = new Set(["pending", "running"]);

let jobsPollTimer = null;
let tasksPollTimer = null;
let liveOutputContainer = null;
let currentPlan = null;

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

async function hydrateTasksFromServer() {
  try {
    const r = await fetch("/api/tasks", { credentials: "same-origin" });
    if (!r.ok) return;
    const j = await r.json();
    tasks.clear();
    for (const task of j.tasks || []) {
      if (task && task.id && !dismissedTaskIds.has(task.id)) tasks.set(task.id, task);
    }
    syncPlanWithTasks();
    renderTasks();
    renderPlan();
  } catch {
    /* ignore */
  }
}

function startJobsPolling() {
  if (jobsPollTimer) clearInterval(jobsPollTimer);
  jobsPollTimer = setInterval(hydrateJobsFromServer, 8000);
}

function startTasksPolling() {
  if (tasksPollTimer) clearInterval(tasksPollTimer);
  tasksPollTimer = setInterval(hydrateTasksFromServer, 8000);
}

function connectEvents() {
  const eventsWs = new WebSocket(wsUrl("/ws/events"));
  eventsWs.onopen = () => {
    setStatus("Connected");
    hydrateJobsFromServer();
    hydrateTasksFromServer();
    startJobsPolling();
    startTasksPolling();
  };
  eventsWs.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      appendLog(data);
      updateJobFromEvent(data);
      updateTaskFromEvent(data);
      updatePlanFromEvent(data);
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
      if (data.type === "watchdog_job_stopped") {
        setStatus("Watchdog stopped job");
        hydrateJobsFromServer();
      }
      if (data.type === "autonomy_loop_broken") {
        setStatus("Loop broken — stopped");
        appendAgent(
          `Stopped autonomous loop (${data.reason || "unknown"}).${
            data.stall_count ? ` Repeated tool stalls: ${data.stall_count}.` : ""
          }`,
          [],
          { autonomous: true, source: "watchdog" },
        );
      }
      if (data.type === "autonomy_aborted") {
        for (const taskId of tasks.keys()) dismissedTaskIds.add(taskId);
        tasks.clear();
        currentPlan = null;
        renderTasks();
        renderPlan();
        hydrateJobsFromServer();
        setStatus("Idle");
      }
      if (data.type === "autonomy_task_started") {
        const label = data.source || "task";
        setStatus(data.proactive ? `Autonomous (${label})…` : "Running…");
      }
      if (data.type === "autonomy_continuation") {
        setStatus(`Continuing (${data.continuation_index || "?"})…`);
      }
      if (data.type === "chat_complete" && data.reply) {
        const showReply =
          data.client_id === pageClientId || data.autonomous || !data.client_id;
        if (showReply) {
          if (liveOutputContainer && liveOutputContainer.isConnected) {
            liveOutputContainer.remove();
          }
          liveOutputContainer = null;
          appendAgent(data.reply, data.outputs || [], {
            autonomous: !!data.autonomous,
            source: data.task_source,
          });
          setStatus("Idle");
        }
      }
      if (data.type === "output_ready" && data.client_id === pageClientId && data.output) {
        const block = ensureLiveOutputContainer();
        renderOutputItem(block, data.output);
        chatStream.scrollTop = chatStream.scrollHeight;
      }
      if ((data.type === "chat_error" && (data.client_id === pageClientId || data.autonomous)) || data.type === "llm_error") {
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
    if (tasksPollTimer) {
      clearInterval(tasksPollTimer);
      tasksPollTimer = null;
    }
    setStatus("Disconnected; reconnecting…");
    setTimeout(connectEvents, 1500);
  };
}

async function appendAgent(text, outputs, meta = {}) {
  const div = document.createElement("div");
  div.className = "msg agent";
  if (meta.autonomous) {
    div.classList.add("msg-autonomous");
    const tag = document.createElement("div");
    tag.className = "msg-tag";
    const source = meta.source || "autonomous";
    tag.textContent =
      source === "job_followup"
        ? "Job follow-up"
        : source === "continuation"
          ? "Continued"
          : "Autonomous";
    div.appendChild(tag);
  }
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

function normalizeStatus(status) {
  return String(status || "pending").toLowerCase();
}

function activeTasks() {
  return Array.from(tasks.values()).filter((task) => ACTIVE_TASK_STATUSES.has(normalizeStatus(task.status)));
}

function syncPlanWithTasks() {
  const active = activeTasks();
  const activeWithPlan = active
    .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))
    .find((task) => task.metadata?.plan);
  if (activeWithPlan?.metadata?.plan) {
    currentPlan = activeWithPlan.metadata.plan;
  } else if (!active.length) {
    currentPlan = null;
  }
}

function updateTaskBadge() {
  if (!autonomySummary) return;
  const values = Array.from(tasks.values());
  const active = activeTasks().length;
  if (!values.length) {
    autonomySummary.classList.add("hidden");
    autonomySummary.textContent = "";
    return;
  }
  autonomySummary.classList.remove("hidden");
  autonomySummary.textContent = active ? `${active} active` : `${values.length} task${values.length === 1 ? "" : "s"}`;
}

function renderPlan() {
  if (!planCard) return;
  const plan = currentPlan;
  if (!plan || !Array.isArray(plan.steps) || !plan.steps.length) {
    const active = activeTasks();
    planCard.className = "plan-card empty";
    planCard.textContent = active.length ? "Mission active" : "No active mission";
    return;
  }
  planCard.className = "plan-card";
  planCard.innerHTML = "";
  const title = document.createElement("div");
  title.className = "plan-title";
  title.textContent = plan.summary || "Mission plan";
  planCard.appendChild(title);
  const steps = document.createElement("div");
  steps.className = "plan-steps";
  for (const step of plan.steps) {
    const row = document.createElement("div");
    const status = normalizeStatus(step.status);
    row.className = `plan-step plan-step--${status}`;
    const badge = document.createElement("span");
    badge.className = "plan-step-status";
    badge.textContent = status;
    const body = document.createElement("div");
    body.className = "plan-step-body";
    const head = document.createElement("div");
    head.className = "plan-step-head";
    head.textContent = `${step.id || "step"} · ${step.role || "execute"}`;
    const goal = document.createElement("div");
    goal.className = "plan-step-goal";
    goal.textContent = step.goal || "";
    body.append(head, goal);
    if (Array.isArray(step.depends_on) && step.depends_on.length) {
      const deps = document.createElement("div");
      deps.className = "plan-step-deps";
      deps.textContent = `after ${step.depends_on.join(", ")}`;
      body.appendChild(deps);
    }
    row.append(badge, body);
    steps.appendChild(row);
  }
  planCard.appendChild(steps);
}

function renderTasks() {
  updateTaskBadge();
  if (!taskList) return;
  taskList.innerHTML = "";
  const sorted = Array.from(tasks.values()).sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  for (const task of sorted.slice(0, 8)) {
    const li = document.createElement("li");
    li.className = "task-card";
    const status = normalizeStatus(task.status);
    if (status === "running") li.classList.add("task-card--running");
    const top = document.createElement("div");
    top.className = "task-card-top";
    const badge = document.createElement("span");
    badge.className = `job-status ${status}`;
    badge.textContent = status;
    const label = document.createElement("span");
    label.className = "task-label";
    label.title = task.message || "";
    label.textContent = task.message || task.id;
    top.append(badge, label);
    const meta = document.createElement("div");
    meta.className = "job-meta";
    const planStatus = task.metadata?.plan_status ? ` · plan: ${task.metadata.plan_status}` : "";
    meta.textContent = `${task.source || "task"} · ${(task.id || "").slice(0, 8)}…${planStatus}`;
    li.append(top, meta);
    taskList.appendChild(li);
  }
}

function setPlanStepStatus(stepId, status) {
  if (!currentPlan || !Array.isArray(currentPlan.steps)) return;
  for (const step of currentPlan.steps) {
    if (step.id === stepId) {
      step.status = status;
      return;
    }
  }
}

function updatePlanFromEvent(data) {
  if (data.client_id && data.client_id !== pageClientId && !data.autonomous) return;
  if (data.type === "task_analyzed" && data.plan) {
    currentPlan = data.plan;
    renderPlan();
    return;
  }
  if (data.type === "plan_created" && data.plan) {
    currentPlan = data.plan;
    renderPlan();
    return;
  }
  if (data.type === "plan_step_started") {
    setPlanStepStatus(data.step_id, "running");
    renderPlan();
  } else if (data.type === "plan_step_retry") {
    setPlanStepStatus(data.step_id, "retrying");
    renderPlan();
  } else if (data.type === "plan_step_complete") {
    setPlanStepStatus(data.step_id, "done");
    renderPlan();
  } else if (data.type === "plan_step_failed") {
    setPlanStepStatus(data.step_id, "failed");
    renderPlan();
  }
}

function updateTaskFromEvent(data) {
  if (data.task_id && dismissedTaskIds.has(data.task_id)) return;
  if (data.type === "autonomy_task_enqueued" && data.task_id) {
    const existing = tasks.get(data.task_id) || {};
    tasks.set(data.task_id, {
      ...existing,
      id: data.task_id,
      status: "pending",
      source: data.source || existing.source || "user",
      message: data.preview || existing.message || "",
      client_id: data.client_id || existing.client_id,
      created_at: existing.created_at || Date.now() / 1000,
      metadata: existing.metadata || {},
    });
    renderTasks();
  } else if (data.type === "autonomy_task_started" && data.task_id) {
    const existing = tasks.get(data.task_id) || {};
    tasks.set(data.task_id, {
      ...existing,
      id: data.task_id,
      status: "running",
      source: data.source || existing.source || "task",
      client_id: data.client_id || existing.client_id,
      created_at: existing.created_at || Date.now() / 1000,
      metadata: existing.metadata || {},
    });
    renderTasks();
  } else if (data.type === "autonomy_task_updated" && data.task_id) {
    const existing = tasks.get(data.task_id) || {};
    tasks.set(data.task_id, {
      ...existing,
      id: data.task_id,
      metadata: data.metadata || existing.metadata || {},
    });
    if (data.metadata?.plan) currentPlan = data.metadata.plan;
    renderTasks();
    renderPlan();
  } else if (data.type === "chat_complete" && data.task_id) {
    const existing = tasks.get(data.task_id) || {};
    tasks.set(data.task_id, { ...existing, id: data.task_id, status: "done" });
    syncPlanWithTasks();
    renderTasks();
    renderPlan();
  } else if (data.type === "chat_error" && data.task_id) {
    const existing = tasks.get(data.task_id) || {};
    tasks.set(data.task_id, { ...existing, id: data.task_id, status: "failed" });
    syncPlanWithTasks();
    renderTasks();
    renderPlan();
  }
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

function updateJobCountBadge() {
  if (!jobCountBadge) return;
  const running = Array.from(jobs.values()).filter((job) => {
    const status = job.status || "running";
    return status === "running" || status === "stopping";
  }).length;
  const total = jobs.size;
  if (!total) {
    jobCountBadge.classList.add("hidden");
    jobCountBadge.textContent = "";
    return;
  }
  jobCountBadge.classList.remove("hidden");
  jobCountBadge.textContent = running ? `${running} running` : `${total} job${total === 1 ? "" : "s"}`;
}

function renderJobs() {
  updateJobCountBadge();
  jobList.innerHTML = "";
  if (!jobs.size) {
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
    } else {
      const data = await resp.json().catch(() => ({}));
      if (data.queued) {
        setStatus("Queued…");
      }
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
  clearJobsBtn.addEventListener("click", async () => {
    try {
      await fetch("/api/jobs/clear", { method: "POST", credentials: "same-origin" });
    } catch {}
    jobs.clear();
    expandedJobIds.clear();
    renderJobs();
  });
}

if (clearTasksBtn) {
  clearTasksBtn.addEventListener("click", async () => {
    const taskIdsToDismiss = Array.from(tasks.keys());
    try {
      const resp = await fetch("/api/tasks/clear?include_active=true", {
        method: "POST",
        credentials: "same-origin",
      });
      if (!resp.ok) return;
      for (const taskId of taskIdsToDismiss) dismissedTaskIds.add(taskId);
      tasks.clear();
      currentPlan = null;
      renderTasks();
      renderPlan();
      setStatus("Idle");
      await hydrateTasksFromServer();
      await hydrateJobsFromServer();
    } catch {}
  });
}

connectEvents();
renderJobs();
renderTasks();
renderPlan();
