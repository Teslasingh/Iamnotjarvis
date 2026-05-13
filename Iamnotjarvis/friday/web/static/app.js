const $ = (sel) => document.querySelector(sel);

const logEl = $("#event-log");
const chatStream = $("#chat-stream");
const jobList = $("#job-list");
const sendBtn = $("#btn-send");
const statusEl = $("#status");
const clearLogBtn = $("#btn-clear-log");
const jobs = new Map();
let cleanupJobsTimer = null;
let acceptingJobEvents = false;

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

function connectEvents() {
  const eventsWs = new WebSocket(wsUrl("/ws/events"));
  eventsWs.onopen = () => setStatus("Connected");
  eventsWs.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      appendLog(data);
      if (data.type === "llm_turn_start") {
        startJobRun();
      }
      updateJobFromEvent(data);
      if (data.type === "chat_complete" && data.reply) {
        appendAgent(data.reply);
        setStatus("Idle");
        scheduleJobCleanup();
      }
      if (data.type === "chat_error" || data.type === "llm_error") {
        setStatus("Error");
        scheduleJobCleanup();
      }
      if (data.type === "assistant_delta" && data.text) {
        setStatus("Thinking…");
      }
    } catch {
      appendLog({ raw: ev.data });
    }
  };
  eventsWs.onclose = () => {
    setStatus("Disconnected; reconnecting…");
    setTimeout(connectEvents, 1500);
  };
}

function appendAgent(text) {
  const div = document.createElement("div");
  div.className = "msg agent";
  div.textContent = text;
  chatStream.appendChild(div);
  chatStream.scrollTop = chatStream.scrollHeight;
}

function appendUser(text) {
  const div = document.createElement("div");
  div.className = "msg user";
  div.textContent = "> " + text;
  chatStream.appendChild(div);
  chatStream.scrollTop = chatStream.scrollHeight;
}

async function refreshJobs() {
  try {
    const r = await fetch("/api/jobs");
    const j = await r.json();
    if (!acceptingJobEvents) return;
    for (const job of j.jobs || []) {
      jobs.set(job.id, job);
    }
    renderJobs();
  } catch {
    /* ignore */
  }
}

function updateJobFromEvent(ev) {
  const id = ev.job_id;
  if (!id) return;
  acceptingJobEvents = true;
  const existing = jobs.get(id) || { id, command: ev.command || "", cwd: ev.cwd || "", status: "running" };
  if (ev.type === "job_created") {
    existing.command = ev.command || existing.command;
    existing.cwd = ev.cwd || existing.cwd;
    existing.status = "running";
    existing.return_code = null;
  } else if (ev.type === "job_output") {
    existing.last_output = ((existing.last_output || "") + ev.text).slice(-500);
  } else if (ev.type === "job_exit") {
    existing.status = ev.return_code === 0 ? "done" : "error";
    existing.return_code = ev.return_code;
  } else if (ev.type === "job_timeout") {
    existing.status = "timeout";
    existing.return_code = -1;
  } else if (ev.type === "job_error") {
    existing.status = "error";
    existing.last_output = ev.error || existing.last_output || "";
  } else {
    return;
  }
  jobs.set(id, existing);
  renderJobs();
}

function startJobRun() {
  if (cleanupJobsTimer) {
    clearTimeout(cleanupJobsTimer);
    cleanupJobsTimer = null;
  }
  acceptingJobEvents = true;
  jobs.clear();
  renderJobs();
}

function scheduleJobCleanup() {
  if (cleanupJobsTimer) clearTimeout(cleanupJobsTimer);
  cleanupJobsTimer = setTimeout(() => {
    acceptingJobEvents = false;
    jobs.clear();
    renderJobs();
  }, 2500);
}

function renderJobs() {
  jobList.innerHTML = "";
  if (!jobs.size) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No shell jobs yet";
    jobList.appendChild(li);
    return;
  }
  for (const job of Array.from(jobs.values()).slice(-30).reverse()) {
    const li = document.createElement("li");
    const status = document.createElement("span");
    status.className = `job-status ${job.status || "running"}`;
    status.textContent = job.status || "running";
    const command = document.createElement("span");
    command.className = "job-command";
    command.textContent = job.command || job.id;
    li.append(status, command);
    if (job.last_output) {
      const output = document.createElement("small");
      output.textContent = job.last_output.trim();
      li.appendChild(output);
    }
    jobList.appendChild(li);
  }
}

$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#chat-input");
  const msg = input.value.trim();
  if (!msg) return;
  appendUser(msg);
  input.value = "";
  startJobRun();
  sendBtn.disabled = true;
  sendBtn.textContent = "Sending…";
  setStatus("Running…");
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg }),
    });
    if (!resp.ok) {
      setStatus("Error");
      appendAgent(`Request failed: ${resp.status}`);
      scheduleJobCleanup();
    }
  } catch (exc) {
    setStatus("Error");
    appendAgent(`Request failed: ${exc}`);
    scheduleJobCleanup();
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Send";
  }
});

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

connectEvents();
renderJobs();
