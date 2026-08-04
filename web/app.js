const state = { messages: [], metadata: null, filter: "all", busy: false };
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const escapeHtml = (value = "") => String(value)
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;").replaceAll("'", "&#039;");

const number = (value) => new Intl.NumberFormat("vi-VN").format(Number(value || 0));

function safeUrl(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch { return ""; }
}

function markdown(text = "") {
  let html = escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(S\d+)\]/g, '<span class="citation">[$1]</span>');
  html = html.split("\n").map((line) => {
    if (/^[-*]\s+/.test(line)) return `<li>${line.replace(/^[-*]\s+/, "")}</li>`;
    if (/^\d+\.\s+/.test(line)) return `<li>${line.replace(/^\d+\.\s+/, "")}</li>`;
    return line ? `<p>${line}</p>` : "";
  }).join("");
  return html.replace(/(?:<li>.*?<\/li>)+/gs, (items) => `<ul>${items}</ul>`);
}

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.style.background = error ? "#a93636" : "";
  element.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove("show"), 2600);
}

function switchView(name) {
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

function renderStatus(metadata) {
  const openai = metadata.api_ready?.openai;
  const pageindex = metadata.api_ready?.pageindex;
  $("#system-dot").classList.toggle("ready", openai);
  $("#system-label").textContent = openai ? "Hệ thống sẵn sàng" : "Thiếu API key";
  $("#openai-state").textContent = `OpenAI · ${openai ? "ready" : "offline"}`;
  $("#pageindex-state").textContent = `PageIndex · ${pageindex ? "ready" : "offline"}`;
  $("#openai-state").classList.toggle("ready", openai);
  $("#pageindex-state").classList.toggle("ready", pageindex);
}

function renderCounts(counts) {
  const ids = {
    "side-legal": counts.legal_pdf, "side-news": counts.news_json,
    "side-chunks": counts.chunks, "side-pageindex": counts.pageindex_docs,
    "data-legal": counts.legal_pdf, "data-news": counts.news_json,
    "data-md": counts.standardized_md, "data-chunks": counts.chunks,
  };
  Object.entries(ids).forEach(([id, value]) => { $(`#${id}`).textContent = number(value); });
}

function renderDocuments() {
  const documents = (state.metadata?.documents || []).filter(
    (document) => state.filter === "all" || document.type === state.filter,
  );
  $("#document-list").innerHTML = documents.map((document) => {
    const url = safeUrl(document.url);
    const isLegal = document.type === "legal";
    return `<div class="document-row ${escapeHtml(document.type)}">
      <span class="doc-icon">${isLegal ? "PDF" : "JSON"}</span>
      <div class="doc-info"><strong>${escapeHtml(document.title || document.name)}</strong>
        <span>${number(document.characters)} ký tự · ${isLegal ? "chính sách" : "tin UIT"}</span></div>
      ${url ? `<a class="doc-link" href="${escapeHtml(url)}" target="_blank" rel="noreferrer" title="Mở nguồn">↗</a>` : ""}
    </div>`;
  }).join("") || '<p style="padding:20px;color:#7a858f">Không có tài liệu.</p>';
}

function renderTechnology() {
  $("#technology-list").innerHTML = (state.metadata?.technology || []).map((item) => `
    <div class="tech-row"><i class="tech-color"></i><div class="tech-info">
      <strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.role)}</span>
    </div><span class="tech-kind">${escapeHtml(item.kind)}</span></div>`).join("");
}

function renderEvaluation() {
  const metrics = state.metadata?.evaluation || [];
  $("#metric-list").innerHTML = metrics.map((metric) => `
    <div class="metric-row"><div class="metric-label"><span>${escapeHtml(metric.name)}</span>
      <strong>${Number(metric.hybrid).toFixed(3)} <small>vs ${Number(metric.dense).toFixed(3)}</small></strong></div>
      <div class="bar-pair"><div class="metric-bar hybrid"><i style="width:${Math.min(100, metric.hybrid * 100)}%"></i></div>
      <div class="metric-bar dense"><i style="width:${Math.min(100, metric.dense * 100)}%"></i></div></div>
    </div>`).join("") || '<p style="color:#7a858f">Chưa đọc được kết quả RAGAS.</p>';
}

async function loadMetadata() {
  try {
    state.metadata = await fetchJson("/api/metadata");
    renderStatus(state.metadata);
    renderCounts(state.metadata.counts || {});
    renderDocuments();
    renderTechnology();
    renderEvaluation();
  } catch (error) {
    $("#system-label").textContent = "Backend mất kết nối";
    toast(error.message, true);
  }
}

function sourceCards(sources) {
  if (!sources.length) return "";
  const cards = sources.slice(0, 5).map((source) => {
    const meta = source.metadata || {};
    const url = safeUrl(meta.url);
    return `<div class="source-item"><span class="source-label">${escapeHtml(meta.citation_label || "S")}</span>
      <div><strong>${escapeHtml(meta.source || "Tài liệu UIT")}</strong>
      <small>${escapeHtml(meta.type || "source")} · score ${Number(source.score || 0).toFixed(3)}</small></div>
      ${url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">↗</a>` : ""}</div>`;
  }).join("");
  return `<details class="sources" open><summary>◫ ${sources.length} nguồn được truy xuất</summary><div class="source-grid">${cards}</div></details>`;
}

function appendMessage(role, content, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.innerHTML = `<div class="avatar">${role === "user" ? "B" : "AI"}</div><div class="message-body">
    <div class="message-meta"><strong>${role === "user" ? "Bạn" : "UIT RAG"}</strong><span>vừa xong</span></div>
    <div class="message-content">${role === "assistant" ? markdown(content) : escapeHtml(content)}</div>
    ${role === "assistant" ? sourceCards(sources) : ""}</div>`;
  $("#messages").append(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
  return article;
}

function appendThinking() {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.innerHTML = '<div class="avatar">AI</div><div class="message-body"><div class="message-meta"><strong>UIT RAG</strong><span>đang xử lý</span></div><div class="message-content thinking"><i></i><i></i><i></i></div></div>';
  $("#messages").append(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
  return article;
}

function showPendingTrace() {
  $("#trace-empty").classList.add("hidden");
  $("#trace-list").classList.remove("hidden");
  $("#retrieval-summary").classList.add("hidden");
  $("#trace-total").textContent = "running";
  const pending = [
    ["Semantic retrieval", "OpenAI embedding → ChromaDB"],
    ["BM25 lexical", "Exact term matching"],
    ["RRF + relevance rerank", "Fusion hai bảng xếp hạng"],
    ["PageIndex fallback", "Kích hoạt nếu cosine thấp"],
    ["Grounded generation", "Answer + citations"],
  ];
  $("#trace-list").innerHTML = pending.map(([label, detail], index) => `
    <div class="trace-step ${index === 0 ? "running" : ""}"><span class="step-indicator">${index + 1}</span>
    <div class="trace-copy"><strong>${label}</strong><span>${detail}</span></div><span class="step-duration">wait</span></div>`).join("");
}

function renderTrace(result) {
  const trace = result.trace || [];
  $("#trace-list").innerHTML = trace.map((step, index) => {
    const status = step.status || "done";
    const indicator = status === "done" ? "✓" : status === "used" ? "F" : "—";
    return `<div class="trace-step ${escapeHtml(status)}"><span class="step-indicator">${indicator}</span>
      <div class="trace-copy"><strong>${escapeHtml(step.label || step.id)}</strong><span>${escapeHtml(step.detail || "")}</span></div>
      <span class="step-duration">${status === "skipped" ? "skip" : `${number(step.duration_ms)} ms`}</span></div>`;
  }).join("");
  $("#trace-total").textContent = `${number(result.total_duration_ms)} ms`;
  const stats = result.retrieval_stats || {};
  const summary = $("#retrieval-summary");
  summary.innerHTML = `<div class="summary-row"><span>Dense candidates</span><strong>${number(stats.dense_candidates)}</strong></div>
    <div class="summary-row"><span>BM25 candidates</span><strong>${number(stats.sparse_candidates)}</strong></div>
    <div class="summary-row"><span>Best cosine</span><strong>${Number(stats.best_dense_score || 0).toFixed(3)}</strong></div>
    <div class="summary-row"><span>PageIndex fallback</span><strong>${stats.fallback_used ? "Có" : "Không"}</strong></div>`;
  summary.classList.remove("hidden");
}

async function ask(rawQuestion) {
  const query = rawQuestion.trim();
  if (!query || state.busy) return;
  state.busy = true;
  $("#send-button").disabled = true;
  $("#query-input").value = "";
  resizeInput();
  const history = state.messages.slice(-6);
  appendMessage("user", query);
  state.messages.push({ role: "user", content: query });
  const thinking = appendThinking();
  showPendingTrace();
  try {
    const result = await fetchJson("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: Number($("#top-k").value), history }),
    });
    thinking.remove();
    appendMessage("assistant", result.answer, result.sources || []);
    state.messages.push({ role: "assistant", content: result.answer });
    renderTrace(result);
    toast("Đã trả lời bằng dữ liệu UIT có trích dẫn");
  } catch (error) {
    thinking.remove();
    appendMessage("assistant", `Không thể hoàn tất truy vấn: ${error.message}`);
    $("#trace-total").textContent = "error";
    toast(error.message, true);
  } finally {
    state.busy = false;
    $("#send-button").disabled = false;
    $("#query-input").focus();
  }
}

function resetChat() {
  state.messages = [];
  $("#messages").innerHTML = `<article class="message assistant"><div class="avatar">AI</div><div class="message-body">
    <div class="message-meta"><strong>UIT RAG</strong><span>vừa xong</span></div><div class="message-content">Chào bạn! Hãy hỏi về chính sách hoặc thông báo UIT. Mình sẽ hiển thị cả nguồn và luồng xử lý.</div></div></article>`;
  $("#trace-list").classList.add("hidden");
  $("#retrieval-summary").classList.add("hidden");
  $("#trace-empty").classList.remove("hidden");
  $("#trace-total").textContent = "idle";
  toast("Đã tạo cuộc hội thoại mới");
}

function resizeInput() {
  const input = $("#query-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 100)}px`;
}

function animateFlow() {
  const nodes = $$(".flow-node");
  nodes.forEach((node) => node.classList.remove("flow-active"));
  $("#animate-flow").disabled = true;
  nodes.forEach((node, index) => setTimeout(() => {
    nodes.forEach((item) => item.classList.remove("flow-active"));
    node.classList.add("flow-active");
    if (index === nodes.length - 1) setTimeout(() => {
      node.classList.remove("flow-active");
      $("#animate-flow").disabled = false;
      toast("Mô phỏng pipeline hoàn tất");
    }, 700);
  }, index * 560));
}

function bindEvents() {
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => switchView(item.dataset.view)));
  $("#chat-form").addEventListener("submit", (event) => { event.preventDefault(); ask($("#query-input").value); });
  $("#query-input").addEventListener("input", resizeInput);
  $("#query-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("#chat-form").requestSubmit(); }
  });
  $$(".suggestion").forEach((button) => button.addEventListener("click", () => ask(button.textContent)));
  $("#reset-chat").addEventListener("click", resetChat);
  $("#animate-flow").addEventListener("click", animateFlow);
  $$(".filter").forEach((button) => button.addEventListener("click", () => {
    state.filter = button.dataset.filter;
    $$(".filter").forEach((item) => item.classList.toggle("active", item === button));
    renderDocuments();
  }));
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    const view = ["chat", "flow", "data", "evaluation"][Number(event.key) - 1];
    if (view) switchView(view);
  });
}

bindEvents();
resizeInput();
loadMetadata();
