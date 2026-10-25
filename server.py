"""Local agent console. Run ``python server.py`` and open http://127.0.0.1:5500.

MODEL_NAME selects the default model; AGENT_MODELS optionally supplies a comma-
or newline-separated dropdown list. API credentials stay in the environment.
Set AGENT_WEB_USERNAME and AGENT_WEB_PASSWORD to unlock the console.
This console executes agent tools on your machine; use HTTPS for remote access.
"""

import argparse
import codecs
import json
import hmac
import math
import secrets
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from datetime import timedelta

from flask import Flask, Response, g, jsonify, redirect, render_template_string, request, send_file, session
from werkzeug.security import check_password_hash, generate_password_hash

AGENT_DIR = Path(__file__).resolve().parent
LOG_PATH = AGENT_DIR / "log"
TEMP_DIR = AGENT_DIR / "temp"
HISTORY_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
EFFORT_LEVELS = ("auto", "none", "minimal", "low", "medium", "high", "xhigh", "max")
RUN_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
ACTIVE_RUN = None
AUTH_LOCK = threading.Lock()
AUTH_SESSIONS = {}
LOGIN_ATTEMPTS = {}
AUTH_TTL = 48 * 60 * 60
LOGIN_WINDOW = 60
LOGIN_LIMIT = 5
WEB_SECRET_ENV = ("AGENT_WEB_USERNAME", "AGENT_WEB_PASSWORD", "AGENT_WEB_SECRET_KEY")


app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=2 * 1024 * 1024,
    TRUSTED_HOSTS=None,  # Trust every Host; legacy allowlist settings are ignored.
    SECRET_KEY=os.environ.get("AGENT_WEB_SECRET_KEY") or secrets.token_hex(32),
    AUTH_USERNAME=os.environ.get("AGENT_WEB_USERNAME", "").strip(),
    AUTH_PASSWORD_HASH=(
        generate_password_hash(os.environ["AGENT_WEB_PASSWORD"])
        if os.environ.get("AGENT_WEB_PASSWORD", "").strip() else None
    ),
    SESSION_COOKIE_NAME="agent_console_session",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("AGENT_WEB_COOKIE_SECURE", "").lower() in {"1", "true", "yes"},
    PERMANENT_SESSION_LIFETIME=timedelta(seconds=AUTH_TTL),
    SESSION_REFRESH_EACH_REQUEST=False,

)

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<meta name="csrf-token" content="{{ csrf_token }}">
<title>Agent / Mission Control</title>
<style>
:root {
  --bg:#080c14; --panel:#101725; --line:#243147; --muted:#91a2bb;
  --text:#e8effa; --accent:#73f5c1; --blue:#80aaff; --danger:#ff879d;
  --log-height:clamp(340px, 64vh, 900px);
}
* { box-sizing:border-box; }
body {
  margin:0; color:var(--text); background:var(--bg);
  background-image:radial-gradient(ellipse at 12% 0%, #1b3c4138, transparent 48%),
    radial-gradient(ellipse at 95% 20%, #23325c38, transparent 45%);
  font:14px/1.6 Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
button, input, select, textarea { font:inherit; }
button, select { cursor:pointer; }
button { transition:background .18s, border-color .18s, transform .18s; }
button:hover:not(:disabled) { border-color:var(--accent); transform:translateY(-1px); }
button:disabled { cursor:not-allowed; opacity:.45; }
:focus-visible { outline:2px solid var(--accent); outline-offset:3px; }
[hidden] { display:none !important; }
.shell { max-width:1600px; margin:auto; padding:30px 36px; }
header { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:28px; }
.brand { display:flex; align-items:center; gap:14px; }
.mark {
  display:grid; place-items:center; width:46px; height:46px; border-radius:14px;
  background:linear-gradient(145deg, #73f5c1, #7aa9ff); color:#0a1724;
  font:bold 24px ui-monospace, monospace; box-shadow:0 0 32px #73f5c11c;
}
h1 { font-size:20px; letter-spacing:.02em; margin:0; }
.eyebrow { font:11px/1.5 ui-monospace, monospace; letter-spacing:.18em; color:var(--muted); }
.badge {
  display:inline-flex; align-items:center; gap:8px; padding:6px 12px;
  border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:12px;
}
.dot { width:7px; height:7px; border-radius:50%; background:var(--accent); }
.intro { display:flex; justify-content:space-between; align-items:end; gap:20px; margin-bottom:22px; }
h2 { margin:0 0 6px; font-size:30px; letter-spacing:-.03em; }
p { margin:0; color:var(--muted); }
.workspace { display:grid; grid-template-columns:350px minmax(0, 1fr); gap:20px; align-items:start; }
.panel { background:linear-gradient(150deg, #131d2d, #0e1522); border:1px solid var(--line); border-radius:18px; box-shadow:0 16px 48px #0002; overflow:hidden; }
.panel-head { padding:18px 20px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; gap:12px; }
.panel-title { font-weight:650; }
.panel-index { color:var(--accent); font:12px ui-monospace, monospace; margin-right:8px; }
.form-body { padding:20px; }
.field { margin-bottom:18px; }
label, .field-title { display:block; font-size:12px; color:#becde0; margin-bottom:7px; }
input:not([type=checkbox]), select, textarea {
  width:100%; padding:10px 12px; border:1px solid #2a3851; border-radius:9px;
  background:#090f1d; color:var(--text);
}
textarea { min-height:175px; resize:vertical; line-height:1.7; }
textarea::placeholder, input::placeholder { color:#687b95; }
.help { font-size:11px; color:var(--muted); margin-top:7px; overflow-wrap:anywhere; }
.row { display:flex; gap:8px; align-items:center; }
.row > input { min-width:0; }
.btn { padding:8px 13px; color:var(--text); background:#182338; border:1px solid #30415d; border-radius:9px; white-space:nowrap; }
.btn.small { padding:5px 10px; font-size:12px; }
.btn.primary { background:linear-gradient(110deg, #73f5c1, #85d7e8); border:0; color:#092021; font-weight:750; flex:1; padding:12px; }
.btn.danger { color:var(--danger); border-color:#704354; }
.advanced { border-top:1px solid var(--line); padding-top:13px; margin-bottom:18px; }
.advanced summary { cursor:pointer; color:var(--muted); font-size:12px; }
.advanced .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
.notice { padding:12px 14px; border:1px solid #6d572c; background:#302713; border-radius:10px; color:#e5c77f; font-size:12px; margin-bottom:18px; }
.console { position:sticky; top:20px; min-width:0; }
.console-top { flex-wrap:wrap; }
.window-dots { display:inline-flex; gap:5px; margin-right:10px; }
.window-dots i { width:8px; height:8px; border-radius:50%; background:#ff7188; }
.window-dots i:nth-child(2) { background:#efc46e; }
.window-dots i:nth-child(3) { background:#73f5c1; }
.state { color:var(--muted); font-size:12px; display:flex; align-items:center; gap:8px; }
.state .dot { background:#74849a; }
.state[data-state=running] { color:var(--accent); }
.state[data-state=running] .dot { background:var(--accent); animation:pulse 1.5s infinite; }
.state[data-state=success] { color:var(--accent); }
.state[data-state=success] .dot { background:var(--accent); }
.state[data-state=error] { color:var(--danger); }
.state[data-state=error] .dot { background:var(--danger); }
.state[data-state=stopped] { color:#efc46e; }
.state[data-state=stopped] .dot { background:#efc46e; }
.toolbar { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; padding:12px 18px; border-bottom:1px solid var(--line); background:#0d1421; }
.tools { display:flex; align-items:center; flex-wrap:wrap; gap:8px; }
.toolbar label { display:inline-flex; align-items:center; gap:6px; margin:0; font-size:12px; }
.toolbar select { width:auto; padding:5px 8px; font-size:12px; }
input[type=checkbox] { accent-color:var(--accent); }
.log-shell { position:relative; background:#080e18; }
.log-viewport { height:var(--log-height); overflow:auto; overscroll-behavior:contain; scrollbar-gutter:stable; padding:22px; scrollbar-color:#344a64 #0b1220; }
.log-viewport::-webkit-scrollbar { width:9px; height:9px; }
.log-viewport::-webkit-scrollbar-thumb { background:#344a64; border-radius:9px; }
#out { margin:0; font:12.5px/1.85 "SFMono-Regular", Consolas, "Liberation Mono", monospace; color:#c9d8ec; white-space:pre-wrap; overflow-wrap:anywhere; tab-size:4; }
#out.nowrap { white-space:pre; overflow-wrap:normal; }
.log-system { color:#82aaff; }
.log-thought { color:#c7a5ff; }
.log-action { color:#f1c878; }
.log-observation { color:#78d4e6; }
.log-final { color:#73f5c1; }
.log-error { color:var(--danger); }
.empty { min-height:100%; display:flex; align-items:center; justify-content:center; flex-direction:column; text-align:center; gap:12px; padding:30px; }
.empty-icon { font:36px ui-monospace, monospace; color:var(--accent); width:84px; height:84px; border:1px solid #29463e; border-radius:22px; display:grid; place-items:center; background:#14282166; margin-bottom:8px; }
.empty strong { font-size:17px; font-weight:550; }
.empty p { font-size:12px; max-width:350px; }
.jump { position:absolute; bottom:18px; right:24px; background:#1d3d36; border-color:#3d7563; box-shadow:0 4px 20px #0006; }
.console-footer { display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; padding:12px 18px; border-top:1px solid var(--line); font:11px/1.6 ui-monospace, monospace; color:var(--muted); }
#run-meta { overflow-wrap:anywhere; min-width:0; }
#status-message { margin-top:12px; font-size:12px; min-height:20px; color:var(--muted); overflow-wrap:anywhere; }
#status-message.error { color:var(--danger); }
.legend { display:flex; flex-wrap:wrap; gap:16px; margin-top:13px; color:var(--muted); font:11px ui-monospace, monospace; }
.legend span::before { content:""; display:inline-block; width:5px; height:5px; margin:0 6px 2px 0; border-radius:50%; background:currentColor; }
footer { display:flex; justify-content:space-between; gap:16px; margin-top:24px; padding-top:18px; border-top:1px solid #1e2a3b; color:#778ba7; font-size:11px; }
#toast { position:fixed; bottom:24px; left:50%; transform:translateX(-50%); padding:10px 20px; border:1px solid #416655; border-radius:10px; background:#192f29; color:#ccfce6; z-index:10; max-width:90vw; }
@keyframes pulse { 50% { box-shadow:0 0 0 5px #73f5c11a; } }
@media (prefers-reduced-motion:reduce) { *, *::before, *::after { animation:none !important; transition:none !important; } }
@media (min-width:1400px) { .workspace { grid-template-columns:380px minmax(0, 1fr); } }
@media (max-width:950px) {
  .shell { padding:22px 18px; } .workspace { grid-template-columns:1fr; }
  .console { position:static; } textarea { min-height:130px; }
  .intro .eyebrow { display:none; } h2 { font-size:25px; }
}
@media (max-width:480px) {
  .shell { padding:18px 12px; } header .badge { display:none; }
  .panel-head, .form-body { padding:16px; } .toolbar { padding:10px 12px; }
  .log-viewport { padding:14px; } footer { flex-direction:column; gap:5px; }
}
</style>
</head>
<body>
<div class="shell">
  <header>
    <div class="brand"><div class="mark" aria-hidden="true">&gt;_</div><div><h1>Agent</h1><div class="eyebrow">MISSION CONTROL / LOCAL WORKSPACE</div></div></div>
    <div class="row"><span class="badge"><span class="dot"></span>{{ username }}</span><button type="button" class="btn small" id="logout">锁定退出</button></div>
  </header>
  <div class="intro"><div><h2>想法进来，行动出去。</h2><p>选择模型，发起任务，在一个窗口里观察 Agent 的每一步。</p></div><div class="eyebrow">THINK → ACT → OBSERVE</div></div>
  <div id="notice" class="notice" hidden></div>
  <main class="workspace">
    <section class="panel" aria-label="运行配置">
      <div class="panel-head"><span class="panel-title"><span class="panel-index">01</span>任务配置</span><span class="eyebrow">CONFIGURE</span></div>
      <form id="form" class="form-body">
        <div class="field">
          <label for="model">模型 / Model</label>
          <select id="model" name="model_choice" disabled><option value="">正在读取配置…</option></select>
          <div id="custom-model-field" hidden style="margin-top:10px">
            <label for="custom-model">自定义模型名称</label>
            <input id="custom-model" maxlength="200" placeholder="输入网关支持的完整模型名" autocomplete="off" spellcheck="false">
          </div>
          <p class="help">内置选项 + MODEL_NAME / AGENT_MODELS；实际可用性由网关决定。</p>
        </div>
        <div class="field">
          <label for="effort">推理强度 / Reasoning effort</label>
          <select id="effort" name="effort"><option value="auto">Auto · 使用服务商默认</option></select>
          <p class="help">不同模型支持的强度不同，不确定时选 Auto。</p>
        </div>
        <div class="field">
          <label for="history">会话 / History</label>
          <div class="row"><input id="history" name="history" value="new" required maxlength="64" pattern="[A-Za-z0-9._\-]{1,64}" list="history-list" spellcheck="false" autocomplete="off"><button class="btn small" id="new-history" type="button">＋ 新会话</button></div>
          <datalist id="history-list"></datalist>
          <p class="help">同名会话继续上下文；“新会话”生成独立名称。new 也是普通会话名。</p>
        </div>
        <div class="field">
          <label for="input">任务指令 / Prompt</label>
          <textarea id="input" name="input" required placeholder="今天让 Agent 帮你完成什么？&#10;&#10;描述目标、约束，以及你期待的结果。"></textarea>
          <p class="help"><span id="input-count">0</span> 字符 · Ctrl / ⌘ + Enter 运行</p>
        </div>
        <details class="advanced">
          <summary>高级运行参数</summary>
          <div class="grid">
            <div><label for="max-steps">最大步数</label><input id="max-steps" name="max_steps" type="number" value="20" min="1" max="1000" required></div>
            <div><label for="max-tokens">最大输出 tokens</label><input id="max-tokens" name="max_tokens" type="number" value="65536" min="1" max="1048576" required></div>
          </div>
        </details>
        <div class="row"><button class="btn primary" id="run" type="submit" disabled>▶ 开始运行</button><button class="btn danger" id="stop" type="button" disabled>■ 停止</button></div>
        <div id="status-message" role="status" aria-live="polite">正在读取本地配置…</div>
      </form>
    </section>
    <section class="console" aria-label="Agent 日志">
      <div class="panel">
        <div class="panel-head console-top">
          <span class="panel-title"><span class="window-dots" aria-hidden="true"><i></i><i></i><i></i></span>Agent Live Log</span>
          <div class="row"><span id="state" class="state" data-state="idle" role="status"><span class="dot"></span><span id="state-label">待命</span></span><button type="button" id="collapse" class="btn small" aria-expanded="true" aria-controls="console-body">收起 ▴</button></div>
        </div>
        <div id="console-body">
          <div class="toolbar">
            <div class="tools">
              <label for="log-size">高度 <select id="log-size"><option value="compact">紧凑 · 360px</option><option value="standard" selected>标准 · 64vh</option><option value="large">沉浸 · 78vh</option></select></label>
              <label><input id="follow" type="checkbox" checked>自动跟随</label>
              <label><input id="wrap" type="checkbox" checked>换行</label>
            </div>
            <div class="tools"><button type="button" class="btn small" id="clear">清屏</button><button type="button" class="btn small" id="copy">复制</button><button type="button" class="btn small" id="download">下载 log ↓</button></div>
          </div>
          <div class="log-shell">
            <div id="viewport" class="log-viewport" tabindex="0" aria-label="日志内容，可滚动查看">
              <div id="empty" class="empty"><div class="empty-icon" aria-hidden="true">/_</div><strong>准备好，开始下一次探索。</strong><p>运行后，模型思考、工具调用和执行结果会实时出现在这里。日志面板保持固定高度，不会把页面撑长。</p></div>
              <pre id="out" hidden></pre>
            </div>
            <button type="button" class="btn small jump" id="jump" hidden>↓ 回到最新日志</button>
          </div>
        </div>
        <div class="console-footer"><span id="run-meta">尚未开始 · 输出同步保存至 ./log</span><span><span id="elapsed">00:00</span> &nbsp; / &nbsp; <span id="line-count">0</span> 行</span></div>
      </div>
      <div class="legend" aria-hidden="true"><span class="log-system">SYSTEM</span><span class="log-thought">THOUGHT</span><span class="log-action">ACTION</span><span class="log-observation">OBSERVATION</span><span class="log-final">FINAL</span></div>
    </section>
  </main>
  <footer><span>仅供可信本地使用 · Agent 可执行本机命令，请确认任务内容。</span><span>NO BUILD STEP. JUST RUN.</span></footer>
</div>
<div id="toast" role="status" hidden></div>
<script>
"use strict";
const $ = id => document.getElementById(id);
const form = $("form"), out = $("out"), viewport = $("viewport");
const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
async function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.method && options.method.toUpperCase() !== "GET") headers.set("X-CSRF-Token", csrfToken);
  const response = await fetch(url, {...options, headers, credentials:"same-origin"});
  if (response.status === 401) {
    running = false; runId = null;
    window.location.replace("/login");
    throw new Error("登录已失效，请重新登录。");
  }
  return response;
}
const CUSTOM = "__custom__";
const STORAGE = "agent-console-preferences-v1";
const MAX_DISPLAY = 1000000; // Bound browser memory; the complete log stays on disk.
let config = null, running = false, runId = null, started = 0, timer = null;
let rawLog = "", pendingOutput = "", paintTimer = null, toastTimer = null;
let preferences = {}, trimmed = false, userScrolled = false;
try { preferences = JSON.parse(localStorage.getItem(STORAGE) || "{}") || {}; } catch (_) {}
function toast(message) {
  $("toast").textContent = message; $("toast").hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3200);
}
function message(text, error = false) {
  $("status-message").textContent = text;
  $("status-message").classList.toggle("error", error);
}
function setState(state, label) {
  $("state").dataset.state = state; $("state-label").textContent = label;
}
function selectedModel() { return $("model").value === CUSTOM ? $("custom-model").value.trim() : $("model").value; }
function savePreferences() {
  preferences = { model:selectedModel(), effort:$("effort").value, size:$("log-size").value, wrap:$("wrap").checked, follow:$("follow").checked };
  try { localStorage.setItem(STORAGE, JSON.stringify(preferences)); } catch (_) {}
}
function updateModel() {
  const custom = $("model").value === CUSTOM;
  $("custom-model-field").hidden = !custom; $("custom-model").required = custom;
}
function setBusy(busy) {
  running = busy;
  for (const el of form.querySelectorAll("input, select, textarea, button")) el.disabled = busy;
  $("run").disabled = busy || !config;
  $("stop").disabled = !busy || !runId;
  $("run").textContent = busy ? "◌ 运行中…" : "▶ 开始运行";
}
function tick() {
  const seconds = Math.max(0, Math.floor((Date.now() - started) / 1000));
  $("elapsed").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
function jumpToBottom() {
  userScrolled = false; viewport.scrollTop = viewport.scrollHeight; $("jump").hidden = true;
}
function classify(line) {
  if (/Traceback|^(?:\w*Error|Exception):/.test(line)) return "log-error";
  if (/==== SYSTEM ====|^history_context_usage:/.test(line)) return "log-system";
  if (/<\|thought\|>|^Thought:/.test(line)) return "log-thought";
  if (/<\|action(?:_input)?\|>|^Action:/.test(line)) return "log-action";
  if (/<\|observation\|>|^Observation:/.test(line)) return "log-observation";
  if (/<\|final_answer\|>|^Final Answer:/.test(line)) return "log-final";
  return "";
}
function spanFor(text) {
  const span = document.createElement("span");
  span.textContent = text; span.className = classify(text); return span;
}
function renderLines(text) {
  // Only update the partial last line; never interpret model output as HTML.
  if (out.lastChild && !out.lastChild.textContent.endsWith("\n")) {
    text = out.lastChild.textContent + text; out.lastChild.remove();
  }
  const fragment = document.createDocumentFragment();
  for (const line of text.match(/[^\n]*\n|[^\n]+$/g) || []) fragment.appendChild(spanFor(line));
  out.appendChild(fragment);
}
function flushOutput() {
  clearTimeout(paintTimer); paintTimer = null;
  if (!pendingOutput) return;
  let text = pendingOutput; pendingOutput = "";
  rawLog += text;
  if (rawLog.length > MAX_DISPLAY) {
    rawLog = rawLog.slice(-MAX_DISPLAY);
    const newline = rawLog.indexOf("\n");
    if (newline >= 0 && newline < MAX_DISPLAY / 2) rawLog = rawLog.slice(newline + 1);
    text = rawLog; out.replaceChildren(); trimmed = true;
  }
  $("empty").hidden = true; out.hidden = false;
  renderLines(text);
  $("line-count").textContent = `${trimmed ? "最近 " : ""}${rawLog.split("\n").length}`;
  if ($("follow").checked) jumpToBottom();
  else $("jump").hidden = false;
}
function appendOutput(text) {
  pendingOutput += text;
  if (!paintTimer) paintTimer = setTimeout(flushOutput, 50);
}
function clearOutput() {
  clearTimeout(paintTimer); paintTimer = null; pendingOutput = ""; rawLog = ""; trimmed = false;
  out.replaceChildren(); out.hidden = true; $("empty").hidden = false;
  $("line-count").textContent = "0"; $("jump").hidden = true;
}
function applyLogSize() {
  const heights = {compact:"360px", standard:"clamp(340px, 64vh, 900px)", large:"78vh"};
  document.documentElement.style.setProperty("--log-height", heights[$("log-size").value] || heights.standard);
  if ($("follow").checked) jumpToBottom();
}
async function responseError(response) {
  const text = await response.text();
  try { return JSON.parse(text).error || `HTTP ${response.status}`; }
  catch (_) { return `HTTP ${response.status}: ${text.slice(0, 200)}`; }
}
async function loadConfig() {
  try {
    const response = await apiFetch("/api/config", {cache:"no-store"});
    if (!response.ok) throw new Error(await responseError(response));
    config = await response.json();
    $("model").replaceChildren();
    for (const name of config.models) $("model").add(new Option(name, name));
    $("model").add(new Option("＋ 自定义模型…", CUSTOM));
    const model = preferences.model || config.default_model;
    if (config.models.includes(model)) $("model").value = model;
    else { $("model").value = CUSTOM; $("custom-model").value = model || ""; }
    updateModel();
    $("effort").replaceChildren();
    for (const effort of config.efforts) $("effort").add(new Option(effort === "auto" ? "Auto · 使用服务商默认" : effort, effort));
    $("effort").value = config.efforts.includes(preferences.effort) ? preferences.effort : config.default_effort;
    $("history-list").replaceChildren();
    for (const name of config.histories) $("history-list").appendChild(new Option(name, name));
    const warnings = [];
    if (!config.credentials_configured) warnings.push("启动前请在服务端环境中配置 BASE_URL 和 API_KEY；密钥不会发送到页面。");
    if (config.busy) warnings.push("服务端已有任务运行，新任务需等待它结束；此页面不会接管其他窗口的任务。");
    $("notice").textContent = warnings.join(" "); $("notice").hidden = !warnings.length;
    setBusy(false); message("准备就绪。选择模型并输入任务。");
  } catch (error) { setState("error", "配置加载失败"); message(`${error.message}，请刷新重试。`, true); }
}
form.addEventListener("submit", async event => {
  event.preventDefault();
  if (running || !config) return;
  if (!$("input").value.trim()) { message("请输入非空的任务指令。", true); $("input").focus(); return; }
  if (!selectedModel()) { message("请选择或输入模型名称。", true); $("custom-model").focus(); return; }
  const body = new FormData(form);
  body.set("model", selectedModel());
  const model = selectedModel(), history = $("history").value;
  savePreferences(); clearOutput(); runId = null; setBusy(true);
  setState("running", "连接中"); message("正在启动 Agent…");
  started = Date.now(); tick(); timer = setInterval(tick, 1000);
  $("run-meta").textContent = `${model} · ${history}`;
  let reader = null, finished = false;
  try {
    const response = await apiFetch("/run", {method:"POST", body});
    if (!response.ok) throw new Error(await responseError(response));
    if (!response.body) throw new Error("浏览器不支持流式响应，请使用现代浏览器。");
    reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    function consume(line) {
      if (!line.trim()) return;
      const item = JSON.parse(line);
      if (item.type === "start") {
        runId = item.id; $("stop").disabled = false; setState("running", "运行中");
        message("输出实时写入 ./log；向上滚动会暂停自动跟随。");
      } else if (item.type === "output") appendOutput(item.text);
      else if (item.type === "error") throw new Error(item.message);
      else if (item.type === "done") {
        finished = true;
        setState(item.stopped ? "stopped" : item.code === 0 ? "success" : "error", item.stopped ? "已停止" : item.code === 0 ? "已完成" : "运行失败");
        message(item.stopped ? "任务已停止，已有日志和 history 保留。" : item.code === 0 ? "运行完成。完整输出已保存到 ./log。" : `Agent 退出码 ${item.code}，请查看日志中的错误信息。`, !item.stopped && item.code !== 0);
      }
    }
    while (true) {
      const {value, done} = await reader.read();
      if (done) { buffer += decoder.decode(); break; }
      buffer += decoder.decode(value, {stream:true});
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline); buffer = buffer.slice(newline + 1); consume(line);
      }
    }
    if (buffer.trim()) consume(buffer);
    if (!finished) throw new Error("日志连接意外中断，未收到完成状态。");
  } catch (error) {
    setState("error", "运行异常"); message(error.message, true);
    // If the transport fails, also request cancellation; never silently leave work running.
    if (runId && !finished) {
      try { await apiFetch("/stop", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({id:runId})}); } catch (_) {}
    }
  } finally {
    if (reader) { try { await reader.cancel(); } catch (_) {} reader.releaseLock(); }
    flushOutput(); clearInterval(timer); tick(); runId = null; setBusy(false);
    if (finished && !Array.from($("history-list").options).some(option => option.value === history)) $("history-list").appendChild(new Option(history, history));
  }
});
$("stop").addEventListener("click", async () => {
  if (!runId) return;
  $("stop").disabled = true; message("正在停止 Agent 及其工具进程…");
  try {
    const response = await apiFetch("/stop", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({id:runId})});
    if (!response.ok) throw new Error(await responseError(response));
  } catch (error) { message(error.message, true); $("stop").disabled = !running || !runId; }
});
$("new-history").addEventListener("click", () => {
  const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "");
  $("history").value = `web-${stamp}-${Math.random().toString(36).slice(2, 6)}`;
  toast("已创建新会话名称；下次运行使用独立上下文。");
});
$("input").addEventListener("input", () => { $("input-count").textContent = $("input").value.length.toLocaleString(); });
$("input").addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !running) { event.preventDefault(); form.requestSubmit(); }
});
$("model").addEventListener("change", () => { updateModel(); savePreferences(); });
$("custom-model").addEventListener("change", savePreferences);
$("effort").addEventListener("change", savePreferences);
$("log-size").addEventListener("change", () => { applyLogSize(); savePreferences(); });
$("wrap").addEventListener("change", () => { out.classList.toggle("nowrap", !$("wrap").checked); savePreferences(); });
$("follow").addEventListener("change", () => { if ($("follow").checked) jumpToBottom(); savePreferences(); });
viewport.addEventListener("wheel", () => { userScrolled = true; }, {passive:true});
viewport.addEventListener("touchstart", () => { userScrolled = true; }, {passive:true});
viewport.addEventListener("pointerdown", () => { userScrolled = true; });
viewport.addEventListener("keydown", () => { userScrolled = true; });
viewport.addEventListener("scroll", () => {
  const atBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 40;
  if (userScrolled && !atBottom && $("follow").checked) { $("follow").checked = false; savePreferences(); }
  $("jump").hidden = atBottom || !rawLog;
}, {passive:true});
$("jump").addEventListener("click", () => { $("follow").checked = true; jumpToBottom(); savePreferences(); });
$("collapse").addEventListener("click", () => {
  const expanded = $("collapse").getAttribute("aria-expanded") !== "true";
  $("collapse").setAttribute("aria-expanded", String(expanded)); $("console-body").hidden = !expanded;
  $("collapse").textContent = expanded ? "收起 ▴" : "展开 ▾";
  if (expanded && $("follow").checked) jumpToBottom();
});
$("clear").addEventListener("click", () => { clearOutput(); toast("仅清空当前显示，不删除磁盘上的 log。"); });
$("copy").addEventListener("click", async () => {
  flushOutput();
  if (!rawLog) { toast("还没有可复制的日志。"); return; }
  try {
    await navigator.clipboard.writeText(rawLog);
    toast(trimmed ? "已复制当前保留的日志；完整内容请下载。" : "日志已复制。");
  } catch (_) { toast("无法使用剪贴板，请手动选择日志或下载。"); }
});
$("download").addEventListener("click", async () => {
  try {
    const response = await apiFetch("/log", {cache:"no-store"});
    if (!response.ok) throw new Error(await responseError(response));
    const url = URL.createObjectURL(await response.blob());
    const a = document.createElement("a"); a.href = url; a.download = "agent.log"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("已下载服务端最近一次运行的 log。");
  } catch (error) { toast(error.message); }
});
$("logout").addEventListener("click", async () => {
  if (running && !window.confirm("锁定退出将停止当前会话的任务，确定退出？")) return;
  $("logout").disabled = true;
  try {
    const response = await apiFetch("/logout", {method:"POST"});
    if (!response.ok) throw new Error(await responseError(response));
    running = false; runId = null; clearOutput();
    window.location.replace("/login");
  } catch (error) { toast(error.message); $("logout").disabled = false; }
});
window.addEventListener("pageshow", event => {
  if (event.persisted) window.location.reload();
});
window.addEventListener("beforeunload", event => {
  if (running) { event.preventDefault(); event.returnValue = ""; }
});
window.addEventListener("pagehide", () => {
  if (runId) navigator.sendBeacon("/stop", new URLSearchParams({id:runId, csrf_token:csrfToken}));
});
if (["compact", "standard", "large"].includes(preferences.size)) $("log-size").value = preferences.size;
if (typeof preferences.wrap === "boolean") $("wrap").checked = preferences.wrap;
if (typeof preferences.follow === "boolean") $("follow").checked = preferences.follow;
out.classList.toggle("nowrap", !$("wrap").checked); applyLogSize(); loadConfig();
</script>
</body>
</html>
"""


def model_names():
    names = [os.environ.get("MODEL_NAME", "")]
    names.extend(re.split(r"[,\n]", os.environ.get("AGENT_MODELS", "")))
    names.append("gpt-5.6-sol")
    return list(dict.fromkeys(name.strip() for name in names if name.strip()))


def event(kind, **fields):
    return json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\n"


class AgentRun:
    """One subprocess, with idempotent cleanup even if the HTTP client disconnects."""

    def __init__(self, text, history, model, effort, max_steps, max_tokens):
        self.id = uuid.uuid4().hex
        self.text = text
        self.history = history
        self.model = model
        self.effort = effort
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.input_path = TEMP_DIR / f"web-input-{self.id}.txt"
        self.process = None
        self.log_file = None
        self.stopped = False
        self.closed = False
        self.lock = threading.Lock()
        self.started = time.monotonic()

    def start(self):
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.input_path.write_text(self.text, encoding="utf-8")
        env = os.environ.copy()
        # Agent tools need model credentials, but never the web console secrets.
        for key in WEB_SECRET_ENV:
            env.pop(key, None)
        env.update(
            PYTHONUNBUFFERED="1", PYTHONUTF8="1",
            MODEL_NAME=self.model, REASONING_EFFORT=self.effort,
        )
        self.log_file = LOG_PATH.open("w", encoding="utf-8", newline="")
        self.process = subprocess.Popen(
            [
                sys.executable, str(AGENT_DIR / "core" / "agent.py"),
                "--input", str(self.input_path),
                "--history", self.history,
                "--effort", self.effort,
                "--max_steps", str(self.max_steps),
                "--max_tokens", str(self.max_tokens),
            ],
            cwd=AGENT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, bufsize=0, env=env,
            start_new_session=(os.name == "posix"),
        )

    def _kill(self):
        if self.process is None:
            return
        try:
            if os.name == "posix":
                # Tools can spawn shells/children; kill the whole isolated group.
                os.killpg(self.process.pid, signal.SIGKILL)
            elif self.process.poll() is None:
                self.process.kill()
        except ProcessLookupError:
            pass

    def stop(self):
        with self.lock:
            if not self.closed:
                self.stopped = True
                self._kill()

    def close(self):
        global ACTIVE_RUN
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                self._kill()
                if self.process is not None:
                    self.process.wait()
                    if self.process.stdout is not None:
                        self.process.stdout.close()
                if self.log_file is not None:
                    self.log_file.close()
            finally:
                try:
                    self.input_path.unlink(missing_ok=True)
                finally:
                    with STATE_LOCK:
                        if ACTIVE_RUN is self:
                            ACTIVE_RUN = None
                    RUN_LOCK.release()

    def stream(self):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            yield event("start", id=self.id, model=self.model, history=self.history)
            while True:
                chunk = self.process.stdout.read(8192)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    self.log_file.write(text)
                    self.log_file.flush()
                    yield event("output", text=text)
            tail = decoder.decode(b"", final=True)
            if tail:
                self.log_file.write(tail)
                self.log_file.flush()
                yield event("output", text=tail)
            code = self.process.wait()
            self.log_file.flush()
            # Release resources before sending completion, so an immediate retry works.
            self.close()
            yield event("done", code=code, stopped=self.stopped,
                        elapsed=round(time.monotonic() - self.started, 3))
        except Exception as exc:
            app.logger.exception("Agent log stream failed")
            yield event("error", message=f"日志流异常：{exc}")
        finally:
            self.close()


LOGIN_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Agent / 解锁控制台</title>
<style>
* { box-sizing:border-box; }
body { margin:0; min-height:100vh; display:grid; place-items:center; padding:24px;
  background:radial-gradient(ellipse at top, #16302e, #080c14 65%);
  color:#e8effa; font:14px/1.6 system-ui, sans-serif; }
.card { width:100%; max-width:420px; padding:34px; border:1px solid #2a3b50;
  border-radius:22px; background:#101725; box-shadow:0 24px 80px #0005; }
.mark { width:52px; height:52px; display:grid; place-items:center; border-radius:15px;
  background:linear-gradient(140deg, #73f5c1, #80aaff); color:#102023; font:bold 25px monospace; }
.eyebrow { color:#91a2bb; font:11px monospace; letter-spacing:.15em; margin-top:22px; }
h1 { font-size:25px; margin:7px 0; }
p { color:#91a2bb; font-size:12px; margin:0 0 22px; }
label { display:block; color:#bdcce0; margin:16px 0 6px; font-size:12px; }
input, button { width:100%; border-radius:10px; padding:12px; font:inherit; }
input { background:#080f1d; color:#e8effa; border:1px solid #30415d; }
button { margin-top:24px; border:0; background:linear-gradient(110deg,#73f5c1,#85d7e8);
  color:#092021; font-weight:700; cursor:pointer; }
button:disabled { opacity:.45; cursor:not-allowed; }
:focus-visible { outline:2px solid #73f5c1; outline-offset:3px; }
.error { padding:12px; background:#36202b; border:1px solid #754354; border-radius:10px;
  color:#ffc1ce; font-size:13px; overflow-wrap:anywhere; }
.note { color:#778ba7; font-size:11px; margin-top:20px; text-align:center; }
</style>
</head>
<body>
<main class="card">
  <div class="mark" aria-hidden="true">&gt;_</div>
  <div class="eyebrow">MISSION CONTROL / LOCKED</div>
  <h1>解锁 Agent 控制台</h1>
  <p>登录后才能查看任务、运行 Agent 或下载日志。</p>
  {% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}
  <form method="post" action="/login">
    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
    <label for="username">账号</label>
    <input id="username" name="username" autocomplete="username" required autofocus>
    <label for="password">密码</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required>
    <button type="submit" {% if not configured %}disabled{% endif %}>解锁控制台 →</button>
  </form>
  <div class="note">凭据由服务端环境变量配置 · 会话有效期 48 小时</div>
</main>
</body>
</html>
"""


def auth_configured():
    return bool(app.config["AUTH_USERNAME"] and app.config["AUTH_PASSWORD_HASH"])


def csrf_token():
    if not session.get("csrf"):
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


def active_auth_id():
    """Server-side registry makes logout revoke even a copied signed cookie."""
    sid = session.get("auth_id")
    now = time.monotonic()
    with AUTH_LOCK:
        for key in list(AUTH_SESSIONS):
            if AUTH_SESSIONS[key] <= now:
                del AUTH_SESSIONS[key]
        return sid if isinstance(sid, str) and sid in AUTH_SESSIONS else None


def csrf_valid():
    provided = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
    expected = session.get("csrf", "")
    return bool(expected and hmac.compare_digest(expected.encode(), provided.encode()))


def reject_post(code, message):
    # Only log diagnostic codes, never submitted passwords, cookies or tokens.
    app.logger.warning("Request rejected (%s): %s %s", code, request.method, request.path)
    if request.endpoint == "login":
        # Render a usable login form with a current token after stale-cookie errors.
        return render_template_string(
            LOGIN_PAGE, configured=True, error=message, csrf_token=csrf_token()
        ), 403
    return jsonify(error=message, code=code), 403


@app.before_request
def require_login():
    # Preserve trust-all Host behavior. Auth is independent of the Host header.
    if not auth_configured():
        error = "控制台已锁定：请设置 AGENT_WEB_USERNAME 和 AGENT_WEB_PASSWORD 后重启服务。"
        if request.endpoint in {"index", "login"}:
            return render_template_string(LOGIN_PAGE, configured=False, error=error, csrf_token=""), 503
        return jsonify(error=error), 503

    # Login is public, but never the console, configuration, logs, or actions.
    g.auth_id = active_auth_id()
    if request.endpoint != "login" and not g.auth_id:
        # Keep the anonymous CSRF session: favicon/API requests may arrive after
        # GET /login, and must not erase the token belonging to its form.
        session.pop("auth_id", None)
        if request.endpoint == "index" and request.method in {"GET", "HEAD"}:
            return redirect("/login")
        return jsonify(error="请先登录或重新解锁控制台"), 401

    if request.method == "POST":
        # Reverse proxies / forwarding services may rewrite Origin or Fetch
        # Metadata. Do not use those headers as a second address allowlist.
        # Every POST, including login, still needs an unpredictable token bound
        # to its signed session cookie. No permissive CORS is enabled.
        if not csrf_valid():
            if not session.get("csrf"):
                return reject_post(
                    "csrf_session_missing",
                    "登录会话 Cookie 缺失或已失效，请重新输入账号密码。请允许此站点的 Cookie；"
                    "若使用 HTTP 访问，不要设置 AGENT_WEB_COOKIE_SECURE=1。"
                    "服务重启后需要重新打开登录页。",
                )
            return reject_post(
                "csrf_invalid",
                "页面校验已失效，请在此页面重新登录；若在控制台操作，请刷新页面。"
                "这不是账号密码错误。",
            )


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.auth_id:
        return redirect("/")
    error = None
    status = 200
    retry_after = None
    if request.method == "POST":
        # Do not trust X-Forwarded-For. Bound state and attempts per direct peer.
        peer = request.remote_addr or "unknown"
        now = time.monotonic()
        with AUTH_LOCK:
            for key in list(LOGIN_ATTEMPTS):
                LOGIN_ATTEMPTS[key] = [t for t in LOGIN_ATTEMPTS[key] if t > now - LOGIN_WINDOW]
                if not LOGIN_ATTEMPTS[key]:
                    del LOGIN_ATTEMPTS[key]
            attempts = LOGIN_ATTEMPTS.get(peer, [])
            if len(attempts) >= LOGIN_LIMIT:
                retry_after = max(1, math.ceil(LOGIN_WINDOW - (now - attempts[0])))
            elif peer not in LOGIN_ATTEMPTS and len(LOGIN_ATTEMPTS) >= 2048:
                retry_after = LOGIN_WINDOW
            else:
                LOGIN_ATTEMPTS.setdefault(peer, []).append(now)
        if retry_after is not None:
            error, status = "登录尝试过于频繁，请稍后重试。", 429
        else:
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            username_ok = hmac.compare_digest(
                username.encode("utf-8"), app.config["AUTH_USERNAME"].encode("utf-8")
            )
            password_ok = check_password_hash(app.config["AUTH_PASSWORD_HASH"], password)
            if username_ok and password_ok:
                session.clear()
                session.permanent = True
                sid = secrets.token_urlsafe(32)
                session["auth_id"] = sid
                csrf_token()  # Rotate CSRF along with the authenticated session.
                with AUTH_LOCK:
                    AUTH_SESSIONS[sid] = time.monotonic() + AUTH_TTL
                    LOGIN_ATTEMPTS.pop(peer, None)
                return redirect("/", code=303)
            error, status = "账号或密码错误。", 401
    response = Response(render_template_string(
        LOGIN_PAGE, configured=True, error=error, csrf_token=csrf_token()
    ), status=status, content_type="text/html; charset=utf-8")
    if retry_after is not None:
        response.headers["Retry-After"] = str(retry_after)
    return response


@app.post("/logout")
def logout():
    with AUTH_LOCK:
        AUTH_SESSIONS.pop(g.auth_id, None)
    with STATE_LOCK:
        current = ACTIVE_RUN
    if current is not None and getattr(current, "owner_session", None) == g.auth_id:
        current.stop()
    session.clear()
    return jsonify(ok=True)


@app.after_request
def response_headers(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
        "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    )
    return response


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="请求过大，请将输入控制在 2 MiB 以内"), 413


@app.get("/")
def index():
    return render_template_string(PAGE, csrf_token=csrf_token(), username=app.config["AUTH_USERNAME"])


@app.get("/api/config")
def configuration():
    names = model_names()
    effort = os.environ.get("REASONING_EFFORT", "auto").strip().lower()
    if effort == "ultra-high":
        effort = "xhigh"
    histories = []
    if TEMP_DIR.is_dir():
        try:
            paths = sorted(TEMP_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            histories = [p.stem for p in paths if HISTORY_NAME.fullmatch(p.stem)][:40]
        except OSError:
            app.logger.warning("Could not list history files", exc_info=True)
    return jsonify(
        models=names, default_model=names[0] if names else "",
        efforts=EFFORT_LEVELS, default_effort=effort if effort in EFFORT_LEVELS else "auto",
        histories=histories, busy=RUN_LOCK.locked(),
        credentials_configured=bool(os.environ.get("BASE_URL") and os.environ.get("API_KEY")),
    )


@app.post("/run")
def run():
    global ACTIVE_RUN
    text = request.form.get("input", "")
    history = request.form.get("history", "new").strip() or "new"
    model = request.form.get("model", os.environ.get("MODEL_NAME", "")).strip()
    effort = request.form.get("effort", os.environ.get("REASONING_EFFORT", "auto")).strip().lower()
    if effort == "ultra-high":
        effort = "xhigh"
    if not text.strip():
        return jsonify(error="请输入非空的任务指令"), 400
    if not HISTORY_NAME.fullmatch(history) or history in {".", ".."}:
        return jsonify(error="history 只能包含 1–64 位字母、数字、点、下划线和短横线，不能是 . 或 .."), 400
    if not model or len(model) > 200 or any(ord(char) < 32 or ord(char) == 127 for char in model):
        return jsonify(error="请输入有效模型名称（1–200 字符，不含控制字符）"), 400
    if effort not in EFFORT_LEVELS:
        return jsonify(error="不支持的推理强度"), 400
    try:
        max_steps = int(request.form.get("max_steps", "20"))
        max_tokens = int(request.form.get("max_tokens", "65536"))
        if not 1 <= max_steps <= 1000 or not 1 <= max_tokens <= 1048576:
            raise ValueError
    except ValueError:
        return jsonify(error="最大步数需为 1–1000，最大输出 tokens 需为 1–1048576 的整数"), 400
    if not RUN_LOCK.acquire(blocking=False):
        return jsonify(error="已有任务正在运行，请等待结束后再试"), 409
    current = AgentRun(text, history, model, effort, max_steps, max_tokens)
    current.owner_session = getattr(g, "auth_id", None)
    try:
        current.start()
        # Logout may race with subprocess startup. Publish only while this
        # session is still valid; otherwise immediately clean up the child.
        with AUTH_LOCK:
            still_authenticated = AUTH_SESSIONS.get(current.owner_session, 0) > time.monotonic()
            if still_authenticated:
                with STATE_LOCK:
                    ACTIVE_RUN = current
        if not still_authenticated:
            current.close()
            return jsonify(error="登录已失效，请重新解锁控制台"), 401
    except Exception as exc:
        app.logger.exception("Could not start agent")
        current.close()
        return jsonify(error=f"无法启动 Agent：{exc}"), 500
    response = Response(
        current.stream(), content_type="application/x-ndjson; charset=utf-8",
        headers={"X-Accel-Buffering": "no"},
    )
    # Also covers responses closed before the generator is first iterated.
    response.call_on_close(current.close)
    return response


@app.post("/stop")
def stop():
    data = request.get_json(silent=True) if request.is_json else request.form
    if not isinstance(data, dict) and request.is_json:
        return jsonify(error="无效的停止请求"), 400
    run_id = (data or {}).get("id")
    with STATE_LOCK:
        current = ACTIVE_RUN
    if current is None or current.id != run_id:
        return jsonify(error="该任务已结束或运行标识不匹配"), 409
    current.stop()
    return jsonify(ok=True)


@app.get("/log")
def download_log():
    if not LOG_PATH.is_file():
        return jsonify(error="尚无日志，请先运行一个任务"), 404
    return send_file(LOG_PATH, mimetype="text/plain", as_attachment=True,
                     download_name="agent.log", conditional=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Agent Mission Control (trusted local use only)")
    parser.add_argument("--port", type=int, default=5500)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address; default 127.0.0.1. Non-loopback requires a trusted network.")
    parser.add_argument("--trusted-host", action="append", default=[],
                        help="Deprecated compatibility option; ignored (all Hosts are trusted)")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        app.logger.warning("Non-loopback binding: this console can execute commands. "
                           "Use only on a trusted network with access controls.")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
