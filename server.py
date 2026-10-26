"""Local agent console. Run ``python server.py`` and open https://127.0.0.1:5500.

MODEL_NAME selects the default model; AGENT_MODELS optionally supplies a comma-
or newline-separated dropdown list. API credentials stay in the environment.
Set AGENT_WEB_USERNAME and AGENT_WEB_PASSWORD to unlock the console.
HTTPS is enabled by default with a persistent, self-signed localhost certificate.
For remote access, supply a trusted certificate with --ssl-cert and --ssl-key.
This console executes agent tools on your machine; restrict remote access.
"""

import argparse
import atexit
import codecs
import json
import hashlib
import hmac
import math
import secrets
import os
import re
import signal
import ssl
import stat
import subprocess
import sys
import threading
import tempfile
import time
import uuid
from pathlib import Path
from datetime import datetime, timedelta

from flask import Flask, Response, g, jsonify, redirect, render_template_string, request, send_file, session
from werkzeug.security import check_password_hash, generate_password_hash

from core.web_tls import create_ssl_context

AGENT_DIR = Path(__file__).resolve().parent
LOG_PATH = AGENT_DIR / "log"
TEMP_DIR = AGENT_DIR / "temp"
HISTORY_PREVIEW_BYTES = 256 * 1024
HISTORY_PREVIEW_LINES = 50
RESUME_PROMPT = (
    "Continue the previous task from the saved conversation and observations. "
    "The previous run may have been interrupted. If a tool call has no saved "
    "observation, its outcome is unknown: inspect the current state before "
    "retrying it, and do not assume it failed or blindly repeat side effects."
)
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
    SESSION_COOKIE_SECURE=True,  # Login sessions must only travel over HTTPS.
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
.log-think { color:#c7a5ff; }
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

.tabs { display:flex; gap:8px; margin:0 0 20px; }
.tabs [aria-selected=true] { color:var(--accent); border-color:#47796a; background:#19342e; }
.history-layout { display:grid; grid-template-columns:320px minmax(0,1fr); gap:20px; align-items:start; }
.history-list { max-height:50vh; overflow:auto; padding:10px; scrollbar-color:#344a64 #0b1220; }
.history-file { display:block; width:100%; text-align:left; padding:12px; margin:0 0 6px;
  border:1px solid transparent; border-radius:10px; background:#0b1320; color:var(--text); }
.history-file[aria-pressed=true] { border-color:#47796a; background:#19342e; }
.history-file strong { display:block; font-size:13px; overflow-wrap:anywhere; }
.history-file small { display:block; color:var(--muted); margin-top:4px; }
.history-preview { height:clamp(360px,65vh,850px); overflow:auto; padding:18px; background:#080e18; scrollbar-color:#344a64 #0b1220; }
.history-preview pre { white-space:pre-wrap; overflow-wrap:anywhere; margin:0; font:12px/1.8 ui-monospace, monospace; color:#c9d8ec; }
.history-entry { margin-bottom:14px; border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.history-entry-head { padding:9px 12px; background:#152034; color:#b9cde8; font-size:11px; overflow-wrap:anywhere; }
.history-entry pre { padding:12px; }
.history-entry details { border-top:1px solid var(--line); }
.history-entry summary { cursor:pointer; color:var(--muted); padding:6px 12px; font-size:11px; }
.history-error { color:var(--danger); font-size:12px; margin:8px 0; }
.history-status { min-height:22px; color:var(--muted); font-size:12px; padding:0 18px 12px; overflow-wrap:anywhere; }
#history-title { overflow-wrap:anywhere; min-width:0; }
#create-history-form textarea { min-height:100px; }
@media (max-width:950px) { .history-layout { grid-template-columns:1fr; } .history-list { max-height:30vh; } }
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
  <nav class="tabs" role="tablist" aria-label="控制台视图">
    <button type="button" class="btn" id="tab-run" role="tab" aria-selected="true" aria-controls="run-panel">▶ 运行控制台</button>
    <button type="button" class="btn" id="tab-history" role="tab" aria-selected="false" aria-controls="history-panel" tabindex="-1">◷ History / 会话文件</button>
  </nav>
  <main id="run-panel" class="workspace" role="tabpanel" aria-labelledby="tab-run">
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
          <div class="row"><input id="history" name="history" value="new" required maxlength="255" list="history-list" spellcheck="false" autocomplete="off"><button class="btn small" id="new-history" type="button">＋ 新会话</button></div>
          <datalist id="history-list"></datalist>
          <p class="help">同名会话继续上下文；卡住时可停止后恢复。刷新或断线不会停止后台任务。</p>
          <button type="button" class="btn small" id="browse-history" style="margin-top:8px">浏览 / 预览会话文件 →</button>
        </div>
        <div class="field">
          <label for="input">任务指令 / Prompt</label>
          <textarea id="input" name="input" placeholder="今天让 Agent 帮你完成什么？&#10;&#10;描述目标、约束，以及你期待的结果。"></textarea>
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
        <div class="row" style="margin-top:10px"><button class="btn" id="resume" type="button" disabled>↻ 恢复所选会话</button><button class="btn small" id="reconnect" type="button">重连后台任务</button></div>
        <p class="help">恢复时留空指令即可继续原任务；也可输入补充要求。从最后保存的上下文继续，不会自动重放结果未知的工具。</p>
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
      <div class="legend" aria-hidden="true"><span class="log-system">SYSTEM</span><span class="log-think">THINK</span><span class="log-action">ACTION</span><span class="log-observation">OBSERVATION</span><span class="log-final">FINAL</span></div>
    </section>
  </main>

  <section id="history-panel" role="tabpanel" aria-labelledby="tab-history" hidden>
    <div class="history-layout">
      <section class="panel" aria-label="History 文件列表">
        <div class="panel-head"><span class="panel-title">temp / *.jsonl <small id="history-total"></small></span><button type="button" class="btn small" id="history-refresh">刷新</button></div>
        <div style="padding:14px 18px">
          <label for="history-search">搜索文件</label><input id="history-search" type="search" placeholder="按文件名搜索…">
          <p class="help">读取当前 server.py 同级的 temp 目录；按修改时间排序。</p>
        </div>
        <div id="history-files" class="history-list" aria-label="会话文件"></div>
        <div id="history-list-status" class="history-status" role="status"></div>
        <details id="history-create" class="form-body" style="border-top:1px solid var(--line)">
          <summary style="cursor:pointer;color:var(--accent)">＋ 新建 / 导入 JSONL 文件</summary>
          <form id="create-history-form" style="margin-top:14px">
            <div class="field"><label for="history-new-name">文件名（自动补 .jsonl）</label><input id="history-new-name" name="name" required maxlength="255" placeholder="例如 experiment-01" autocomplete="off"></div>
            <div class="field"><label for="history-upload">导入本地文件（可选）</label><input id="history-upload" type="file" accept=".jsonl,application/json,text/plain"></div>
            <div class="field"><label for="history-new-content">初始 JSONL 内容（可选）</label><textarea id="history-new-content" name="content" spellcheck="false" placeholder='每行一个 JSON：{"role":"user","content":"任务内容"}'></textarea>
            <p class="help">留空创建空会话；导入记录需有 role 和 content。只新建文件，不覆盖或改写原有记录。</p></div>
            <button type="submit" class="btn" id="history-create-submit">创建文件</button>
            <div id="history-create-status" class="help" role="status"></div>
          </form>
        </details>
      </section>
      <section class="panel" aria-label="History 预览">
        <div class="panel-head console-top"><span class="panel-title" id="history-title">选择一个 JSONL 文件</span><div class="tools"><button type="button" class="btn small" id="history-use" disabled>恢复此会话 →</button><button type="button" class="btn small danger" id="history-delete" disabled>删除文件…</button></div></div>
        <div class="toolbar">
          <div class="tools"><label for="history-view">预览方式 <select id="history-view"><option value="readable">会话内容</option><option value="raw">原始 JSONL</option></select></label></div>
          <button type="button" class="btn small" id="history-preview-refresh" disabled>重新读取</button>
        </div>
        <div id="history-preview" class="history-preview" tabindex="0"><p>在左侧选择文件，查看消息、工具输出及原始 JSONL。</p></div>
        <div class="console-footer"><span id="history-page-info">每页最多 50 行 / 256 KiB，不修改原文件。</span><div class="tools"><button type="button" class="btn small" id="history-prev" disabled>上一页</button><button type="button" class="btn small" id="history-next" disabled>下一页</button></div></div>
        <div id="history-preview-status" class="history-status" role="status"></div>
        <p class="help" style="padding:0 18px 14px">删除需输入完整文件名确认；任务运行期间不可删除。仅删除 temp 下的正式 history，任务日志和 web-runs 副本仍保留，不是完整隐私清理。</p>
      </section>
    </div>
  </section>
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
    cancelStream(); running = false; runId = null;
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
let streamController = null, eventOffset = 0, reconnectTimer = null, restoring = false;
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
  updateHistoryUse();
  for (const el of form.querySelectorAll("input, select, textarea, button")) el.disabled = busy;
  $("run").disabled = busy || !config;
  $("stop").disabled = !busy || !runId;
  $("resume").disabled = busy || !config;
  $("reconnect").disabled = !config;
  $("run").textContent = busy ? "◌ 运行中…" : "▶ 开始运行";
}
function tick() {
  const seconds = Math.max(0, Math.floor((Date.now() - started) / 1000));
  $("elapsed").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
function jumpToBottom() {
  userScrolled = false; viewport.scrollTop = viewport.scrollHeight; $("jump").hidden = true;
}
// BEGIN LOG HIGHLIGHTER
const LOG_COLORS = {
  think:"log-think", action:"log-action", action_input:"log-action",
  observation:"log-observation", final_answer:"log-final", system:"log-system",
};
function newLogState() {
  return {block:"", legacy:"", fence:"", fenceLength:0, lineStart:true};
}
function logColor(state) { return LOG_COLORS[state.block] || state.legacy; }
function colorLogLine(line, initial) {
  const state = {...initial}, runs = [];
  const atStart = state.lineStart;
  state.lineStart = line.endsWith("\n");
  function emit(text, color = logColor(state)) {
    if (!text) return;
    const last = runs[runs.length - 1];
    if (last && last.color === color) last.text += text;
    else runs.push({text, color});
  }
  // Fenced code belongs to its enclosing block, even when it quotes ReAct tags.
  if (state.fence) {
    emit(line);
    const closing = atStart && line.match(/^[ \t]*(`{3,}|~{3,})[ \t]*(?:\r?\n)?$/);
    if (closing && closing[1][0] === state.fence && closing[1].length >= state.fenceLength) {
      state.fence = ""; state.fenceLength = 0;
    }
    return {runs, state};
  }
  const fence = atStart && line.match(/^[ \t]*(`{3,}|~{3,})/);
  if (fence) {
    state.fence = fence[1][0]; state.fenceLength = fence[1].length;
    emit(line); return {runs, state};
  }
  // Untagged/legacy output uses a heading until the next heading. Never let
  // "Action:" or a traceback inside an observation override its block color.
  if (!state.block && atStart) {
    const heading = line.match(/^[ \t]*(Think|Action[ \t_]+Input|Action|Observation|Final[ \t_]+Answer|System)[ \t]*:/i);
    if (heading) {
      state.legacy = LOG_COLORS[heading[1].toLowerCase().replace(/[ \t]+/g, "_")];
    } else if (/^[ \t]*(?:==== SYSTEM ====|history_context_usage:)/i.test(line)) {
      state.legacy = "log-system";
    } else if (/^[ \t]*(?:Traceback\b|(?:[\w.]*Error|Exception):)/.test(line)) {
      state.legacy = "log-error";
    }
  }
  const tags = /<([|\\/])(think|action_input|action|observation|final_answer|system)\|>/gi;
  let offset = 0;
  for (const match of line.matchAll(tags)) {
    emit(line.slice(offset, match.index));
    const name = match[2].toLowerCase(), opening = match[1] === "|";
    if (opening && !state.block) {
      state.block = name; state.legacy = "";
      emit(match[0]);
    } else if (!opening && state.block === name) {
      emit(match[0]); // The closing tag has the same color as its body.
      state.block = ""; state.legacy = "";
    } else {
      // Tags quoted inside another block are literal content, not nested blocks.
      emit(match[0]);
    }
    offset = match.index + match[0].length;
  }
  emit(line.slice(offset));
  return {runs, state};
}
function advanceLogState(text, initial) {
  let state = {...initial};
  for (const line of text.match(/[^\n]*\n|[^\n]+$/g) || []) state = colorLogLine(line, state).state;
  return state;
}
function safeLogCut(text, cut) {
  // Don't cut through a delimiter when trimming a very long unbroken line.
  const start = text.lastIndexOf("<", cut - 1);
  if (start >= 0 && cut - start < 32) {
    const tag = text.slice(start).match(/^<([|\\/])(think|action_input|action|observation|final_answer|system)\|>/i);
    if (tag && start + tag[0].length > cut) return start;
    const tail = text.slice(start).toLowerCase();
    if (Object.keys(LOG_COLORS).some(name =>
      ["|", "\\", "/"].some(marker => (`<${marker}${name}|>`).startsWith(tail)))) return start;
  }
  return cut;
}
class LogHighlighter {
  constructor(container, limit) {
    this.container = container; this.limit = limit; this.reset();
  }
  reset(keepContext = false) {
    let state = newLogState(), prefix = "";
    if (keepContext) {
      // Retain a small hidden suffix so a tag split at the clear button still
      // completes correctly, without retaining the entire cleared log.
      const cut = safeLogCut(this.partial, Math.max(0, this.partial.length - 64));
      state = advanceLogState(this.partial.slice(0, cut), this.lineState);
      prefix = this.partial.slice(cut);
    }
    this.headState = {...state}; this.headPrefix = prefix;
    this.lineState = {...state}; this.partial = prefix; this.hidden = prefix.length;
    this.partialNode = null; this.text = ""; this.trimmed = false;
    this.container.replaceChildren();
  }
  append(text) {
    this.text += text;
    if (this.text.length > this.limit) {
      let cut = this.text.length - this.limit;
      const newline = this.text.indexOf("\n", cut);
      if (newline >= 0 && newline - cut < this.limit / 2) cut = newline + 1;
      cut = safeLogCut(this.text, cut);
      // Advance over discarded text BEFORE rebuilding, so clipping away an
      // opening tag does not remove the color of its remaining body.
      this.headState = advanceLogState(this.headPrefix + this.text.slice(0, cut), this.headState);
      this.headPrefix = ""; this.text = this.text.slice(cut);
      this.lineState = {...this.headState}; this.partial = ""; this.hidden = 0;
      this.partialNode = null; this.container.replaceChildren(); this.trimmed = true;
      this.render(this.text);
    } else {
      this.render(text);
    }
  }
  render(text) {
    // Reparse only the unfinished last line. This handles streamed delimiters
    // split at ANY character, without repeatedly rebuilding the whole log.
    text = this.partial + text;
    if (this.partialNode) this.partialNode.remove();
    this.partialNode = null; this.partial = "";
    const fragment = document.createDocumentFragment();
    for (const line of text.match(/[^\n]*\n|[^\n]+$/g) || []) {
      const {runs, state} = colorLogLine(line, this.lineState);
      const node = document.createElement("span");
      let skip = this.hidden;
      for (const run of runs) {
        if (skip >= run.text.length) { skip -= run.text.length; continue; }
        const span = document.createElement("span");
        span.className = run.color; span.textContent = run.text.slice(skip); skip = 0;
        node.appendChild(span); // Never use innerHTML for model/tool output.
      }
      if (node.hasChildNodes()) fragment.appendChild(node);
      if (line.endsWith("\n")) {
        this.lineState = state; this.hidden = Math.max(0, this.hidden - line.length);
      } else {
        this.partial = line; this.partialNode = node.hasChildNodes() ? node : null;
      }
    }
    this.container.appendChild(fragment);
  }
}
// END LOG HIGHLIGHTER
const logHighlighter = new LogHighlighter(out, MAX_DISPLAY);
function flushOutput() {
  clearTimeout(paintTimer); paintTimer = null;
  if (!pendingOutput) return;
  const text = pendingOutput; pendingOutput = "";
  logHighlighter.append(text);
  rawLog = logHighlighter.text; trimmed = logHighlighter.trimmed;
  $("empty").hidden = true; out.hidden = false;
  $("line-count").textContent = `${trimmed ? "最近 " : ""}${rawLog.split("\n").length}`;
  if ($("follow").checked) jumpToBottom();
  else $("jump").hidden = false;
}
function appendOutput(text) {
  pendingOutput += text;
  if (!paintTimer) paintTimer = setTimeout(flushOutput, 50);
}
function clearOutput(keepContext = false) {
  if (keepContext) flushOutput();
  logHighlighter.reset(keepContext);
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
    $("notice").textContent = warnings.join(" "); $("notice").hidden = !warnings.length;
    message("准备就绪。选择模型并输入任务。");
    await restoreRun();
  } catch (error) { setState("error", "配置加载失败"); message(`${error.message}，请刷新重试。`, true); }
}
function applyRunSettings(data, clearPrompt = true) {
  if (config.models.includes(data.model)) $("model").value = data.model;
  else { $("model").value = CUSTOM; $("custom-model").value = data.model; }
  updateModel();
  $("history").value = data.history;
  $("effort").value = data.effort;
  $("max-steps").value = data.max_steps;
  $("max-tokens").value = data.max_tokens;
  if (clearPrompt) $("input").value = "";
  $("input-count").textContent = $("input").value.length.toLocaleString();
  $("run-meta").textContent = `${data.model} · ${data.history}`;
}
function cancelStream() {
  clearTimeout(reconnectTimer);
  if (streamController) streamController.abort();
  streamController = null;
}
function finishRun(item) {
  if (item.error && item.code === 0) item.code = -1;
  flushOutput(); clearInterval(timer);
  $("elapsed").textContent = `${Math.floor((item.elapsed || 0) / 60).toString().padStart(2, "0")}:${Math.floor((item.elapsed || 0) % 60).toString().padStart(2, "0")}`;
  const interrupted = item.interrupted;
  setState(interrupted || item.stopped ? "stopped" : item.code === 0 ? "success" : "error",
    interrupted ? "已中断" : item.stopped ? "已停止" : item.code === 0 ? "已完成" : "运行失败");
  message(interrupted ? "服务曾重启或运行中断。可点击“恢复所选会话”，从已保存的 history 继续。" :
    item.stopped ? "任务已停止，日志和 history 已保留。留空指令后点击“恢复所选会话”即可继续。" :
    item.code === 0 ? "运行完成。可在同一会话输入新指令继续。" :
    (item.error || `Agent 退出码 ${item.code}`) + "。修复问题后可恢复所选会话。", !item.stopped && !interrupted && item.code !== 0);
  if (item.history_warning) message($("status-message").textContent + " " + item.history_warning);
  runId = null; setBusy(false);
  if (!$("history-panel").hidden) refreshHistories();
}
async function readRunStream(response, controller) {
  if (!response.ok) throw new Error(await responseError(response));
  if (!response.body) throw new Error("浏览器不支持流式响应，请使用现代浏览器。");
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = "", finished = false;
  function consume(line) {
    if (!line.trim() || streamController !== controller) return;
    const item = JSON.parse(line);
    if (item.type === "start") {
      runId = item.id; applyRunSettings(item); setBusy(true);
      started = Date.parse(item.started_at) || Date.now();
      clearInterval(timer); tick(); timer = setInterval(tick, 1000);
      setState("running", "运行中");
      message("已连接后台任务；刷新/断线不会停止运行，卡住时可点“停止”后恢复。");
    } else if (item.type === "output") appendOutput(item.text);
    else if (item.type === "done") { finished = true; finishRun(item); }
    else if (item.type === "error") throw new Error(item.message);
    if (Number.isSafeInteger(item.offset)) eventOffset = item.offset;
  }
  try {
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
    if (!finished && streamController === controller) throw new Error("日志连接中断");
  } finally {
    try { await reader.cancel(); } catch (_) {}
    reader.releaseLock();
  }
}
function streamFailed(error, controller) {
  if (streamController !== controller || error.name === "AbortError") return;
  flushOutput();
  setState("stopped", "等待重连");
  message(`${error.message}；3 秒后重新连接。后台任务不会因断线被取消。`, true);
  // Query the backend even if POST /run lost its response before the start
  // event; never retry POST automatically (that could launch a duplicate task).
  reconnectTimer = setTimeout(() => {
    if (runId) connectRun(runId);
    else restoreRun();
  }, 3000);
}
async function connectRun(id) {
  cancelStream();
  const controller = new AbortController(); streamController = controller;
  runId = id; setBusy(true); setState("running", "连接中");
  try {
    const response = await apiFetch(`/api/runs/${id}/events?offset=${eventOffset}`, {cache:"no-store", signal:controller.signal});
    if (response.status === 404 || response.status === 400) {
      // Lost disk state or an invalidated cursor: rediscover instead of retrying
      // the same cursor forever.
      eventOffset = 0; runId = null;
      throw new Error(await responseError(response));
    }
    if (response.ok) {
      setState("running", "运行中"); message("已重新连接后台日志。");
    }
    await readRunStream(response, controller);
  } catch (error) { streamFailed(error, controller); }
}
async function restoreRun() {
  if (restoring) return;
  restoring = true;
  cancelStream();
  const controller = new AbortController(); streamController = controller;
  setBusy(true);
  try {
    const response = await apiFetch("/api/runs/latest", {cache:"no-store", signal:controller.signal});
    if (!response.ok) throw new Error(await responseError(response));
    const {run: data} = await response.json();
    if (streamController !== controller) return;
    if (!data) { runId = null; clearInterval(timer); setBusy(false); setState("idle", "待命"); message("没有已保存的 Web 运行；可从 History 选择旧会话恢复。"); return; }
    applyRunSettings(data);
    clearOutput(); eventOffset = 0;
    started = Date.parse(data.started_at) || Date.now();
    clearInterval(timer); tick(); timer = setInterval(tick, 1000);
    // This also replays completed/interrupted runs without executing tools.
    connectRun(data.id);
  } catch (error) { streamFailed(error, controller); }
  finally { restoring = false; }
}
async function startRun(resume = false) {
  if (running || !config) return;
  if (!resume && !$("input").value.trim()) { message("请输入任务指令，或点击“恢复所选会话”。", true); $("input").focus(); return; }
  if (!form.reportValidity()) return;
  if (!selectedModel()) { message("请选择或输入模型名称。", true); $("custom-model").focus(); return; }
  const body = new FormData(form);
  body.set("model", selectedModel());
  if (resume) body.set("resume", "1");
  const model = selectedModel(), history = $("history").value;
  cancelStream();
  const controller = new AbortController(); streamController = controller;
  savePreferences(); clearOutput(); eventOffset = 0; runId = null; setBusy(true);
  setState("running", "连接中"); message(resume ? "正在从 history 恢复 Agent…" : "正在启动 Agent…");
  started = Date.now(); tick(); clearInterval(timer); timer = setInterval(tick, 1000);
  $("run-meta").textContent = `${model} · ${history}`;
  try {
    const response = await apiFetch("/run", {method:"POST", body, signal:controller.signal});
    if (!response.ok) {
      const error = await responseError(response);
      if (response.status === 409) { await restoreRun(); return; }
      if (response.status >= 500) throw new Error(error);
      clearInterval(timer); setBusy(false); setState("error", "启动失败"); message(error, true);
      return;
    }
    await readRunStream(response, controller);
  } catch (error) { streamFailed(error, controller); }
}
form.addEventListener("submit", event => { event.preventDefault(); startRun(); });
$("resume").addEventListener("click", () => startRun(true));
$("reconnect").addEventListener("click", restoreRun);
$("stop").addEventListener("click", async () => {
  if (!runId) return;
  $("stop").disabled = true; message("正在停止 Agent 及其工具进程…");
  try {
    const response = await apiFetch("/stop", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({id:runId})});
    if (!response.ok) throw new Error(await responseError(response));
  } catch (error) { message(error.message, true); $("stop").disabled = !running || !runId; }
});
$("new-history").addEventListener("click", () => {
  showTab("history");
  $("history-create").open = true;
  $("history-new-name").value = `web-${new Date().toISOString().replace(/[-:.TZ]/g, "")}`;
  $("history-new-name").focus();
});
$("browse-history").addEventListener("click", () => {
  const name = $("history").value.endsWith(".jsonl") ? $("history").value : $("history").value + ".jsonl";
  showTab("history");
  selectHistory(name);
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
$("clear").addEventListener("click", () => { clearOutput(running); toast("仅清空当前显示，不删除磁盘上的 log。"); });
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
    cancelStream(); running = false; runId = null; clearOutput();
    window.location.replace("/login");
  } catch (error) { toast(error.message); $("logout").disabled = false; }
});
window.addEventListener("pageshow", event => {
  if (event.persisted) window.location.reload();
});
// Navigation disconnects only the viewer. Explicit Stop/Logout still cancels.
window.addEventListener("pagehide", cancelStream);

// BEGIN HISTORY UI
let historyFiles = [], selectedHistory = null, previewData = null;
let previewPages = [], previewOffset = 0, previewLoading = false;
let listRequestId = 0, previewRequestId = 0, historyDeleting = false;
function historyStatus(id, text, error = false) {
  $(id).textContent = text; $(id).classList.toggle("history-error", error);
}
function sizeLabel(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KiB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MiB`;
}
function showTab(name) {
  for (const key of ["run", "history"]) {
    const active = key === name;
    $(`tab-${key}`).setAttribute("aria-selected", String(active));
    $(`tab-${key}`).tabIndex = active ? 0 : -1;
    $(`${key}-panel`).hidden = !active;
  }
  if (name === "history") refreshHistories();
  else if ($("follow").checked) jumpToBottom();
}
for (const name of ["run", "history"]) {
  $(`tab-${name}`).addEventListener("click", () => showTab(name));
  $(`tab-${name}`).addEventListener("keydown", event => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const next = event.key === "Home" ? "run" : event.key === "End" ? "history" : name === "run" ? "history" : "run";
      showTab(next); $(`tab-${next}`).focus();
    }
  });
}
function renderHistoryList() {
  const query = $("history-search").value.toLocaleLowerCase();
  const shown = historyFiles.filter(file => file.name.toLocaleLowerCase().includes(query));
  $("history-total").textContent = `(${historyFiles.length})`;
  const fragment = document.createDocumentFragment();
  for (const file of shown) {
    const button = document.createElement("button"); button.type = "button"; button.className = "history-file";
    button.setAttribute("aria-pressed", String(file.name === selectedHistory));
    const title = document.createElement("strong"); title.textContent = file.name;
    const meta = document.createElement("small");
    meta.textContent = `${sizeLabel(file.size)} · ${new Date(file.modified).toLocaleString()}${file.active ? " · 正在写入" : ""}`;
    button.append(title, meta);
    button.addEventListener("click", () => selectHistory(file.name));
    fragment.appendChild(button);
  }
  $("history-files").replaceChildren(fragment);
  historyStatus("history-list-status", historyFiles.length ? `${shown.length} / ${historyFiles.length} 个文件` : "还没有 JSONL 文件，可以在下方新建。");
}
async function refreshHistories() {
  const requestId = ++listRequestId;
  $("history-refresh").disabled = true;
  historyStatus("history-list-status", "正在读取文件列表…");
  try {
    const response = await apiFetch("/api/histories", {cache:"no-store"});
    if (!response.ok) throw new Error(await responseError(response));
    const data = await response.json();
    if (requestId !== listRequestId) return;
    historyFiles = data.files;
    if (selectedHistory && !historyFiles.some(file => file.name === selectedHistory)) {
      clearHistorySelection("所选文件已不存在，请选择其他会话。");
    }
    renderHistoryList(); updatePreviewControls();
    $("history-list").replaceChildren();
    for (const file of historyFiles) $("history-list").appendChild(new Option(file.name, file.name));
  } catch (error) {
    if (requestId === listRequestId) historyStatus("history-list-status", error.message, true);
  } finally { if (requestId === listRequestId) $("history-refresh").disabled = false; }
}
function updateHistoryUse() {
  $("history-use").disabled = running || historyDeleting || previewLoading || !previewData;
  $("history-use").title = running ? "任务运行期间不能切换会话，可继续浏览预览。" : "";
  const active = previewData?.file.active || historyFiles.some(file => file.name === selectedHistory && file.active);
  $("history-delete").disabled = running || historyDeleting || previewLoading || !previewData ||
    previewData.file.name !== selectedHistory || !previewData.file.version || active;
  $("history-delete").title = running || active ? "请等待后台任务结束并刷新列表后再删除。" :
    "仅删除正式 history，需输入完整文件名确认；日志和任务工作副本保留。";
}
function updatePreviewControls() {
  updateHistoryUse();
  $("history-prev").disabled = previewLoading || !previewPages.length;
  $("history-next").disabled = previewLoading || !previewData?.has_more;
  $("history-preview-refresh").disabled = previewLoading || !selectedHistory;
}
function renderHistoryPreview() {
  const container = $("history-preview");
  container.replaceChildren();
  if (!previewData) return;
  if (!previewData.raw) {
    const empty = document.createElement("p"); empty.textContent = "这是一个空会话文件。选择“恢复此会话”后，Agent 会在其中追加记录。";
    container.appendChild(empty); return;
  }
  if ($("history-view").value === "raw") {
    const raw = document.createElement("pre"); raw.textContent = previewData.raw;
    container.appendChild(raw); return;
  }
  for (const entry of previewData.entries) {
    if (!entry.raw.trim()) continue;
    const card = document.createElement("article"); card.className = "history-entry";
    const head = document.createElement("div"); head.className = "history-entry-head";
    const record = entry.record;
    const isMessage = record && typeof record === "object" && typeof record.content === "string";
    head.textContent = isMessage ? [record.role || "message", record.state, record.timestamp].filter(Boolean).join(" · ") : "JSONL 记录";
    card.appendChild(head);
    if (entry.error) {
      const warning = document.createElement("p"); warning.className = "history-error"; warning.style.padding = "0 12px";
      warning.textContent = entry.error; card.appendChild(warning);
    }
    const content = document.createElement("pre");
    if (isMessage) new LogHighlighter(content, 2 * 1024 * 1024).append(record.content);
    else content.textContent = entry.error ? entry.raw : JSON.stringify(record, null, 2);
    card.appendChild(content);
    if (isMessage) {
      const details = document.createElement("details"), summary = document.createElement("summary"), raw = document.createElement("pre");
      summary.textContent = "查看该条原始 JSON"; raw.textContent = entry.raw;
      details.append(summary, raw); card.appendChild(details);
    }
    container.appendChild(card);
  }
}
async function loadHistoryPreview(offset, pages) {
  if (!selectedHistory) return;
  const name = selectedHistory, requestId = ++previewRequestId;
  previewLoading = true; updatePreviewControls();
  historyStatus("history-preview-status", "正在读取预览…");
  try {
    const response = await apiFetch(`/api/histories/preview?${new URLSearchParams({name, offset})}`, {cache:"no-store"});
    if (!response.ok) throw new Error(await responseError(response));
    const data = await response.json();
    if (requestId !== previewRequestId) return;
    previewData = data; previewOffset = offset; previewPages = pages;
    $("history-title").textContent = data.file.name;
    $("history-page-info").textContent = `第 ${pages.length + 1} 页 · 字节 ${offset}–${data.next_offset} / ${data.file.size} · ${sizeLabel(data.file.size)}`;
    renderHistoryPreview(); $("history-preview").scrollTop = 0;
    historyStatus("history-preview-status", data.file.active ? "Agent 正在追加此文件，点击“重新读取”可查看更新。" : "只读预览，不会修改原有 history。");
  } catch (error) {
    if (requestId !== previewRequestId) return;
    previewData = null; $("history-page-info").textContent = "预览未加载，请重新读取。";
    $("history-preview").replaceChildren();
    historyStatus("history-preview-status", error.message, true);
  } finally {
    if (requestId === previewRequestId) { previewLoading = false; updatePreviewControls(); }
  }
}
function selectHistory(name) {
  selectedHistory = name; previewData = null; previewPages = []; previewOffset = 0;
  $("history-title").textContent = name; $("history-preview").replaceChildren();
  renderHistoryList(); loadHistoryPreview(0, []);
}
function clearHistorySelection(status = "") {
  ++previewRequestId; // Ignore any in-flight preview for the deleted file.
  selectedHistory = null; previewData = null; previewPages = []; previewOffset = 0; previewLoading = false;
  $("history-title").textContent = "选择一个 JSONL 文件";
  $("history-page-info").textContent = "每页最多 50 行 / 256 KiB，不修改原文件。";
  const hint = document.createElement("p"); hint.textContent = "在左侧选择文件，查看消息、工具输出及原始 JSONL。";
  $("history-preview").replaceChildren(hint);
  historyStatus("history-preview-status", status);
  updatePreviewControls();
}
async function deleteSelectedHistory() {
  if (running || historyDeleting || previewLoading || !previewData ||
      previewData.file.name !== selectedHistory || previewData.file.active) return;
  const {name, version} = previewData.file;
  if (!version) return;
  const confirmation = window.prompt(
    `确认永久删除正式历史文件？\n\ntemp/${name}\n\n` +
    "删除后不能直接从此文件恢复会话，本界面没有撤销功能。\n" +
    "仅删除该文件；temp/web-runs 中的任务记录、历史工作副本及 log 仍保留，不是完整隐私清理。\n\n" +
    `如需删除，请输入完整文件名（含 .jsonl）：\n${name}`, ""
  );
  if (confirmation === null) return;
  if (confirmation !== name) { toast("文件名不一致，已取消删除；文件未改动。"); return; }
  historyDeleting = true; updatePreviewControls();
  historyStatus("history-preview-status", `正在删除 temp/${name}…`);
  try {
    const response = await apiFetch("/api/histories/delete", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({name, confirm_name:confirmation, version})
    });
    if (!response.ok) throw new Error(await responseError(response));
    const data = await response.json();
    if (selectedHistory === name) clearHistorySelection(`已删除 ${name}；任务日志和工作副本仍保留。`);
    const configured = $("history").value;
    if (!running && (configured.endsWith(".jsonl") ? configured : configured + ".jsonl") === name) {
      $("history").value = "";
      message("所选 history 已删除，请选择或新建会话后再运行。");
    }
    toast(data.warning || `已删除 ${name}；仅删除正式 history。`);
    await refreshHistories();
  } catch (error) {
    await refreshHistories(); // Never automatically retry a destructive request.
    historyStatus("history-preview-status", `${error.message} 请核对文件列表，重新读取预览后再操作。`, true);
  } finally {
    historyDeleting = false; updatePreviewControls();
  }
}
$("history-delete").addEventListener("click", deleteSelectedHistory);
$("history-search").addEventListener("input", renderHistoryList);
$("history-refresh").addEventListener("click", refreshHistories);
$("history-preview-refresh").addEventListener("click", () => loadHistoryPreview(0, []));
$("history-view").addEventListener("change", renderHistoryPreview);
$("history-next").addEventListener("click", () => {
  if (previewData?.has_more && !previewLoading) loadHistoryPreview(previewData.next_offset, [...previewPages, previewOffset]);
});
$("history-prev").addEventListener("click", () => {
  if (previewPages.length && !previewLoading) loadHistoryPreview(previewPages[previewPages.length - 1], previewPages.slice(0, -1));
});
$("history-use").addEventListener("click", async () => {
  if (running || historyDeleting || !previewData) return;
  const name = previewData.file.name;
  cancelStream(); setBusy(true);
  try {
    const response = await apiFetch(`/api/runs/latest?${new URLSearchParams({history:name})}`, {cache:"no-store"});
    if (!response.ok) throw new Error(await responseError(response));
    const {run: data} = await response.json();
    if (data) applyRunSettings(data);
    else { $("input").value = ""; $("input-count").textContent = "0"; }
    $("history").value = name;
    showTab("run");
    message(`已载入 ${name}。点击“恢复所选会话”继续原任务，或输入补充指令。`);
    toast("会话已选择，尚未执行任何工具。");
  } catch (error) { message(error.message, true); }
  finally { setBusy(false); }
});
$("history-upload").addEventListener("change", async () => {
  const file = $("history-upload").files[0]; if (!file) return;
  try {
    if (file.size > 1536 * 1024) throw new Error("导入文件请控制在 1.5 MiB 内；更大的已有文件可直接放入 temp 后刷新列表。");
    $("history-new-content").value = await file.text();
    $("history-new-name").value = file.name;
    historyStatus("history-create-status", "已读取本地文件，点击“创建文件”后保存。");
  } catch (error) { historyStatus("history-create-status", error.message, true); }
});
$("create-history-form").addEventListener("submit", async event => {
  event.preventDefault();
  const body = new FormData(event.currentTarget);
  $("history-create-submit").disabled = true;
  historyStatus("history-create-status", "正在创建…");
  try {
    const response = await apiFetch("/api/histories", {method:"POST", body});
    if (!response.ok) throw new Error(await responseError(response));
    const data = await response.json();
    $("create-history-form").reset();
    historyStatus("history-create-status", `已创建 ${data.file.name}，原有文件未改动。`);
    $("history-search").value = "";
    await refreshHistories(); selectHistory(data.file.name);
  } catch (error) { historyStatus("history-create-status", error.message, true); }
  finally { $("history-create-submit").disabled = false; }
});
// END HISTORY UI

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


def run_path(run_id, suffix):
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("无效的运行标识")
    return TEMP_DIR / "web-runs" / f"{run_id}.{suffix}"


def read_run(run_id):
    path = run_path(run_id, "json")
    with STATE_LOCK:
        data = json.loads(path.read_text(encoding="utf-8"))
        active = ACTIVE_RUN is not None and ACTIVE_RUN.id == run_id
    if active:
        return {**data, "status": "running"}
    # Never resurrect a PID after a server restart: only saved context is safe
    # to resume. Tool side effects before the last checkpoint may be unknown.
    if data["status"] == "running" and not active:
        data = {**data, "status": "interrupted"}
    return data


def latest_run(history=None):
    directory = TEMP_DIR / "web-runs"
    if not directory.exists():
        return None
    latest = None
    for path in directory.glob("*.json"):
        try:
            data = read_run(path.stem)
            if history is not None and data["history"] != history:
                continue
            if latest is None or data["started_at"] > latest["started_at"]:
                latest = data
        except (OSError, ValueError, KeyError):
            app.logger.warning("Could not read run metadata: %s", path.name)
    return latest


def completion_event(data):
    return dict(
        code=data.get("code"), stopped=data["status"] == "stopped",
        interrupted=data["status"] == "interrupted",
        elapsed=data.get("elapsed", 0), error=data.get("error"),
        history_warning=data.get("history_warning"),
    )


def stream_run_events(run_id, offset=0):
    """Replay a disk-backed stream. Disconnecting only closes this reader."""
    with run_path(run_id, "events").open("rb") as file:
        file.seek(offset)
        while True:
            position = file.tell()
            line = file.readline()
            if line.endswith(b"\n"):
                item = json.loads(line)
                item["offset"] = file.tell()
                kind = item.pop("type")
                yield event(kind, **item)
                if kind == "done":
                    return
                continue
            # A writer may be between writes, or the server died mid-record.
            file.seek(position)
            data = read_run(run_id)
            if data["status"] != "running":
                # Recheck EOF after metadata: the writer publishes completion
                # only after flushing all output.
                if file.readline().endswith(b"\n"):
                    file.seek(position)
                    continue
                yield event("done", offset=position, **completion_event(data))
                return
            yield event("heartbeat", offset=position)
            time.sleep(1)


def fsync_directory(path):
    """Persist a rename, not just the contents of its new inode (POSIX)."""
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write(path, writer, before_replace=None):
    """Readers see either the previous file or the complete, fsynced new file."""
    fd, temporary = tempfile.mkstemp(prefix=".agent-publish-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as file:
            writer(file)
            file.flush()
            os.fsync(file.fileno())
        if before_replace is not None:
            before_replace()
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def validate_history_record(raw, description):
    try:
        record = strict_json_loads(raw.decode("utf-8"))
        if (not isinstance(record, dict) or record.get("role") not in {"user", "assistant"}
                or not isinstance(record.get("content"), str)):
            raise ValueError("需包含 role 和字符串 content")
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"{description}不是完整有效的 UTF-8 JSONL 消息；不会忽略或修补记录") from exc


class HistoryCheckpoint:
    """Agent writes only a private working copy; server publishes valid prefixes.

    No changes to Agent or its save_history method are necessary. A killed agent
    may leave a partial record in the working copy, but never in the public
    history. Keep that working copy for diagnosis rather than silently fixing it.
    """

    def __init__(self, name, work_path):
        self.path = history_path(name)
        self.work_path = work_path
        self.version = self.target_version()
        self.validated = 0
        self.published = 0
        self.pending_bytes = 0
        self.line_number = 0

    def target_version(self):
        if self.path.is_symlink():
            raise ValueError("不允许写入符号链接")
        try:
            info = self.path.stat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("History 必须是普通文件")
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def check_target(self):
        if self.target_version() != self.version:
            raise ValueError("History 被其他进程修改，已停止提交以避免覆盖；工作副本已保留")

    def prepare(self):
        def seed(destination):
            if self.version is None:
                return
            # Read binary: never substitute invalid UTF-8 or skip malformed JSON.
            with self.path.open("rb") as source:
                for line_number, raw in enumerate(source, 1):
                    if raw.strip():
                        validate_history_record(raw, f"History 第 {line_number} 行")
                    destination.write(raw)
                    self.validated += len(raw)
                    self.line_number = line_number
                    if not raw.endswith(b"\n"):
                        # A valid imported final JSON record need not have LF.
                        # Normalize only the private copy before Agent appends.
                        destination.write(b"\n")
                        self.validated += 1
            self.check_target()
        atomic_write(self.work_path, seed)
        self.published = self.validated if self.version and self.version[2] == self.validated else -1

    def commit(self):
        with self.work_path.open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            if size < self.validated:
                raise ValueError("History 工作副本意外变短；停止提交")
            cursor, line_number = self.validated, self.line_number
            source.seek(cursor)
            while source.tell() < size:
                raw = source.readline(size - source.tell())
                if not raw.endswith(b"\n"):
                    break  # No commit marker: writer is still in this record.
                line_number += 1
                if raw.strip():
                    validate_history_record(raw, f"History 工作副本第 {line_number} 行")
                cursor = source.tell()
            self.pending_bytes = size - cursor
            self.check_target()
            if cursor == self.published:
                return

            def copy_prefix(destination):
                source.seek(0)
                remaining = cursor
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise OSError("History 工作副本在提交期间变短")
                    destination.write(chunk)
                    remaining -= len(chunk)

            # Even SIGKILL or a failed write cannot expose half a JSON record:
            # fsync the entire snapshot before a same-filesystem atomic rename.
            atomic_write(self.path, copy_prefix, before_replace=self.check_target)
            self.version = self.target_version()
            self.validated = self.published = cursor
            self.line_number = line_number

    def discard_committed_work_copy(self):
        """Remove only an exact, fully published duplicate after successful exit.

        Call only after all history writers stopped/joined, while holding
        RUN_LOCK. Retaining a copy is safer than deleting uncertain data.
        """
        if self.pending_bytes or self.published != self.validated:
            raise ValueError("History 尚未完整提交，保留工作副本")
        self.check_target()
        if self.version is None:
            raise ValueError("正式 history 不存在，保留工作副本")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(self.work_path, flags), "rb") as work:
            work_info = os.fstat(work.fileno())
            if not stat.S_ISREG(work_info.st_mode) or work_info.st_size != self.published:
                raise ValueError("History 工作副本不是完整提交的普通文件，保留副本")
            with os.fdopen(os.open(self.path, flags), "rb") as published:
                published_info = os.fstat(published.fileno())
                if (not stat.S_ISREG(published_info.st_mode)
                        or published_info.st_size != self.published):
                    raise ValueError("正式 history 大小不匹配，保留工作副本")
                # Bound memory use. Check actual bytes, not just lengths.
                while True:
                    chunk = work.read(1024 * 1024)
                    if published.read(len(chunk)) != chunk:
                        raise ValueError("正式 history 与工作副本内容不同，保留副本")
                    if not chunk:
                        break
                if (history_version(os.fstat(work.fileno())) != history_version(work_info)
                        or history_version(os.fstat(published.fileno())) != history_version(published_info)):
                    raise ValueError("History 在核对期间变化，保留工作副本")
        self.check_target()
        if history_version(self.work_path.lstat()) != history_version(work_info):
            raise ValueError("History 工作副本已被替换，保留副本")
        # Only this run's copy, never metadata/events/formal history.
        # No startup sweep: old/interrupted copies remain for diagnosis.
        self.work_path.unlink()


class AgentRun:
    """A background subprocess whose lifetime is independent of HTTP readers."""

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
        self.checkpoint = None
        self.checkpoint_stop = threading.Event()
        self.checkpoint_error = None
        self.checkpoint_worker = None
        self.stopped = False
        self.closed = False
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.metadata = dict(
            id=self.id, input=text, history=history, model=model, effort=effort,
            max_steps=max_steps, max_tokens=max_tokens, status="running",
            started_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
        )

    def save_metadata(self):
        data = json.dumps(self.metadata, ensure_ascii=False).encode("utf-8")
        atomic_write(run_path(self.id, "json"), lambda file: file.write(data))

    def emit(self, kind, **fields):
        with run_path(self.id, "events").open("a", encoding="utf-8") as file:
            file.write(event(kind, **fields))
            file.flush()
            os.fsync(file.fileno())

    def start(self):
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        run_path(self.id, "json").parent.mkdir(mode=0o700, exist_ok=True)
        self.checkpoint = HistoryCheckpoint(self.history, run_path(self.id, "history.jsonl"))
        self.checkpoint.prepare()
        self.input_path.write_text(self.text, encoding="utf-8")
        env = os.environ.copy()
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
                "--history", str(self.checkpoint.work_path),
                "--effort", self.effort,
                "--max_steps", str(self.max_steps),
                "--max_tokens", str(self.max_tokens),
            ],
            cwd=AGENT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, bufsize=0, env=env,
            start_new_session=(os.name == "posix"),
        )
        self.emit("start", **self.metadata)
        self.save_metadata()

    def _kill(self):
        if self.process is None:
            return
        try:
            if os.name == "posix":
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

    def watch_history(self):
        try:
            while not self.checkpoint_stop.is_set():
                self.checkpoint.commit()
                self.checkpoint_stop.wait(0.2)
        except Exception as exc:
            self.checkpoint_error = exc
            # Do not keep executing tools if progress cannot be committed.
            self._kill()

    def collect(self):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        code, error = None, None
        try:
            self.checkpoint_worker = threading.Thread(
                target=self.watch_history, name=f"history-{self.id}", daemon=True,
            )
            self.checkpoint_worker.start()
            while True:
                chunk = self.process.stdout.read(8192)
                text = decoder.decode(chunk, final=not chunk)
                if text:
                    self.log_file.write(text)
                    self.log_file.flush()
                    self.emit("output", text=text)
                if not chunk:
                    break
            code = self.process.wait()
        except Exception as exc:
            app.logger.exception("Agent log collection failed")
            error = f"日志收集异常：{exc}"
        finally:
            # Freeze the writer before the final checkpoint. Stop/Logout never
            # release RUN_LOCK until that checkpoint has been fully published.
            self._kill()
            self.process.wait()
            self.checkpoint_stop.set()
            if self.checkpoint_worker and self.checkpoint_worker.ident is not None:
                self.checkpoint_worker.join()
            try:
                if self.checkpoint_error:
                    raise self.checkpoint_error
                self.checkpoint.commit()
                self.metadata.update(
                    history_committed_bytes=self.checkpoint.published,
                    history_pending_bytes=self.checkpoint.pending_bytes,
                )
                if self.checkpoint.pending_bytes:
                    warning = (
                        "中断时未写完的记录仅保留在运行工作副本中，未提交到正式 history；"
                        "恢复将从最后一条完整记录继续。"
                    )
                    self.metadata["history_warning"] = warning
                    if not self.stopped:
                        error = error or warning
            except Exception as exc:
                app.logger.exception("Could not commit agent history")
                error = f"History 提交失败：{exc}。正式 history 保持完整，工作副本已保留。"
            try:
                self.log_file.flush()
                os.fsync(self.log_file.fileno())
            except OSError as exc:
                error = error or f"日志落盘失败：{exc}"
            try:
                # Serialize Stop/Logout with the success decision and cleanup.
                # RUN_LOCK remains held until close(): a next run or UI deletion
                # cannot change the formal history during this comparison.
                with self.lock:
                    self.metadata.update(
                        status="failed" if error else "stopped" if self.stopped else "completed" if code == 0 else "failed",
                        code=code, error=error,
                        elapsed=round(time.monotonic() - self.started, 3),
                    )
                    try:
                        self.save_metadata()
                    except OSError:
                        # Keep the working copy if completion is not durable.
                        app.logger.exception("Could not save run completion")
                    else:
                        if (self.metadata["status"] == "completed"
                                and self.checkpoint_error is None
                                and self.checkpoint.pending_bytes == 0):
                            try:
                                self.checkpoint.discard_committed_work_copy()
                            except (OSError, ValueError):
                                # Reclaiming space is optional, not task success.
                                app.logger.warning(
                                    "Could not safely clean history work copy for run %s; "
                                    "leaving it for inspection", self.id, exc_info=True,
                                )
            finally:
                self.close()
            try:
                self.emit("done", **completion_event(self.metadata))
            except OSError:
                app.logger.exception("Could not save completion event")

    def stream(self):
        return stream_run_events(self.id)


def shutdown_active_run():
    """Normal server shutdown must not leave tool children writing history."""
    with STATE_LOCK:
        current = ACTIVE_RUN
    if current is not None:
        current.stop()
        worker = getattr(current, "worker", None)
        if worker is not None and worker is not threading.current_thread():
            worker.join()


atexit.register(shutdown_active_run)


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
                    "请使用 HTTPS 地址访问，登录会话 Cookie 仅通过 HTTPS 发送。"
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


def history_filename(name, add_suffix=False):
    """Accept a single filename, never a path; Unicode filenames are supported."""
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ValueError("请输入 JSONL 文件名")
    if any(char in name for char in ("/", "\\", ":")) or any(
        ord(char) < 32 or ord(char) == 127 for char in name
    ):
        raise ValueError("文件名不能包含路径分隔符或控制字符")
    if add_suffix and not name.endswith(".jsonl"):
        name += ".jsonl"
    try:
        encoded_name = name.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("文件名包含无效 Unicode 字符")
    if not name.endswith(".jsonl") or len(encoded_name) > 255:
        raise ValueError("文件名必须以 .jsonl 结尾，且 UTF-8 长度不超过 255 字节")
    return name


def history_path(name):
    path = TEMP_DIR / history_filename(name)
    if path.is_symlink():
        raise ValueError("不允许访问符号链接")
    if path.exists() and not path.is_file():
        raise ValueError("该名称不是普通文件")
    return path


def history_version(info):
    """Opaque optimistic-concurrency token; not an authentication credential."""
    identity = (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    )
    return hashlib.sha256(repr(identity).encode("ascii")).hexdigest()


def history_info(path, info=None):
    info = info or path.stat(follow_symlinks=False)
    with STATE_LOCK:
        current = ACTIVE_RUN
        active = current is not None and (
            current.history if current.history.endswith(".jsonl") else current.history + ".jsonl"
        ) == path.name
    return {
        "name": path.name, "size": info.st_size,
        "modified": datetime.fromtimestamp(info.st_mtime).astimezone().isoformat(timespec="seconds"),
        "active": active, "version": history_version(info),
    }


def list_history_files():
    if not TEMP_DIR.exists():
        return []
    files = []
    for path in TEMP_DIR.iterdir():
        if not path.name.endswith(".jsonl"):
            continue
        try:
            info = path.stat(follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                files.append(history_info(path, info))
        except FileNotFoundError:
            continue  # A file was removed during directory refresh.
    return sorted(files, key=lambda item: (item["modified"], item["name"]), reverse=True)


def strict_json_loads(text):
    def invalid_constant(_value):
        raise ValueError("非标准 JSON 常量")
    return json.loads(text, parse_constant=invalid_constant)


@app.get("/api/histories")
def histories():
    try:
        return jsonify(files=list_history_files(), directory="temp")
    except OSError:
        app.logger.exception("Could not list histories")
        return jsonify(error="无法读取 temp 目录，请检查文件权限"), 500


@app.post("/api/histories")
def create_history():
    data = request.get_json(silent=True) if request.is_json else request.form
    if not isinstance(data, dict) and request.is_json:
        return jsonify(error="请提交文件名和可选的 JSONL 内容"), 400
    data = data or {}
    try:
        name = history_filename(data.get("name", "").strip(), add_suffix=True)
        content = data.get("content", "")
        if not isinstance(content, str):
            raise ValueError("JSONL 内容必须是文本")
        records = []
        for line_number, line in enumerate(content.lstrip("\ufeff").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = strict_json_loads(line)
            except (ValueError, RecursionError):
                raise ValueError(f"第 {line_number} 行不是有效 JSON")
            if (not isinstance(record, dict) or record.get("role") not in {"user", "assistant"}
                    or not isinstance(record.get("content"), str)):
                raise ValueError(f"第 {line_number} 行需包含 role（user / assistant）和字符串 content")
            records.append(json.dumps(record, ensure_ascii=False))
        content = "".join(line + "\n" for line in records)
        try:
            content.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("JSONL 内容包含无效 Unicode 字符")
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        path = history_path(name)
        # Exclusive creation: never overwrite an existing history, even in races.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return jsonify(file=history_info(path)), 201
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except AttributeError:
        return jsonify(error="请输入有效的文件名"), 400
    except FileExistsError:
        return jsonify(error="同名文件已存在，请更换名称；不会覆盖原有记录"), 409
    except OSError:
        app.logger.exception("Could not create history")
        return jsonify(error="创建失败，请检查 temp 目录权限和磁盘空间"), 500


@app.post("/api/histories/delete")
def delete_history():
    """Delete one explicitly confirmed formal history, never logs/run snapshots."""
    data = request.get_json(silent=True) if request.is_json else request.form
    if not isinstance(data, dict) and request.is_json:
        return jsonify(error="请提交文件名、确认文件名和预览版本"), 400
    data = data or {}
    try:
        name = history_filename(data.get("name", ""))  # Exact name; no trimming/suffix guessing.
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if data.get("confirm_name") != name:
        return jsonify(error="删除未确认：请输入完整且一致的文件名（含 .jsonl）"), 400
    version = data.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{64}", version):
        return jsonify(error="缺少有效的预览版本，请重新读取文件后确认删除"), 400

    # Share startup/cleanup's lock, including windows before ACTIVE_RUN is set
    # and after Stop while the final history checkpoint is still being published.
    if not RUN_LOCK.acquire(blocking=False):
        return jsonify(error="后台任务正在启动、运行或收尾，暂不可删除 history；请等待结束后重试"), 409
    directory_fd = None
    try:
        # Anchor operations to an opened directory on POSIX; do not follow a
        # substituted TEMP_DIR symlink. Only unlink a direct regular-file entry.
        if os.name == "posix":
            directory_fd = os.open(
                TEMP_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        else:
            info = (TEMP_DIR / name).lstat()
        if not stat.S_ISREG(info.st_mode):
            return jsonify(error="仅允许删除 temp 目录直属的普通 JSONL 文件，不允许符号链接或目录"), 400
        if not hmac.compare_digest(history_version(info), version):
            return jsonify(error="文件在预览后已发生变化，未删除；请重新读取并确认"), 409
        if directory_fd is not None:
            os.unlink(name, dir_fd=directory_fd)
        else:
            (TEMP_DIR / name).unlink()
        warning = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                app.logger.warning("Deleted history but directory sync failed", exc_info=True)
                warning = "文件已删除，但目录落盘确认失败；任务日志和工作副本仍保留。"
        return jsonify(ok=True, deleted=name, warning=warning)
    except FileNotFoundError:
        return jsonify(error="文件不存在或已被删除，请刷新列表"), 404
    except OSError:
        app.logger.exception("Could not delete formal history")
        return jsonify(error="删除失败，请检查目录权限并刷新列表确认文件状态"), 500
    finally:
        try:
            if directory_fd is not None:
                os.close(directory_fd)
        finally:
            RUN_LOCK.release()


@app.get("/api/histories/preview")
def preview_history():
    try:
        name = history_filename(request.args.get("name", ""))
        offset = int(request.args.get("offset", "0"))
        if offset < 0:
            raise ValueError("预览偏移量不能小于 0")
        path = history_path(name)
        # NOFOLLOW closes the symlink race; NONBLOCK avoids hanging on a FIFO.
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("只能预览普通 JSONL 文件")
            size = info.st_size  # Read a bounded snapshot; the agent may append.
            if offset > size:
                raise ValueError("文件已变短，请刷新预览")
            partial_start = False
            if offset:
                file.seek(offset - 1)
                partial_start = file.read(1) != b"\n"
            file.seek(offset)
            entries, raw_parts = [], []
            for _ in range(HISTORY_PREVIEW_LINES):
                remaining = min(HISTORY_PREVIEW_BYTES - (file.tell() - offset), size - file.tell())
                if remaining <= 0:
                    break
                chunk = file.readline(remaining)
                if not chunk:
                    break
                # Keep page boundaries on UTF-8 character boundaries.
                if file.tell() < size:
                    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                    decoder.decode(chunk, final=False)
                    pending, _ = decoder.getstate()
                    if pending:
                        chunk = chunk[:-len(pending)]
                        file.seek(-len(pending), os.SEEK_CUR)
                    if not chunk:
                        break
                raw = chunk.decode("utf-8", errors="replace")
                partial_end = not chunk.endswith(b"\n") and file.tell() < size
                partial = partial_start or partial_end
                record, error = None, None
                if partial:
                    error = "超长记录分段展示，可切换原始 JSONL 并继续翻页"
                elif raw.strip():
                    try:
                        record = strict_json_loads(raw)
                    except (ValueError, RecursionError):
                        error = "此行不是有效 JSON（运行中的末行可能尚未写完）"
                entries.append({"raw": raw, "record": record, "error": error, "partial": partial})
                raw_parts.append(raw)
                partial_start = partial_end
                if partial_end:
                    break
            return jsonify(
                file=history_info(path, info), entries=entries, raw="".join(raw_parts),
                offset=offset, next_offset=file.tell(), has_more=file.tell() < size,
            )
    except FileNotFoundError:
        return jsonify(error="文件不存在，可能已被移动或删除，请刷新列表"), 404
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except OSError:
        app.logger.exception("Could not preview history")
        return jsonify(error="无法预览文件，请检查权限或刷新列表"), 400


@app.get("/api/config")
def configuration():
    names = model_names()
    effort = os.environ.get("REASONING_EFFORT", "auto").strip().lower()
    if effort == "ultra-high":
        effort = "xhigh"
    try:
        histories = [item["name"] for item in list_history_files()]
    except OSError:
        histories = []
        app.logger.warning("Could not list history files", exc_info=True)
    return jsonify(
        models=names, default_model=names[0] if names else "",
        efforts=EFFORT_LEVELS, default_effort=effort if effort in EFFORT_LEVELS else "auto",
        histories=histories, busy=RUN_LOCK.locked(),
        credentials_configured=bool(os.environ.get("BASE_URL") and os.environ.get("API_KEY")),
    )


@app.get("/api/runs/latest")
def get_latest_run():
    try:
        history = request.args.get("history")
        if history is not None:
            history = history_filename(history, add_suffix=True)
        with STATE_LOCK:
            current = ACTIVE_RUN
        data = read_run(current.id) if current and history is None else latest_run(history)
        return jsonify(run=data)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except OSError:
        return jsonify(error="无法读取保存的运行状态"), 500


@app.get("/api/runs/<run_id>/events")
def reconnect_run(run_id):
    try:
        read_run(run_id)
        offset = int(request.args.get("offset", "0"))
        with run_path(run_id, "events").open("rb") as file:
            size = file.seek(0, os.SEEK_END)
            if offset < 0 or offset > size:
                raise ValueError("日志偏移量无效，请从头重新连接")
            if offset:
                file.seek(offset - 1)
                if file.read(1) != b"\n":
                    raise ValueError("日志偏移量必须位于事件边界")
    except FileNotFoundError:
        return jsonify(error="找不到该次运行，可从 History 选择会话继续"), 404
    except (ValueError, OSError) as exc:
        return jsonify(error=str(exc)), 400
    return Response(
        stream_run_events(run_id, offset),
        content_type="application/x-ndjson; charset=utf-8",
        headers={"X-Accel-Buffering": "no"},
    )


@app.post("/run")
def run():
    global ACTIVE_RUN
    text = request.form.get("input", "")
    history = request.form.get("history", "new") or "new"
    model = request.form.get("model", os.environ.get("MODEL_NAME", "")).strip()
    effort = request.form.get("effort", os.environ.get("REASONING_EFFORT", "auto")).strip().lower()
    if effort == "ultra-high":
        effort = "xhigh"
    resume = request.form.get("resume") == "1"
    if not text.strip() and not resume:
        return jsonify(error="请输入非空的任务指令，或选择恢复会话"), 400
    try:
        history = history_filename(history, add_suffix=True)
        path = history_path(history)
        if resume and (not path.is_file() or path.stat().st_size == 0):
            return jsonify(error="恢复需要已有的非空 history，请先选择之前的会话"), 400
    except (ValueError, OSError) as exc:
        return jsonify(error=str(exc)), 400
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
    if resume:
        text = RESUME_PROMPT + ("\n\nAdditional instructions:\n" + text if text.strip() else "")
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
        current.worker = threading.Thread(target=current.collect, name=f"agent-{current.id}", daemon=True)
        current.worker.start()
    except ValueError as exc:
        current.close()
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        app.logger.exception("Could not start agent")
        current.close()
        return jsonify(error=f"无法启动 Agent：{exc}"), 500
    response = Response(
        current.stream(), content_type="application/x-ndjson; charset=utf-8",
        headers={"X-Accel-Buffering": "no"},
    )
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
    parser.add_argument("--ssl-cert", default=os.environ.get("AGENT_WEB_SSL_CERT") or None,
                        help="PEM certificate/full-chain path (or AGENT_WEB_SSL_CERT); requires --ssl-key.")
    parser.add_argument("--ssl-key", default=os.environ.get("AGENT_WEB_SSL_KEY") or None,
                        help="PEM private key path (or AGENT_WEB_SSL_KEY); requires --ssl-cert.")
    args = parser.parse_args()
    try:
        tls_context = create_ssl_context(
            args.ssl_cert, args.ssl_key, cert_dir=AGENT_DIR / ".certs"
        )
    except (OSError, ValueError, RuntimeError, ssl.SSLError) as exc:
        parser.error(f"HTTPS setup failed: {exc}")
    if not args.ssl_cert:
        app.logger.warning(
            "Using a self-signed localhost certificate from %s. "
            "Browsers will warn until it is trusted; use a trusted certificate for remote access.",
            AGENT_DIR / ".certs",
        )
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        app.logger.warning("Non-loopback binding: this console can execute commands. "
                           "Use only on a trusted network with access controls.")
    def terminate(_signal, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False,
                ssl_context=tls_context)
    finally:
        shutdown_active_run()
