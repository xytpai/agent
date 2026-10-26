"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const script = fs.readFileSync(process.argv[2], "utf8");
function section(start, end) {
  const a = script.indexOf(start), b = script.indexOf(end, a);
  assert.ok(a >= 0 && b > a);
  return script.slice(a, b);
}
const code =
  section("function clearOutput(", "function applyLogSize(") +
  section("function cancelStream(", "function finishRun(") +
  section('$("history-use").addEventListener', '$("history-upload").addEventListener');
function setup(mode = "existing") {
  const nodes = new Map();
  function $(id) {
    if (!nodes.has(id)) nodes.set(id, {
      value:id === "history" ? "old.jsonl" : "",
      textContent:id === "elapsed" ? "03:45" : "",
      hidden:false, content:"previous rendered log",
      replaceChildren() { this.content = ""; },
      addEventListener(type, callback) { this.callback = callback; },
    });
    return nodes.get(id);
  }
  const oldController = new AbortController();
  const ctx = {
    $, AbortController, URLSearchParams,
    running:false, historyDeleting:false, previewLoading:false,
    previewData:{file:{name:"新的 history.jsonl"}},
    streamController:oldController, reconnectTimer:101,
    runId:"old-run", eventOffset:800, timer:102, started:123456,
    paintTimer:103, pendingOutput:"old pending output", rawLog:"old raw log", trimmed:true,
    out:$("out"), calls:[], statuses:[], messages:[], toasts:[], tabs:[], resets:[],
    oldController, intervals:[], timeouts:[],
    clearTimeout(id) { ctx.timeouts.push(id); },
    clearInterval(id) { ctx.intervals.push(id); },
    logHighlighter:{reset(keepContext) { ctx.resets.push(keepContext); }},
    flushOutput() { throw new Error("History load must not flush old output back into display"); },
    setBusy(value) { ctx.running = value; },
    setState(...args) { ctx.statuses.push(args); },
    selectedModel() { return "selected-model"; },
    applyRunSettings(data) {
      ctx.applied = data;
      $("history").value = data.history;
      $("input").value = "";
      $("input-count").textContent = "0";
      $("run-meta").textContent = "old metadata that must be replaced";
    },
    showTab(name) { ctx.tabs.push(name); },
    message(...args) { ctx.messages.push(args); },
    toast(value) { ctx.toasts.push(value); },
    async responseError() { return "load failed"; },
    async apiFetch(url, options) {
      ctx.calls.push({url, options});
      if (mode === "network-error") throw new Error("network error");
      if (mode === "abort") {
        const e = new Error("cancelled"); e.name = "AbortError"; throw e;
      }
      if (mode === "superseded") ctx.streamController = new AbortController();
      if (mode === "cancelled") ctx.cancelStream();
      if (mode === "deferred") return new Promise(resolve => { ctx.resolveFetch = resolve; });
      return {ok:mode !== "http-error", async json() {
        return {run:mode === "new" ? null : {id:"old-selected-task", history:"新的 history.jsonl", model:"saved-model"}};
      }};
    },
  };
  vm.createContext(ctx); vm.runInContext(code, ctx);
  ctx.click = () => $("history-use").callback();
  return ctx;
}
function assertCleared(c) {
  assert.equal(c.oldController.signal.aborted,true);
  assert.deepEqual(c.resets,[false]);
  assert.equal(c.pendingOutput,""); assert.equal(c.rawLog,"");
  assert.equal(c.paintTimer,null); assert.equal(c.trimmed,false);
  assert.equal(c.out.content,""); assert.equal(c.out.hidden,true);
  assert.equal(c.$("empty").hidden,false);
  assert.equal(c.$("line-count").textContent,"0");
  assert.equal(c.$("elapsed").textContent,"00:00");
  assert.equal(c.runId,null); assert.equal(c.eventOffset,0);
  assert.equal(c.timer,null); assert.equal(c.started,0);
  assert.equal(c.$("history").value,"新的 history.jsonl");
  assert.equal(c.$("run-meta").textContent,"selected-model · 新的 history.jsonl · 尚未运行");
  assert.deepEqual(c.statuses,[["idle","会话已载入"]]);
  assert.deepEqual(c.tabs,["run"]); assert.equal(c.running,false);
  assert.equal(c.streamController,null);
  assert.ok(c.timeouts.includes(101)); assert.ok(c.timeouts.includes(103));
  assert.ok(c.intervals.includes(102));
  assert.equal(c.calls.length,1);
  const url=new URL(c.calls[0].url,"https://localhost");
  assert.equal(url.pathname,"/api/runs/latest");
  assert.equal(url.searchParams.get("history"),"新的 history.jsonl");
  assert.ok(c.calls[0].options.signal);
  assert.equal(c.calls[0].options.method,undefined); // GET only, never run/stop/delete.
}
(async () => {
  let c = setup(); await c.click(); assertCleared(c); assert.ok(c.applied);
  console.log("PASS: loading existing history clears previous live log, parser context, task ID and timer.");
  c=setup("new"); await c.click(); assertCleared(c); assert.equal(c.applied,undefined);
  assert.equal(c.$("input").value,""); assert.equal(c.$("input-count").textContent,"0");
  console.log("PASS: history without a saved run also clears previous display.");
  for (const mode of ["http-error","network-error","abort"]) {
    c=setup(mode); await c.click();
    assert.equal(c.resets.length,0); assert.equal(c.rawLog,"old raw log");
    assert.equal(c.runId,"old-run"); assert.equal(c.$("history").value,"old.jsonl");
    assert.equal(c.running,false);
    if(mode==="abort") assert.equal(c.messages.length,0);
  }
  console.log("PASS: failed/cancelled loads do not clear or replace the existing display.");
  for (const mode of ["superseded","cancelled"]) {
    c=setup(mode); await c.click();
    assert.equal(c.resets.length,0); assert.equal(c.rawLog,"old raw log");
    assert.equal(c.applied,undefined); assert.equal(c.tabs.length,0);
    assert.equal(c.running,true); // Obsolete request must not unlock a newer operation.
  }
  console.log("PASS: stale responses cannot switch the selected history or overwrite newer UI state.");
  for(const flag of ["running","historyDeleting","previewLoading"]) {
    c=setup(); c[flag]=true; await c.click();
    assert.equal(c.calls.length,0); assert.equal(c.resets.length,0);
    assert.equal(c.oldController.signal.aborted,false);
  }
  c=setup(); c.previewData=null; await c.click(); assert.equal(c.calls.length,0);
  console.log("PASS: running tasks, pending previews and deletions block history loading.");
  c=setup("deferred"); const first=c.click(); await c.click();
  assert.equal(c.calls.length,1);
  c.resolveFetch({ok:true,async json(){return {run:null};}});
  await first; assertCleared(c);
  console.log("PASS: repeated click while loading does not send duplicate requests.");
})().catch(error => { console.error(error); process.exitCode=1; });
