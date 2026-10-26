"use strict";
// Lightweight DOM tests, no network/browser credentials/model calls.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.children = []; this.parentNode = null;
    this.className = ""; this.dataset = {}; this.open = false; this._text = "";
  }
  appendChild(node) {
    node.remove();
    node.parentNode = this; this.children.push(node);
    return node;
  }
  get previousElementSibling() {
    if (!this.parentNode) return null;
    const nodes = this.parentNode.children;
    return nodes[nodes.indexOf(this) - 1] || null;
  }
  get nextElementSibling() {
    if (!this.parentNode) return null;
    const nodes = this.parentNode.children;
    return nodes[nodes.indexOf(this) + 1] || null;
  }
  append(...nodes) { for (const node of nodes) this.appendChild(node); }
  remove() {
    if (!this.parentNode) return;
    const siblings = this.parentNode.children;
    siblings.splice(siblings.indexOf(this), 1); this.parentNode = null;
  }
  replaceChildren(...nodes) {
    for (const child of [...this.children]) child.remove();
    this._text = ""; this.append(...nodes);
  }
  hasChildNodes() { return this.children.length > 0 || this._text.length > 0; }
  set textContent(value) { this.replaceChildren(); this._text = String(value); }
  get textContent() { return this._text + this.children.map(n => n.textContent).join(""); }
  set innerHTML(_) { throw new Error("Untrusted log must never use innerHTML"); }
}
const document = {
  createElement(tag) { return new Element(tag); },
  createDocumentFragment() { return new Element("fragment"); },
};
const context = vm.createContext({document, console});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8") +
  "\nglobalThis.Renderer = FoldedLogHighlighter; globalThis.PlainRenderer = LogHighlighter;", context);
function make(limit=1000000) {
  const out = new Element("div");
  return {out, h:new context.Renderer(out, limit)};
}
function raw(node) {
  if (node.tagName === "SUMMARY") return "";
  return node._text + node.children.map(raw).join("");
}
function descendants(node, tag) {
  return node.children.flatMap(child => [
    ...(child.tagName === tag.toUpperCase() ? [child] : []), ...descendants(child, tag),
  ]);
}
function folds(out) { return descendants(out, "details"); }
function assertText(h, out, text) {
  assert.equal(h.text, text, "copy buffer must retain exact raw text");
  assert.equal(raw(out), text, "rendered bodies must retain raw text/order");
}
let count = 0;
function test(name, fn) {
  fn(); count++; console.log(`PASS: ${name}`);
}
const input = "<|action_input|>";
const inputEnd = "<\\action_input|>";
const obs = "<|observation|>";
const obsEnd = "<\\observation|>";
const sample = "<|think|>检查一下<\\think|><|action|>run_cmd<\\action|>" +
  input + "\n```text\nprintf '你好'\n```\n" + inputEnd +
  "\n" + obs + "command succeeded\nstdout:\n你好\n" + obsEnd +
  "\n<|final_answer|>完成了<\\final_answer|>";

test("default collapsed blocks and lossless raw text", () => {
  const {out,h}=make(); h.append(sample); assertText(h,out,sample);
  assert.deepEqual(folds(out).map(f=>f.dataset.kind),["action_input","observation"]);
  assert.ok(folds(out).every(f=>!f.open));
  assert.ok(folds(out)[0].children[0].textContent.includes("ACTION INPUT"));
  assert.ok(folds(out)[1].children[0].textContent.includes("OBSERVATION"));
});
test("every two-part split, including delimiters and code fences", () => {
  for (let i=0;i<=sample.length;i++) {
    const {out,h}=make(); h.append(sample.slice(0,i)); h.append(sample.slice(i));
    assertText(h,out,sample);
    assert.equal(folds(out).length,2);
  }
});
test("one-character streaming keeps content order on every paint", () => {
  const {out,h}=make(); let text="";
  for (const char of sample) {
    text+=char; h.append(char); assertText(h,out,text);
  }
  assert.equal(folds(out).length,2);
});
test("expanded same-line block retains node and open state", () => {
  const {out,h}=make();
  const a="<|think|>visible<\\think|>"+input+"print(";
  h.append(a);
  const fold=folds(out)[0]; fold.open=true;
  h.append("'ok')");
  assert.strictEqual(folds(out)[0],fold); assert.ok(fold.open);
  h.append(inputEnd+obs+"result");
  assert.strictEqual(folds(out)[0],fold); assert.ok(fold.open);
  assert.equal(folds(out).length,2);
  assert.ok(!folds(out)[1].open);
  assertText(h,out,a+"'ok')"+inputEnd+obs+"result");
});
test("expanded multi-line observation stays open through updates", () => {
  const {out,h}=make(); h.append(obs+"first\n");
  const fold=folds(out)[0]; fold.open=true;
  h.append("second\nthird"); h.append(obsEnd+"\nfinal");
  assert.strictEqual(folds(out)[0],fold); assert.ok(fold.open);
  assertText(h,out,obs+"first\nsecond\nthird"+obsEnd+"\nfinal");
});
test("several inline folds and plain text never reorder on partial reparse", () => {
  const text="prefix"+input+"a"+inputEnd+"middle"+obs+"b"+obsEnd+"tail";
  const {out,h}=make(); let acc="";
  for (const char of text) { acc+=char; h.append(char); assertText(h,out,acc); }
  assert.equal(folds(out).length,2);
});
test("quoted tags inside another tagged block do not fold", () => {
  const text="<|final_answer|>Example: "+input+"x"+inputEnd+obs+"y"+obsEnd+"<\\final_answer|>";
  const {out,h}=make(); for(const char of text) h.append(char);
  assertText(h,out,text); assert.equal(folds(out).length,0);
});
test("fenced examples stay literal; enclosing input stays folded", () => {
  const text="```text\n"+obs+"quoted"+obsEnd+"\n```\n"+input+
    "\n```python\nprint('"+inputEnd+"')\nprint('"+obs+"')\n```\n"+inputEnd+
    "\n<|final_answer|>visible<\\final_answer|>";
  const {out,h}=make(); for(const char of text) h.append(char);
  assertText(h,out,text); assert.equal(folds(out).length,1);
  assert.equal(folds(out)[0].dataset.kind,"action_input");
});
test("case-insensitive and slash closing tags", () => {
  const text="<|ACTION_INPUT|>x</ACTION_INPUT|>before<|Observation|>y</Observation|>after";
  const {out,h}=make(); h.append(text);
  assertText(h,out,text); assert.equal(folds(out).length,2);
});
test("legacy Action Input and Observation headings fold until next heading", () => {
  const text="Think: check\nAction: run_cmd\nAction Input: print('ok')\n" +
    "Observation: result\nTraceback (most recent call last):\nValueError: example\nFinal Answer: done\n";
  const {out,h}=make(); for(const char of text) h.append(char);
  assertText(h,out,text); assert.equal(folds(out).length,2);
  assert.ok(raw(folds(out)[1]).includes("ValueError"));
  assert.ok(!raw(folds(out)[1]).includes("Final Answer"));
});
test("hostile HTML remains inert text", () => {
  const text=obs+'<img src=x onerror="alert(1)"><script>alert(1)</script>'+obsEnd;
  const {out,h}=make(); h.append(text); assertText(h,out,text);
  assert.equal(descendants(out,"img").length,0);
  assert.equal(descendants(out,"script").length,0);
});
test("clear while tag is split retains parser context without old text", () => {
  const {out,h}=make(); h.append("discarded\n<|observ");
  h.reset(true); h.append("ation|>new output"+obsEnd+"\nafter");
  assertText(h,out,"ation|>new output"+obsEnd+"\nafter");
  assert.equal(folds(out).length,1);
  assert.equal(folds(out)[0].dataset.kind,"observation");
});
test("clear inside folded fenced code preserves context", () => {
  const {out,h}=make(); h.append(input+"\n```python\nold\n");
  h.reset(true);
  const next="new\nprint('"+obs+"literal')\n```\n"+inputEnd+"\nafter";
  h.append(next); assertText(h,out,next);
  assert.equal(folds(out).length,1);
  assert.equal(folds(out)[0].dataset.kind,"action_input");
});
test("trimming retains block identity and user's expanded state", () => {
  const {out,h}=make(120);
  h.append(obs+"\n"+"a".repeat(80)+"\n");
  folds(out)[0].open=true;
  h.append("b".repeat(85)+"\n");
  assert.ok(h.trimmed); assert.equal(raw(out),h.text);
  assert.equal(folds(out).length,1); assert.ok(folds(out)[0].open);
  h.append(obsEnd+"\n<|final_answer|>done<\\final_answer|>");
  assert.equal(raw(out),h.text); assert.ok(folds(out)[0].open);
});
test("long-line truncation keeps future boundaries and a bounded buffer", () => {
  const {out,h}=make(90);
  h.append(input+"x".repeat(180)); assert.ok(h.trimmed);
  h.append(inputEnd+obs+"y".repeat(100)); h.append(obsEnd+"end");
  assert.equal(raw(out),h.text);
  assert.ok(h.text.length<=122);
  assert.equal(folds(out).length,1);
  assert.equal(folds(out)[0].dataset.kind,"observation");
});
test("evicted panels do not leak, new panels default collapsed", () => {
  const {out,h}=make(120);
  h.append(obs+"one"+obsEnd+"\n"); folds(out)[0].open=true;
  h.append("plain ".repeat(50)+"\n"+obs+"two"+obsEnd);
  assert.equal(raw(out),h.text);
  assert.equal(folds(out).length,1); assert.equal(h.folds.size,1);
  assert.equal(h.savedOpen.size,0); assert.ok(!folds(out)[0].open);
});
test("new task or replay resets collapsed state", () => {
  const {out,h}=make(); h.append(sample); folds(out)[0].open=true;
  h.reset(); assertText(h,out,"");
  h.append(sample); assertText(h,out,sample);
  assert.ok(folds(out).every(f=>!f.open));
});
test("randomized chunk sizes match a complete replay", () => {
  const text=(sample+"\n").repeat(4);
  for(let seed=1;seed<15;seed++) {
    const {out,h}=make(); let index=0, n=seed;
    while(index<text.length) { n=(n*16807)%2147483647; const size=1+n%31; h.append(text.slice(index,index+size)); index+=size; }
    assertText(h,out,text); assert.equal(folds(out).length,8);
  }
});

function visibleOutside(out) {
  return out.children.filter(node => !node.hidden && node.tagName !== "DETAILS").map(raw).join("");
}
function separatorNodes(out) { return out.children.filter(node => node.dataset.logGap === "true"); }
test("fold separators and trailing newlines occupy no visible text lines", () => {
  const text=input+"body"+inputEnd+"\n\n"+obs+"result"+obsEnd+"\n\n";
  const {out,h}=make(); h.append(text);
  assertText(h,out,text);
  assert.equal(folds(out).length,2);
  assert.equal(visibleOutside(out),"");
  assert.ok(separatorNodes(out).length>0);
  assert.ok(separatorNodes(out).every(node=>node.hidden));
});
test("separators stay compact for every streamed split and fresh replay", () => {
  const text="before\n\n"+input+"body"+inputEnd+"\n\n"+obs+"result"+obsEnd+"\r\n\r\nafter";
  const whole=make(); whole.h.append(text);
  for(let split=0;split<=text.length;split++) {
    const {out,h}=make(); h.append(text.slice(0,split)); h.append(text.slice(split));
    assertText(h,out,text);
    assert.equal(visibleOutside(out),visibleOutside(whole.out));
    assert.ok(separatorNodes(out).every(node=>node.hidden));
  }
});
test("late whitespace-only chunks after last fold remain hidden", () => {
  const {out,h}=make(); const text=obs+"result"+obsEnd;
  h.append(text);
  for(const part of ["\n","\r","\n"," ","\t","\n"]) h.append(part);
  assert.equal(visibleOutside(out),"");
  assertText(h,out,text+"\n\r\n \t\n");
});
test("whitespace inside expanded fold is preserved exactly", () => {
  const body="\n```python\n\n    if True:\n        print('ok')\n\n```\n\n";
  const text=input+body+inputEnd+"\n\n";
  const {out,h}=make(); h.append(text);
  const fold=folds(out)[0]; fold.open=true; h.append("\n");
  assert.strictEqual(folds(out)[0],fold); assert.ok(fold.open);
  assert.equal(raw(fold),input+body+inputEnd);
  assert.equal(visibleOutside(out),"");
  assertText(h,out,text+"\n");
});
test("ordinary paragraphs and final answer blank lines are not compacted", () => {
  const {out,h}=make();
  const plain="paragraph one\n\nparagraph two\n\n";
  h.append(plain); assert.equal(visibleOutside(out),plain);
  const final="<|final_answer|>first paragraph\n\nsecond paragraph\n\n<\\final_answer|>";
  h.append(obs+"tool"+obsEnd+"\n\n"+final);
  // Only the last blank separator immediately before <details> is hidden.
  // The blank line between ordinary paragraphs and those inside final remain.
  assert.equal(visibleOutside(out),plain.slice(0,-1)+final);
  assertText(h,out,plain+obs+"tool"+obsEnd+"\n\n"+final);
});
test("text arriving after temporary whitespace remains visible and indented", () => {
  const {out,h}=make(); const prefix=obs+"result"+obsEnd;
  h.append(prefix+"  ");
  assert.ok(separatorNodes(out)[0].hidden);
  h.append("indented text\n\nmore");
  assert.equal(visibleOutside(out),"  indented text\n\nmore");
  assertText(h,out,prefix+"  indented text\n\nmore");
});
test("spacing compaction survives trimming and clear with parser context", () => {
  const {out,h}=make(100);
  h.append(obs+"x".repeat(200)+obsEnd+"\n\n");
  assert.equal(raw(out),h.text);
  assert.equal(visibleOutside(out),"");
  assert.ok(separatorNodes(out).every(node=>node.hidden));
  h.reset(true); h.append(input+"new"+inputEnd+"\n\n");
  assertText(h,out,input+"new"+inputEnd+"\n\n");
  assert.equal(visibleOutside(out),"");
});
test("unknown/incomplete tag text after a fold is never hidden", () => {
  const {out,h}=make();
  const prefix=obs+"result"+obsEnd+"\n\n";
  h.append(prefix+"<|observ");
  assert.equal(visibleOutside(out),"<|observ");
  h.append("ation|>again"+obsEnd+"\n");
  assert.equal(visibleOutside(out),"");
  assert.equal(folds(out).length,2);
  assertText(h,out,prefix+obs+"again"+obsEnd+"\n");
});

console.log(`Completed ${count} main-log folding tests.`);
