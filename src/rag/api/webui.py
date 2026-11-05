"""`GET /` 的最小聊天页：**零依赖、离线可用**（不引任何 CDN，内网/断网也能开）。

为什么要有它：CLI 和 `/docs`（Swagger）能测，但都不像"聊天"——想看流式效果、想点开引用、
想按停止键，还是得有个页面。这个页面只做一件事：把 `/api/v1/ask/stream` 的 SSE 渲染出来。

刻意不做的：不做历史会话（服务端本来就无状态）、不做 markdown 渲染（会引入依赖、
而答案里的 `[证据 n]` 反而是要看清的东西）、不做鉴权（O13 之前它只服务 localhost）。
"""

from __future__ import annotations

HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>本地 RAG 问答</title>
<style>
  :root { --fg:#1f2328; --dim:#6b7280; --line:#e5e7eb; --bg:#ffffff; --accent:#2563eb; --warn:#b45309; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
         color:var(--fg); background:var(--bg); }
  header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; gap:12px;
           align-items:baseline; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .meta { color:var(--dim); font-size:12px; }
  main { max-width:860px; margin:0 auto; padding:20px; }
  .row { display:flex; gap:8px; }
  textarea { flex:1; min-height:64px; padding:10px 12px; border:1px solid var(--line);
             border-radius:10px; font:inherit; resize:vertical; }
  textarea:focus { outline:2px solid #bfdbfe; border-color:var(--accent); }
  button { padding:10px 16px; border-radius:10px; border:1px solid var(--line);
           background:#f9fafb; font:inherit; cursor:pointer; white-space:nowrap; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .status { color:var(--dim); font-size:13px; min-height:20px; margin:10px 0 0; }
  .banner { margin:12px 0 0; padding:10px 12px; border-radius:10px; font-size:13px;
            background:#fffbeb; border:1px solid #fde68a; color:var(--warn); white-space:pre-wrap; }
  .answer { margin:16px 0 0; white-space:pre-wrap; word-break:break-word; }
  .answer .cite { color:var(--accent); font-size:.9em; }
  section { margin:18px 0 0; }
  section > h2 { font-size:13px; color:var(--dim); font-weight:600; margin:0 0 6px;
                 text-transform:uppercase; letter-spacing:.04em; }
  ol { padding-left:20px; margin:0; font-size:13px; }
  li { margin-bottom:4px; }
  li .path { color:var(--dim); }
  .facts { font-size:12px; color:var(--dim); display:flex; gap:14px; flex-wrap:wrap; }
  .hint { color:var(--dim); font-size:12px; margin-top:8px; }
</style>
</head>
<body>
<header>
  <h1>本地 RAG 问答</h1>
  <span class="meta">三路召回 + RRF + 重排 · 引用体检 · 流式输出</span>
</header>
<main>
  <form class="row" id="form">
    <textarea id="q" placeholder="问点什么，比如：切分过短会有什么问题？（Enter 发送，Shift+Enter 换行）"></textarea>
    <div>
      <button class="primary" id="send" type="submit">发送</button>
      <div style="height:8px"></div>
      <button id="stop" type="button" disabled>停止</button>
    </div>
  </form>
  <p class="status" id="status"></p>
  <div class="banner" id="banner" hidden></div>
  <article class="answer" id="answer"></article>
  <section id="sourcesBox" hidden>
    <h2>来源（证据编号对应正文里的 [证据 n]）</h2>
    <ol id="sources"></ol>
  </section>
  <section id="factsBox" hidden>
    <h2>这一次的体检</h2>
    <div class="facts" id="facts"></div>
  </section>
  <p class="hint">答案完全来自你自己的文档；没有依据时它会弃答，而不是编。</p>
</main>
<script>
const $ = id => document.getElementById(id);
const form = $('form'), qEl = $('q'), sendBtn = $('send'), stopBtn = $('stop');
const statusEl = $('status'), bannerEl = $('banner'), answerEl = $('answer');
const sourcesBox = $('sourcesBox'), sourcesEl = $('sources'), factsBox = $('factsBox'), factsEl = $('facts');
let es = null, started = 0;

function reset() {
  statusEl.textContent = ''; bannerEl.hidden = true; bannerEl.textContent = '';
  answerEl.textContent = ''; sourcesEl.innerHTML = ''; sourcesBox.hidden = true;
  factsEl.innerHTML = ''; factsBox.hidden = true;
}

function showDegraded(info) {
  if (!info) return;
  bannerEl.hidden = false;
  bannerEl.textContent = '降级：' + info.reason + '（停用 ' + (info.routes || []).join(', ') + '）\\n'
    + (info.fix ? '修复：\\n' + info.fix : '');
}

function renderSources(data) {
  showDegraded(data.degraded);
  sourcesEl.innerHTML = '';
  (data.sources || []).forEach(s => {
    const li = document.createElement('li');
    const path = (s.section_path || []).join(' > ');
    li.innerHTML = '<b>[证据 ' + s.index + ']</b> ' + (s.title || '(无标题)')
      + (path ? ' <span class="path">· ' + path + '</span>' : '')
      + (s.source_uri ? ' <span class="path">· ' + s.source_uri + '</span>' : '');
    sourcesEl.appendChild(li);
  });
  sourcesBox.hidden = (data.sources || []).length === 0;
}

function renderDone(data) {
  const cited = (data.cited || []).length ? '引用 [证据 ' + data.cited.join('、') + ']' : '未标注引用';
  const bad = (data.out_of_range || []).length ? '｜越界 ' + data.out_of_range.join(',') : '';
  const t = data.timings || {};
  const facts = [
    cited + bad,
    '解析成功率 ' + (data.parse_rate || 0).toFixed(2),
    '上下文 ' + data.block_count + ' 块 / ' + data.context_tokens + ' token',
    '候选 ' + data.candidates + ' 条',
    '检索 ' + Math.round(t.retrieve_rerank_ms || 0) + ' ms',
    '首字 ' + Math.round(t.first_delta_ms || 0) + ' ms',
    '生成 ' + Math.round(t.generate_ms || 0) + ' ms',
    '总耗时 ' + Math.round(performance.now() - started) + ' ms'
  ];
  if (data.llm_called === false) facts.push('（只检索，未调用模型）');
  factsEl.innerHTML = facts.map(f => '<span>' + f + '</span>').join('');
  factsBox.hidden = false;
}

function finish(message) {
  if (es) { es.close(); es = null; }
  stopBtn.disabled = true; sendBtn.disabled = false;
  statusEl.textContent = message || '';
}

form.addEventListener('submit', ev => {
  ev.preventDefault();
  const q = qEl.value.trim();
  if (!q || es) return;
  reset();
  started = performance.now();
  sendBtn.disabled = true; stopBtn.disabled = false;

  const url = '/api/v1/ask/stream?q=' + encodeURIComponent(q) + '&k=4';
  es = new EventSource(url);

  es.addEventListener('queued', ev => {
    const d = JSON.parse(ev.data);
    statusEl.textContent = d.will_wait
      ? '排队中：' + d.inflight + ' 个在跑 / ' + d.waiting + ' 个等待…'
      : '检索中…';
  });

  es.addEventListener('sources', ev => {
    const d = JSON.parse(ev.data);
    statusEl.textContent = d.blocks > 0
      ? '已找到 ' + d.blocks + ' 块证据，正在生成…'
      : '没有找到证据，模型不会作答';
    renderSources(d);
  });

  es.addEventListener('delta', ev => {
    answerEl.textContent += JSON.parse(ev.data).text;
  });

  es.addEventListener('done', ev => {
    const d = JSON.parse(ev.data);
    if (!answerEl.textContent) answerEl.textContent = d.answer || '';
    renderDone(d);
    finish(d.abstained ? '弃答（' + (d.abstain_reason || 'no_evidence') + '）：知识库里没有足够依据' : '完成');
  });

  es.addEventListener('error', ev => {
    // 注意：EventSource 自己的连接错误也会走 'error'，那时 ev.data 是 undefined
    if (ev.data === undefined) {
      finish('连接中断');
      return;
    }
    const d = JSON.parse(ev.data);
    bannerEl.hidden = false;
    bannerEl.textContent = '出错（status ' + (d.status || 500) + '）：' + (d.detail || '')
      + (d.retry_after_s ? '\\n建议 ' + d.retry_after_s + ' 秒后重试' : '');
    if (d.partial) answerEl.textContent = d.partial;
    finish('已中断，可重试');
  });
});

stopBtn.addEventListener('click', () => finish('已停止（服务端会同时取消生成）'));
window.addEventListener('beforeunload', () => { if (es) es.close(); });
</script>
</body>
</html>
"""


def render() -> str:
    return HTML
