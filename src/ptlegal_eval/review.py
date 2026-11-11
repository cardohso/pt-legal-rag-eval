"""Human review of golden-set drafts: a local page, one question at a time, and a merge step.

`build` writes a self-contained HTML file that shows each draft with its gold article text. It
makes no network requests and keeps decisions in the browser's localStorage; "Download" saves them
as JSON. `apply` merges that JSON into the draft items. The page embeds the law text, so it is
written under `review/`, which git ignores.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ptlegal_eval.chunks import Chunk, read_chunks
from ptlegal_eval.golden_check import resolve_gold

STATUSES = {"accept", "edit", "reject"}


@dataclass
class ApplyResult:
    items: list[dict[str, Any]]  # rejected items removed; pending ones kept, still unreviewed
    rejected: list[str] = field(default_factory=list)
    edited: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)


def _text(value: object, fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def apply_decisions(
    items: Sequence[Mapping[str, Any]], decisions: Mapping[str, Mapping[str, Any]], reviewer: str
) -> ApplyResult:
    known = {it["id"] for it in items}
    for iid, d in decisions.items():
        if iid not in known:
            raise ValueError(f"decision for unknown item {iid}: is this the right drafts file?")
        if d.get("status") not in STATUSES:
            raise ValueError(f"item {iid}: unknown status {d.get('status')!r}")

    result = ApplyResult(items=[])
    for original in items:
        it = copy.deepcopy(dict(original))
        d = decisions.get(it["id"])
        if d is None:
            result.pending.append(it["id"])
            result.items.append(it)
            continue
        if d["status"] == "reject":
            result.rejected.append(it["id"])
            continue
        question = _text(d.get("question"), it["question"])
        reference = _text(d.get("reference_answer"), it["reference_answer"])
        meta = it.setdefault("metadata", {})
        if (question, reference) != (it["question"], it["reference_answer"]):
            it["question"], it["reference_answer"] = question, reference
            meta["edited"] = True
            result.edited.append(it["id"])
        meta["reviewed"] = True
        meta["reviewed_by"] = reviewer
        if _text(d.get("comment"), ""):
            meta["review_comment"] = d["comment"].strip()
        result.items.append(it)
    return result


CORE_JS = """
function isDecided(d) { return !!d && ['accept', 'edit', 'reject'].includes(d.status); }
function summarize(items, decisions) {
  const s = { total: items.length, accepted: 0, edited: 0, rejected: 0, pending: 0 };
  for (const it of items) {
    const d = decisions[it.id];
    if (!isDecided(d)) s.pending++;
    else if (d.status === 'accept') s.accepted++;
    else if (d.status === 'edit') s.edited++;
    else s.rejected++;
  }
  return s;
}
function nextPending(items, decisions, from) {
  const n = items.length;
  for (let k = 1; k <= n; k++) {
    const i = (from + k + n) % n;
    if (!isDecided(decisions[items[i].id])) return i;
  }
  return -1;
}
function pick(x, fallback) { return (typeof x === 'string' && x.trim()) ? x.trim() : fallback; }
function effective(item, d) {
  return {
    question: pick(d && d.question, item.question),
    reference_answer: pick(d && d.reference_answer, item.reference_answer),
    comment: (d && d.comment) || '',
  };
}
"""

APP_JS = """
(function () {
  const DATA = JSON.parse(document.getElementById('data').textContent);
  const items = DATA.items;
  const KEY = 'ptlegal-review-v1';
  const $ = (id) => document.getElementById(id);
  let storageOk = true;
  let decisions = {};
  try { decisions = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { storageOk = false; }
  const first = nextPending(items, decisions, -1);
  let cur = first === -1 ? 0 : first;

  function save() {
    if (storageOk) { try { localStorage.setItem(KEY, JSON.stringify(decisions)); } catch (e) { storageOk = false; } }
    $('storage').textContent = storageOk
      ? 'Autosaved in this browser. Download a backup now and then.'
      : 'Autosave is OFF in this browser: download your decisions often.';
  }
  function el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    for (const kid of kids) e.append(kid);
    return e;
  }
  function render() {
    const it = items[cur], d = decisions[it.id], eff = effective(it, d), s = summarize(items, decisions);
    const done = s.total - s.pending;
    $('progress').textContent = done + ' of ' + s.total + ' reviewed \\u00b7 ' + s.edited + ' edited \\u00b7 '
      + s.rejected + ' rejected \\u00b7 ' + s.pending + ' to go';
    $('bar').max = s.total; $('bar').value = done;
    $('pos').textContent = 'Item ' + (cur + 1) + ' of ' + items.length + ' \\u00b7 ' + it.id;
    $('badges').replaceChildren(
      el('span', { class: 'badge' }, it.metadata.category),
      el('span', { class: 'badge' }, 'expected: ' + it.metadata.behaviour));
    $('question').value = eff.question;
    $('reference').value = eff.reference_answer;
    $('comment').value = eff.comment;
    $('refLabel').textContent = it.metadata.behaviour === 'refuse'
      ? 'Why it cannot be answered (reference)' : 'Reference answer';
    $('status').textContent = d && d.status ? 'Decided: ' + d.status : 'Not reviewed yet';
    $('notes').textContent = it.notes ? 'Drafter note: ' + it.notes : '';
    const gold = $('gold');
    gold.replaceChildren();
    if (!it.gold.length) {
      gold.append(el('p', { class: 'muted' }, 'No gold article: this should not be answerable from the C\\u00f3digo.'));
    }
    for (const g of it.gold) {
      gold.append(el('details', { open: '' },
        el('summary', {}, 'Artigo ' + g.article + ' \\u2014 ' + g.epigraph),
        el('p', { class: 'muted' }, g.section),
        el('pre', {}, g.text)));
    }
  }
  function decide(status) {
    const it = items[cur];
    const q = $('question').value.trim(), r = $('reference').value.trim();
    let st = status;
    if (st === 'accept' && (q !== it.question || r !== it.reference_answer)) st = 'edit';
    decisions[it.id] = { status: st, question: q, reference_answer: r, comment: $('comment').value.trim() };
    save();
    const n = nextPending(items, decisions, cur);
    if (n !== -1) cur = n; else if (cur < items.length - 1) cur++;
    render();
  }
  function go(delta) { cur = (cur + delta + items.length) % items.length; render(); }
  function download() {
    const body = JSON.stringify({ exported_at: new Date().toISOString(), decisions: decisions }, null, 2);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([body], { type: 'application/json' }));
    a.download = 'review_decisions.json';
    document.body.append(a); a.click(); a.remove();
  }
  $('accept').addEventListener('click', () => decide('accept'));
  $('reject').addEventListener('click', () => decide('reject'));
  $('prev').addEventListener('click', () => go(-1));
  $('next').addEventListener('click', () => go(1));
  $('todo').addEventListener('click', () => { const n = nextPending(items, decisions, cur); if (n !== -1) { cur = n; render(); } });
  $('download').addEventListener('click', download);
  document.addEventListener('keydown', (e) => {
    const typing = ['TEXTAREA', 'INPUT'].includes(document.activeElement && document.activeElement.tagName);
    if (typing) { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); decide('accept'); } return; }
    if (e.key === 'a') decide('accept');
    else if (e.key === 'r') decide('reject');
    else if (e.key === 'ArrowRight' || e.key === 'n') go(1);
    else if (e.key === 'ArrowLeft' || e.key === 'p') go(-1);
  });
  save();
  render();
})();
"""

PAGE = (
    """<!doctype html>
<html lang="pt-PT">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Golden set review</title>
<style>
:root { --bg:#fafaf8; --fg:#1d1d1b; --muted:#6a6a64; --card:#fff; --line:#d9d9d2; --accent:#1f5fbf; --bad:#b3261e; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161615; --fg:#ecece8; --muted:#a3a39b; --card:#1f1f1d; --line:#3a3a36; --accent:#7aa7f0; --bad:#f2928b; }
}
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
main { max-width:820px; margin:0 auto; padding:16px; }
h1 { font-size:1.1rem; margin:0 0 4px; }
.muted { color:var(--muted); font-size:.9rem; margin:.2rem 0; }
progress { width:100%; height:10px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; margin:12px 0; }
.badge { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:0 10px; margin-right:6px; font-size:.85rem; }
label { display:block; font-weight:600; margin:12px 0 4px; }
textarea { width:100%; box-sizing:border-box; font:inherit; color:inherit; background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:8px; }
pre { white-space:pre-wrap; font:inherit; margin:4px 0 8px; }
summary { cursor:pointer; font-weight:600; }
.row { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
button { font:inherit; padding:8px 14px; border-radius:6px; border:1px solid var(--line); background:var(--card); color:var(--fg); cursor:pointer; }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
button.danger { color:var(--bad); border-color:var(--bad); }
</style>
</head>
<body>
<main>
  <h1>Golden set review</h1>
  <p id="progress" class="muted" aria-live="polite"></p>
  <progress id="bar" value="0" max="1"></progress>
  <p id="storage" class="muted"></p>
  <div class="card">
    <p class="muted"><span id="pos"></span> \\u00b7 <span id="status"></span></p>
    <p id="badges"></p>
    <label for="question">Question</label>
    <textarea id="question" rows="3"></textarea>
    <label for="reference" id="refLabel">Reference answer</label>
    <textarea id="reference" rows="5"></textarea>
    <p id="notes" class="muted"></p>
    <h2 style="font-size:1rem;margin:16px 0 4px">Gold article text</h2>
    <div id="gold"></div>
    <label for="comment">Comment (optional)</label>
    <textarea id="comment" rows="2"></textarea>
    <div class="row">
      <button id="accept" class="primary" title="Shortcut: a, or Ctrl+Enter in a text box">Accept (a)</button>
      <button id="reject" class="danger" title="Shortcut: r">Reject (r)</button>
      <button id="prev">\\u2190 Prev</button>
      <button id="next">Next \\u2192</button>
      <button id="todo">Next to review</button>
      <button id="download">Download decisions</button>
    </div>
  </div>
  <p class="muted">Edit the question or the reference answer in place, then Accept: it is saved as an edit.</p>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>__CORE__</script>
<script>__APP__</script>
</body>
</html>
""".replace("\\u00b7", "\u00b7")
    .replace("\\u2190", "\u2190")
    .replace("\\u2192", "\u2192")
)


def render_review_page(items: Sequence[Mapping[str, Any]], chunks: Sequence[Chunk]) -> str:
    by_id = {c.id: c for c in chunks}
    payload = []
    for it in items:
        gold_ids = resolve_gold(it.get("gold_passage_ids", []), by_id.keys())
        gold = [
            {
                "id": by_id[g].id,
                "article": by_id[g].article,
                "epigraph": by_id[g].epigraph,
                "section": by_id[g].section,
                "text": by_id[g].text,
            }
            for g in gold_ids
        ]
        payload.append({**it, "gold": gold})
    data = json.dumps({"items": payload}, ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace("__DATA__", data).replace("__CORE__", CORE_JS).replace("__APP__", APP_JS)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the local review page")
    b.add_argument("drafts", type=Path)
    b.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    b.add_argument("--out", type=Path, default=Path("review/review.html"))
    a = sub.add_parser("apply", help="merge downloaded decisions into the drafts")
    a.add_argument("drafts", type=Path)
    a.add_argument("decisions", type=Path)
    a.add_argument("--reviewer", required=True)
    a.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    items = _read_jsonl(args.drafts)
    if args.cmd == "build":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_review_page(items, read_chunks(args.chunks)), encoding="utf-8")
        print(f"{len(items)} items -> {args.out}")
        return
    raw = json.loads(args.decisions.read_text(encoding="utf-8"))
    result = apply_decisions(items, raw.get("decisions", raw), args.reviewer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in result.items), encoding="utf-8"
    )
    print(
        f"kept {len(result.items)}, rejected {len(result.rejected)}, "
        f"edited {len(result.edited)}, still pending {len(result.pending)}"
    )


if __name__ == "__main__":
    main()
