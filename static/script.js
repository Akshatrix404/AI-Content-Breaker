const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// -------------------------------------------------------------- mode tabs
$$('.mode-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    $$('.mode-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const mode = tab.dataset.mode;
    $$('.mode-panel').forEach(p => p.hidden = true);
    $(`#mode-${mode}`).hidden = false;
    if (mode === 'history') renderHistory();
  });
});

function verdictInfo(verdict) {
  if (verdict === 'Likely Human-Written') return { color: 'var(--green)', cls: 'human' };
  if (verdict === 'Mixed / Inconclusive') return { color: 'var(--amber)', cls: 'mixed' };
  return { color: 'var(--red)', cls: '' };
}

function escapeHtml(str) {
  return String(str).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Request failed.');
  return data;
}

function setLoading(btn, isLoading, idleLabel, busyLabel) {
  btn.disabled = isLoading;
  const spinner = btn.querySelector('.btn-spinner');
  if (spinner) spinner.hidden = !isLoading;
  const label = btn.querySelector('.btn-label');
  if (label) label.textContent = isLoading ? busyLabel : idleLabel;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// word-level diff (LCS-based), used by the humanizer's track-changes view
function diffWords(a, b) {
  const wa = a.split(/(\s+)/);
  const wb = b.split(/(\s+)/);
  const n = wa.length, m = wb.length;
  if (n * m > 3_000_000) return null; // safety valve for pathological inputs
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = wa[i] === wb[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  let i = 0, j = 0;
  const out = [];
  while (i < n && j < m) {
    if (wa[i] === wb[j]) { out.push({ type: 'same', text: wa[i] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push({ type: 'removed', text: wa[i] }); i++; }
    else { out.push({ type: 'added', text: wb[j] }); j++; }
  }
  while (i < n) { out.push({ type: 'removed', text: wa[i] }); i++; }
  while (j < m) { out.push({ type: 'added', text: wb[j] }); j++; }
  return out;
}

function renderDiff(a, b) {
  const segs = diffWords(a, b);
  if (!segs) return `<div>${escapeHtml(a)}</div><div>${escapeHtml(b)}</div>`;
  return segs.map(s => {
    if (s.type === 'same') return escapeHtml(s.text);
    if (s.type === 'removed') return `<span class="tc-removed">${escapeHtml(s.text)}</span>`;
    return `<span class="tc-added">${escapeHtml(s.text)}</span>`;
  }).join('');
}

// ============================================================== SINGLE MODE
const sourceText = $('#sourceText');
const charCount = $('#charCount');
const analyzeBtn = $('#analyzeBtn');
const errorMsg = $('#errorMsg');
const consolePanel = $('#consolePanel');
const ledgerList = $('#ledgerList');
const totalTime = $('#totalTime');
const results = $('#results');
let lastResult = null;

sourceText.addEventListener('input', () => {
  charCount.textContent = `${sourceText.value.length.toLocaleString()} characters`;
});

$('#clearBtn').addEventListener('click', () => {
  sourceText.value = '';
  charCount.textContent = '0 characters';
  results.hidden = true;
  consolePanel.hidden = true;
  errorMsg.hidden = true;
});

$('#loadHuman').addEventListener('click', () => loadSample('human'));
$('#loadAI').addEventListener('click', () => loadSample('ai'));

$('#fileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  sourceText.value = await file.text();
  sourceText.dispatchEvent(new Event('input'));
});

async function loadSample(kind) {
  const res = await fetch(`/api/sample/${kind}`);
  const data = await res.json();
  sourceText.value = data.text;
  sourceText.dispatchEvent(new Event('input'));
}

analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  const text = sourceText.value.trim();
  errorMsg.hidden = true;
  if (!text) {
    errorMsg.textContent = 'Please enter or load some text first.';
    errorMsg.hidden = false;
    return;
  }
  setLoading(analyzeBtn, true, 'Break it down', 'Breaking it down…');
  results.hidden = true;
  consolePanel.hidden = true;
  $('#humanizeSection').hidden = true;

  try {
    const data = await postJSON('/api/analyze', { text });
    await playLedger(data.pipeline_stages);
    lastResult = data;
    renderAll(data);
    saveToHistory(text, data);
    results.hidden = false;
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    await playJarSimulation(data);
    await playFormulaSimulation(data.report);
  } catch (err) {
    errorMsg.textContent = err.message;
    errorMsg.hidden = false;
  } finally {
    setLoading(analyzeBtn, false, 'Break it down', 'Breaking it down…');
  }
}

async function playLedger(stages) {
  consolePanel.hidden = false;
  ledgerList.innerHTML = '';
  for (let i = 0; i < stages.length; i++) {
    const s = stages[i];
    const li = document.createElement('li');
    li.style.animationDelay = `${i * 0.05}s`;
    li.innerHTML = `<span class="lg-num mono">${i + 1}</span>` +
      `<span class="lg-stage">${s.stage}</span>` +
      `<span class="lg-detail">${s.detail || ''}</span>` +
      `<span class="lg-time">${s.elapsed_ms}ms</span>`;
    ledgerList.appendChild(li);
    await sleep(90);
  }
  const last = stages[stages.length - 1];
  totalTime.textContent = last ? `total ${last.elapsed_ms}ms` : '';
}

// ---- token sort simulation (tokenize -> classify -> drop into jar) ----
async function playJarSimulation(data) {
  const streamEl = $('#jarStream');
  const humanBody = $('#jarHumanBody');
  const aiBody = $('#jarAIBody');
  const humanCountEl = $('#jarHumanCount');
  const aiCountEl = $('#jarAICount');
  const noteEl = $('#jarFormulaNote');
  streamEl.innerHTML = '';
  humanBody.innerHTML = '';
  aiBody.innerHTML = '';
  humanCountEl.textContent = '0';
  aiCountEl.textContent = '0';
  noteEl.textContent = '';

  // Phase A: show the raw token stream appearing (the "breaking into tokens" step)
  const allTokens = [];
  data.sentences.forEach(s => s.tokens.forEach(t => allTokens.push(t.value)));
  const streamDelay = allTokens.length > 80 ? 8 : 22;
  for (let i = 0; i < allTokens.length; i++) {
    const chip = document.createElement('span');
    chip.className = 'stream-token';
    chip.style.animationDelay = '0s';
    chip.textContent = allTokens[i];
    streamEl.appendChild(chip);
    if (i % 3 === 0) await sleep(streamDelay);
  }
  await sleep(400);

  // Phase B: fade the stream out, then sort classified tokens into jars one at a time
  streamEl.style.transition = 'opacity 0.3s ease';
  streamEl.style.opacity = '0.25';

  const tj = data.token_jars;
  const tokens = tj.tokens;
  const sortDelay = tokens.length > 60 ? 28 : 65;
  let hCount = 0, aCount = 0;
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    const chip = document.createElement('span');
    chip.className = `jar-token jt-${t.jar}`;
    chip.textContent = t.value;
    chip.title = `AI-likelihood score for this token = ${t.ai_score}`;
    if (t.jar === 'ai') {
      aiBody.appendChild(chip);
      aCount++;
      aiCountEl.textContent = aCount;
    } else {
      humanBody.appendChild(chip);
      hCount++;
      humanCountEl.textContent = hCount;
    }
    aiBody.scrollTop = aiBody.scrollHeight;
    humanBody.scrollTop = humanBody.scrollHeight;
    await sleep(sortDelay);
  }
  streamEl.style.opacity = '1';

  noteEl.innerHTML =
    `<strong>${tj.ai_count}</strong> of ${tj.total_classified} classified tokens (<strong>${tj.ai_jar_pct}%</strong>) sorted into the AI jar · ` +
    `<strong>${tj.human_count}</strong> (<strong>${tj.human_jar_pct}%</strong>) into the Human jar. ` +
    `Each token's jar comes from a fixed formula: start at the document's calibrated AI score (re-centred on the 24% AI threshold), ` +
    `nudge by its sentence's suspicion score, then nudge again for transition words, repeated function words, ` +
    `hapax rarity, contractions and hedge words — score ≥ 50% lands in the AI jar. Click a token below in ` +
    `"Every token, analyzed" to see a token's exact arithmetic.`;
}

// ---- formula simulation (score x weight = contribution, running total) ----
async function playFormulaSimulation(report) {
  const stepsEl = $('#formulaSteps');
  const totalEl = $('#formulaTotal');
  stepsEl.innerHTML = '';
  totalEl.innerHTML = '';
  let running = 0;
  const items = [...report.breakdown].sort((a, b) => b.weight_pct - a.weight_pct);
  for (const f of items) {
    running += f.contribution_pct;
    const row = document.createElement('div');
    row.className = 'formula-step';
    row.innerHTML = `
      <span class="fs-label">${escapeHtml(f.label)}</span>
      <span class="fs-expr mono">${f.score_pct}% score × ${f.weight_pct}% weight = ${f.contribution_pct} pts</span>
      <span class="fs-running mono">running total → ${running.toFixed(1)}%</span>
    `;
    stepsEl.appendChild(row);
    stepsEl.scrollTop = stepsEl.scrollHeight;
    await sleep(160);
  }
  const { color, cls } = verdictInfo(report.verdict);
  totalEl.innerHTML = `
    <span class="ft-label">Final AI-content score</span>
    <span class="ft-value" style="color:${color}">${report.percentage}%</span>
    <span class="ft-verdict">${escapeHtml(report.verdict)} · confidence ${escapeHtml(report.confidence)}</span>
  `;
}

$('#replayJarBtn').addEventListener('click', () => { if (lastResult) playJarSimulation(lastResult); });
$('#replayFormulaBtn').addEventListener('click', () => { if (lastResult) playFormulaSimulation(lastResult.report); });

function renderAll(data) {
  renderVerdict(data.report, data.input_stats);
  renderAnnotatedDoc(data.sentences);
  renderReceipt(data.report);
  renderDiagnostics(data.diagnostics);
  renderSymtab(data.symbol_table);
  renderMetrics(data.linguistics, data.passive_ratio);
  renderSentenceTrees(data.sentences);
  renderLL1(data.ll1_analysis, data.sentences);
  renderIR(data.ir_analysis);
  renderDataflow(data.dataflow_analysis);
  renderCFG(data.cfg_analysis);
}

// ---- 1. Formal grammar verification (LL(1)) ----
function renderLL1(ll1, sentences) {
  const statRow = $('#ll1Stats');
  const conflictsEl = $('#ll1Conflicts');
  const tableEl = $('#ll1SentenceTable');
  if (!ll1) { statRow.innerHTML = ''; conflictsEl.innerHTML = ''; tableEl.innerHTML = ''; return; }

  const conflictCls = ll1.conflicts === 0 ? 'good' : 'warn';
  statRow.innerHTML = `
    <div class="ci-stat"><strong>${ll1.grammar.production_count}</strong>productions, ${ll1.grammar.non_terminal_count} non-terminals</div>
    <div class="ci-stat"><strong>${ll1.grammar.table_size}</strong>parsing-table cells</div>
    <div class="ci-stat ${conflictCls}"><strong>${ll1.conflicts}</strong>table conflict(s) ${ll1.is_ll1 ? '(grammar is LL(1))' : '(grammar is not strictly LL(1))'}</div>
    <div class="ci-stat"><strong>${ll1.rejected} / ${ll1.total}</strong>sentence(s) rejected by the table parser</div>
  `;
  conflictsEl.innerHTML = (ll1.grammar.conflicts || []).map(c => `
    <div class="ll1-conflict-item">conflict @ (${c.nonterminal}, ${c.terminal}): '${c.existing}' vs '${c.new}' — classic PP-attachment-style ambiguity</div>
  `).join('');
  tableEl.innerHTML = sentences.map((s, i) => {
    const r = s.ll1 || { accepted: false, errors: [], productions_applied: 0 };
    const verdict = r.accepted
      ? `<span class="ll1-verdict accept">ACCEPT · ${r.productions_applied} productions</span>`
      : `<span class="ll1-verdict reject">REJECT · ${r.errors.length} error(s)</span>`;
    return `<div class="ll1-row"><span class="ll1-idx mono">${i + 1}</span><span class="ll1-text">${escapeHtml(s.text.slice(0, 90))}${s.text.length > 90 ? '…' : ''}</span>${verdict}</div>`;
  }).join('');
}

// ---- 2. Intermediate representation + optimizer ----
function renderIR(ir) {
  const statRow = $('#irStats');
  const listEl = $('#irListing');
  if (!ir) { statRow.innerHTML = ''; listEl.innerHTML = ''; return; }
  statRow.innerHTML = `
    <div class="ci-stat"><strong>${ir.total}</strong>IR instructions generated</div>
    <div class="ci-stat ${ir.cse_eliminated ? 'warn' : 'good'}"><strong>${ir.cse_eliminated}</strong>eliminated by CSE</div>
    <div class="ci-stat ${ir.dce_removed ? 'warn' : 'good'}"><strong>${ir.dce_removed}</strong>removed by DCE</div>
    <div class="ci-stat"><strong>${ir.live}</strong>live instructions remain</div>
  `;
  listEl.innerHTML = ir.instructions.map(instr => {
    const expr = instr.op === 'ASSERT'
      ? `(${instr.arg1 ?? '?'}, ${instr.arg2 ?? '?'}, ${instr.result ?? '∅'})`
      : `MOD(${instr.arg1 ?? '?'}, ${instr.arg2 ?? '?'})`;
    return `
      <div class="ir-row">
        <span class="ir-op">${instr.op}</span>
        <span class="ir-expr" title="${escapeHtml(instr.reason || instr.note || '')}">s${instr.sentence_index + 1} · ${escapeHtml(expr)}</span>
        <span class="ir-status ${instr.status}">${instr.status}</span>
      </div>`;
  }).join('');
}

// ---- 3. Data-flow analysis (reaching definitions) ----
function renderDataflow(df) {
  const statRow = $('#dfStats');
  const tableEl = $('#dfTable');
  if (!df) { statRow.innerHTML = ''; tableEl.innerHTML = ''; return; }
  statRow.innerHTML = `
    <div class="ci-stat"><strong>${df.iterations}</strong>worklist iterations to fixed point</div>
    <div class="ci-stat"><strong>${df.total_pronouns}</strong>pronoun use(s) checked</div>
    <div class="ci-stat ${df.unresolved_references.length ? 'bad' : 'good'}"><strong>${df.unresolved_references.length}</strong>unresolved reference(s)</div>
  `;
  const rows = [`<div class="df-row df-head"><span>#</span><span>GEN</span><span>KILL</span><span>IN</span><span>OUT</span></div>`];
  df.per_sentence.forEach(s => {
    rows.push(`<div class="df-row">
      <span>${s.index + 1}</span>
      <span title="${escapeHtml(s.gen.join(', '))}">{${escapeHtml(s.gen.join(', '))}}</span>
      <span title="${escapeHtml(s.kill.join(', '))}">{${escapeHtml(s.kill.join(', '))}}</span>
      <span title="${escapeHtml(s.in.join(', '))}">{${escapeHtml(s.in.join(', '))}}</span>
      <span title="${escapeHtml(s.out.join(', '))}">{${escapeHtml(s.out.join(', '))}}</span>
    </div>`);
  });
  tableEl.innerHTML = rows.join('');
  $('#dfUnresolved').innerHTML = df.unresolved_references.length
    ? df.unresolved_references.map(u => `sentence ${u.sentence_index + 1}: '${escapeHtml(u.pronoun)}' — ${escapeHtml(u.reason)}`).join('<br>')
    : '<span class="faint">No unresolved pronoun references detected.</span>';
}

// ---- 4. Discourse control-flow graph ----
function renderCFG(cfg) {
  const statRow = $('#cfgStats');
  const listEl = $('#cfgEdgeList');
  if (!cfg) { statRow.innerHTML = ''; listEl.innerHTML = ''; return; }
  statRow.innerHTML = `
    <div class="ci-stat"><strong>${cfg.node_count}</strong>nodes (sentences)</div>
    <div class="ci-stat"><strong>${cfg.edge_count}</strong>edges</div>
    <div class="ci-stat"><strong>M = ${cfg.cyclomatic_complexity}</strong>cyclomatic complexity (E − N + 2P)</div>
  `;
  listEl.innerHTML = cfg.edges.map(e => `
    <div class="cfg-edge-row">
      <span>s${e.src + 1} → s${e.dst + 1}</span>
      <span class="cfg-edge-kind ${e.kind}">${e.kind}${e.entity ? ' · ' + escapeHtml(e.entity) : ''}</span>
    </div>
  `).join('');
}

function renderVerdict(report, stats) {
  const { color, cls } = verdictInfo(report.verdict);
  const stamp = $('#verdictStamp');
  stamp.textContent = report.verdict;
  stamp.className = `stamp ${cls}`;
  const score = $('#verdictScore');
  score.textContent = `${report.percentage}%`;
  score.style.color = color;
  $('#verdictStats').innerHTML =
    `${stats.sentences} sentences · ${stats.words} words · ${stats.characters} characters<br>` +
    `Confidence: <strong>${report.confidence}</strong>`;
}

// ---- annotated manuscript (interlinear gloss + margin notes) ----
function tokenFlagClasses(t) {
  const c = [];
  if (t.is_transition_word) c.push('fl-transition');
  if (t.is_hedge_word) c.push('fl-hedge');
  if (t.is_contraction) c.push('fl-contraction');
  if (t.is_hapax) c.push('fl-hapax');
  return c.join(' ');
}

function tokenFlagLabels(t) {
  const labels = [];
  if (t.is_stopword) labels.push('stopword');
  if (t.is_transition_word) labels.push('discourse connective');
  if (t.is_hedge_word) labels.push('hedge word');
  if (t.is_contraction) labels.push('contraction');
  if (t.is_hapax) labels.push('appears once in document (hapax)');
  else if (t.doc_frequency != null) labels.push(`appears ${t.doc_frequency}x in document`);
  if (t.syllables != null) labels.push(`${t.syllables} syllable${t.syllables === 1 ? '' : 's'}`);
  return labels.length ? labels.join(' · ') : 'punctuation / structural token — not scored individually';
}

function renderAnnotatedDoc(sentences) {
  const el = $('#annotatedDoc');
  el.innerHTML = sentences.map((s, si) => {
    const glossWords = s.tokens.map((t, ti) => `
      <span class="gloss-word" data-si="${si}" data-ti="${ti}">
        <span class="gw-text ${tokenFlagClasses(t)}">${escapeHtml(t.value)}</span>
        <span class="gw-tag">${t.pos}</span>
      </span>`).join('');

    const notes = s.notes && s.notes.length
      ? `<ul>${s.notes.map(n => `<li>${escapeHtml(n)}</li>`).join('')}</ul>`
      : '<p class="no-notes">nothing unusual detected</p>';

    return `
      <div class="annotated-line">
        <span class="al-index mono">${si + 1}</span>
        <div class="al-text-col">
          <div class="gloss-row">${glossWords}</div>
          <div class="token-detail" id="token-detail-${si}"></div>
        </div>
        <div class="al-notes-col">${notes}</div>
      </div>
    `;
  }).join('');

  // delegate click handling for token inspection
  el.querySelectorAll('.gloss-word').forEach(gw => {
    gw.addEventListener('click', () => {
      const si = gw.dataset.si, ti = gw.dataset.ti;
      const sentence = lastResult.sentences[si];
      const tok = sentence.tokens[ti];
      const detailBox = $(`#token-detail-${si}`);
      const alreadySelected = gw.classList.contains('selected');

      el.querySelectorAll(`.annotated-line:nth-child(${Number(si) + 1}) .gloss-word`)
        .forEach(w => w.classList.remove('selected'));

      if (alreadySelected) {
        detailBox.classList.remove('open');
        detailBox.innerHTML = '';
        return;
      }
      gw.classList.add('selected');
      detailBox.innerHTML =
        `<span class="td-word">${escapeHtml(tok.value)}</span>` +
        `<span class="mono" style="color:var(--navy)">${tok.pos}</span> — ` +
        `<span class="td-flags">${escapeHtml(tokenFlagLabels(tok))}</span>`;
      detailBox.classList.add('open');
    });
  });
}

// ---- score receipt ----
function renderReceipt(report) {
  const el = $('#receipt');
  const byCategory = {};
  report.breakdown.forEach(f => {
    (byCategory[f.category_label] ||= []).push(f);
  });
  let html = '';
  Object.entries(byCategory).forEach(([label, feats]) => {
    html += `<div class="receipt-cat">${escapeHtml(label)}</div>`;
    feats.forEach(f => {
      html += `
        <div class="receipt-row">
          <span class="rr-label">${escapeHtml(f.label)}</span>
          <span class="rr-fill"></span>
          <span class="rr-value">+${f.contribution_pct}</span>
        </div>
        <div class="receipt-note">${escapeHtml(f.note)}</div>
      `;
    });
  });
  html += `
    <div class="receipt-total">
      <span>AI-likelihood total</span>
      <span>${report.percentage}%</span>
    </div>
  `;
  el.innerHTML = html;
}

// ---- diagnostics ----
function renderDiagnostics(diags) {
  const el = $('#diagList');
  el.innerHTML = diags.map(d => `
    <li class="diag-item ${d.severity}">
      <span class="diag-tag">${d.severity}</span>
      <span class="diag-msg">${escapeHtml(d.message)} <span class="faint mono" style="font-size:10px">[${d.category}]</span></span>
    </li>
  `).join('');
}

// ---- symbol table + metrics (deep dive) ----
function renderSymtab(sym) {
  const el = $('#symtabGrid');
  const li = (arr) => arr.map(([w, c]) => `<li><span>${escapeHtml(String(w))}</span><span>${c}</span></li>`).join('');
  const openers = sym.repeated_openers.length ? li(sym.repeated_openers) : '<li><span class="faint">none repeated</span></li>';
  el.innerHTML = `
    <div class="symtab-card">
      <div class="symtab-metric">${sym.vocab_size}</div>
      <div class="symtab-metric-label">distinct words / ${sym.total_words} total</div>
      <h3 style="margin-top:14px">Type-token ratio</h3>
      <div class="symtab-metric" style="font-size:16px">${sym.type_token_ratio} <span class="symtab-metric-label">(hapax ${sym.hapax_ratio})</span></div>
    </div>
    <div class="symtab-card"><h3>Top words</h3><ul>${li(sym.top_words)}</ul></div>
    <div class="symtab-card"><h3>Top bigrams</h3><ul>${li(sym.top_bigrams)}</ul></div>
    <div class="symtab-card"><h3>Repeated grammar-rule signatures</h3><ul>${li(sym.top_rule_signatures)}</ul></div>
    <div class="symtab-card"><h3>Repeated sentence openers</h3><ul>${openers}</ul></div>
  `;
}

function renderMetrics(ling, passiveRatio) {
  const el = $('#metricsGrid');
  const cards = [
    ['Flesch reading ease', ling.flesch_reading_ease],
    ['Flesch-Kincaid grade', ling.flesch_kincaid_grade],
    ['Avg sentence length', `${ling.avg_sentence_length_words}w`],
    ['Avg word length', `${ling.avg_word_length_chars} chars`],
    ['Long-word ratio', `${ling.long_word_ratio_pct}%`],
    ['Stopword ratio', `${ling.stopword_ratio_pct}%`],
    ['Hedge words / 100w', ling.hedge_words_per_100],
    ['Contractions / 100w', ling.contractions_per_100_words],
    ['POS-bigram entropy', `${ling.pos_bigram_entropy_bits} bits`],
    ['Compression ratio', ling.compression_ratio_pct != null ? `${ling.compression_ratio_pct}%` : '—'],
    ['Passive voice ratio', passiveRatio != null ? `${Math.round(passiveRatio * 100)}%` : '—'],
    ['1st-person pronouns', `${ling.pronoun_person_pct.first}%`],
  ];
  el.innerHTML = cards.map(([label, val]) => `
    <div class="metric-card"><div class="metric-value">${val}</div><div class="metric-label">${label}</div></div>
  `).join('');
}

// ---- full parse trees (deep dive) ----
function renderSentenceTrees(sentences) {
  const el = $('#sentenceTrees');
  el.innerHTML = sentences.map((s, i) => `
    <div style="margin-bottom:14px">
      <span class="sentence-toggle" data-i="${i}">▸ Sentence ${i + 1} — ${escapeHtml(s.text.slice(0, 60))}${s.text.length > 60 ? '…' : ''}
        <span class="faint mono">(depth ${s.tree_depth}, clause ${s.clause_depth})</span></span>
      <div class="tree-wrap" id="tree-${i}" hidden></div>
    </div>
  `).join('');
  el.querySelectorAll('.sentence-toggle').forEach(t => {
    t.addEventListener('click', () => {
      const i = t.dataset.i;
      const box = $(`#tree-${i}`);
      box.hidden = !box.hidden;
      if (!box.hidden && !box.dataset.rendered) {
        box.innerHTML = renderTree(sentences[i].tree, true);
        box.dataset.rendered = '1';
      }
    });
  });
}

function renderTree(node, isRoot = false) {
  if (!node) return '';
  const a = node.attrs || {};
  const attrStr = [
    a.depth != null ? `depth=${a.depth}` : null,
    a.width != null ? `width=${a.width}` : null,
    a.voice ? `voice=${a.voice}` : null,
    a.clause_depth != null ? `clause=${a.clause_depth}` : null,
  ].filter(Boolean).join(' ');
  const labelHtml = node.label.includes(':')
    ? `<span class="tree-symbol">${node.symbol}</span><span class="tree-token">${escapeHtml(node.label.split(':').slice(1).join(':'))}</span>`
    : `<span class="tree-symbol">${node.symbol}</span>`;
  const childrenHtml = (node.children || []).map(c => renderTree(c, false)).join('');
  return `
    <div class="tree-node ${isRoot ? 'root' : ''}">
      <div class="tree-node-label">${labelHtml} <span class="tree-attrs">${attrStr}</span></div>
      ${childrenHtml}
    </div>
  `;
}

// ---- export ----
$('#exportJsonBtn').addEventListener('click', () => {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ai-content-breaker-report-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});
$('#printBtn').addEventListener('click', () => window.print());

// ---- humanizer ----
$('#humanizeBtn').addEventListener('click', async () => {
  const text = sourceText.value.trim();
  if (!text) return;
  const btn = $('#humanizeBtn');
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Retrieving examples, rewriting & re-scoring… (~15-20 s)';
  try {
    const data = await postJSON('/api/humanize', { text });
    renderHumanize(data);
    $('#humanizeSection').hidden = false;
    $('#humanizeSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) {
    errorMsg.textContent = e.message;
    errorMsg.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
});

function renderHumanize(data) {
  const before = verdictInfo(data.original_verdict);
  const after = verdictInfo(data.humanized_verdict || data.original_verdict);

  $('#humBefore').textContent = `${data.original_score}%`;
  $('#humBefore').style.color = before.color;
  $('#humBeforeVerdict').textContent = data.original_verdict;

  $('#humAfter').textContent = data.humanized_score != null ? `${data.humanized_score}%` : '—';
  $('#humAfter').style.color = after.color;
  $('#humAfterVerdict').textContent = data.humanized_verdict || '—';

  const delta = data.score_delta;
  const deltaEl = $('#humDelta');
  if (delta == null) { deltaEl.textContent = 'no change measured'; }
  else if (delta > 0) {
    deltaEl.textContent = `−${delta} pts`;
    deltaEl.style.color = 'var(--green)'; deltaEl.style.borderColor = 'var(--green)';
  } else if (delta < 0) {
    deltaEl.textContent = `+${Math.abs(delta)} pts`;
    deltaEl.style.color = 'var(--red)'; deltaEl.style.borderColor = 'var(--red)';
  } else { deltaEl.textContent = 'no change'; }

  $('#trackChanges').innerHTML = renderDiff(data.original_text, data.humanized_text) || '';

  const log = $('#editLog');
  const dl = data.deep_log
    ? `<div class="edit-entry"><span class="ee-pass">Local LLM (Ollama) + RAG · scorer-verified</span><div class="ee-rule">${data.deep_log.map(c => `round ${c.round}.${c.candidate}: ${c.score ?? 'n/a'}% ${c.kept ? '✓ kept' : '✗ rejected'}`).join(' &nbsp;|&nbsp; ')}</div></div>`
    : (data.deep_error ? `<div class="edit-entry"><span class="ee-pass">Local LLM unavailable</span><div class="ee-rule">${escapeHtml(data.deep_error)} — showing rule-based result only.</div></div>` : '');
  if (!data.edits.length) {
    log.innerHTML = dl;
    log.innerHTML = '<p class="muted">No rule-based transformations applied — this text didn\'t match any humanizer pattern (no formal connectives, no long compound sentences, no repeated openers to fix).</p>';
    return;
  }
  log.innerHTML = dl + data.edits.map(e => `
    <div class="edit-entry">
      <span class="ee-pass">${e.pass.replaceAll('_', ' ')} · sentence ${e.sentence_index + 1}</span>
      <div class="ee-rule">${escapeHtml(e.rule)}</div>
      <div class="mono" style="font-size:12px">${renderDiff(e.before, e.after)}</div>
    </div>
  `).join('');
}

// ============================================================== COMPARE MODE
$('#compareBtn').addEventListener('click', async () => {
  const a = $('#compareA').value.trim();
  const b = $('#compareB').value.trim();
  const err = $('#compareError');
  err.hidden = true;
  if (!a || !b) {
    err.textContent = 'Please provide both Text A and Text B.';
    err.hidden = false;
    return;
  }
  const btn = $('#compareBtn');
  setLoading(btn, true, 'Compare', 'Comparing…');
  try {
    const [ra, rb] = await Promise.all([postJSON('/api/analyze', { text: a }), postJSON('/api/analyze', { text: b })]);
    renderCompare(ra, rb);
    $('#compareResults').hidden = false;
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    setLoading(btn, false, 'Compare', 'Comparing…');
  }
});

function renderCompare(ra, rb) {
  const diff = Math.abs(ra.report.percentage - rb.report.percentage).toFixed(1);
  const higher = ra.report.percentage >= rb.report.percentage ? 'Text A' : 'Text B';
  $('#compareSummary').innerHTML =
    `<strong>${higher}</strong> scores ${diff} points higher on AI-likelihood. ` +
    `A: ${ra.report.percentage}% (${ra.report.verdict}) · B: ${rb.report.percentage}% (${rb.report.verdict})`;

  const card = (label, r) => {
    const top = r.report.breakdown.slice(0, 4);
    return `
      <div class="compare-card">
        <h3>${label}</h3>
        <div class="cc-score" style="color:${verdictInfo(r.report.verdict).color}">${r.report.percentage}%</div>
        <div class="muted" style="font-size:12px">${r.report.verdict} · confidence ${r.report.confidence}</div>
        <ul>${top.map(f => `<li><span>${f.label}</span><span>+${f.contribution_pct}</span></li>`).join('')}</ul>
      </div>
    `;
  };
  $('#compareCards').innerHTML = card('Text A', ra) + card('Text B', rb);
}

// ============================================================== BATCH MODE
$('#batchBtn').addEventListener('click', async () => {
  const text = $('#batchText').value.trim();
  const err = $('#batchError');
  err.hidden = true;
  if (!text) {
    err.textContent = 'Please paste a multi-paragraph document first.';
    err.hidden = false;
    return;
  }
  const btn = $('#batchBtn');
  setLoading(btn, true, 'Analyze all paragraphs', 'Analyzing…');
  try {
    const data = await postJSON('/api/analyze_batch', { text });
    renderBatch(data);
    $('#batchResults').hidden = false;
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    setLoading(btn, false, 'Analyze all paragraphs', 'Analyzing…');
  }
});

function renderBatch(data) {
  $('#batchOverall').textContent = `overall avg ${data.overall_average}% · ${data.count} paragraphs`;
  $('#batchTable').innerHTML = data.paragraphs.map(p => {
    const { color } = verdictInfo(p.verdict);
    return `
      <div class="batch-row">
        <span class="bi mono">${p.index + 1}</span>
        <span class="bp">${escapeHtml(p.preview)}</span>
        <span class="bscore" style="color:${color}">${p.percentage}%</span>
        <span class="bverdict" style="color:${color}">${p.verdict}</span>
      </div>
    `;
  }).join('');
}

// ============================================================== HISTORY
const HISTORY_KEY = 'aicb_history_v2';

function saveToHistory(text, data) {
  try {
    const hist = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    hist.unshift({
      preview: text.slice(0, 100),
      text,
      percentage: data.report.percentage,
      verdict: data.report.verdict,
      time: Date.now(),
    });
    localStorage.setItem(HISTORY_KEY, JSON.stringify(hist.slice(0, 25)));
  } catch (e) { /* localStorage unavailable -- ignore */ }
}

function renderHistory() {
  let hist = [];
  try { hist = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch (e) {}
  const el = $('#historyList');
  if (!hist.length) {
    el.innerHTML = '<p class="muted">No analyses yet — results from "Analyze text" are saved here automatically.</p>';
    return;
  }
  el.innerHTML = hist.map((h, i) => {
    const { color } = verdictInfo(h.verdict);
    const t = new Date(h.time);
    return `
      <div class="history-row" data-index="${i}">
        <span class="hscore" style="color:${color}">${h.percentage}%</span>
        <span class="hpreview">${escapeHtml(h.preview)}</span>
        <span class="htime">${t.toLocaleDateString()} ${t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
      </div>
    `;
  }).join('');
  $$('.history-row', el).forEach(row => {
    row.addEventListener('click', () => {
      const h = hist[Number(row.dataset.index)];
      $$('.mode-tab').forEach(t => t.classList.remove('active'));
      $('.mode-tab[data-mode="single"]').classList.add('active');
      $$('.mode-panel').forEach(p => p.hidden = true);
      $('#mode-single').hidden = false;
      sourceText.value = h.text;
      sourceText.dispatchEvent(new Event('input'));
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  });
}

$('#clearHistoryBtn').addEventListener('click', () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});
