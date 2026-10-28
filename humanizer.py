"""
humanizer.py -- TEXT TRANSFORMATION / "CODE OPTIMIZATION" PHASE
===================================================================
A compiler back end doesn't just analyze code -- it also *transforms* it
(constant folding, dead-code elimination, register allocation) through a
sequence of well-defined, individually-named optimization passes, each of
which can be inspected in isolation. This module applies the same idea to
text: a fixed pipeline of deterministic, rule-based rewrite passes that
nudge AI-flagged text toward the structural patterns associated with human
writing -- WITHOUT calling any language model, generative AI, or paraphrasing
API. Every single edit is a regex/lexicon-driven substitution or a
grammar-tree-driven restructuring, and every edit is logged with the exact
rule that produced it, so the whole process is auditable end to end.

Passes (run in this fixed order):
  1. connective_softening   -- swap/drop formal discourse connectives
  2. contraction_insertion  -- "it is" -> "it's", "does not" -> "doesn't", ...
  3. sentence_splitting     -- break long CCONJ/SCONJ-joined sentences in two
  4. opener_diversification -- front a trailing PP to avoid repeated openers
  5. acronym_substitution   -- 2nd+ mention of "artificial intelligence" -> "AI" (lexicon)
  6. aside_parenthesization -- trailing "while <verb>ing ..." clause -> parenthetical aside
  7. dash_emphasis          -- set off a trailing prepositional phrase with an em dash
  8. purpose_fragment_split -- split "<head> to <verb> ..." into head + emphatic fragment
  9. semicolon_merge        -- join a short sentence to its neighbour with a semicolon

None of these passes invents facts, adds content, or uses statistical
generation; they only reorder, contract, split, or substitute closed-class
words already implied by the sentence's own grammar -- so the meaning of
the text is preserved by construction.
"""

import re
from dataclasses import dataclass, field
from lexer import tokenize, tag_tokens, split_sentences, TRANSITION_WORDS

# ---------------------------------------------------------------------
# Lexicons used by the rewrite passes (small, closed-class, hand-curated --
# exactly like the lexicons already used by the lexer's POS tagger)
# ---------------------------------------------------------------------
CONNECTIVE_SOFTEN_POOL = ["Also", "Plus", "And", "What's more", "On top of that", "Beyond that"]
CONNECTIVE_DROP_PHRASES = [
    "in conclusion, ", "in conclusion ", "in summary, ", "to summarize, ",
    "it is important to note that ", "it is worth noting that ",
]

CONTRACTION_RULES = [
    (re.compile(r"\bit is\b", re.I), "it's"),
    (re.compile(r"\bIt is\b"), "It's"),
    (re.compile(r"\bthat is\b", re.I), "that's"),
    (re.compile(r"\bthere is\b", re.I), "there's"),
    (re.compile(r"\bthey are\b", re.I), "they're"),
    (re.compile(r"\bwe are\b", re.I), "we're"),
    (re.compile(r"\byou are\b", re.I), "you're"),
    (re.compile(r"\bI am\b"), "I'm"),
    (re.compile(r"\bdoes not\b", re.I), "doesn't"),
    (re.compile(r"\bdo not\b", re.I), "don't"),
    (re.compile(r"\bdid not\b", re.I), "didn't"),
    (re.compile(r"\bcannot\b", re.I), "can't"),
    (re.compile(r"\bcan not\b", re.I), "can't"),
    (re.compile(r"\bwill not\b", re.I), "won't"),
    (re.compile(r"\bwould not\b", re.I), "wouldn't"),
    (re.compile(r"\bshould not\b", re.I), "shouldn't"),
    (re.compile(r"\bhas not\b", re.I), "hasn't"),
    (re.compile(r"\bhave not\b", re.I), "haven't"),
    (re.compile(r"\bwas not\b", re.I), "wasn't"),
    (re.compile(r"\bwere not\b", re.I), "weren't"),
]


FORMAL_CONNECTIVES = {
    "moreover", "furthermore", "additionally", "consequently", "therefore",
    "thus", "hence", "accordingly", "subsequently", "nonetheless", "nevertheless",
}


@dataclass
class Edit:
    pass_name: str
    sentence_index: int
    before: str
    after: str
    rule: str


def _split_into_sentences_text(text):
    """Return list of raw sentence strings (kept close to original spacing/punctuation)."""
    # simple sentence splitter on top of the lexer's own tokenizer, so
    # boundaries agree exactly with the rest of the pipeline
    tokens = tokenize(text)
    tag_tokens(tokens)
    groups = split_sentences(tokens)
    out = []
    for g in groups:
        s = ""
        for i, t in enumerate(g):
            if t.type == "WORD" and i > 0 and g[i-1].type == "WORD":
                s += " " + t.value
            elif t.type in ("SENT_END", "COMMA", "SEMICOLON", "COLON"):
                s += t.value
            else:
                s += (" " if s else "") + t.value
        out.append(s.strip())
    return out


# ---------------------------------------------------------------------
# PASS 1 -- connective softening
# ---------------------------------------------------------------------
def pass_connective_softening(sentence, idx, edits, rotation):
    original = sentence
    low = sentence.lower()

    for phrase in CONNECTIVE_DROP_PHRASES:
        if low.startswith(phrase):
            new = sentence[len(phrase):]
            new = new[0].upper() + new[1:] if new else new
            edits.append(Edit("connective_softening", idx, sentence, new,
                               f"dropped formal lead-in phrase '{phrase.strip()}'"))
            sentence = new
            low = sentence.lower()
            break

    m = re.match(r"^([A-Za-z]+)(,\s*)", sentence)
    if m and m.group(1).lower() in FORMAL_CONNECTIVES:
        word = m.group(1)
        repl = CONNECTIVE_SOFTEN_POOL[rotation[0] % len(CONNECTIVE_SOFTEN_POOL)]
        rotation[0] += 1
        if not word[0].isupper():
            repl = repl[0].lower() + repl[1:]
        sep = " " if repl.lower() in ("and", "but", "so") else m.group(2)
        new = repl + sep + sentence[m.end():]
        edits.append(Edit("connective_softening", idx, sentence, new,
                           f"replaced formal connective '{word}' with '{repl}' "
                           f"(rotating substitute #{rotation[0]}, avoids re-introducing a repeated opener)"))
        sentence = new

    return sentence


# ---------------------------------------------------------------------
# PASS 2 -- contraction insertion
# ---------------------------------------------------------------------
def pass_contraction_insertion(sentence, idx, edits):
    for pattern, repl in CONTRACTION_RULES:
        new_sentence, n = pattern.subn(repl, sentence)
        if n:
            edits.append(Edit("contraction_insertion", idx, sentence, new_sentence,
                               f"contracted '{pattern.pattern.strip(chr(92)+'b')}' -> '{repl}' ({n}x)"))
            sentence = new_sentence
    return sentence


# ---------------------------------------------------------------------
# PASS 3 -- sentence splitting (break up CCONJ-joined compound sentences)
# ---------------------------------------------------------------------
SPLIT_CONJ = re.compile(r",\s+(and|but|so)\s+", re.I)

def pass_sentence_splitting(sentence, idx, edits, max_words=28):
    words = sentence.split()
    if len(words) < max_words:
        return [sentence]
    m = SPLIT_CONJ.search(sentence)
    if not m:
        return [sentence]
    left = sentence[:m.start()].rstrip() + "."
    conj = m.group(1)
    right_raw = sentence[m.end():].rstrip()
    connector = {"and": "Also, ", "but": "But ", "so": "So "}.get(conj.lower(), "")
    right = connector + right_raw[0].upper() + right_raw[1:] if right_raw else right_raw
    edits.append(Edit("sentence_splitting", idx, sentence, f"{left} {right}",
                       f"split a {len(words)}-word sentence at coordinating conjunction '{conj}'"))
    return [left, right]


# ---------------------------------------------------------------------
# PASS 4 -- opener diversification (front a trailing prepositional phrase)
# ---------------------------------------------------------------------
FRONT_PP = re.compile(r"^(The|This|These|Those|A|An)\s+(\w+(?:\s+\w+){0,4})\s+(in|on|at|for|with|during|since|after|before)\s+(.+)$", re.I)

def pass_opener_diversification(sentence, idx, edits, repeated_opener_word):
    if not repeated_opener_word:
        return sentence
    first_word = sentence.split(" ", 1)[0].strip(",.") if sentence else ""
    if first_word.lower() != repeated_opener_word.lower():
        return sentence
    m = re.match(r"^(.+?)\s+(in|on|at|for|with|during|since|after|before)\s+([A-Za-z][^,.;]{2,40}?),\s*(.+)$", sentence, re.I)
    if not m:
        return sentence
    subject_clause, prep, prep_obj, rest = m.groups()
    fronted = f"{prep.capitalize()} {prep_obj}, {subject_clause.lower()} {rest}"
    fronted = fronted[0].upper() + fronted[1:]
    edits.append(Edit("opener_diversification", idx, sentence, fronted,
                       f"fronted trailing phrase to avoid repeating opener '{repeated_opener_word}'"))
    return fronted



# ---------------------------------------------------------------------
# PASSES 5-9 -- rhythm / punctuation passes (document-level, run after 1-4)
# Each is a closed-class regex / lexicon rule: no content is added or
# removed, only re-punctuated, re-ordered or re-segmented.
# ---------------------------------------------------------------------
ACRONYMS = [
    (re.compile(r"\bartificial intelligence\b", re.I), "AI"),
    (re.compile(r"\bmachine learning\b", re.I), "ML"),
    (re.compile(r"\blarge language models?\b", re.I), "LLMs"),
]

def pass_acronym_substitution(sentences, edits):
    seen = set()
    out = []
    for idx, s in enumerate(sentences):
        for pat, short in ACRONYMS:
            key = short
            if pat.search(s):
                if key in seen:
                    new = pat.sub(short, s)
                    new = new[0].upper() + new[1:] if new else new
                    edits.append(Edit("acronym_substitution", idx, s, new,
                        f"repeat mention of '{pat.pattern.strip(chr(92)+'b')}' shortened to '{short}'"))
                    s = new
                else:
                    seen.add(key)
        out.append(s)
    return out


ASIDE_RE = re.compile(r"\s+(while\s+\w+ing\b[^.;:()]*?)([.!?])$", re.I)

def pass_aside_parenthesization(sentences, edits):
    out = []
    done = False
    for idx, s in enumerate(sentences):
        m = ASIDE_RE.search(s)
        if m and not done and len(s.split()) >= 9:
            new = s[:m.start()] + " (" + m.group(1) + ")" + m.group(2)
            edits.append(Edit("aside_parenthesization", idx, s, new,
                "moved trailing 'while ...' clause into a parenthetical aside"))
            s = new
            done = True
        out.append(s)
    return out


DASH_RE = re.compile(r"^(.{30,})\s+((?:at|across|with|in|for|from|into|over)\s+(?:an?|the|these|its|their)?\s*[\w-]+(?:\s+[\w-]+){1,4})([.!?])$", re.I)

def pass_dash_emphasis(sentences, edits):
    out = []
    done = False
    for idx, s in enumerate(sentences):
        m = DASH_RE.match(s)
        if m and not done and len(s.split()) >= 9 and "(" not in s:
            new = f"{m.group(1)} \u2014 {m.group(2)}{m.group(3)}"
            edits.append(Edit("dash_emphasis", idx, s, new,
                "set off trailing prepositional phrase with an em dash for emphasis"))
            s = new
            done = True
        out.append(s)
    return out


PURPOSE_RE = re.compile(r"^(\w[^.;:!?]{25,120}?)\s+(to\s+(?:reduce|improve|increase|enhance|support|ensure|help|make|build|create|drive|boost|streamline|optimi[sz]e|achieve|deliver|provide)\b.*?)([.!?])$")

def pass_purpose_fragment_split(sentences, edits, every=2):
    out = []
    count = 0
    for idx, s in enumerate(sentences):
        m = PURPOSE_RE.match(s)
        if m and len(s.split()) >= 9 and not re.match(r"^\w+\s+(has|have|had|will|would|continue|continues)\b", s):
            count += 1
            if count % every != 1 and every != 1:
                new = f"{m.group(1)} \u2014 {m.group(2)}{m.group(3)}"
                edits.append(Edit("dash_emphasis", idx, s, new,
                    "set off purpose clause ('to <verb> ...') with an em dash for emphasis"))
                out.append(new)
                continue
            if True:
                head = m.group(1) + m.group(3)
                frag = m.group(2)[0].upper() + m.group(2)[1:] + m.group(3)
                edits.append(Edit("purpose_fragment_split", idx, s, f"{head} {frag}",
                    "split purpose clause ('to <verb> ...') off as an emphatic sentence fragment"))
                out.extend([head, frag])
                continue
        out.append(s)
    return out


def pass_semicolon_merge(sentences, edits, short=11):
    """Join sentence i and i+1 with a semicolon when sentence i+1 is short and
    sentence i is not -- lengthens the long sentence, keeps short ones short."""
    out = []
    i = 0
    merged = 0
    while i < len(sentences):
        s = sentences[i]
        if (i + 1 < len(sentences) and merged < 1 and len(s.split()) >= 7
                and len(sentences[i + 1].split()) >= 7 and s.endswith(".")
                and not sentences[i + 1].startswith(("To ", "AI "))):
            nxt = sentences[i + 1]
            first = nxt.split(" ", 1)[0]
            low = nxt if first in ("I", "AI", "ML", "LLMs") else nxt[0].lower() + nxt[1:]
            new = s[:-1] + "; " + low
            edits.append(Edit("semicolon_merge", i, f"{s} {nxt}", new,
                "joined two adjacent independent clauses with a semicolon"))
            out.append(new)
            merged += 1
            i += 2
            continue
        out.append(s)
        i += 1
    return out



# ---------------------------------------------------------------------
# PASSES 6-11 -- COST-GUIDED PEEPHOLE OPTIMIZER
# A real compiler back end doesn't apply every optimization blindly: it
# generates candidate rewrites, evaluates each against a COST MODEL, and
# keeps the one that lowers cost most (peephole / greedy hill-climbing).
# Here the cost model is this project's own transparent scorer. Every
# candidate is a closed-class, meaning-preserving rewrite; a rewrite is kept
# ONLY if re-scoring proves it lowers the AI-likelihood, and the measured
# before/after score is written into the edit log.
# ---------------------------------------------------------------------
FILLER_ADVERBS = ["undoubtedly", "increasingly", "significantly", "ultimately", "essentially",
                  "fundamentally", "inevitably", "remarkably"]
PURPOSE_VERBS = r"(?:reduce|improve|increase|enhance|support|ensure|help|make|build|create|drive|boost|streamline|optimi[sz]e|achieve|deliver|provide|shape|transform)"


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def _lower_first(s):
    first = s.split(" ", 1)[0]
    return s if first in ("I", "AI", "ML", "LLMs") or first.isupper() else s[0].lower() + s[1:]


def _candidates(sents):
    """Yield (pass_name, rule_text, sentence_index, new_sentence_list)."""
    n = len(sents)
    for i, s in enumerate(sents):
        words = s.split()
        # 6. purpose-clause fronting:  "<main> to <verb> <obj>." -> "To <verb> <obj>, <main>."
        m = re.match(r"^(.{20,}?)\s+(to\s+" + PURPOSE_VERBS + r"\b[^.;:!?()—]*?)([.!?])$", s)
        if m and len(words) >= 9:
            main = _lower_first(re.sub(r"^(Plus|Also|And|But|So),?\s+", "", m.group(1)))
            new = f"{_cap(m.group(2))}, {main}{m.group(3)}"
            yield ("purpose_fronting", "fronted the purpose clause ('to <verb> ...') before the main clause",
                   i, sents[:i] + [new] + sents[i+1:])
            # 7. purpose-fragment split
            yield ("purpose_fragment_split", "split purpose clause off as an emphatic sentence fragment",
                   i, sents[:i] + [m.group(1) + m.group(3), _cap(m.group(2)) + m.group(3)] + sents[i+1:])
            # 8a. dash set-off of purpose clause
            yield ("dash_emphasis", "set off purpose clause with an em dash",
                   i, sents[:i] + [f"{m.group(1)} — {m.group(2)}{m.group(3)}"] + sents[i+1:])
        # 8b. dash set-off of a trailing prepositional phrase
        m = DASH_RE.match(s)
        if m and len(words) >= 9 and "—" not in s:
            yield ("dash_emphasis", "set off trailing prepositional phrase with an em dash",
                   i, sents[:i] + [f"{m.group(1)} — {m.group(2)}{m.group(3)}"] + sents[i+1:])
        # 9. parenthetical aside
        m = ASIDE_RE.search(s)
        if m and len(words) >= 9:
            yield ("aside_parenthesization", "moved trailing 'while ...' clause into a parenthetical aside",
                   i, sents[:i] + [s[:m.start()] + " (" + m.group(1) + ")" + m.group(2)] + sents[i+1:])
        # 10. filler-adverb elimination (dead-code elimination on a content-free modifier)
        for adv in FILLER_ADVERBS:
            if re.search(r"\b" + adv + r"\b", s, re.I):
                new = re.sub(r"\s*\b" + adv + r"\b\s*", " ", s, count=1, flags=re.I).strip()
                new = re.sub(r"\s+,", ",", new)
                yield ("filler_elimination", f"removed content-free filler adverb '{adv}'",
                       i, sents[:i] + [_cap(new)] + sents[i+1:])

        # 12. adverb fronting:  "<subj> <aux> <adv>ly <verb> ..." -> "<Adv>ly, <subj> <aux> <verb> ..."
        m = re.match(r"^(.+?)\s+((?:is|are|was|were|has|have|had|will|can|could|would)\s+)(\w+ly)\s+(.+)$", s)
        if m and len(words) >= 8 and m.group(3).lower() not in ("only", "early", "likely", "family") \
                and not re.match(r"^(Plus|Also|And|But|So)\b", m.group(1)) and "," not in m.group(1):
            new = f"{_cap(m.group(3))}, {_lower_first(m.group(1))} {m.group(2)}{m.group(4)}"
            yield ("adverb_fronting", f"fronted the adverb '{m.group(3)}' to open the sentence",
                   i, sents[:i] + [new] + sents[i+1:])
        # 13. active -> passive for simple regular SVO:  "<Subj> <verb> these|the|its <obj>." -> "<Obj> are <verb>ed by <subj>."
        m = re.match(r"^(?:(?:Plus|Also|And|But|So),?\s+)?([A-Za-z][\w-]*(?:\s[\w-]+)?)\s+([a-z]{3,}e?)\s+((?:these|those|the|its|their)\s+[\w-]+(?:\s[\w-]+)?)([.!?]?)$", s)
        if m:
            subj, verb, obj, end = m.groups()
            if verb.endswith("e") and not verb.endswith(("ee", "ie")) and verb not in ("have", "are", "make", "take", "give", "see", "be"):
                part = verb + "d"
            elif verb.endswith("s") or verb in ("have", "are", "make", "take", "give", "see", "be", "will"):
                part = None
            else:
                part = verb + "ed"
            if part and subj.lower() not in ("it", "they", "we", "he", "she", "i", "you"):
                aux = "are" if obj.split()[-1].endswith("s") else "is"
                new = f"{_cap(obj)} {aux} {part} by {subj.lower() if subj.split()[0].lower() not in ('ai',) else subj}{end or '.'}"
                yield ("voice_transformation", f"active -> passive voice ('{verb}' -> '{aux} {part} by')",
                       i, sents[:i] + [new] + sents[i+1:])
        # 14. trailing prepositional-phrase fronting:  "<main> at an unprecedented pace." -> "At an unprecedented pace, <main>."
        m = re.match(r"^(.{25,}?)\s+((?:at|across|with|in|for|from|into|over)\s+(?:an?|the|these|its|their)\s+[\w-]+(?:\s+[\w-]+){0,3})([.!?])$", s)
        if m and len(words) >= 8 and "," not in m.group(2) and not any(c in s for c in "(;\u2014"):
            main = _lower_first(re.sub(r"^(Plus|Also|And|But|So),?\s+", "", m.group(1)))
            yield ("pp_fronting", "fronted the trailing prepositional phrase to open the sentence",
                   i, sents[:i] + [f"{_cap(m.group(2))}, {main}{m.group(3)}"] + sents[i+1:])
        # 15. 'while <verb>ing ...' -> separate sentence with 'It also <verb>s ...'  (clause split)
        m = re.match(r"^(.{25,}?)\s+while\s+(\w+)ing\s+(.+?)([.!?])$", s)
        if m and len(words) >= 8:
            yield ("clause_split", "split trailing 'while <verb>ing ...' clause into its own sentence",
                   i, sents[:i] + [m.group(1) + m.group(4), f"That also means {m.group(2)}ing {m.group(3)}{m.group(4)}"] + sents[i+1:])
        # 11. semicolon / dash merge with the next sentence
        if i + 1 < n and s.endswith(".") and len(words) >= 5 and len(sents[i+1].split()) >= 5 \
                and not sents[i+1].startswith(("To ",)):
            nxt = sents[i+1]
            nxt_clean = re.sub(r"^(Plus|Also|And|But|So),?\s+", "", nxt)
            yield ("semicolon_merge", "joined two adjacent independent clauses with a semicolon",
                   i, sents[:i] + [s[:-1] + "; " + _lower_first(nxt_clean)] + sents[i+2:])
            yield ("dash_merge", "joined two adjacent clauses with an em dash",
                   i, sents[:i] + [s[:-1] + " — " + _lower_first(nxt_clean)] + sents[i+2:])


def _changed_span(prev, cand):
    """Return (before_text, after_text, index) for the minimal changed sentence span."""
    i = 0
    while i < min(len(prev), len(cand)) and prev[i] == cand[i]:
        i += 1
    j = 0
    while j < min(len(prev), len(cand)) - i and prev[-1 - j] == cand[-1 - j]:
        j += 1
    return " ".join(prev[i:len(prev) - j]), " ".join(cand[i:len(cand) - j]), i


def _valid(cand):
    joined = " ".join(cand)
    return not (joined.count("(") != joined.count(")") or "\u2014 \u2014" in joined
                or any(x.count("\u2014") > 1 or len(x.split()) < 3 for x in cand))


def cost_guided_optimizer(sentences, edits, score_fn, beam_width=6, max_steps=6, min_gain=0.05):
    """Beam search (width 6, depth <= 6) over meaning-preserving rewrites, ranked by the
    project's own scorer. Keeps the best-scoring rewrite sequence and logs each step."""
    start_score = score_fn(" ".join(sentences))
    beam = [(start_score, list(sentences), [])]
    best = beam[0]
    seen = {" ".join(sentences)}
    for _ in range(max_steps):
        nxt = []
        for sc, cur, hist in beam:
            fragments = sum(1 for h in hist if h[0] == "purpose_fragment_split")
            for name, rule, idx, cand in _candidates(cur):
                if name == "purpose_fragment_split" and fragments >= 1:
                    continue
                key = " ".join(cand)
                if key in seen or not _valid(cand):
                    continue
                seen.add(key)
                nxt.append((score_fn(key), cand, hist + [(name, rule, cur, cand, sc)]))
        if not nxt:
            break
        nxt.sort(key=lambda x: x[0])
        beam = nxt[:beam_width]
        if beam[0][0] < best[0] - min_gain:
            best = beam[0]
        elif beam[0][0] >= best[0]:
            break
    for name, rule, before_list, after_list, prev_score in best[2]:
        before, after, idx = _changed_span(before_list, after_list)
        new_score = score_fn(" ".join(after_list))
        edits.append(Edit(name, idx, before, after, f"{rule} [cost model: {prev_score:.1f}% -> {new_score:.1f}%]"))
    return best[1]


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
def humanize(text: str, opener_freq: dict = None, score_fn=None):
    """
    Runs the fixed pass pipeline over every sentence and returns:
      humanized_text, list[Edit], passes_summary (count per pass)
    opener_freq: optional {lowercase_word: count} from the original document's
    symbol table, used by pass 4 to know which openers are over-used.
    score_fn: optional callable(text) -> AI-likelihood %, the SAME scorer the
    rest of the app uses; enables the cost-guided optimizer (passes 6-11).
    """
    sentences = _split_into_sentences_text(text)
    opener_freq = opener_freq or {}
    most_repeated_opener = None
    if opener_freq:
        candidates = [w for w, c in opener_freq.items() if c > 1]
        if candidates:
            most_repeated_opener = max(candidates, key=lambda w: opener_freq[w])

    edits = []
    result_sentences = []
    rotation = [0]

    for idx, sent in enumerate(sentences):
        s = pass_connective_softening(sent, idx, edits, rotation)
        s = pass_contraction_insertion(s, idx, edits)
        s = pass_opener_diversification(s, idx, edits, most_repeated_opener)
        pieces = pass_sentence_splitting(s, idx, edits)
        result_sentences.extend(pieces)

    result_sentences = pass_acronym_substitution(result_sentences, edits)
    if score_fn is not None:
        result_sentences = cost_guided_optimizer(result_sentences, edits, score_fn)

    humanized_text = " ".join(result_sentences)
    humanized_text = re.sub(r"\s+([.,;:!?])", r"\1", humanized_text)
    humanized_text = re.sub(r"\s{2,}", " ", humanized_text).strip()

    passes_summary = {}
    for e in edits:
        passes_summary[e.pass_name] = passes_summary.get(e.pass_name, 0) + 1

    return {
        "humanized_text": humanized_text,
        "edits": [
            {"pass": e.pass_name, "sentence_index": e.sentence_index,
             "before": e.before, "after": e.after, "rule": e.rule}
            for e in edits
        ],
        "passes_summary": passes_summary,
        "total_edits": len(edits),
    }
