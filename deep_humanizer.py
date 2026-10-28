"""
DEEP mode = local RAG + Ollama, no API key, no internet.

 Retrieve : rag.py pulls the human-written passages most similar to the input (TF-IDF cosine).
 Generate : a local Ollama model rewrites the text, using those passages as style exemplars
            and being told which scorer features are still flagged.
 Verify   : the project's own compiler-based scorer accepts a rewrite only if the score drops.

Env: OLLAMA_URL (default http://localhost:11434), AICB_MODEL (default llama3.2).
"""
import json, os, time, urllib.request, threading
from rag import retrieve

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")


BUDGET = float(os.environ.get("AICB_BUDGET", "20"))   # max seconds of LLM time per click


def warmup():
    """Load the model into memory at app start so the first click isn't slow."""
    def _w():
        try:
            _llm("hi", max_tokens=1)
        except Exception:
            pass
    threading.Thread(target=_w, daemon=True).start()


def _llm(prompt, temperature=1.0, max_tokens=500, timeout=120):
    body = json.dumps({"model": os.environ.get("AICB_MODEL", "llama3.2"), "prompt": prompt,
                       "stream": False, "keep_alive": "30m",
                       "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 2048}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/generate", body, {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["response"].strip()
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA} ({e}). Run `ollama serve` and `ollama pull "
                           f"{os.environ.get('AICB_MODEL', 'llama3.2')}`.")


def _prompt(text, worst, examples):
    ex = "\n\n".join(f"EXAMPLE {i+1}:\n{d}" for i, (_, d) in enumerate(examples))
    return ("You rewrite text so it reads like a real person wrote it. Below are examples of genuine human "
            "writing. Copy their rhythm: uneven sentence lengths, short fragments, contractions, casual "
            "phrasing, varied punctuation.\n\n" + ex +
            "\n\nNow rewrite the TEXT below in that voice. Keep every fact and the full meaning. Do not use "
            "moreover/furthermore/additionally/in conclusion. No lists, no headings. "
            f"A detector still flags: {'; '.join(worst) or 'nothing'}. "
            "Output ONLY the rewritten text.\n\nTEXT:\n" + text)


def deep_humanize(text, score_fn, breakdown_fn, polish_fn=None, baseline=None, n=1, rounds=3, target=2.0):
    """Returns (best_text, best_score, log, retrieved)."""
    best, best_s, log = (baseline or text), score_fn(baseline or text), []
    t0 = time.time()
    cap = int(len(text.split()) * 2.2) + 40   # output length cap -> faster generation
    src = text   # LLM rewrites from this; refreshed whenever a better draft is found
    examples = retrieve(text, k=3)
    for r in range(rounds):
        worst = [f"{b['label']} ({b['score_pct']:.0f}%)"
                 for b in sorted(breakdown_fn(src), key=lambda b: -b["contribution_pct"])[:3]]
        for k in range(n):
            if time.time() - t0 > BUDGET:
                break
            cand = _llm(_prompt(src, worst, examples), max_tokens=cap,
                        timeout=max(5, BUDGET - (time.time() - t0)))   # hard stop at the time budget
            if len(cand.split()) < 0.5 * len(text.split()):   # reject truncated / meaning-losing output
                log.append({"round": r + 1, "candidate": k + 1, "score": None, "kept": False})
                continue
            if polish_fn:   # run the rule-based passes on the LLM output too (hybrid)
                polished = polish_fn(cand)
                if score_fn(polished) < score_fn(cand):
                    cand = polished
            s = score_fn(cand)
            log.append({"round": r + 1, "candidate": k + 1, "score": s, "kept": s < best_s})
            if s < best_s:
                best, best_s, src = cand, s, cand
        if best_s <= target or time.time() - t0 > BUDGET:
            break
    return best, best_s, log, examples
