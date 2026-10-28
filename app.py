"""
app.py -- WEB APPLICATION LAYER
==================================
Thin Flask wrapper around pipeline.analyze_text(). Serves the single-page
UI and a JSON API endpoint the frontend calls asynchronously.
"""

from flask import Flask, request, jsonify, render_template
from pipeline import analyze_text, analyze_batch, humanize_and_compare

app = Flask(__name__)

SAMPLE_AI = ("Artificial intelligence has become an integral part of modern society. "
             "Moreover, it is transforming industries across the globe at an unprecedented pace. "
             "Furthermore, businesses are increasingly adopting AI-driven solutions to improve "
             "operational efficiency. Additionally, organizations leverage these tools to reduce "
             "costs while enhancing customer outcomes. In conclusion, artificial intelligence will "
             "undoubtedly continue to shape the future of global technology and innovation.")

SAMPLE_HUMAN = ("I stayed up way too late last night finishing that book. Couldn't put it down "
                "honestly, even though the ending kind of let me down. My sister called around "
                "midnight asking if I wanted to grab breakfast tomorrow, so now I'm regretting the "
                "whole thing. Coffee first. Definitely coffee first before we talk about anything else.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Please provide non-empty text."}), 400
    if len(text) > 20000:
        return jsonify({"error": "Text too long (max 20,000 characters)."}), 400
    result = analyze_text(text)
    return jsonify(result)


@app.route("/api/analyze_batch", methods=["POST"])
def api_analyze_batch():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Please provide non-empty text."}), 400
    if len(text) > 40000:
        return jsonify({"error": "Text too long (max 40,000 characters for batch mode)."}), 400
    result = analyze_batch(text)
    return jsonify(result)


@app.route("/api/humanize", methods=["POST"])
def api_humanize():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Please provide non-empty text."}), 400
    if len(text) > 20000:
        return jsonify({"error": "Text too long (max 20,000 characters)."}), 400
    result = humanize_and_compare(text, deep=data.get('deep', True))
    if "error" in result:
        return jsonify({"error": "Could not process this text."}), 400
    return jsonify(result)


@app.route("/api/sample/<kind>")
def api_sample(kind):
    return jsonify({"text": SAMPLE_AI if kind == "ai" else SAMPLE_HUMAN})


if __name__ == "__main__":
    from deep_humanizer import warmup
    warmup()
    app.run(host="0.0.0.0", port=5000, debug=True)
