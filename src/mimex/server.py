import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from mimex.detector import load, predict

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mimex</title>
<style>body{font:17px system-ui;background:#131816;color:#e9eee9;max-width:800px;margin:70px auto;padding:24px}h1{font-size:48px;margin-bottom:0}p{line-height:1.6;color:#b5c4b9}textarea{box-sizing:border-box;width:100%;height:260px;background:#202a24;color:inherit;border:1px solid #53695b;border-radius:12px;padding:18px;font:inherit}button{background:#b6efba;color:#122216;border:0;border-radius:8px;padding:14px 24px;font:inherit;margin-top:16px;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:20px;background:#202a24;border-radius:12px}small{color:#b5c4b9}</style>
<h1>Mimex<span style="color:#b6efba">.</span></h1><p>A local, experimental AI text detector.</p>
<label for="text">Paste English text (at least 50 words)</label><p><textarea id="text" placeholder="Your text stays on this computer."></textarea></p>
<button id="check">Analyze text</button><p role="status" id="status"></p>
<section id="result" hidden><h2 id="verdict"></h2><p id="explanation"></p><details><summary>Model details</summary><pre id="details"></pre></details></section>
<small>Experimental detector. Scores are not proof of authorship. Current generators and mixed authorship are not validated. __MODEL_SCOPE__.</small>
<script>
const button = document.querySelector('#check');
const result = document.querySelector('#result');
const status = document.querySelector('#status');
button.onclick = async () => {
  button.disabled = true;
  result.hidden = true;
  status.textContent = 'Analyzing…';
  try {
    const response = await fetch('/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: document.querySelector('#text').value})
    });
    const data = await response.json();
    if (!response.ok) {
      status.textContent = data.error || 'Unable to analyze';
      return;
    }
    if (data.verdict === 'insufficient_text') {
      status.textContent = 'Add more text: at least 50 words are needed.';
      return;
    }
    document.querySelector('#verdict').textContent = data.verdict === 'ai_signal'
      ? 'AI-like patterns detected' : 'Below the AI detection threshold';
    document.querySelector('#explanation').textContent =
      'This experimental result cannot establish who wrote the text.' +
      (data.windows_analyzed
        ? ` Analyzed all ${data.tokens_scored} tokens in ${data.windows_analyzed} overlapping windows.`
        : (data.truncated ? ' Only the first 256 tokens were analyzed.' : '')) +
      (data.outside_evaluated_length_range ? ` This is outside the evaluated ${data.evaluated_word_range[0]}–${data.evaluated_word_range[1]}-word range.` : '');
    document.querySelector('#details').textContent = JSON.stringify(data, null, 2);
    result.hidden = false;
    status.textContent = 'Analysis complete';
  } catch (e) {
    status.textContent = 'Could not connect to the local server.';
  } finally {
    button.disabled = false;
  }
};
</script></html>"""


def serve(checkpoint, port):
    loaded = load(checkpoint)
    import html

    page = PAGE.replace("__MODEL_SCOPE__", html.escape(loaded[3]["scope"]))

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, body, content_type="application/json"):
            payload = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/":
                self.respond(200, page, "text/html")
            elif self.path == "/health":
                self.respond(
                    200,
                    json.dumps(
                        {
                            "status": "ready",
                            "scope": loaded[3]["scope"],
                            "scoring": loaded[3].get("scoring", "prefix"),
                        }
                    ),
                )
            else:
                self.respond(404, json.dumps({"error": "Not found"}))

        def do_POST(self):
            if self.path != "/predict":
                self.respond(404, json.dumps({"error": "Not found"}))
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 100_000:
                    self.respond(
                        413, json.dumps({"error": "Request must be 1–100,000 bytes"})
                    )
                    return
                if (
                    self.headers.get("Content-Type", "").split(";")[0]
                    != "application/json"
                ):
                    self.respond(415, json.dumps({"error": "Use application/json"}))
                    return
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict) or not isinstance(body.get("text"), str):
                    raise TypeError("text must be a string")
                self.respond(200, json.dumps(predict(body["text"], loaded)))
            except (ValueError, UnicodeError, TypeError) as error:
                self.respond(400, json.dumps({"error": str(error)}))

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Mimex is ready at http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
