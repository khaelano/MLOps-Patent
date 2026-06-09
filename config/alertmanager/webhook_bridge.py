#!/usr/bin/env python3
"""Alertmanager webhook bridge — receives drift alerts and triggers retraining.

Listens on ``localhost:9099/drift-alert`` for Alertmanager webhook POSTs.
When a ``PatentDrift*`` alert fires, invokes the continuous training pipeline
with ``--trigger drift``.

Usage::

    python config/alertmanager/webhook_bridge.py

Or run in the background::

    nohup python config/alertmanager/webhook_bridge.py &
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOST = os.getenv("BRIDGE_HOST", "127.0.0.1")
PORT = int(os.getenv("BRIDGE_PORT", "9099"))


class BridgeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, "Invalid JSON")
            return

        alerts = payload.get("alerts", [])
        drift_alerts = [
            a for a in alerts if a.get("labels", {}).get("alertname", "").startswith("PatentDrift")
        ]

        if not drift_alerts:
            self._respond(200, "No patent drift alerts")
            return

        names = [a["labels"]["alertname"] for a in drift_alerts]
        print(f"[webhook-bridge] Drift alerts received: {names}", flush=True)

        try:
            # Use module form — works in both Docker (pip-installed) and local (source checkout)
            subprocess.run(
                [sys.executable, "-m", "patent.cli", "continuous", "--trigger", "drift"],
                cwd=str(PROJECT_ROOT),
                check=True,
                timeout=3600,
            )
            self._respond(200, "Retraining triggered")
        except subprocess.CalledProcessError as e:
            print(f"[webhook-bridge] Retraining failed: {e}", flush=True)
            self._respond(500, f"Retraining failed: {e}")
        except subprocess.TimeoutExpired:
            print("[webhook-bridge] Retraining timed out", flush=True)
            self._respond(504, "Retraining timed out")

    def _respond(self, status, message):
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(f"{message}\n".encode("utf-8"))

    def log_message(self, format, *args):  # noqa: A002
        print(f"[webhook-bridge] {args[0]}", flush=True)


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), BridgeHandler)
    print(f"[webhook-bridge] Listening on {HOST}:{PORT}/drift-alert", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[webhook-bridge] Shutting down")
        server.shutdown()
