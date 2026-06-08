#!/usr/bin/env python3
"""Alertmanager → GitHub repository_dispatch webhook relay."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.request import Request, urlopen


_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
_GITHUB_REPO = os.getenv("GITHUB_REPO", "")
_WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
_WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8080"))


def _dispatch_github(
    event_type: str,
    payload: dict[str, Any],
) -> tuple[int, str]:
    """Kirim ``repository_dispatch`` event ke GitHub API.

    Returns
    -------
    (status_code, response_body)
    """
    if not _GITHUB_TOKEN:
        return 401, "GITHUB_TOKEN not set"
    if not _GITHUB_REPO:
        return 400, "GITHUB_REPO not set"

    url = f"https://api.github.com/repos/{_GITHUB_REPO}/dispatches"
    body = json.dumps({
        "event_type": event_type,
        "client_payload": payload,
    }).encode("utf-8")

    req = Request(
        url,
        data=body,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {_GITHUB_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "patent-ct-webhook/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )

    try:
        with urlopen(req, timeout=15) as resp:
            status = resp.status
            response_text = resp.read().decode()
            return status, response_text
    except Exception as exc:
        return 500, str(exc)


def _map_alert_to_event(alert: dict[str, Any]) -> str:
    """Petakan label ``trigger_type`` ke GitHub event type."""
    labels = alert.get("labels", {})
    trigger = labels.get("trigger_type", "unknown")
    mapping = {
        "data_drift": "ct-data-drift-trigger",
        "data_arrival": "ct-data-arrival-trigger",
    }
    return mapping.get(trigger, "ct-trigger")


def _extract_summary(alert: dict[str, Any]) -> dict[str, Any]:
    """Ekstrak informasi penting dari alert untuk payload."""
    annotations = alert.get("annotations", {})
    labels = alert.get("labels", {})
    return {
        "alert_name": labels.get("alertname", "unknown"),
        "severity": labels.get("severity", "unknown"),
        "trigger_type": labels.get("trigger_type", "unknown"),
        "summary": annotations.get("summary", ""),
        "instance": labels.get("instance", ""),
        "fired_at": alert.get("startsAt", datetime.now(timezone.utc).isoformat()),
    }


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP handler untuk menerima webhook Alertmanager."""

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path != "/webhook":
            self._respond(404, {"error": "not found"})
            return

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid JSON"})
            return

        # Alertmanager mengirimkan {"version": "4", "alerts": [...]}
        alerts = data.get("alerts", [])
        if not alerts:
            self._respond(200, {"message": "no alerts to process"})
            return

        print(f"[{datetime.now().isoformat()}] Received {len(alerts)} alert(s)")

        results = []
        for alert in alerts:
            status = alert.get("status", "")
            if status == "resolved":
                print(f"  ⏭  Skipping resolved alert: {alert.get('labels', {}).get('alertname')}")
                continue

            event_type = _map_alert_to_event(alert)
            payload = _extract_summary(alert)
            payload["raw_alert"] = alert

            print(f"  →  Dispatching '{event_type}' to GitHub...")
            code, resp = _dispatch_github(event_type, payload)
            print(f"     GitHub API response: {code} {resp[:200]}")
            results.append({"event_type": event_type, "status": code})

        self._respond(200, {"message": "processed", "dispatches": results})

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[webhook] {format % args}")

    def _respond(self, code: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Alertmanager → GitHub repository_dispatch webhook relay"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_WEBHOOK_PORT,
        help=f"Listen port (default: {_WEBHOOK_PORT})",
    )
    parser.add_argument(
        "--host",
        default=_WEBHOOK_HOST,
        help=f"Bind address (default: {_WEBHOOK_HOST})",
    )
    args = parser.parse_args()

    if not _GITHUB_TOKEN:
        print("[warn] GITHUB_TOKEN not set — GitHub dispatch will fail", file=sys.stderr)
    if not _GITHUB_REPO:
        print("[warn] GITHUB_REPO not set — GitHub dispatch will fail", file=sys.stderr)

    server = HTTPServer((args.host, args.port), WebhookHandler)
    print(f"[webhook] Listening on {args.host}:{args.port}")
    print(f"[webhook] GitHub repo: {_GITHUB_REPO or '(not set)'}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[webhook] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
