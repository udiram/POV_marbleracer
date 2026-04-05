from __future__ import annotations

import argparse
import html
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from marbleracer.training_monitor import build_dashboard_summary, build_generation_index


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Marbleracer Training Monitor</title>
  <style>
    :root {
      --bg: #0b1117;
      --panel: #111a23;
      --panel-alt: #162231;
      --border: #284056;
      --text: #e7eef6;
      --muted: #92a6ba;
      --accent: #4cc9f0;
      --good: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #112132 0%, var(--bg) 42%);
      color: var(--text);
    }
    .shell {
      max-width: 1480px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 20px;
    }
    .hero h1 { margin: 0; font-size: 28px; }
    .hero p { margin: 6px 0 0; color: var(--muted); }
    .badge {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 13px;
      background: rgba(17, 26, 35, 0.75);
    }
    .grid {
      display: grid;
      gap: 16px;
    }
    .cards {
      grid-template-columns: repeat(6, minmax(0, 1fr));
      margin-bottom: 16px;
    }
    .two-col {
      grid-template-columns: 1.3fr 1fr;
      margin-bottom: 16px;
    }
    .panel {
      background: linear-gradient(180deg, rgba(22,34,49,0.92), rgba(17,26,35,0.92));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      overflow: hidden;
      box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }
    .panel h2, .panel h3 { margin: 0 0 12px; font-size: 16px; }
    .metric-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-value { font-size: 28px; margin-top: 6px; }
    .status-active { color: var(--good); }
    .status-idle { color: var(--warn); }
    .plot-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .plot-grid img, .hero-frame {
      width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.08);
      background: #0b1117;
    }
    video {
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.08);
      background: #05080c;
    }
    .chart svg { width: 100%; height: 220px; display: block; }
    .chart-legend {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .chart-legend span::before {
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 8px;
      vertical-align: middle;
    }
    .legend-progress::before { background: #4cc9f0; }
    .legend-reward::before { background: #22c55e; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      padding: 10px 8px;
      border-top: 1px solid rgba(255,255,255,0.06);
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 600; }
    tbody tr { cursor: pointer; }
    tbody tr:hover { background: rgba(255,255,255,0.03); }
    tbody tr.selected { background: rgba(76,201,240,0.08); }
    .pill {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid rgba(255,255,255,0.08);
      margin-right: 6px;
    }
    .pill.latest { color: var(--accent); }
    .pill.best { color: var(--good); }
    .muted { color: var(--muted); }
    .links a {
      color: var(--accent);
      text-decoration: none;
      margin-right: 12px;
      font-size: 13px;
    }
    .empty {
      padding: 32px;
      text-align: center;
      color: var(--muted);
    }
    @media (max-width: 1180px) {
      .cards { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .two-col { grid-template-columns: 1fr; }
      .plot-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .hero { flex-direction: column; align-items: start; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div>
        <h1>Marbleracer Training Monitor</h1>
        <p>Live metrics, replay media, and generation history from the current training run.</p>
      </div>
      <div class="badge" id="status-badge">Loading...</div>
    </div>

    <div class="grid cards" id="cards"></div>

    <div class="grid two-col">
      <section class="panel">
        <h2>Latest Replay</h2>
        <video id="latest-video" controls preload="metadata"></video>
        <div class="links" style="margin-top:12px">
          <a id="video-link" href="#" target="_blank" rel="noopener noreferrer">Open video</a>
          <a id="frame-link" href="#" target="_blank" rel="noopener noreferrer">Open frame</a>
        </div>
      </section>
      <section class="panel">
        <h2>Selected Generation</h2>
        <div id="selected-generation" class="empty">No generation selected yet.</div>
      </section>
    </div>

    <div class="grid two-col">
      <section class="panel">
        <h2>Training Curves</h2>
        <div class="chart" id="curve-chart"></div>
        <div class="chart-legend">
          <span class="legend-progress">Max progress</span>
          <span class="legend-reward">Reward</span>
        </div>
      </section>
      <section class="panel">
        <h2>Generated Plots</h2>
        <div class="plot-grid" id="plot-grid"></div>
      </section>
    </div>

    <section class="panel">
      <h2>Checkpoint Browser</h2>
      <div id="generation-table-wrap" class="empty">Loading generations…</div>
    </section>
  </div>

  <script>
    const state = {
      summary: null,
      generations: [],
      selectedCheckpoint: null,
      assetVersion: null,
    };

    function fmtNumber(value, digits = 2) {
      return value == null ? "--" : Number(value).toFixed(digits);
    }

    function fmtDuration(value) {
      if (value == null) return "--";
      const total = Math.max(0, Math.round(Number(value)));
      const hours = Math.floor(total / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const seconds = total % 60;
      if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
      if (minutes > 0) return `${minutes}m ${seconds}s`;
      return `${seconds}s`;
    }

    function fmtBool(value) {
      return value ? "yes" : "no";
    }

    function withBust(path, token) {
      return `${path}?v=${encodeURIComponent(token ?? "static")}`;
    }

    function card(label, value, extra = "") {
      return `
        <section class="panel">
          <div class="metric-label">${label}</div>
          <div class="metric-value">${value}</div>
          ${extra ? `<div class="muted" style="margin-top:8px">${extra}</div>` : ""}
        </section>
      `;
    }

    function renderCards() {
      const summary = state.summary;
      const status = summary && summary.trainer_active ? '<span class="status-active">active</span>' : '<span class="status-idle">idle</span>';
      document.getElementById("cards").innerHTML = [
        card("Trainer", status, summary?.device ? `device ${summary.device}` : ""),
        card("Latest Gen", summary?.latest_generation ?? "--", `timesteps ${summary?.latest_timesteps ?? "--"}`),
        card("Training Gen", summary?.in_progress_generation ?? "--", summary?.current_timesteps == null ? "" : `live timesteps ${summary.current_timesteps}`),
        card("Next Checkpoint", summary?.next_checkpoint_timesteps ?? "--", summary?.timesteps_to_next_checkpoint == null ? "" : `${summary.timesteps_to_next_checkpoint} to go`),
        card("ETA", fmtDuration(summary?.eta_seconds_to_next_checkpoint), summary?.current_fps == null ? "" : `at ${summary.current_fps} fps`),
        card("Latest Progress", summary?.latest_max_progress == null ? "--" : `${fmtNumber(summary.latest_max_progress)} m`, summary?.latest_termination_reason ?? ""),
        card("Latest Reward", summary?.latest_reward == null ? "--" : fmtNumber(summary.latest_reward), summary?.latest_finish_time == null ? "" : `finish ${fmtNumber(summary.latest_finish_time)}s`),
        card("Best Progress", summary?.best_progress_value == null ? "--" : `${fmtNumber(summary.best_progress_value)} m`, summary?.best_progress_generation == null ? "" : `gen ${summary.best_progress_generation}`),
        card("Checkpoints", summary?.checkpoint_count ?? "--", summary?.workers == null ? "" : `${summary.workers} workers`)
      ].join("");
      document.getElementById("status-badge").innerHTML = summary && summary.trainer_active
        ? `Trainer active • gen ${summary?.in_progress_generation ?? "--"} in progress • ETA ${fmtDuration(summary?.eta_seconds_to_next_checkpoint)}`
        : `Trainer idle • latest gen ${summary?.latest_generation ?? "--"}`;
    }

    function renderLatestReplay() {
      const summary = state.summary;
      if (!summary) return;
      const videoPath = "/artifacts/run/replay_3d_latest/training_ghosts_latest_active.mp4";
      const framePath = "/artifacts/run/replay_3d_latest/training_ghosts_latest_active.png";
      const video = document.getElementById("latest-video");
      const version = summary.asset_version || "static";
      const videoUrl = withBust(videoPath, version);
      if (video.dataset.src !== videoUrl) {
        video.dataset.src = videoUrl;
        video.src = videoUrl;
      }
      document.getElementById("video-link").href = videoUrl;
      document.getElementById("frame-link").href = withBust(framePath, version);
    }

    function renderPlots() {
      const version = state.summary?.asset_version || "static";
      const plots = [
        ["checkpoint_dashboard.png", "Checkpoint dashboard"],
        ["reward_curve.png", "Reward curve"],
        ["max_progress_curve.png", "Progress curve"],
        ["section_success_curve.png", "Section reach"],
        ["failure_histogram.png", "Termination histogram"],
      ];
      document.getElementById("plot-grid").innerHTML = plots.map(([file, label]) => `
        <a href="${withBust(`/artifacts/run/plots/${file}`, version)}" target="_blank" rel="noopener noreferrer" title="${label}">
          <img src="${withBust(`/artifacts/run/plots/${file}`, version)}" alt="${label}">
        </a>
      `).join("");
    }

    function renderCurveChart() {
      const rows = state.generations;
      if (!rows.length) {
        document.getElementById("curve-chart").innerHTML = '<div class="empty">No generation data yet.</div>';
        return;
      }
      const width = 760;
      const height = 220;
      const pad = 24;
      const maxX = Math.max(...rows.map((row) => row.timesteps));
      const minX = Math.min(...rows.map((row) => row.timesteps));
      const maxProgress = Math.max(...rows.map((row) => row.max_progress));
      const minProgress = Math.min(...rows.map((row) => row.max_progress));
      const maxReward = Math.max(...rows.map((row) => row.reward));
      const minReward = Math.min(...rows.map((row) => row.reward));
      const xScale = (value) => pad + ((value - minX) / Math.max(1, maxX - minX)) * (width - pad * 2);
      const yScale = (value, minY, maxY) => height - pad - ((value - minY) / Math.max(1e-6, maxY - minY)) * (height - pad * 2);
      const pathFor = (items, yGetter, minY, maxY) => items.map((row, index) => `${index === 0 ? "M" : "L"} ${xScale(row.timesteps).toFixed(2)} ${yScale(yGetter(row), minY, maxY).toFixed(2)}`).join(" ");
      const progressPath = pathFor(rows, (row) => row.max_progress, minProgress, maxProgress);
      const rewardPath = pathFor(rows, (row) => row.reward, minReward, maxReward);
      const selected = rows.find((row) => row.checkpoint_path === state.selectedCheckpoint) || rows[rows.length - 1];
      const selectedX = xScale(selected.timesteps);
      document.getElementById("curve-chart").innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="training curves">
          <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
          <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="rgba(255,255,255,0.15)"/>
          <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="rgba(255,255,255,0.15)"/>
          <path d="${progressPath}" fill="none" stroke="#4cc9f0" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>
          <path d="${rewardPath}" fill="none" stroke="#22c55e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>
          <line x1="${selectedX}" y1="${pad}" x2="${selectedX}" y2="${height - pad}" stroke="rgba(255,255,255,0.28)" stroke-dasharray="6 6"></line>
        </svg>
      `;
    }

    function renderSelectedGeneration() {
      const row = state.generations.find((entry) => entry.checkpoint_path === state.selectedCheckpoint) || state.generations[state.generations.length - 1];
      const container = document.getElementById("selected-generation");
      if (!row) {
        container.className = "empty";
        container.textContent = "No generation selected yet.";
        return;
      }
      container.className = "";
      const pills = [];
      if (row.is_latest) pills.push('<span class="pill latest">latest</span>');
      if (row.is_best_progress) pills.push('<span class="pill best">best progress</span>');
      if (row.is_best_reward) pills.push('<span class="pill best">best reward</span>');
      const checkpointHref = `/artifacts/report/${row.checkpoint_path.replace(state.summary.report_dir + "/", "").replace(/^\\/+/, "")}`;
      container.innerHTML = `
        <div style="margin-bottom:10px">${pills.join("")}</div>
        <div class="metric-label">Generation</div>
        <div class="metric-value" style="font-size:22px">${row.generation}</div>
        <div class="muted" style="margin-top:8px">timesteps ${row.timesteps}</div>
        <div style="margin-top:16px; line-height:1.7">
          <div><strong>Progress:</strong> ${fmtNumber(row.max_progress)} m</div>
          <div><strong>Reward:</strong> ${fmtNumber(row.reward)}</div>
          <div><strong>Termination:</strong> ${row.termination_reason}</div>
          <div><strong>Steps:</strong> ${row.steps}</div>
          <div><strong>Resets:</strong> ${row.resets}</div>
          <div><strong>Finished:</strong> ${fmtBool(row.finished)}</div>
          <div><strong>Finish time:</strong> ${row.finish_time == null ? "--" : `${fmtNumber(row.finish_time)} s`}</div>
        </div>
        <div class="links" style="margin-top:16px">
          <a href="${checkpointHref}" target="_blank" rel="noopener noreferrer">Open checkpoint</a>
        </div>
      `;
    }

    function renderGenerationTable() {
      const rows = state.generations;
      const wrap = document.getElementById("generation-table-wrap");
      if (!rows.length) {
        wrap.className = "empty";
        wrap.textContent = "No checkpoint data yet.";
        return;
      }
      wrap.className = "";
      wrap.innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Gen</th>
              <th>Timesteps</th>
              <th>Progress</th>
              <th>Reward</th>
              <th>Termination</th>
              <th>Flags</th>
            </tr>
          </thead>
          <tbody>
            ${rows.slice().reverse().map((row) => `
              <tr data-checkpoint="${row.checkpoint_path}" class="${row.checkpoint_path === state.selectedCheckpoint ? "selected" : ""}">
                <td>${row.generation}</td>
                <td>${row.timesteps}</td>
                <td>${fmtNumber(row.max_progress)} m</td>
                <td>${fmtNumber(row.reward)}</td>
                <td>${row.termination_reason}</td>
                <td>
                  ${row.is_latest ? '<span class="pill latest">latest</span>' : ""}
                  ${row.is_best_progress ? '<span class="pill best">best progress</span>' : ""}
                  ${row.is_best_reward ? '<span class="pill best">best reward</span>' : ""}
                </td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
      wrap.querySelectorAll("tbody tr").forEach((row) => {
        row.addEventListener("click", () => {
          state.selectedCheckpoint = row.dataset.checkpoint;
          renderSelectedGeneration();
          renderGenerationTable();
          renderCurveChart();
        });
      });
    }

    async function refresh() {
      const [summaryResp, generationsResp] = await Promise.all([
        fetch(`/api/summary?ts=${Date.now()}`),
        fetch(`/api/generations?ts=${Date.now()}`)
      ]);
      state.summary = await summaryResp.json();
      const generationPayload = await generationsResp.json();
      state.generations = generationPayload.generations || [];
      if (!state.selectedCheckpoint && state.generations.length) {
        const latest = state.generations.find((row) => row.is_latest) || state.generations[state.generations.length - 1];
        state.selectedCheckpoint = latest.checkpoint_path;
      }
      renderCards();
      renderLatestReplay();
      renderPlots();
      renderCurveChart();
      renderSelectedGeneration();
      renderGenerationTable();
    }

    refresh().catch((error) => {
      document.body.innerHTML = `<pre style="padding:24px;color:#ef4444">${error}</pre>`;
    });
    setInterval(() => refresh().catch(() => {}), 5000);
  </script>
</body>
</html>
"""


def _safe_join(root: Path, rel_path: str) -> Path | None:
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


class TrainingMonitorHandler(BaseHTTPRequestHandler):
    run_dir: Path
    report_dir: Path

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_body: str) -> None:
        body = html_body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        mime_type, _ = mimetypes.guess_type(str(file_path))
        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/":
            self._send_html(HTML_PAGE)
            return
        if route == "/api/summary":
            self._send_json(build_dashboard_summary(self.run_dir, self.report_dir))
            return
        if route == "/api/generations":
            self._send_json(build_generation_index(self.report_dir))
            return
        if route.startswith("/artifacts/run/"):
            rel_path = route.removeprefix("/artifacts/run/")
            target = _safe_join(self.run_dir, rel_path)
            if target is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid run artifact path")
                return
            self._send_file(target)
            return
        if route.startswith("/artifacts/report/"):
            rel_path = route.removeprefix("/artifacts/report/")
            target = _safe_join(self.report_dir, rel_path)
            if target is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid report artifact path")
                return
            self._send_file(target)
            return
        self.send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {html.escape(route)}")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def make_handler(run_dir: Path, report_dir: Path):
    class BoundTrainingMonitorHandler(TrainingMonitorHandler):
        pass

    BoundTrainingMonitorHandler.run_dir = run_dir
    BoundTrainingMonitorHandler.report_dir = report_dir
    return BoundTrainingMonitorHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local training monitor UI for marbleracer training artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Snapshot directory containing plots, summary, and replay_3d_latest.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Report directory containing checkpoint_metrics.csv and checkpoints.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    report_dir = (args.report_dir if args.report_dir is not None else run_dir.parent / "report").resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir does not exist: {run_dir}")
    if not report_dir.exists():
        raise FileNotFoundError(f"Report dir does not exist: {report_dir}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(run_dir, report_dir))
    print(f"Training monitor available at http://{args.host}:{args.port}")
    print(f"Run dir: {run_dir}")
    print(f"Report dir: {report_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
