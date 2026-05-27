"""Send the deadline-eve picks email.

Triggered 90 minutes before each MD deadline by GitHub Actions. Reads
the latest snapshot at `data/latest_recommendation.json`, renders a
mobile-friendly HTML email, and sends it via Gmail SMTP.

Environment
-----------
EMAIL_FROM      Gmail address that owns the app password
EMAIL_PASSWORD  16-char Gmail app password (NOT the account password)
EMAIL_TO        recipient address (or comma-separated list)

Optional:
EMAIL_REPLY_TO    if set, used as the Reply-To header (useful for tests)

CLI
---
    python -m scripts.email_picks                 # uses the next upcoming MD
    python -m scripts.email_picks --md 3          # explicit
    python -m scripts.email_picks --dry-run       # render to stdout only
    python -m scripts.email_picks --to test@me.com

Failure modes
-------------
* Snapshot missing → log + write to data/email_failures.log + exit 1.
* SMTP auth/connection failure → log + write to data/email_failures.log + exit 1.
* HTML render fails → caught + logged; tries a plain-text fallback.

The script is intentionally read-only with respect to FIFA — it only
sends information. End-user applies picks manually on play.fifa.com.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from config import DATA_DIR
from model.schedule import MD_DEADLINES_UTC, next_deadline

log = logging.getLogger(__name__)

LATEST_REC = DATA_DIR / "latest_recommendation.json"
EMAIL_FAIL_LOG = DATA_DIR / "email_failures.log"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

# Friendly greeting card. Override via EMAIL_GREETING env var (or leave blank
# to skip). Keep it short — the picks themselves are the main event.
DEFAULT_GREETING = (
    "Hey Dad — this is your amazing son's AI 🤖⚽ "
    "Here's exactly what to do before the deadline."
)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------
def _load_snapshot() -> dict:
    if not LATEST_REC.exists():
        raise FileNotFoundError(
            f"{LATEST_REC} missing — run scripts.refresh_pipeline first."
        )
    return json.loads(LATEST_REC.read_text())


def _pos_sort(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (POS_ORDER.get(r.get("position"), 99),
                                        -float(r.get("ev_md1", 0))))


# ---------------------------------------------------------------------------
# HTML email rendering
# ---------------------------------------------------------------------------
CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f6f7f9; margin: 0; padding: 0; color: #1a1a1a; }
.container { max-width: 620px; margin: 0 auto; padding: 16px; }
.card { background: white; border-radius: 12px; padding: 20px;
        margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
h1 { font-size: 22px; margin: 0 0 8px; color: #0e3a1f; }
h2 { font-size: 17px; margin: 0 0 12px; color: #0e3a1f; }
.hero { background: #0e3a1f; color: white; }
.hero h1 { color: white; }
.hero .num { font-size: 32px; font-weight: 700; margin: 8px 0 0; }
.hero .sub { font-size: 13px; opacity: 0.85; }
.row { display: flex; justify-content: space-between; padding: 6px 0;
       border-bottom: 1px solid #eee; font-size: 15px; }
.row:last-child { border-bottom: none; }
.row .name { font-weight: 600; }
.row .meta { color: #666; font-size: 13px; }
.pos-label { display: inline-block; width: 38px; color: #888;
             font-size: 12px; font-weight: 600; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 700; margin-left: 6px; }
.badge.C { background: #ffd700; color: #1a1a1a; }
.badge.VC { background: #c0c0c0; color: #1a1a1a; }
.bench-row { padding: 6px 0; font-size: 14px; }
.bench-row .first-sub { color: #c92a2a; font-weight: 600; font-size: 12px; }
.risks li { margin-bottom: 6px; font-size: 14px; }
.footer { text-align: center; font-size: 12px; color: #888;
          margin-top: 16px; padding: 12px; }
@media (max-width: 480px) {
    .container { padding: 8px; }
    .card { padding: 14px; }
    h1 { font-size: 18px; }
}
"""


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _format_xi(xi_rows: list[dict], formation: list[int],
               captain: str, vice: str, md_col: str) -> str:
    """Render the starting XI grouped by formation row (GK / DEF / MID / FWD)."""
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for r in xi_rows:
        by_pos.setdefault(r.get("position", "MID"), []).append(r)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda r: -float(r.get(md_col, 0)))

    sections = []
    n_gk, n_def, n_mid, n_fwd = formation
    for label, players in [("GK", by_pos["GK"][:n_gk]),
                            ("DEF", by_pos["DEF"][:n_def]),
                            ("MID", by_pos["MID"][:n_mid]),
                            ("FWD", by_pos["FWD"][:n_fwd])]:
        for p in players:
            name = _esc(p.get("name"))
            team = _esc(p.get("team"))
            price = p.get("price")
            price_str = f"${float(price):.1f}M" if price is not None else "—"
            badge = ""
            if p.get("name") == captain:
                badge = '<span class="badge C">C</span>'
            elif p.get("name") == vice:
                badge = '<span class="badge VC">VC</span>'
            sections.append(
                f'<div class="row">'
                f'<div><span class="pos-label">{label}</span>'
                f'<span class="name">{name}</span>{badge}</div>'
                f'<div class="meta">{team} · {price_str}</div>'
                f'</div>'
            )
    return "".join(sections)


def _format_bench(bench_rows: list[dict], md_col: str) -> str:
    bench_rows = sorted(bench_rows, key=lambda r: -float(r.get(md_col, 0)))
    parts = []
    for i, p in enumerate(bench_rows):
        sub_label = '<span class="first-sub">First sub if anyone DNPs</span><br>' if i == 0 else ""
        parts.append(
            f'<div class="bench-row">{sub_label}'
            f'<strong>{_esc(p.get("name"))}</strong> '
            f'<span class="meta">({_esc(p.get("team"))}, {_esc(p.get("position"))})</span>'
            f'</div>'
        )
    return "".join(parts)


def _format_chip(chip_plan: list[dict], matchday: int) -> str:
    """Render the chip recommendation for THIS matchday only."""
    for p in chip_plan or []:
        if p.get("matchday") == matchday and p.get("expected_lift", 0) > 0:
            chip = _esc(p["chip"])
            lift = float(p["expected_lift"])
            reason = _esc(p["reasoning"])
            return (
                f'<div class="card">'
                f'<h2>Chip — USE {chip} this round</h2>'
                f'<p style="margin:0 0 6px;font-size:15px;">'
                f'Projected lift: <strong>+{lift:.1f} pts</strong></p>'
                f'<p style="margin:0;color:#666;font-size:14px;">{reason}</p>'
                f'</div>'
            )
    return (
        '<div class="card">'
        '<h2>Chip — HOLD all chips this round</h2>'
        '<p style="margin:0;color:#666;font-size:14px;">'
        'No chip is expected to lift this matchday\'s score enough to '
        'beat saving it for a higher-leverage week.</p>'
        '</div>'
    )


def _format_news(news_context: dict, squad_names: set[str]) -> str:
    """Late-news section. Empty string if no relevant news."""
    if not news_context:
        return ""
    rows = []
    for player, snippets in news_context.items():
        if player not in squad_names:
            continue
        for s in snippets[:1]:  # cap one snippet per player in the email
            rows.append(f'<li><strong>{_esc(player)}:</strong> {_esc(s)}</li>')
    if not rows:
        return ""
    return (
        '<div class="card">'
        '<h2>Late news</h2>'
        f'<ul class="risks" style="padding-left:18px;margin:0;">{"".join(rows)}</ul>'
        '</div>'
    )


def render_email_html(snapshot: dict, news_context: dict | None = None,
                      greeting: str | None = None) -> str:
    matchday = snapshot["matchday"]
    stage = snapshot["stage"]
    formation = snapshot["formation"]
    captain = snapshot["captain_rec"]["primary_captain"]
    vice = snapshot["captain_rec"]["vice_captain"]
    md_col = f"ev_md{matchday}"

    # Hero numbers
    xi_rows = snapshot["xi"]
    bench_rows = snapshot["bench"]
    xi_score = sum(float(r.get(md_col, 0)) for r in xi_rows)
    capt_row = next((r for r in xi_rows if r.get("name") == captain), None)
    cb = float(capt_row.get(md_col, 0)) if capt_row else 0.0
    md_score = xi_score + cb
    point_estimate = snapshot.get("point_estimate", 0)

    advice = snapshot.get("advice") or {}
    overall_strategy = advice.get("overall_strategy", "")
    captain_reasoning = advice.get(
        "captain_reasoning",
        snapshot["captain_rec"].get("captain_reasoning", ""),
    )
    key_risks = advice.get("key_risks", [])

    # Transfers — for now we only show the chip + captain + XI; the LP
    # picks a fresh 15 each round, so explicit OUT/IN lines need
    # diffing against the previous MD's snapshot, deferred to a future
    # iteration. For now we put a placeholder if no transfer data.
    transfers_section = ""  # TODO: diff against previous MD's snapshot

    squad_names = {r.get("name") for r in snapshot.get("squad", [])}
    news_html = _format_news(news_context or {}, squad_names)

    formation_str = f"{formation[1]}-{formation[2]}-{formation[3]}"
    greeting_text = greeting if greeting is not None else os.environ.get(
        "EMAIL_GREETING", DEFAULT_GREETING
    )
    greeting_html = (
        f'<div class="card" style="background:#fff8e1;border-left:4px solid #ffd700;">'
        f'<p style="margin:0;font-size:15px;line-height:1.5;">{_esc(greeting_text)}</p>'
        f'</div>'
    ) if greeting_text else ""

    body = f"""
<div class="container">
  {greeting_html}

  <div class="card hero">
    <h1>World Cup MD{matchday} — Apply on FIFA in 90 min</h1>
    <div class="sub">Stage: {stage} · Formation: {formation_str}</div>
    <div class="num">{md_score:.0f} pts</div>
    <div class="sub">Projected this matchday · Tournament total: {point_estimate:.0f}</div>
  </div>

  <div class="card">
    <h2>Starting XI ({formation_str})</h2>
    {_format_xi(xi_rows, formation, captain, vice, md_col)}
  </div>

  <div class="card">
    <h2>Bench (sub priority)</h2>
    {_format_bench(bench_rows, md_col)}
  </div>

  <div class="card">
    <h2>Captain — {_esc(captain)} (Vice: {_esc(vice)})</h2>
    <p style="margin:0;font-size:14px;color:#444;">{_esc(captain_reasoning)}</p>
  </div>

  {_format_chip(snapshot.get("chip_plan", []), matchday)}

  {transfers_section}

  {f'<div class="card"><h2>Strategy</h2><p style="margin:0;font-size:14px;">{_esc(overall_strategy)}</p></div>' if overall_strategy else ""}

  {f'<div class="card"><h2>Top risks</h2><ul class="risks" style="padding-left:18px;margin:0;">{"".join(f"<li>{_esc(r)}</li>" for r in key_risks)}</ul></div>' if key_risks else ""}

  {news_html}

  <div class="card">
    <h2>How to apply</h2>
    <p style="margin:0;font-size:14px;">Open the FIFA app or
    <a href="https://play.fifa.com">play.fifa.com</a>, tap each player to set
    the XI, mark your captain, save. Should take about 2 minutes.</p>
  </div>

  <div class="footer">
    Generated {snapshot.get("generated_at", "—")} ·
    World Cup Fantasy Advisor
  </div>
</div>
"""
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WC MD{matchday}</title>
<style>{CSS}</style></head><body>{body}</body></html>"""


def render_email_text(snapshot: dict) -> str:
    """Plain-text fallback for clients that don't render HTML."""
    matchday = snapshot["matchday"]
    md_col = f"ev_md{matchday}"
    xi_rows = sorted(snapshot["xi"],
                     key=lambda r: (POS_ORDER.get(r.get("position"), 99),
                                     -float(r.get(md_col, 0))))
    bench_rows = sorted(snapshot["bench"], key=lambda r: -float(r.get(md_col, 0)))
    captain = snapshot["captain_rec"]["primary_captain"]
    vice = snapshot["captain_rec"]["vice_captain"]

    lines = [f"World Cup MD{matchday} picks", ""]
    lines.append("Starting XI:")
    for r in xi_rows:
        tag = " (C)" if r["name"] == captain else " (VC)" if r["name"] == vice else ""
        lines.append(f"  {r.get('position')} {r['name']} ({r['team']}, ${float(r.get('price') or 0):.1f}M){tag}")
    lines.append("")
    lines.append("Bench:")
    for i, r in enumerate(bench_rows):
        prefix = "  [First sub] " if i == 0 else "  "
        lines.append(f"{prefix}{r['name']} ({r['team']}, {r.get('position')})")
    lines.append("")
    lines.append(f"Captain: {captain} (Vice: {vice})")
    lines.append("")
    lines.append("Open the FIFA app or play.fifa.com, set the XI and captain, save.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SMTP send
# ---------------------------------------------------------------------------
def send_email(html_body: str, text_body: str, subject: str,
               recipients: list[str]) -> None:
    sender = os.environ.get("EMAIL_FROM")
    password = os.environ.get("EMAIL_PASSWORD")
    if not sender or not password:
        raise RuntimeError(
            "EMAIL_FROM / EMAIL_PASSWORD env vars not set. "
            "Use a Gmail app password (not your account password)."
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    reply_to = os.environ.get("EMAIL_REPLY_TO")
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
        s.starttls()
        s.login(sender, password)
        s.sendmail(sender, recipients, msg.as_string())


# ---------------------------------------------------------------------------
# Failure log
# ---------------------------------------------------------------------------
def _log_failure(reason: str) -> None:
    EMAIL_FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EMAIL_FAIL_LOG.open("a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} {reason}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=int, default=None,
                        help="matchday (default: next upcoming)")
    parser.add_argument("--to", default=None,
                        help="override EMAIL_TO with this comma-separated list")
    parser.add_argument("--dry-run", action="store_true",
                        help="render HTML to stdout, don't send")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        snapshot = _load_snapshot()
    except Exception as exc:  # noqa: BLE001
        log.exception("could not load snapshot")
        _log_failure(f"snapshot load failed: {exc}")
        return 1

    if args.md is not None:
        # Verify the snapshot matches the requested MD; otherwise warn.
        if int(snapshot.get("matchday", -1)) != args.md:
            log.warning(
                "snapshot is MD%s but --md=%s requested. Send anyway.",
                snapshot.get("matchday"), args.md,
            )

    # News context (optional)
    news_context = {}
    news_path = DATA_DIR / "news_context.json"
    if news_path.exists():
        try:
            news_context = json.loads(news_path.read_text())
        except Exception:  # noqa: BLE001
            log.warning("could not parse news_context.json; skipping news section")

    try:
        html_body = render_email_html(snapshot, news_context)
    except Exception as exc:  # noqa: BLE001
        log.exception("html render failed; using plaintext only")
        html_body = "<html><body><pre>" + html.escape(render_email_text(snapshot)) + "</pre></body></html>"
    text_body = render_email_text(snapshot)

    matchday = snapshot["matchday"]
    subject = f"World Cup MD{matchday} — Apply these picks on FIFA in 90 min"

    if args.dry_run:
        print(html_body)
        return 0

    to_raw = args.to or os.environ.get("EMAIL_TO")
    if not to_raw:
        log.error("EMAIL_TO not set and --to not provided")
        _log_failure("no recipient configured")
        return 1
    recipients = [s.strip() for s in to_raw.split(",") if s.strip()]

    try:
        send_email(html_body, text_body, subject, recipients)
        log.info("sent MD%d email to %s", matchday, ", ".join(recipients))
        return 0
    except Exception as exc:  # noqa: BLE001
        log.exception("send failed")
        _log_failure(f"smtp send failed for MD{matchday}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
