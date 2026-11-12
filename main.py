"""Probabilistic forecaster for the Roblox Up-and-Coming top 10."""
import json
import math
import os
from datetime import datetime

PLACES = 10
STEPS = 72          # hourly slots in the forecast window
RESERVE = 0.15      # mass per place left unnamed for newly charting games
MAX_ROWS = 100      # max rows per place
H_CAP = 12.0        # spread stops growing past this many hours


def parse_t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load(in_dir):
    snaps = []
    with open(os.path.join(in_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("ok") is False or not o.get("up_and_coming"):
                continue
            ranks = {g["universe_id"]: g["rank"] for g in o["up_and_coming"]}
            snaps.append((parse_t(o["at"]), ranks))
    snaps.sort(key=lambda s: s[0])
    with open(os.path.join(in_dir, "task.json")) as f:
        task = json.load(f)
    return snaps, task


def estimate(snaps):
    """Per-hour drift and diffusion scale (per sqrt-hour) of rank changes."""
    drifts, var_terms = [], []
    for (t0, a), (t1, b) in zip(snaps, snaps[1:]):
        dt = (t1 - t0).total_seconds() / 3600
        if dt <= 0:
            continue
        for uid, r0 in a.items():
            if uid in b:
                d = b[uid] - r0
                drifts.append(d / dt)
                var_terms.append((d, dt))
    mu = sum(drifts) / len(drifts) if drifts else 0.0
    # diffusion: variance of (change - drift*dt) per hour
    num = sum((d - mu * dt) ** 2 / dt for d, dt in var_terms)
    sigma = math.sqrt(num / len(var_terms)) if var_terms else 5.0
    return mu, max(sigma, 0.5)


def place_probs(r, mu, sigma, h0):
    """P(game with base rank r sits at place k), averaged over the window."""
    out = [0.0] * PLACES
    for i in range(STEPS):
        h = h0 + i
        m = r + mu * h
        s = sigma * math.sqrt(min(h, H_CAP))  # bounded chart: spread saturates
        for k in range(1, PLACES + 1):
            out[k - 1] += phi((k + 0.5 - m) / s) - phi((k - 0.5 - m) / s)
    return [x / STEPS for x in out]


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    snaps, task = load(in_dir)
    mu, sigma = estimate(snaps)

    last_t, base = snaps[-1]
    start = parse_t(task["window"]["from"])
    h0 = max((start - last_t).total_seconds() / 3600, 1.0)  # gap to window

    # probability matrix: game -> per-place probability
    table = {uid: place_probs(r, mu, sigma, h0) for uid, r in base.items()}

    rows, grid = [], {}
    for k in range(PLACES):
        col = {uid: p[k] for uid, p in table.items()}
        total = sum(col.values())
        cap = 1.0 - RESERVE
        scale = cap / total if total > cap else 1.0  # keep reserve free
        col = {u: p * scale for u, p in col.items()}
        grid[k] = col
        best = sorted(col.items(), key=lambda kv: -kv[1])[:MAX_ROWS]
        for uid, p in best:
            if p > 1e-9:
                rows.append({"universe_id": uid, "place": k + 1, "p": p})

    # top10: greedy, most likely unused game for each place
    top, used = [], set()
    for k in range(PLACES):
        for uid, _ in sorted(grid[k].items(), key=lambda kv: -kv[1]):
            if uid not in used:
                top.append(uid)
                used.add(uid)
                break

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "forecast.json"), "w") as f:
        json.dump({"probabilities": rows, "top10": top}, f)


if __name__ == "__main__":
    main()
