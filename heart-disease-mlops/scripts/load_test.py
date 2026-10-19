"""Send a stream of random (valid + some invalid) requests so the Grafana
dashboard has traffic to show. Standard library only.

    python scripts/load_test.py --url http://localhost:8080 --n 300
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request


def random_patient() -> dict:
    return {
        "age": random.randint(30, 77), "sex": random.randint(0, 1),
        "cp": random.randint(1, 4), "trestbps": random.randint(95, 190),
        "chol": random.randint(150, 400), "fbs": random.randint(0, 1),
        "restecg": random.choice([0, 2]), "thalach": random.randint(90, 200),
        "exang": random.randint(0, 1), "oldpeak": round(random.uniform(0, 4), 1),
        "slope": random.randint(1, 3), "ca": random.randint(0, 3),
        "thal": random.choice([3, 6, 7]),
    }


def post(url: str, payload: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--invalid-rate", type=float, default=0.05)
    ap.add_argument("--delay", type=float, default=0.05)
    args = ap.parse_args()

    stats: dict[int, int] = {}
    for i in range(args.n):
        payload = random_patient()
        if random.random() < args.invalid_rate:
            payload["age"] = 500  # triggers a 422 so the error panel is non-empty
        code = post(f"{args.url}/predict", payload)
        stats[code] = stats.get(code, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{args.n} sent, status counts: {stats}")
        time.sleep(args.delay)
    print("done:", stats)


if __name__ == "__main__":
    main()
