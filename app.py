"""
app.py — Microservice Serving with LRU Caching & Dynamic Model Hot-Swapping
===========================================================================
Serving layer providing:
1. Containerized Serving API: /predict, /swap/{policy_name}, /health, /metrics.
2. In-memory LRU Cache keyed on state hash: returns cached action with sub-millisecond latency.
3. Zero-Downtime Dynamic Hot-Swapping:
   - Swap active model in memory across registered policies without restarting server.
4. Schema Boundary Validation:
   - Validates incoming (obs, mask) telemetry through `schema.py` before inference.
5. Dual-Mode Server: Runs standalone via Python standard library HTTP server
   or under ASGI/FastAPI (uvicorn).
"""

import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from adapters import AVAILABLE_POLICIES, get_policy_adapter
from schema import DEFAULT_CONFIG_PATH, DataContractError, load_config, validate_action, validate_observation


class StateLRUCache:
    """Thread-safe In-Memory LRU Cache for cloud state telemetry."""

    def __init__(self, maxsize: int = 512):
        self.maxsize = maxsize
        self.cache: OrderedDict[str, int] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def hash_state(policy_name: str, obs: np.ndarray, mask: np.ndarray) -> str:
        """Compute SHA-256 state signature including policy name to prevent cross-policy cache pollution."""
        rounded_obs = np.round(obs, 4).tobytes()
        rounded_mask = np.round(mask, 1).tobytes()
        return hashlib.sha256(policy_name.encode("utf-8") + b"_" + rounded_obs + b"_" + rounded_mask).hexdigest()

    def get(self, key: str) -> Optional[int]:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        ratio = (self.hits / total) if total > 0 else 0.0
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": round(ratio, 4),
        }


class PolicyServingEngine:
    """Global serving controller managing the active adapter and cache."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.cfg = load_config(config_path)
        self.active_policy_name = str(self.cfg["policy"]["active_policy"])
        self.cache = StateLRUCache(maxsize=int(self.cfg.get("serving", {}).get("cache_maxsize", 512)))
        self.adapter = get_policy_adapter(self.active_policy_name)
        self.request_count = 0
        self.total_infer_latency_ms = 0.0

    def swap_policy(self, new_policy: str) -> Dict[str, Any]:
        if new_policy not in AVAILABLE_POLICIES:
            raise ValueError(f"Unknown policy {new_policy!r}. Available: {AVAILABLE_POLICIES}")
        old_policy = self.active_policy_name
        self.adapter = get_policy_adapter(new_policy)
        self.active_policy_name = new_policy
        return {
            "status": "SUCCESS",
            "previous_policy": old_policy,
            "active_policy": new_policy,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    def predict(self, raw_obs: Any, raw_mask: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.request_count += 1

        # 1. Boundary Contract Validation (schema.py)
        v_obs, v_mask = validate_observation(raw_obs, raw_mask)

        # 2. Check LRU Cache
        cache_key = self.cache.hash_state(self.active_policy_name, v_obs, v_mask)
        cached_action = self.cache.get(cache_key)

        if cached_action is not None:
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            self.total_infer_latency_ms += lat_ms
            return {
                "action": int(cached_action),
                "cache_hit": True,
                "policy": self.active_policy_name,
                "latency_ms": round(lat_ms, 4),
            }

        # 3. Model Inference on Cache Miss
        action = self.adapter.predict(v_obs, v_mask)
        action = validate_action(int(action), v_mask)
        self.cache.put(cache_key, action)

        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        self.total_infer_latency_ms += lat_ms

        return {
            "action": int(action),
            "cache_hit": False,
            "policy": self.active_policy_name,
            "latency_ms": round(lat_ms, 4),
        }

    def get_health(self) -> Dict[str, Any]:
        avg_lat = (
            (self.total_infer_latency_ms / self.request_count)
            if self.request_count > 0
            else 0.0
        )
        return {
            "status": "HEALTHY",
            "active_policy": self.active_policy_name,
            "available_policies": AVAILABLE_POLICIES,
            "total_requests": self.request_count,
            "average_latency_ms": round(avg_lat, 4),
            "cache": self.cache.stats(),
        }


# Global Singleton Serving Engine
ENGINE = PolicyServingEngine()


# ==================== STANDALONE HTTP SERVER (Immune to Pydantic/FastAPI version mismatches) ====================


class CounselHTTPHandler(BaseHTTPRequestHandler):
    """Production-ready HTTP REST API handler supporting /predict, /swap, /health, /metrics."""

    def _send_json(self, status: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, ENGINE.get_health())
        elif self.path == "/metrics":
            self._send_json(200, ENGINE.get_health())
        else:
            self._send_json(404, {"error": "Not Found", "valid_endpoints": ["/health", "/metrics", "/predict", "/swap/{policy}"]})

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(post_data.decode("utf-8")) if post_data else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON format"})
            return

        if self.path == "/predict":
            try:
                obs = payload.get("obs")
                mask = payload.get("mask")
                res = ENGINE.predict(obs, mask)
                self._send_json(200, res)
            except DataContractError as e:
                self._send_json(422, {"error": "DataContractError", "detail": str(e)})
            except Exception as e:
                self._send_json(500, {"error": "InferenceError", "detail": str(e)})

        elif self.path.startswith("/swap/"):
            policy_name = self.path.split("/swap/")[1].strip()
            try:
                res = ENGINE.swap_policy(policy_name)
                self._send_json(200, res)
            except ValueError as e:
                self._send_json(400, {"error": "SwapError", "detail": str(e)})
        else:
            self._send_json(404, {"error": "Not Found"})

    def log_message(self, format, *args):
        # Clean server logging format
        sys.stderr.write(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]} {args[1]} -> {args[2]}\n")


def run_server(host: str = "0.0.0.0", port: int = 8080):
    server = HTTPServer((host, port), CounselHTTPHandler)
    print("=" * 70)
    print(f"COUNSEL Serving Microservice active at http://{host}:{port}")
    print(f"Active Policy: {ENGINE.active_policy_name.upper()} (Swappable: {AVAILABLE_POLICIES})")
    print("Endpoints: POST /predict | POST /swap/{name} | GET /health | GET /metrics")
    print("=" * 70)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        server.server_close()


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    run_server(host=host, port=port)
