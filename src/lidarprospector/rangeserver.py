"""A local HTTP file server with Range support, a shared bandwidth cap and per-request latency, so
streaming strategies can be measured honestly (real range requests, real byte counts) on a simulated
slow connection. Bandwidth is shared by all connections: parallel requests hide latency but cannot add
bandwidth, exactly like a real home line."""

import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Meter:
    def __init__(self, bytes_per_s: float | None, latency_s: float):
        self.rate, self.latency = bytes_per_s, latency_s
        self._lock = threading.Lock()
        self._free_at = 0.0
        self.bytes = 0
        self.requests = 0

    def reset(self) -> None:
        with self._lock:
            self.bytes = self.requests = 0

    def request(self) -> None:
        with self._lock:
            self.requests += 1
        time.sleep(self.latency)

    def send(self, n: int) -> None:
        """Account for n bytes and wait until the shared link has carried them."""
        with self._lock:
            self.bytes += n
            if not self.rate:
                return
            now = time.monotonic()
            start = max(now, self._free_at)
            self._free_at = start + n / self.rate
            wait = start - now
        time.sleep(wait)


def serve(
    path: Path, bytes_per_s: float | None = None, latency_s: float = 0.0
) -> tuple[ThreadingHTTPServer, Meter, str]:
    """Serve `path` at any URL path on localhost. Returns (server, meter, base url)."""
    meter = Meter(bytes_per_s, latency_s)
    size = path.stat().st_size

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _head(self, status, length, extra=None):
            self.send_response(status)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "image/tiff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def do_HEAD(self):
            meter.request()
            self._head(200, size)

        def do_GET(self):
            meter.request()
            m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
            lo, hi = (int(m.group(1)), int(m.group(2)) if m.group(2) else size - 1) if m else (0, size - 1)
            hi = min(hi, size - 1)
            extra = {"Content-Range": f"bytes {lo}-{hi}/{size}"} if m else None
            try:
                self._head(206 if m else 200, hi - lo + 1, extra)
            except (BrokenPipeError, ConnectionResetError):
                return  # the client gave up (a cancelled read); nothing to send
            with open(path, "rb") as f:
                f.seek(lo)
                left = hi - lo + 1
                while left > 0:
                    chunk = f.read(min(32768, left))
                    meter.send(len(chunk))
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, meter, f"http://127.0.0.1:{server.server_address[1]}"
