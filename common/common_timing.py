# common_timing.py
from datetime import datetime, timezone
import time

class Timer:
    def __enter__(self):
        self.start_iso = datetime.now(timezone.utc).isoformat()
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self._t1 = time.perf_counter()
        self.end_iso = datetime.now(timezone.utc).isoformat()
        self.elapsed = self._t1 - self._t0
        print(f"⏱ STT 시작(UTC): {self.start_iso}")
        print(f"⏱ STT 종료(UTC): {self.end_iso}")
        print(f"⏱ STT 경과: {self.elapsed:.3f}s")
