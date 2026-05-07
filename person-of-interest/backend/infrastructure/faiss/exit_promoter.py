"""Background thread that promotes exit embeddings to FAISS after persons leave.

When a person leaves the camera view, the SceneScape tracker stops emitting their
track ID and the Redis gate key (detection:track:seen:{track_id}) expires.
The exit vector (detection:exit_vec:{track_id}) has a longer TTL so it survives
the gate expiry.  This promoter scans periodically, detects gate-expired tracks,
and adds their exit vector to FAISS as a second permanent embedding, giving each
person appearance both an entry and an exit embedding for better search recall.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger("poi.exit_promoter")


class ExitPromoterThread(threading.Thread):
    """Daemon thread that calls detection_index.promote_exits() on an interval."""

    def __init__(self, detection_index, interval_sec: int = 30) -> None:
        super().__init__(name="exit-promoter", daemon=True)
        self._index = detection_index
        self._interval = interval_sec
        self._stop_event = threading.Event()

    def run(self) -> None:
        log.info("ExitPromoterThread started (interval=%ds)", self._interval)
        while not self._stop_event.wait(timeout=self._interval):
            try:
                promoted = self._index.promote_exits()
                if promoted:
                    log.info("ExitPromoter: promoted %d exit embedding(s) to FAISS", promoted)
            except Exception:
                log.warning("ExitPromoter: error during promotion cycle", exc_info=True)
        log.info("ExitPromoterThread stopped")

    def stop(self) -> None:
        self._stop_event.set()
