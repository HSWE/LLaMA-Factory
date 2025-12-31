# session_store.py
import threading
import time

from .engine_custom import Engine


class SessionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._engines: dict[str, Engine] = {}
        self._last_seen: dict[str, float] = {}

    # ===== Engine lifecycle =====

    def get_or_create_engine(
        self,
        session_id: str,
        *,
        demo_mode: bool = False,
        pure_chat: bool = False,
    ) -> Engine:
        """Get existing Engine for session_id,or create a new one if not exists."""
        with self._lock:
            engine = self._engines.get(session_id)
            if engine is None:
                engine = Engine(
                    demo_mode=demo_mode,
                    pure_chat=pure_chat,
                )
                self._engines[session_id] = engine

            self._last_seen[session_id] = time.time()
            return engine

    def get_engine(self, session_id: str) -> Engine | None:
        with self._lock:
            engine = self._engines.get(session_id)
            if engine is not None:
                self._last_seen[session_id] = time.time()
            return engine

    # ===== Optional cleanup (之後可用) =====

    def cleanup_expired(self, ttl_seconds: int = 3600):
        """Remove engines not used for ttl_seconds."""
        now = time.time()
        with self._lock:
            expired = [sid for sid, ts in self._last_seen.items() if now - ts > ttl_seconds]
            for sid in expired:
                engine = self._engines.pop(sid, None)
                self._last_seen.pop(sid, None)
                # 如果之後 Engine 裡有 subprocess，這裡可以順便 kill


# 全域 singleton（整個 Gradio app 共用）
session_store = SessionStore()
