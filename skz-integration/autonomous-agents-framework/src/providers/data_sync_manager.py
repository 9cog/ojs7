import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

try:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore
except Exception:
    psycopg2 = None

try:
    import redis  # type: ignore
except Exception:
    redis = None


class NoOpLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class RedisLock:
    def __init__(self, client, key: str, ttl: int = 30):
        self.client = client
        self.key = key
        self.ttl = ttl
        self._token = None

    def __enter__(self):
        if not self.client:
            return self
        try:
            self._token = os.urandom(16).hex()
            acquired = self.client.set(self.key, self._token, nx=True, ex=self.ttl)
            if not acquired:
                raise RuntimeError("lock_not_acquired")
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.client or not self._token:
            return False
        try:
            val = self.client.get(self.key)
            if val and val.decode() == self._token:
                self.client.delete(self.key)
        except Exception:
            pass
        return False


class DataSyncManager:
    def __init__(self, dsn: Optional[str] = None, redis_url: Optional[str] = None):
        self.dsn = dsn or os.getenv("POSTGRES_DSN")
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self._redis = self._init_redis()

    def _init_redis(self):
        if not self.redis_url or not redis:
            return None
        try:
            return redis.StrictRedis.from_url(self.redis_url)
        except Exception:
            return None

    @contextmanager
    def _conn(self) -> Generator[Any, None, None]:
        if not self.dsn or not psycopg2:
            yield None
            return
        conn = None
        try:
            conn = psycopg2.connect(self.dsn)
            yield conn
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def lock(self, key: str, ttl: int = 30):
        if self._redis:
            return RedisLock(self._redis, f"lock:{key}", ttl=ttl)
        return NoOpLock()

    def upsert_agent_decision(self, agent_id: str, decision: Dict[str, Any]) -> bool:
        with self.lock(f"agent_decision:{agent_id}"):
            with self._conn() as conn:
                if not conn:
                    return True
                try:
                    cur = conn.cursor()
                    cur.execute(
                        """
                        INSERT INTO agent_decisions (agent_id, decision_type, context_data, decision_result, confidence_score)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            agent_id,
                            str(decision.get("decision")),
                            json.dumps(decision.get("context") or {}),
                            json.dumps(decision),
                            float(decision.get("confidence") or 0.0),
                        ),
                    )
                    conn.commit()
                    return True
                except Exception:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    return False

    def record_performance_metric(self, agent_id: str, metric_name: str, value: float) -> bool:
        with self.lock(f"metric:{agent_id}:{metric_name}"):
            with self._conn() as conn:
                if not conn:
                    return True
                try:
                    cur = conn.cursor()
                    cur.execute(
                        """
                        INSERT INTO agent_performance (agent_id, metric_name, metric_value)
                        VALUES (%s, %s, %s)
                        """,
                        (agent_id, metric_name, float(value)),
                    )
                    conn.commit()
                    return True
                except Exception:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    return False
