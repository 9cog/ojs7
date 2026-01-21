import os

try:
    import psycopg2  # type: ignore
except Exception:
    psycopg2 = None  # type: ignore


DDL = [
    """
    CREATE TABLE IF NOT EXISTS agent_decisions (
        id SERIAL PRIMARY KEY,
        agent_id TEXT NOT NULL,
        decision_type TEXT NOT NULL,
        context_data JSONB,
        decision_result JSONB,
        confidence_score DOUBLE PRECISION,
        created_at TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_performance (
        id SERIAL PRIMARY KEY,
        agent_id TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value DOUBLE PRECISION,
        created_at TIMESTAMPTZ DEFAULT NOW()
    )
    """,
]


def run_guarded_migrations(dsn: str | None = None) -> bool:
    if psycopg2 is None:
        return False
    dsn = dsn or os.getenv("POSTGRES_DSN")
    if not dsn:
        return False
    conn = None
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        for stmt in DDL:
            cur.execute(stmt)
        conn.commit()
        return True
    except Exception:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
