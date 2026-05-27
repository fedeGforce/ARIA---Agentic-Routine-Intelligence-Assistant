# =============================================================
# aria/db/database.py
# =============================================================
# This module owns everything related to the database:
#   - opening and closing connections
#   - initializing the schema on first run
#   - CRUD helpers for projects, tasks, and ai_logs
#
# Design principle: no AI logic lives here. This layer only
# knows about SQL. The agent layer calls these functions and
# decides what to do with the results.
# =============================================================

import sqlite3
from pathlib import Path

# The database file lives next to this module.
# Path(__file__).parent resolves to aria/db/
DB_PATH = Path(__file__).parent / "aria.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


# -------------------------------------------------------------
# Connection management
# -------------------------------------------------------------


def get_connection() -> sqlite3.Connection:
    """
    Open a connection to the SQLite database.

    Two important settings applied on every connection:
    - foreign_keys = ON  : enforces FK constraints (SQLite disables
                           them by default for backwards compatibility)
    - row_factory        : makes rows behave like dicts, so you can
                           access columns by name: row['title']
                           instead of by index: row[2]
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row  # dict-like row access
    return conn


def init_db() -> None:
    """
    Initialize the database by running schema.sql.

    Safe to call multiple times — all CREATE statements use
    IF NOT EXISTS, so running this on an existing DB is a no-op.
    Called automatically when the CLI starts.
    """
    schema = SCHEMA_PATH.read_text()
    with get_connection() as conn:
        conn.executescript(schema)
    print(f"[DB] Database ready at {DB_PATH}")


# =============================================================
# PROJECT helpers
# =============================================================


def create_project(name: str, description: str = "", category: str = "general") -> int:
    """
    Insert a new project and return its auto-generated id.

    Args:
        name:        Short display name for the project.
        description: Optional longer description.
        category:    Free-form label — 'work', 'personal', 'learning', etc.

    Returns:
        The integer id of the newly created project.
    """
    sql = """
        INSERT INTO projects (name, description, category)
        VALUES (?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (name, description, category))
        return cursor.lastrowid


def get_project(project_id: int) -> sqlite3.Row | None:
    """
    Fetch a single project by id. Returns None if not found.
    """
    sql = "SELECT * FROM projects WHERE id = ?"
    with get_connection() as conn:
        return conn.execute(sql, (project_id,)).fetchone()


def list_projects(status: str = "active") -> list[sqlite3.Row]:
    """
    Return all projects filtered by status.
    Default shows only active projects; pass status='archived'
    to see archived ones, or status='all' to see everything.
    """
    if status == "all":
        sql = "SELECT * FROM projects ORDER BY created_at DESC"
        params = ()
    else:
        sql = "SELECT * FROM projects WHERE status = ? ORDER BY created_at DESC"
        params = (status,)

    with get_connection() as conn:
        return conn.execute(sql, params).fetchall()


def archive_project(project_id: int) -> bool:
    """
    Mark a project as archived rather than deleting it.
    Returns True if a row was updated, False if id not found.
    """
    sql = """
        UPDATE projects
        SET status = 'archived', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (project_id,))
        return cursor.rowcount > 0


# =============================================================
# TASK helpers
# =============================================================


def create_task(
    project_id: int,
    title: str,
    description: str = "",
    priority: int | None = None,  # 1=low, 2=medium, 3=high
    recurrence: str = "none",
    due_date: str | None = None,  # 'YYYY-MM-DD' string or None
) -> int:
    """
    Insert a new task and return its auto-generated id.

    priority and due_date are optional because the AI suggestion
    action can fill them in later if not provided at creation time.
    """
    sql = """
        INSERT INTO tasks (project_id, title, description, priority, recurrence, due_date)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (project_id, title, description, priority, recurrence, due_date))
        return cursor.lastrowid


def get_task(task_id: int) -> sqlite3.Row | None:
    """
    Fetch a single task by id. Returns None if not found.
    Joins project name so callers don't need a second query.
    """
    sql = """
        SELECT t.*, p.name AS project_name
        FROM tasks t
        JOIN projects p ON t.project_id = p.id
        WHERE t.id = ?
    """
    with get_connection() as conn:
        return conn.execute(sql, (task_id,)).fetchone()


def list_tasks(
    project_id: int | None = None,
    status: str | None = None,
) -> list[sqlite3.Row]:
    """
    Return tasks, optionally filtered by project and/or status.
    Results are ordered by priority (high first) then due_date (soonest first).
    Null priorities are sorted last.
    """
    conditions = []
    params = []

    if project_id is not None:
        conditions.append("t.project_id = ?")
        params.append(project_id)

    if status is not None:
        conditions.append("t.status = ?")
        params.append(status)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT t.*, p.name AS project_name
        FROM tasks t
        JOIN projects p ON t.project_id = p.id
        {where_clause}
        ORDER BY
            CASE WHEN t.priority IS NULL THEN 1 ELSE 0 END,  -- nulls last
            t.priority DESC,                                   -- 3 > 2 > 1
            t.due_date ASC                                     -- soonest first
    """
    with get_connection() as conn:
        return conn.execute(sql, params).fetchall()


def update_task_status(task_id: int, new_status: str) -> bool:
    """
    Update the lifecycle status of a task.
    Returns True if a row was updated, False if id not found.
    """
    sql = """
        UPDATE tasks
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (new_status, task_id))
        return cursor.rowcount > 0


def update_task_priority(task_id: int, priority: int) -> bool:
    """
    Update the priority of a task (typically called by the AI suggest action).
    Returns True if a row was updated, False if id not found.
    """
    sql = """
        UPDATE tasks
        SET priority = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (priority, task_id))
        return cursor.rowcount > 0


def get_overdue_tasks() -> list[sqlite3.Row]:
    """
    Return all non-done tasks whose due_date is in the past.
    Used by the AI 'warn' action to flag neglected work.

    date('now') in SQLite returns today's date in YYYY-MM-DD format,
    which compares correctly against stored due_date values.
    """
    sql = """
        SELECT t.*, p.name AS project_name
        FROM tasks t
        JOIN projects p ON t.project_id = p.id
        WHERE t.due_date < date('now')
          AND t.status NOT IN ('done', 'cancelled')
        ORDER BY t.due_date ASC
    """
    with get_connection() as conn:
        return conn.execute(sql).fetchall()


def get_tasks_for_summary(period: str) -> list[sqlite3.Row]:
    """
    Return tasks relevant to a given summary period.

    period='day'   → tasks updated today
    period='week'  → tasks updated in the last 7 days
    period='month' → tasks updated in the last 30 days

    This is used by the AI 'summarize' action to give the model
    a snapshot of what was worked on in the requested window.
    """
    period_filter = {
        "day": "date(t.updated_at) = date('now')",
        "week": "t.updated_at >= datetime('now', '-7 days')",
        "month": "t.updated_at >= datetime('now', '-30 days')",
    }

    # Fallback to 'week' if an unrecognized period is passed
    condition = period_filter.get(period, period_filter["week"])

    sql = f"""
        SELECT t.*, p.name AS project_name
        FROM tasks t
        JOIN projects p ON t.project_id = p.id
        WHERE {condition}
        ORDER BY t.updated_at DESC
    """
    with get_connection() as conn:
        return conn.execute(sql).fetchall()


# =============================================================
# AI LOG helpers
# =============================================================


def log_ai_action(
    action: str,
    prompt: str,
    response: str,
    task_id: int | None = None,
    model_used: str = "phi3.5",
) -> int:
    """
    Record an AI interaction in the audit log.
    Returns the id of the created log entry.

    This should be called after every AI action, regardless of
    whether the result was acted upon. The log is append-only —
    never update or delete entries.
    """
    sql = """
        INSERT INTO ai_logs (task_id, action, prompt, response, model_used)
        VALUES (?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (task_id, action, prompt, response, model_used))
        return cursor.lastrowid


def get_ai_logs(task_id: int | None = None, action: str | None = None) -> list[sqlite3.Row]:
    """
    Retrieve AI log entries, optionally filtered by task and/or action type.
    Useful for reviewing what the model has suggested for a specific task.
    """
    conditions = []
    params = []

    if task_id is not None:
        conditions.append("task_id = ?")
        params.append(task_id)

    if action is not None:
        conditions.append("action = ?")
        params.append(action)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT * FROM ai_logs
        {where_clause}
        ORDER BY created_at DESC
    """
    with get_connection() as conn:
        return conn.execute(sql, params).fetchall()
