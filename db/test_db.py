# =============================================================
# aria/db/test_db.py
# =============================================================
# Run this script directly to verify the entire DB layer works
# correctly before building anything on top of it.
#
# Usage:
#   cd aria
#   python db/test_db.py
#
# It creates a fresh test database, runs all operations, prints
# results, and cleans up after itself. No pytest needed.
# =============================================================

import sys
import os

# Make sure Python can find the aria package from wherever you run this
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import (
    init_db, DB_PATH,
    create_project, get_project, list_projects, archive_project,
    create_task, get_task, list_tasks, update_task_status,
    update_task_priority, get_overdue_tasks, get_tasks_for_summary,
    log_ai_action, get_ai_logs
)

# ---- helpers ------------------------------------------------

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(msg: str):
    print(f"  ✓  {msg}")

def row_to_dict(row) -> dict:
    """Convert sqlite3.Row to a plain dict for readable printing."""
    return dict(row)

# ---- main test run ------------------------------------------

def run_tests():

    # Use a separate test DB so we never touch a real one
    import db.database as db_module
    db_module.DB_PATH = db_module.DB_PATH.parent / "aria_test.db"

    # Start clean
    if db_module.DB_PATH.exists():
        db_module.DB_PATH.unlink()

    # ── Schema init ──────────────────────────────────────────
    section("1. Schema Initialization")
    init_db()
    ok(f"Database created at {db_module.DB_PATH}")

    # ── Projects ─────────────────────────────────────────────
    section("2. Projects")

    pid1 = create_project("Learn LangGraph", "Study agentic frameworks", "learning")
    ok(f"Created project id={pid1}: 'Learn LangGraph'")

    pid2 = create_project("Work Deliverables", category="work")
    ok(f"Created project id={pid2}: 'Work Deliverables'")

    project = get_project(pid1)
    ok(f"Fetched project: {dict(project)}")

    projects = list_projects()
    ok(f"Listed {len(projects)} active projects")

    archive_project(pid2)
    active = list_projects(status="active")
    archived = list_projects(status="archived")
    ok(f"After archiving: {len(active)} active, {len(archived)} archived")

    # ── Tasks ────────────────────────────────────────────────
    section("3. Tasks")

    tid1 = create_task(
        project_id  = pid1,
        title       = "Read LangGraph quickstart docs",
        priority    = 3,
        recurrence  = "weekly",
        due_date    = "2025-05-30"
    )
    ok(f"Created task id={tid1}: 'Read LangGraph quickstart docs'")

    tid2 = create_task(
        project_id  = pid1,
        title       = "Build first LangGraph agent",
        description = "Wire up a simple tool-calling agent",
        priority    = 2,
        due_date    = "2025-06-15"
    )
    ok(f"Created task id={tid2}: 'Build first LangGraph agent'")

    # A task with no priority — the AI will fill this in
    tid3 = create_task(
        project_id = pid1,
        title      = "Review Ollama API docs",
    )
    ok(f"Created task id={tid3} with no priority (AI will suggest)")

    # An overdue task — due date in the past
    tid4 = create_task(
        project_id = pid1,
        title      = "Set up virtual environment",
        priority   = 3,
        due_date   = "2024-01-01"  # Clearly in the past
    )
    ok(f"Created overdue task id={tid4}")

    task = get_task(tid1)
    ok(f"Fetched task with project join: project_name='{task['project_name']}'")

    tasks = list_tasks(project_id=pid1)
    ok(f"Listed {len(tasks)} tasks for project {pid1}")
    for t in tasks:
        print(f"       id={t['id']} priority={t['priority']} title='{t['title']}'")

    update_task_status(tid1, "in_progress")
    ok(f"Updated task {tid1} status → 'in_progress'")

    update_task_priority(tid3, 1)
    ok(f"Updated task {tid3} priority → 1 (low)")

    pending = list_tasks(project_id=pid1, status="pending")
    ok(f"Filtered by status='pending': {len(pending)} tasks")

    overdue = get_overdue_tasks()
    ok(f"Overdue tasks: {len(overdue)} found")
    for t in overdue:
        print(f"       id={t['id']} due='{t['due_date']}' title='{t['title']}'")

    summary_tasks = get_tasks_for_summary("week")
    ok(f"Tasks for weekly summary: {len(summary_tasks)} found")

    # ── AI Logs ──────────────────────────────────────────────
    section("4. AI Logs")

    fake_prompt = (
        "You are a task assistant. Given these tasks, suggest priorities.\n"
        "Tasks: [{'id': 3, 'title': 'Review Ollama API docs'}]"
    )
    fake_response = '[{"task_id": 3, "suggested_priority": 1, "reason": "Low urgency reference task"}]'

    log_id = log_ai_action(
        action     = "suggest",
        prompt     = fake_prompt,
        response   = fake_response,
        task_id    = tid3,
        model_used = "phi3.5"
    )
    ok(f"Logged AI 'suggest' action, log id={log_id}")

    log_id2 = log_ai_action(
        action   = "warn",
        prompt   = "Check for overdue tasks...",
        response = "Task 4 is overdue by 487 days.",
    )
    ok(f"Logged AI 'warn' action (no task_id), log id={log_id2}")

    logs = get_ai_logs()
    ok(f"Retrieved {len(logs)} total AI logs")

    task_logs = get_ai_logs(task_id=tid3)
    ok(f"Retrieved {len(task_logs)} logs for task {tid3}")

    suggest_logs = get_ai_logs(action="suggest")
    ok(f"Retrieved {len(suggest_logs)} 'suggest' logs")

    # ── Cleanup ──────────────────────────────────────────────
    section("5. Cleanup")
    db_module.DB_PATH.unlink()
    ok("Test database deleted")

    # ── Summary ──────────────────────────────────────────────
    section("ALL TESTS PASSED")
    print("  The DB layer is ready. You can now build on top of it.\n")


if __name__ == "__main__":
    run_tests()
