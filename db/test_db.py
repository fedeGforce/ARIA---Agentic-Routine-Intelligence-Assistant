# =============================================================
# aria/db/test_db.py
# =============================================================
# Pytest test suite for the DB layer.
#
# Usage:
#   cd aria
#   pytest db/test_db.py -v
#
# Each test function gets a completely isolated in-memory
# SQLite database via the `db` fixture — no files created,
# no cleanup needed, no test order dependencies.
# =============================================================

import pytest
import db.database as db_module
from db.database import (
    init_db,
    create_project, get_project, list_projects, archive_project,
    create_task, get_task, list_tasks, update_task_status,
    update_task_priority, get_overdue_tasks, get_tasks_for_summary,
    log_ai_action, get_ai_logs,
)


# =============================================================
# FIXTURES
# =============================================================

@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    """
    Redirect every test to its own isolated SQLite file under
    pytest's temporary directory, then initialise the schema.

    autouse=True means this fixture runs automatically for every
    test in this module — no need to declare it as a parameter
    unless you need the path itself.

    monkeypatch ensures the override is rolled back after each
    test, so tests can never bleed state into each other.
    """
    test_db_path = tmp_path / "aria_test.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db_path)
    init_db()
    return test_db_path


@pytest.fixture
def project(db):
    """
    A ready-made project for tests that need one without caring
    about its specific attributes.
    """
    pid = create_project("Learn LangGraph", "Study agentic frameworks", "learning")
    return pid


@pytest.fixture
def task_with_priority(project):
    """
    A ready-made high-priority task with a future due date.
    """
    return create_task(
        project_id=project,
        title="Read LangGraph quickstart docs",
        priority=3,
        recurrence="weekly",
        due_date="2099-05-30",   # Far future — never overdue in tests
    )


@pytest.fixture
def overdue_task(project):
    """
    A task whose due_date is firmly in the past, guaranteed to
    appear in get_overdue_tasks().
    """
    return create_task(
        project_id=project,
        title="Set up virtual environment",
        priority=3,
        due_date="2024-01-01",
    )


# =============================================================
# SCHEMA
# =============================================================

class TestSchema:

    def test_db_file_is_created(self, db):
        """init_db() must create the SQLite file on disk."""
        assert db.exists()

    def test_init_db_is_idempotent(self, db):
        """
        Calling init_db() a second time must not raise — all
        CREATE statements use IF NOT EXISTS.
        """
        init_db()   # second call
        assert db.exists()


# =============================================================
# PROJECTS
# =============================================================

class TestProjects:

    def test_create_project_returns_integer_id(self):
        pid = create_project("My Project")
        assert isinstance(pid, int)
        assert pid > 0

    def test_create_project_defaults(self):
        pid = create_project("Minimal Project")
        row = get_project(pid)
        assert row["category"] == "general"
        assert row["status"]   == "active"

    def test_create_project_with_all_fields(self):
        pid = create_project("Full Project", description="A description", category="work")
        row = get_project(pid)
        assert row["name"]        == "Full Project"
        assert row["description"] == "A description"
        assert row["category"]    == "work"

    def test_get_project_returns_correct_row(self, project):
        row = get_project(project)
        assert row["id"]   == project
        assert row["name"] == "Learn LangGraph"

    def test_get_project_returns_none_for_missing_id(self):
        assert get_project(9999) is None

    def test_list_projects_returns_only_active_by_default(self):
        pid1 = create_project("Active Project")
        pid2 = create_project("To Be Archived")
        archive_project(pid2)

        active = list_projects()
        ids = [r["id"] for r in active]
        assert pid1 in ids
        assert pid2 not in ids

    def test_list_projects_archived_filter(self):
        pid = create_project("Soon Archived")
        archive_project(pid)

        archived = list_projects(status="archived")
        assert any(r["id"] == pid for r in archived)

    def test_list_projects_all_filter(self):
        pid1 = create_project("Active")
        pid2 = create_project("Archived")
        archive_project(pid2)

        all_projects = list_projects(status="all")
        ids = [r["id"] for r in all_projects]
        assert pid1 in ids
        assert pid2 in ids

    def test_archive_project_returns_true_on_success(self, project):
        assert archive_project(project) is True

    def test_archive_project_returns_false_for_missing_id(self):
        assert archive_project(9999) is False

    def test_archive_project_changes_status(self, project):
        archive_project(project)
        row = get_project(project)
        assert row["status"] == "archived"

    def test_multiple_projects_get_unique_ids(self):
        ids = [create_project(f"Project {i}") for i in range(5)]
        assert len(set(ids)) == 5


# =============================================================
# TASKS
# =============================================================

class TestTasks:

    def test_create_task_returns_integer_id(self, project):
        tid = create_task(project_id=project, title="My Task")
        assert isinstance(tid, int)
        assert tid > 0

    def test_create_task_defaults(self, project):
        tid = create_task(project_id=project, title="Minimal Task")
        row = get_task(tid)
        assert row["status"]     == "pending"
        assert row["recurrence"] == "none"
        assert row["priority"]   is None
        assert row["due_date"]   is None

    def test_create_task_with_all_fields(self, project):
        tid = create_task(
            project_id=project,
            title="Full Task",
            description="Details here",
            priority=2,
            recurrence="weekly",
            due_date="2099-12-31",
        )
        row = get_task(tid)
        assert row["title"]       == "Full Task"
        assert row["description"] == "Details here"
        assert row["priority"]    == 2
        assert row["recurrence"]  == "weekly"
        assert row["due_date"]    == "2099-12-31"

    def test_get_task_includes_project_name(self, project, task_with_priority):
        row = get_task(task_with_priority)
        assert row["project_name"] == "Learn LangGraph"

    def test_get_task_returns_none_for_missing_id(self):
        assert get_task(9999) is None

    def test_list_tasks_returns_all_for_project(self, project):
        create_task(project_id=project, title="Task A")
        create_task(project_id=project, title="Task B")
        tasks = list_tasks(project_id=project)
        assert len(tasks) == 2

    def test_list_tasks_filters_by_status(self, project):
        tid1 = create_task(project_id=project, title="Will be in_progress")
        tid2 = create_task(project_id=project, title="Stays pending")
        update_task_status(tid1, "in_progress")

        pending = list_tasks(project_id=project, status="pending")
        ids = [r["id"] for r in pending]
        assert tid2 in ids
        assert tid1 not in ids

    def test_list_tasks_ordered_by_priority_desc(self, project):
        """
        High priority (3) tasks must appear before low priority (1).
        Null priorities must appear last.
        """
        create_task(project_id=project, title="Low",    priority=1)
        create_task(project_id=project, title="High",   priority=3)
        create_task(project_id=project, title="No prio")           # priority=None

        tasks = list_tasks(project_id=project)
        priorities = [r["priority"] for r in tasks]

        # High must come before Low
        assert priorities.index(3) < priorities.index(1)
        # None must be last
        assert priorities[-1] is None

    def test_update_task_status_returns_true_on_success(self, project, task_with_priority):
        assert update_task_status(task_with_priority, "in_progress") is True

    def test_update_task_status_persists(self, project, task_with_priority):
        update_task_status(task_with_priority, "done")
        row = get_task(task_with_priority)
        assert row["status"] == "done"

    def test_update_task_status_returns_false_for_missing_id(self):
        assert update_task_status(9999, "done") is False

    def test_update_task_priority_returns_true_on_success(self, project):
        tid = create_task(project_id=project, title="No priority yet")
        assert update_task_priority(tid, 2) is True

    def test_update_task_priority_persists(self, project):
        tid = create_task(project_id=project, title="No priority yet")
        update_task_priority(tid, 2)
        row = get_task(tid)
        assert row["priority"] == 2

    def test_update_task_priority_returns_false_for_missing_id(self):
        assert update_task_priority(9999, 1) is False

    def test_get_overdue_tasks_returns_past_due_dates(self, project, overdue_task):
        overdue = get_overdue_tasks()
        ids = [r["id"] for r in overdue]
        assert overdue_task in ids

    def test_get_overdue_tasks_excludes_done_tasks(self, project, overdue_task):
        update_task_status(overdue_task, "done")
        overdue = get_overdue_tasks()
        ids = [r["id"] for r in overdue]
        assert overdue_task not in ids

    def test_get_overdue_tasks_excludes_cancelled_tasks(self, project, overdue_task):
        update_task_status(overdue_task, "cancelled")
        overdue = get_overdue_tasks()
        ids = [r["id"] for r in overdue]
        assert overdue_task not in ids

    def test_get_overdue_tasks_excludes_future_due_dates(self, project, task_with_priority):
        """task_with_priority has due_date=2099-05-30, must not appear."""
        overdue = get_overdue_tasks()
        ids = [r["id"] for r in overdue]
        assert task_with_priority not in ids

    def test_get_tasks_for_summary_day(self, project):
        create_task(project_id=project, title="Today's task")
        tasks = get_tasks_for_summary("day")
        assert len(tasks) >= 1

    def test_get_tasks_for_summary_week(self, project):
        create_task(project_id=project, title="This week's task")
        tasks = get_tasks_for_summary("week")
        assert len(tasks) >= 1

    def test_get_tasks_for_summary_month(self, project):
        create_task(project_id=project, title="This month's task")
        tasks = get_tasks_for_summary("month")
        assert len(tasks) >= 1

    def test_get_tasks_for_summary_unknown_period_falls_back_to_week(self, project):
        """Unknown period strings must not crash — they fall back to 'week'."""
        create_task(project_id=project, title="Some task")
        tasks = get_tasks_for_summary("unknown_period")
        assert isinstance(tasks, list)


# =============================================================
# AI LOGS
# =============================================================

class TestAiLogs:

    def test_log_ai_action_returns_integer_id(self, project):
        log_id = log_ai_action(
            action="suggest",
            prompt="Suggest priorities",
            response="[{}]",
        )
        assert isinstance(log_id, int)
        assert log_id > 0

    def test_log_ai_action_with_task_id(self, project, task_with_priority):
        log_id = log_ai_action(
            action="suggest",
            prompt="Suggest priority for task",
            response='[{"task_id": 1, "priority": 2}]',
            task_id=task_with_priority,
            model_used="phi3.5",
        )
        logs = get_ai_logs(task_id=task_with_priority)
        assert any(r["id"] == log_id for r in logs)

    def test_log_ai_action_without_task_id(self):
        """Span-level actions (warn, summarize) have no task_id."""
        log_id = log_ai_action(
            action="warn",
            prompt="Check overdue tasks",
            response="2 tasks are overdue.",
        )
        logs = get_ai_logs()
        ids = [r["id"] for r in logs]
        assert log_id in ids

    def test_get_ai_logs_returns_all_when_no_filters(self):
        log_ai_action(action="warn",      prompt="p1", response="r1")
        log_ai_action(action="summarize", prompt="p2", response="r2")
        logs = get_ai_logs()
        assert len(logs) == 2

    def test_get_ai_logs_filters_by_task_id(self, project, task_with_priority):
        tid2 = create_task(project_id=project, title="Another task")

        log_ai_action(action="suggest", prompt="p", response="r", task_id=task_with_priority)
        log_ai_action(action="suggest", prompt="p", response="r", task_id=tid2)

        logs = get_ai_logs(task_id=task_with_priority)
        assert len(logs) == 1
        assert logs[0]["task_id"] == task_with_priority

    def test_get_ai_logs_filters_by_action(self):
        log_ai_action(action="suggest",  prompt="p", response="r")
        log_ai_action(action="summarize", prompt="p", response="r")

        suggest_logs = get_ai_logs(action="suggest")
        assert all(r["action"] == "suggest" for r in suggest_logs)
        assert len(suggest_logs) == 1

    def test_get_ai_logs_filters_by_task_and_action(self, project, task_with_priority):
        log_ai_action(action="suggest",  prompt="p", response="r", task_id=task_with_priority)
        log_ai_action(action="warn",     prompt="p", response="r", task_id=task_with_priority)

        logs = get_ai_logs(task_id=task_with_priority, action="suggest")
        assert len(logs) == 1
        assert logs[0]["action"] == "suggest"

    def test_get_ai_logs_returns_empty_list_when_no_match(self):
        logs = get_ai_logs(task_id=9999)
        assert logs == []

    def test_log_stores_model_used(self):
        log_id = log_ai_action(
            action="summarize",
            prompt="p",
            response="r",
            model_used="llama3.2:3b",
        )
        logs = get_ai_logs()
        match = next(r for r in logs if r["id"] == log_id)
        assert match["model_used"] == "llama3.2:3b"

    def test_default_model_is_phi35(self):
        log_id = log_ai_action(action="warn", prompt="p", response="r")
        logs = get_ai_logs()
        match = next(r for r in logs if r["id"] == log_id)
        assert match["model_used"] == "phi3.5"
