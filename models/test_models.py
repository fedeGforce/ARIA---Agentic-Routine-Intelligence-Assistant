# =============================================================
# aria/models/test_models.py
# =============================================================
# Pytest test suite for the dataclasses in models/task.py.
#
# Usage:
#   cd aria
#   pytest models/test_models.py -v
#
# These tests verify:
#   - Dataclass construction (direct and via from_row())
#   - Default field values match the schema
#   - Enum parsing from raw DB strings
#   - Optional fields handle None correctly
#   - from_row() works end-to-end with real DB rows
# =============================================================

from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

import db.database as db_module
from db.database import (
    create_project,
    create_task,
    get_ai_logs,
    get_project,
    get_task,
    init_db,
    log_ai_action,
)
from models.task import (
    AiAction,
    AiLog,
    Project,
    ProjectStatus,
    Recurrence,
    Task,
    TaskPriority,
    TaskStatus,
    _parse_date,
    _parse_datetime,
)

# =============================================================
# FIXTURES
# =============================================================


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    """Isolated DB for every test — same pattern as the DB suite."""
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "aria_test.db")
    init_db()


def make_row(data: dict):
    """
    Build a minimal sqlite3.Row-like mock from a dict.
    Supports both dict-style access (row["key"]) and row.keys().
    """
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: data[key]
    mock.keys = lambda: list(data.keys())
    return mock


# =============================================================
# ENUMS
# =============================================================


class TestEnums:
    def test_project_status_values(self):
        assert ProjectStatus.ACTIVE.value == "active"
        assert ProjectStatus.ARCHIVED.value == "archived"

    def test_task_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.DONE.value == "done"
        assert TaskStatus.CANCELLED.value == "cancelled"

    def test_task_priority_is_int_subclass(self):
        """TaskPriority inherits from int for numeric comparisons."""
        assert TaskPriority.LOW == 1
        assert TaskPriority.MEDIUM == 2
        assert TaskPriority.HIGH == 3

    def test_task_priority_comparison(self):
        assert TaskPriority.HIGH > TaskPriority.LOW
        assert TaskPriority.MEDIUM > TaskPriority.LOW
        assert TaskPriority.HIGH > TaskPriority.MEDIUM

    def test_recurrence_values(self):
        assert Recurrence.NONE.value == "none"
        assert Recurrence.DAILY.value == "daily"
        assert Recurrence.WEEKLY.value == "weekly"
        assert Recurrence.MONTHLY.value == "monthly"

    def test_ai_action_values(self):
        assert AiAction.CREATE.value == "create"
        assert AiAction.SUGGEST.value == "suggest"
        assert AiAction.WARN.value == "warn"
        assert AiAction.SUMMARIZE.value == "summarize"

    def test_enums_are_str_comparable(self):
        """str-based enums must compare equal to their raw string values."""
        assert ProjectStatus.ACTIVE == "active"
        assert TaskStatus.PENDING == "pending"
        assert Recurrence.WEEKLY == "weekly"
        assert AiAction.SUMMARIZE == "summarize"


# =============================================================
# PARSE HELPERS
# =============================================================


class TestParseHelpers:
    def test_parse_date_valid_string(self):
        result = _parse_date("2025-06-01")
        assert result == date(2025, 6, 1)

    def test_parse_date_none_returns_none(self):
        assert _parse_date(None) is None

    def test_parse_date_empty_string_returns_none(self):
        assert _parse_date("") is None

    def test_parse_datetime_with_space_separator(self):
        """SQLite stores datetimes as 'YYYY-MM-DD HH:MM:SS' (space, not T)."""
        result = _parse_datetime("2025-06-01 14:30:00")
        assert result == datetime(2025, 6, 1, 14, 30, 0)

    def test_parse_datetime_with_t_separator(self):
        """ISO format with T separator must also be handled."""
        result = _parse_datetime("2025-06-01T14:30:00")
        assert result == datetime(2025, 6, 1, 14, 30, 0)

    def test_parse_datetime_none_returns_none(self):
        assert _parse_datetime(None) is None

    def test_parse_datetime_empty_string_returns_none(self):
        assert _parse_datetime("") is None


# =============================================================
# PROJECT DATACLASS
# =============================================================


class TestProject:
    def test_direct_construction_minimal(self):
        p = Project(name="My Project")
        assert p.name == "My Project"
        assert p.category == "general"
        assert p.status == ProjectStatus.ACTIVE
        assert p.id is None

    def test_direct_construction_all_fields(self):
        p = Project(
            name="Work Project",
            category="work",
            description="Important stuff",
            status=ProjectStatus.ARCHIVED,
            id=42,
        )
        assert p.id == 42
        assert p.category == "work"
        assert p.description == "Important stuff"
        assert p.status == ProjectStatus.ARCHIVED

    def test_from_row_parses_all_fields(self):
        row = make_row(
            {
                "id": 1,
                "name": "Learn LangGraph",
                "description": "Study agentic frameworks",
                "category": "learning",
                "status": "active",
                "created_at": "2025-05-01 10:00:00",
                "updated_at": "2025-05-02 11:00:00",
            }
        )
        p = Project.from_row(row)
        assert p.id == 1
        assert p.name == "Learn LangGraph"
        assert p.description == "Study agentic frameworks"
        assert p.category == "learning"
        assert p.status == ProjectStatus.ACTIVE
        assert isinstance(p.created_at, datetime)
        assert isinstance(p.updated_at, datetime)

    def test_from_row_parses_archived_status(self):
        row = make_row(
            {
                "id": 2,
                "name": "Old Project",
                "description": None,
                "category": "work",
                "status": "archived",
                "created_at": "2025-01-01 00:00:00",
                "updated_at": "2025-01-01 00:00:00",
            }
        )
        p = Project.from_row(row)
        assert p.status == ProjectStatus.ARCHIVED

    def test_from_row_null_description_becomes_empty_string(self):
        row = make_row(
            {
                "id": 3,
                "name": "No Desc",
                "description": None,
                "category": "general",
                "status": "active",
                "created_at": "2025-01-01 00:00:00",
                "updated_at": "2025-01-01 00:00:00",
            }
        )
        p = Project.from_row(row)
        assert p.description == ""

    def test_from_row_with_real_db_row(self):
        """Round-trip test: write to DB, read back, convert to dataclass."""
        pid = create_project("Real Project", "From DB", "learning")
        row = get_project(pid)
        p = Project.from_row(row)
        assert p.id == pid
        assert p.name == "Real Project"
        assert p.category == "learning"
        assert p.status == ProjectStatus.ACTIVE


# =============================================================
# TASK DATACLASS
# =============================================================


class TestTask:
    def test_direct_construction_minimal(self):
        t = Task(project_id=1, title="My Task")
        assert t.title == "My Task"
        assert t.project_id == 1
        assert t.status == TaskStatus.PENDING
        assert t.recurrence == Recurrence.NONE
        assert t.priority is None
        assert t.due_date is None
        assert t.id is None

    def test_direct_construction_all_fields(self):
        t = Task(
            project_id=1,
            title="Full Task",
            description="Details",
            priority=TaskPriority.HIGH,
            recurrence=Recurrence.WEEKLY,
            due_date=date(2099, 12, 31),
            status=TaskStatus.IN_PROGRESS,
            id=7,
            project_name="My Project",
        )
        assert t.priority == TaskPriority.HIGH
        assert t.recurrence == Recurrence.WEEKLY
        assert t.due_date == date(2099, 12, 31)
        assert t.status == TaskStatus.IN_PROGRESS
        assert t.project_name == "My Project"

    def test_from_row_parses_all_fields(self):
        row = make_row(
            {
                "id": 10,
                "project_id": 1,
                "title": "Read docs",
                "description": "Important reading",
                "priority": 3,
                "recurrence": "weekly",
                "due_date": "2099-06-01",
                "status": "in_progress",
                "project_name": "Learn LangGraph",
                "created_at": "2025-05-01 09:00:00",
                "updated_at": "2025-05-10 17:00:00",
            }
        )
        t = Task.from_row(row)
        assert t.id == 10
        assert t.priority == TaskPriority.HIGH
        assert t.recurrence == Recurrence.WEEKLY
        assert t.due_date == date(2099, 6, 1)
        assert t.status == TaskStatus.IN_PROGRESS
        assert t.project_name == "Learn LangGraph"
        assert isinstance(t.created_at, datetime)

    def test_from_row_null_priority_stays_none(self):
        row = make_row(
            {
                "id": 1,
                "project_id": 1,
                "title": "No Prio",
                "description": None,
                "priority": None,
                "recurrence": "none",
                "due_date": None,
                "status": "pending",
                "project_name": "P",
                "created_at": "2025-01-01 00:00:00",
                "updated_at": "2025-01-01 00:00:00",
            }
        )
        t = Task.from_row(row)
        assert t.priority is None

    def test_from_row_null_due_date_stays_none(self):
        row = make_row(
            {
                "id": 1,
                "project_id": 1,
                "title": "No Due",
                "description": None,
                "priority": 1,
                "recurrence": "none",
                "due_date": None,
                "status": "pending",
                "project_name": "P",
                "created_at": "2025-01-01 00:00:00",
                "updated_at": "2025-01-01 00:00:00",
            }
        )
        t = Task.from_row(row)
        assert t.due_date is None

    def test_from_row_null_description_becomes_empty_string(self):
        row = make_row(
            {
                "id": 1,
                "project_id": 1,
                "title": "T",
                "description": None,
                "priority": None,
                "recurrence": "none",
                "due_date": None,
                "status": "pending",
                "project_name": "P",
                "created_at": "2025-01-01 00:00:00",
                "updated_at": "2025-01-01 00:00:00",
            }
        )
        t = Task.from_row(row)
        assert t.description == ""

    def test_from_row_without_project_name_key(self):
        """
        get_task() always JOINs project_name, but a raw tasks row
        won't have it. from_row() must handle that gracefully.
        """
        row = make_row(
            {
                "id": 1,
                "project_id": 1,
                "title": "T",
                "description": None,
                "priority": None,
                "recurrence": "none",
                "due_date": None,
                "status": "pending",
                "created_at": "2025-01-01 00:00:00",
                "updated_at": "2025-01-01 00:00:00",
            }
        )
        t = Task.from_row(row)
        assert t.project_name is None

    def test_from_row_all_recurrence_values(self):
        for value, expected in [
            ("none", Recurrence.NONE),
            ("daily", Recurrence.DAILY),
            ("weekly", Recurrence.WEEKLY),
            ("monthly", Recurrence.MONTHLY),
        ]:
            row = make_row(
                {
                    "id": 1,
                    "project_id": 1,
                    "title": "T",
                    "description": None,
                    "priority": None,
                    "recurrence": value,
                    "due_date": None,
                    "status": "pending",
                    "project_name": "P",
                    "created_at": "2025-01-01 00:00:00",
                    "updated_at": "2025-01-01 00:00:00",
                }
            )
            assert Task.from_row(row).recurrence == expected

    def test_from_row_all_status_values(self):
        for value, expected in [
            ("pending", TaskStatus.PENDING),
            ("in_progress", TaskStatus.IN_PROGRESS),
            ("done", TaskStatus.DONE),
            ("cancelled", TaskStatus.CANCELLED),
        ]:
            row = make_row(
                {
                    "id": 1,
                    "project_id": 1,
                    "title": "T",
                    "description": None,
                    "priority": None,
                    "recurrence": "none",
                    "due_date": None,
                    "status": value,
                    "project_name": "P",
                    "created_at": "2025-01-01 00:00:00",
                    "updated_at": "2025-01-01 00:00:00",
                }
            )
            assert Task.from_row(row).status == expected

    def test_from_row_with_real_db_row(self):
        """Round-trip test: write to DB, read back, convert to dataclass."""
        pid = create_project("Test Project")
        tid = create_task(
            project_id=pid,
            title="Real Task",
            priority=2,
            recurrence="monthly",
            due_date="2099-12-31",
        )
        row = get_task(tid)
        t = Task.from_row(row)
        assert t.id == tid
        assert t.title == "Real Task"
        assert t.priority == TaskPriority.MEDIUM
        assert t.recurrence == Recurrence.MONTHLY
        assert t.due_date == date(2099, 12, 31)
        assert t.project_name == "Test Project"


# =============================================================
# AILOG DATACLASS
# =============================================================


class TestAiLog:
    def test_direct_construction_minimal(self):
        log = AiLog(action=AiAction.WARN, prompt="p", response="r")
        assert log.action == AiAction.WARN
        assert log.task_id is None
        assert log.model_used == "phi3.5"
        assert log.id is None

    def test_direct_construction_all_fields(self):
        log = AiLog(
            action=AiAction.SUGGEST,
            prompt="Suggest priority",
            response='[{"priority": 3}]',
            task_id=5,
            model_used="llama3.2:3b",
            id=99,
        )
        assert log.task_id == 5
        assert log.model_used == "llama3.2:3b"
        assert log.id == 99

    def test_from_row_parses_all_fields(self):
        row = make_row(
            {
                "id": 1,
                "task_id": 3,
                "action": "suggest",
                "prompt": "Suggest priority for task",
                "response": '[{"priority": 2}]',
                "model_used": "phi3.5",
                "created_at": "2025-05-01 10:00:00",
            }
        )
        log = AiLog.from_row(row)
        assert log.id == 1
        assert log.task_id == 3
        assert log.action == AiAction.SUGGEST
        assert log.model_used == "phi3.5"
        assert isinstance(log.created_at, datetime)

    def test_from_row_null_task_id_stays_none(self):
        """Span-level actions (warn, summarize) have no task_id."""
        row = make_row(
            {
                "id": 2,
                "task_id": None,
                "action": "summarize",
                "prompt": "p",
                "response": "r",
                "model_used": "phi3.5",
                "created_at": "2025-05-01 10:00:00",
            }
        )
        log = AiLog.from_row(row)
        assert log.task_id is None

    def test_from_row_all_action_values(self):
        for value, expected in [
            ("create", AiAction.CREATE),
            ("suggest", AiAction.SUGGEST),
            ("warn", AiAction.WARN),
            ("summarize", AiAction.SUMMARIZE),
        ]:
            row = make_row(
                {
                    "id": 1,
                    "task_id": None,
                    "action": value,
                    "prompt": "p",
                    "response": "r",
                    "model_used": "phi3.5",
                    "created_at": "2025-01-01 00:00:00",
                }
            )
            assert AiLog.from_row(row).action == expected

    def test_from_row_with_real_db_row(self):
        """Round-trip test: write to DB, read back, convert to dataclass."""
        pid = create_project("Test Project")
        tid = create_task(project_id=pid, title="Test Task")
        log_ai_action(
            action="suggest",
            prompt="Suggest priority",
            response='[{"priority": 1}]',
            task_id=tid,
            model_used="phi3.5",
        )
        rows = get_ai_logs(task_id=tid)
        log = AiLog.from_row(rows[0])
        assert log.action == AiAction.SUGGEST
        assert log.task_id == tid
        assert log.model_used == "phi3.5"
        assert log.prompt == "Suggest priority"
