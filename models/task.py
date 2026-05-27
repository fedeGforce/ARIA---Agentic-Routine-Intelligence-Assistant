# =============================================================
# aria/models/task.py
# =============================================================
# Pure dataclasses — no logic, no DB calls, no imports from
# other aria modules. These are typed containers that represent
# one row from each table, making it easy to pass structured
# data between the DB layer and the agent layer without
# carrying raw sqlite3.Row objects everywhere.
#
# Each class maps 1-to-1 with a table in schema.sql.
# Field names, types, and defaults mirror the schema exactly
# so there is never ambiguity about what a value means.
#
# Enums are defined here as the single source of truth for
# allowed values — the same constraints that live in the SQL
# CHECK() clauses.
# =============================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

# =============================================================
# ENUMS
# Defined once here; referenced by both the dataclasses below
# and by the agent layer when it needs to validate or compare
# values without hard-coding strings everywhere.
# =============================================================


class ProjectStatus(str, Enum):
    """Lifecycle states for a project."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class TaskStatus(str, Enum):
    """Lifecycle states for a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """
    Numeric priority levels.
    Inherits from int so comparisons like priority > TaskPriority.LOW
    work naturally without casting.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Recurrence(str, Enum):
    """How often a completed task should reappear."""

    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class AiAction(str, Enum):
    """The type of AI action recorded in an ai_log entry."""

    CREATE = "create"  # natural language → new task
    SUGGEST = "suggest"  # AI assigned priority / deadline
    WARN = "warn"  # AI flagged overdue / neglected tasks
    SUMMARIZE = "summarize"  # AI generated a progress summary


# =============================================================
# DATACLASSES
# =============================================================


@dataclass
class Project:
    """
    Represents one row from the `projects` table.

    id is Optional because a Project object can be constructed
    before it has been saved to the DB (id is assigned by
    SQLite's AUTOINCREMENT on INSERT).
    """

    name: str
    category: str = "general"
    description: str = ""
    status: ProjectStatus = ProjectStatus.ACTIVE
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row) -> Project:
        """
        Construct a Project from a sqlite3.Row (or any dict-like object).
        Converts raw strings into their enum equivalents.
        """
        return cls(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            category=row["category"],
            status=ProjectStatus(row["status"]),
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
        )


@dataclass
class Task:
    """
    Represents one row from the `tasks` table.

    project_name is not a DB column — it comes from the JOIN
    that get_task() and list_tasks() perform against the
    projects table. It is Optional because a Task constructed
    manually (not from a DB row) won't have it.
    """

    project_id: int
    title: str
    description: str = ""
    priority: TaskPriority | None = None
    recurrence: Recurrence = Recurrence.NONE
    due_date: date | None = None
    status: TaskStatus = TaskStatus.PENDING
    id: int | None = None
    project_name: str | None = None  # populated by JOIN queries
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row) -> Task:
        """
        Construct a Task from a sqlite3.Row.
        Handles optional fields (priority, due_date, project_name)
        that may be NULL in the DB.
        """
        return cls(
            id=row["id"],
            project_id=row["project_id"],
            title=row["title"],
            description=row["description"] or "",
            priority=TaskPriority(row["priority"]) if row["priority"] else None,
            recurrence=Recurrence(row["recurrence"]),
            due_date=_parse_date(row["due_date"]),
            status=TaskStatus(row["status"]),
            project_name=row["project_name"] if "project_name" in row.keys() else None,
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
        )


@dataclass
class AiLog:
    """
    Represents one row from the `ai_logs` table.

    prompt and response store the full text of what was sent
    to and received from the model — the complete audit trail
    of every AI decision in the system.
    """

    action: AiAction
    prompt: str
    response: str
    task_id: int | None = None
    model_used: str = "phi3.5"
    id: int | None = None
    created_at: datetime | None = None

    @classmethod
    def from_row(cls, row) -> AiLog:
        """
        Construct an AiLog from a sqlite3.Row.
        task_id is nullable — span-level actions have no task_id.
        """
        return cls(
            id=row["id"],
            task_id=row["task_id"],
            action=AiAction(row["action"]),
            prompt=row["prompt"],
            response=row["response"],
            model_used=row["model_used"],
            created_at=_parse_datetime(row["created_at"]),
        )


# =============================================================
# PRIVATE HELPERS
# Used only by the from_row() methods above to parse the
# ISO-8601 strings that SQLite returns for DATE and DATETIME
# columns into proper Python date/datetime objects.
# =============================================================


def _parse_date(value: str | None) -> date | None:
    """
    Parse a 'YYYY-MM-DD' string into a date object.
    Returns None if value is None or empty.
    """
    if not value:
        return None
    return date.fromisoformat(value)


def _parse_datetime(value: str | None) -> datetime | None:
    """
    Parse a 'YYYY-MM-DD HH:MM:SS' string into a datetime object.
    SQLite stores datetimes without the 'T' separator, so we
    replace the space with 'T' before calling fromisoformat().
    Returns None if value is None or empty.
    """
    if not value:
        return None
    return datetime.fromisoformat(value.replace(" ", "T"))
