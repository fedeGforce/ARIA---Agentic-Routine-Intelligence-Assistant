# =============================================================
# aria/test_aria.py
# =============================================================
# Pytest test suite for the CLI layer (aria.py).
#
# Usage:
#   cd aria
#   pytest test_aria.py -v
#
# Click's CliRunner invokes every command in-process without
# spawning a subprocess, so tests are fast and output is
# captured cleanly.
#
# Two external boundaries are mocked throughout:
#   1. DB functions  — patched per test via monkeypatch or
#                      a real isolated DB for integration tests.
#   2. AI actions    — patched so no Ollama server is needed.
# =============================================================

from unittest.mock import patch

import pytest
from click.testing import CliRunner

import db.database as db_module
from agent.actions import OverdueWarning, ParsedTask, PrioritySuggestion, ProgressSummary
from aria_cli import cli
from db.database import init_db

# =============================================================
# FIXTURES
# =============================================================


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """
    Point every test at its own throwaway SQLite file and
    initialise the schema. Same pattern as the DB test suite.
    """
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "aria_test.db")
    init_db()


@pytest.fixture
def runner():
    """Click's CliRunner — invokes CLI commands in-process."""
    return CliRunner()


@pytest.fixture
def seeded_project(runner):
    """Create one project and return its id (always 1 on a fresh DB)."""
    runner.invoke(cli, ["project", "add", "Test Project", "--category", "work"])
    return 1


@pytest.fixture
def seeded_task(runner, seeded_project):
    """Create one task in the seeded project and return its id (always 1)."""
    runner.invoke(cli, ["task", "add", "Test task", "--project", "1"])
    return 1


# =============================================================
# HELPERS
# =============================================================


def invoke(runner, *args):
    """Shorthand: invoke the CLI and return the result."""
    return runner.invoke(cli, list(args))


# =============================================================
# PROJECT COMMANDS
# =============================================================


class TestProjectAdd:
    def test_exit_code_zero(self, runner):
        result = invoke(runner, "project", "add", "My Project")
        assert result.exit_code == 0

    def test_output_contains_project_name(self, runner):
        result = invoke(runner, "project", "add", "Learn LangGraph")
        assert "Learn LangGraph" in result.output

    def test_output_contains_id(self, runner):
        result = invoke(runner, "project", "add", "My Project")
        assert "[1]" in result.output

    def test_with_category(self, runner):
        result = invoke(runner, "project", "add", "Work Tasks", "--category", "work")
        assert result.exit_code == 0
        assert "work" in result.output

    def test_with_description(self, runner):
        result = invoke(runner, "project", "add", "My Project", "--description", "Some details")
        assert result.exit_code == 0

    def test_second_project_gets_id_2(self, runner):
        invoke(runner, "project", "add", "First")
        result = invoke(runner, "project", "add", "Second")
        assert "[2]" in result.output


class TestProjectList:
    def test_empty_db_shows_no_projects_message(self, runner):
        result = invoke(runner, "project", "list")
        assert result.exit_code == 0
        assert "No" in result.output

    def test_lists_created_project(self, runner, seeded_project):
        result = invoke(runner, "project", "list")
        assert "Test Project" in result.output

    def test_default_shows_only_active(self, runner):
        invoke(runner, "project", "add", "Active")
        invoke(runner, "project", "add", "Soon archived")
        invoke(runner, "project", "archive", "2")
        result = invoke(runner, "project", "list")
        assert "Active" in result.output
        assert "Soon archived" not in result.output

    def test_status_all_shows_archived(self, runner):
        invoke(runner, "project", "add", "Active")
        invoke(runner, "project", "add", "Archived one")
        invoke(runner, "project", "archive", "2")
        result = invoke(runner, "project", "list", "--status", "all")
        assert "Active" in result.output
        assert "Archived one" in result.output

    def test_status_archived_filter(self, runner):
        invoke(runner, "project", "add", "Will archive")
        invoke(runner, "project", "archive", "1")
        result = invoke(runner, "project", "list", "--status", "archived")
        assert "Will archive" in result.output


class TestProjectArchive:
    def test_exit_code_zero(self, runner, seeded_project):
        result = invoke(runner, "project", "archive", "1")
        assert result.exit_code == 0

    def test_output_confirms_archive(self, runner, seeded_project):
        result = invoke(runner, "project", "archive", "1")
        assert "archived" in result.output.lower()
        assert "Test Project" in result.output

    def test_archived_project_disappears_from_active_list(self, runner, seeded_project):
        invoke(runner, "project", "archive", "1")
        result = invoke(runner, "project", "list")
        assert "Test Project" not in result.output

    def test_missing_id_exits_nonzero(self, runner):
        result = invoke(runner, "project", "archive", "9999")
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


# =============================================================
# TASK COMMANDS
# =============================================================


class TestTaskAdd:
    def test_exit_code_zero(self, runner, seeded_project):
        result = invoke(runner, "task", "add", "My task", "--project", "1")
        assert result.exit_code == 0

    def test_output_contains_task_title(self, runner, seeded_project):
        result = invoke(runner, "task", "add", "Write docs", "--project", "1")
        assert "Write docs" in result.output

    def test_output_contains_task_id(self, runner, seeded_project):
        result = invoke(runner, "task", "add", "Write docs", "--project", "1")
        assert "[1]" in result.output

    def test_with_priority(self, runner, seeded_project):
        result = invoke(runner, "task", "add", "Urgent thing", "--project", "1", "--priority", "3")
        assert result.exit_code == 0

    def test_with_due_date(self, runner, seeded_project):
        result = invoke(
            runner, "task", "add", "Deadline task", "--project", "1", "--due", "2099-12-31"
        )
        assert result.exit_code == 0

    def test_with_recurrence(self, runner, seeded_project):
        result = invoke(
            runner, "task", "add", "Weekly review", "--project", "1", "--recurrence", "weekly"
        )
        assert result.exit_code == 0

    def test_missing_project_flag_fails(self, runner):
        result = invoke(runner, "task", "add", "No project task")
        assert result.exit_code != 0

    def test_nonexistent_project_exits_nonzero(self, runner):
        result = invoke(runner, "task", "add", "Orphan task", "--project", "9999")
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_project_name_shown_in_output(self, runner, seeded_project):
        result = invoke(runner, "task", "add", "Task A", "--project", "1")
        assert "Test Project" in result.output


class TestTaskList:
    def test_empty_db_shows_no_tasks_message(self, runner):
        result = invoke(runner, "task", "list")
        assert result.exit_code == 0
        assert "No tasks" in result.output

    def test_lists_created_task(self, runner, seeded_task):
        result = invoke(runner, "task", "list")
        assert "Test task" in result.output

    def test_filter_by_project(self, runner, seeded_task):
        result = invoke(runner, "task", "list", "--project", "1")
        assert "Test task" in result.output

    def test_filter_by_status_pending(self, runner, seeded_task):
        result = invoke(runner, "task", "list", "--status", "pending")
        assert "Test task" in result.output

    def test_done_task_excluded_from_pending_filter(self, runner, seeded_task):
        invoke(runner, "task", "done", "1")
        result = invoke(runner, "task", "list", "--status", "pending")
        assert "Test task" not in result.output

    def test_exit_code_zero(self, runner):
        result = invoke(runner, "task", "list")
        assert result.exit_code == 0


class TestTaskDone:
    def test_exit_code_zero(self, runner, seeded_task):
        result = invoke(runner, "task", "done", "1")
        assert result.exit_code == 0

    def test_output_confirms_done(self, runner, seeded_task):
        result = invoke(runner, "task", "done", "1")
        assert "Done" in result.output
        assert "Test task" in result.output

    def test_task_appears_as_done_in_list(self, runner, seeded_task):
        invoke(runner, "task", "done", "1")
        result = invoke(runner, "task", "list", "--status", "done")
        assert "Test task" in result.output

    def test_missing_id_exits_nonzero(self, runner):
        result = invoke(runner, "task", "done", "9999")
        assert result.exit_code != 0


class TestTaskStatus:
    def test_set_in_progress(self, runner, seeded_task):
        result = invoke(runner, "task", "status", "1", "in_progress")
        assert result.exit_code == 0
        assert "in progress" in result.output.lower()

    def test_set_cancelled(self, runner, seeded_task):
        result = invoke(runner, "task", "status", "1", "cancelled")
        assert result.exit_code == 0

    def test_invalid_status_fails(self, runner, seeded_task):
        result = invoke(runner, "task", "status", "1", "flying")
        assert result.exit_code != 0

    def test_missing_task_exits_nonzero(self, runner):
        result = invoke(runner, "task", "status", "9999", "done")
        assert result.exit_code != 0


class TestTaskParse:
    @patch("cli.parse_tasks")
    @patch("cli._make_client")
    def test_exit_code_zero(self, mock_client, mock_parse, runner, seeded_project):
        mock_parse.return_value = [
            ParsedTask(
                title="Review notes",
                recurrence="weekly",
                suggested_due="2099-06-01",
                priority=2,
                created_id=1,
            )
        ]
        result = invoke(runner, "task", "parse", "Review notes every week", "--project", "1")
        assert result.exit_code == 0

    @patch("cli.parse_tasks")
    @patch("cli._make_client")
    def test_output_shows_task_titles(self, mock_client, mock_parse, runner, seeded_project):
        mock_parse.return_value = [
            ParsedTask(
                title="Deploy service",
                recurrence="none",
                suggested_due=None,
                priority=3,
                created_id=1,
            )
        ]
        result = invoke(runner, "task", "parse", "Deploy service urgently", "--project", "1")
        assert "Deploy service" in result.output

    @patch("cli.parse_tasks")
    @patch("cli._make_client")
    def test_output_shows_task_count(self, mock_client, mock_parse, runner, seeded_project):
        mock_parse.return_value = [
            ParsedTask(title="A", recurrence="none", suggested_due=None, priority=1, created_id=1),
            ParsedTask(title="B", recurrence="none", suggested_due=None, priority=1, created_id=2),
        ]
        result = invoke(runner, "task", "parse", "A and B", "--project", "1")
        assert "2" in result.output

    @patch("cli._make_client")
    def test_nonexistent_project_exits_nonzero(self, mock_client, runner):
        result = invoke(runner, "task", "parse", "Some text", "--project", "9999")
        assert result.exit_code != 0

    @patch("cli.parse_tasks", side_effect=Exception("connection refused"))
    @patch("cli._make_client")
    def test_ollama_error_exits_nonzero(self, mock_client, mock_parse, runner, seeded_project):
        from agent.client import OllamaClientError

        mock_parse.side_effect = OllamaClientError("connection refused")
        result = invoke(runner, "task", "parse", "Any text", "--project", "1")
        assert result.exit_code != 0
        assert "ollama" in result.output.lower()


# =============================================================
# AI COMMANDS
# =============================================================


class TestAiSuggest:
    @patch("cli.suggest_priorities")
    @patch("cli._make_client")
    def test_exit_code_zero(self, mock_client, mock_suggest, runner):
        mock_suggest.return_value = []
        result = invoke(runner, "ai", "suggest")
        assert result.exit_code == 0

    @patch("cli.suggest_priorities")
    @patch("cli._make_client")
    def test_no_suggestions_message(self, mock_client, mock_suggest, runner):
        mock_suggest.return_value = []
        result = invoke(runner, "ai", "suggest")
        assert "no suggestions" in result.output.lower()

    @patch("cli.suggest_priorities")
    @patch("cli._make_client")
    def test_displays_suggestions(self, mock_client, mock_suggest, runner):
        mock_suggest.return_value = [
            PrioritySuggestion(
                task_id=1,
                title="Deploy service",
                old_priority=1,
                new_priority=3,
                reason="Due tomorrow.",
                applied=False,
            )
        ]
        result = invoke(runner, "ai", "suggest")
        assert "Deploy service" in result.output

    @patch("cli.suggest_priorities")
    @patch("cli._make_client")
    def test_apply_flag_passed_to_action(self, mock_client, mock_suggest, runner):
        mock_suggest.return_value = []
        invoke(runner, "ai", "suggest", "--apply")
        _, kwargs = mock_suggest.call_args
        assert kwargs.get("auto_apply") is True

    @patch("cli.suggest_priorities")
    @patch("cli._make_client")
    def test_project_filter_passed_to_action(self, mock_client, mock_suggest, runner):
        mock_suggest.return_value = []
        invoke(runner, "ai", "suggest", "--project", "3")
        _, kwargs = mock_suggest.call_args
        assert kwargs.get("project_id") == 3

    @patch("cli.suggest_priorities")
    @patch("cli._make_client")
    def test_preview_hint_shown_without_apply(self, mock_client, mock_suggest, runner):
        mock_suggest.return_value = [
            PrioritySuggestion(
                task_id=1, title="T", old_priority=1, new_priority=2, reason="r", applied=False
            )
        ]
        result = invoke(runner, "ai", "suggest")
        assert "--apply" in result.output


class TestAiWarn:
    @patch("cli.warn_overdue")
    @patch("cli._make_client")
    def test_exit_code_zero(self, mock_client, mock_warn, runner):
        mock_warn.return_value = []
        result = invoke(runner, "ai", "warn")
        assert result.exit_code == 0

    @patch("cli.warn_overdue")
    @patch("cli._make_client")
    def test_no_overdue_shows_success_message(self, mock_client, mock_warn, runner):
        mock_warn.return_value = []
        result = invoke(runner, "ai", "warn")
        assert "no overdue" in result.output.lower()

    @patch("cli.warn_overdue")
    @patch("cli._make_client")
    def test_displays_warnings(self, mock_client, mock_warn, runner):
        mock_warn.return_value = [
            OverdueWarning(
                task_id=2,
                title="Stale task",
                due_date="2024-01-01",
                days_overdue=150,
                urgency="high",
                suggestion="Reschedule or cancel.",
            )
        ]
        result = invoke(runner, "ai", "warn")
        assert "Stale task" in result.output
        assert "Reschedule or cancel." in result.output

    @patch("cli.warn_overdue")
    @patch("cli._make_client")
    def test_urgency_label_shown(self, mock_client, mock_warn, runner):
        mock_warn.return_value = [
            OverdueWarning(
                task_id=1,
                title="T",
                due_date="2024-01-01",
                days_overdue=10,
                urgency="high",
                suggestion="s",
            )
        ]
        result = invoke(runner, "ai", "warn")
        assert "HIGH" in result.output


class TestAiSummarize:
    @patch("cli.summarize_progress")
    @patch("cli._make_client")
    def test_exit_code_zero(self, mock_client, mock_sum, runner):
        mock_sum.return_value = ProgressSummary(
            period="week",
            summary_text="Good progress.",
            done_count=2,
            pending_count=1,
            in_progress_count=0,
        )
        result = invoke(runner, "ai", "summarize")
        assert result.exit_code == 0

    @patch("cli.summarize_progress")
    @patch("cli._make_client")
    def test_summary_text_shown(self, mock_client, mock_sum, runner):
        mock_sum.return_value = ProgressSummary(
            period="week",
            summary_text="Solid week of work.",
            done_count=3,
            pending_count=0,
            in_progress_count=1,
        )
        result = invoke(runner, "ai", "summarize")
        assert "Solid week of work." in result.output

    @patch("cli.summarize_progress")
    @patch("cli._make_client")
    def test_counts_shown(self, mock_client, mock_sum, runner):
        mock_sum.return_value = ProgressSummary(
            period="week",
            summary_text="Summary.",
            done_count=4,
            pending_count=2,
            in_progress_count=1,
        )
        result = invoke(runner, "ai", "summarize")
        assert "4" in result.output
        assert "2" in result.output
        assert "1" in result.output

    @patch("cli.summarize_progress")
    @patch("cli._make_client")
    def test_period_flag_passed_to_action(self, mock_client, mock_sum, runner):
        mock_sum.return_value = ProgressSummary(
            period="month",
            summary_text=".",
            done_count=0,
            pending_count=0,
            in_progress_count=0,
        )
        invoke(runner, "ai", "summarize", "--period", "month")
        args, kwargs = mock_sum.call_args
        assert kwargs.get("period") == "month" or args[0] == "month"

    @patch("cli.summarize_progress")
    @patch("cli._make_client")
    def test_project_filter_passed_to_action(self, mock_client, mock_sum, runner):
        mock_sum.return_value = ProgressSummary(
            period="week",
            summary_text=".",
            done_count=0,
            pending_count=0,
            in_progress_count=0,
        )
        invoke(runner, "ai", "summarize", "--project", "2")
        _, kwargs = mock_sum.call_args
        assert kwargs.get("project_id") == 2

    @patch("cli.summarize_progress")
    @patch("cli._make_client")
    def test_period_label_shown_in_output(self, mock_client, mock_sum, runner):
        for period, label in [("day", "Today"), ("week", "This week"), ("month", "This month")]:
            mock_sum.return_value = ProgressSummary(
                period=period,
                summary_text=".",
                done_count=0,
                pending_count=0,
                in_progress_count=0,
            )
            result = invoke(runner, "ai", "summarize", "--period", period)
            assert label in result.output
