# =============================================================
# aria/agent/test_actions.py
# =============================================================
# Pytest test suite for agent/actions.py.
#
# Usage:
#   cd aria
#   pytest agent/test_actions.py -v
#
# Three external boundaries are mocked in every test:
#
#   1. OllamaClient  — no real LLM calls; responses are
#                      controlled strings defined per test.
#   2. DB functions  — patched so tests never touch disk.
#   3. log_ai_action — patched globally; verified where relevant.
#
# Each test class maps to one action (parse, suggest, warn,
# summarize) plus a shared class for the internal helpers.
# =============================================================

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from agent.actions import (  # result types; helpers (tested directly); actions
    OverdueWarning,
    ParsedTask,
    PrioritySuggestion,
    ProgressSummary,
    _extract_json,
    _serialize_tasks,
    _today_plus,
    parse_tasks,
    suggest_priorities,
    summarize_progress,
    warn_overdue,
)

# =============================================================
# SHARED FIXTURES
# =============================================================


@pytest.fixture
def client():
    """
    A mock OllamaClient. Tests set .chat.return_value or
    .stream.return_value to control what the 'model' responds.
    """
    mock = MagicMock()
    mock.model = "phi3.5"

    # Convenience constructors must return Message-like dicts
    mock.system.side_effect = lambda c: {"role": "system", "content": c}
    mock.user.side_effect = lambda c: {"role": "user", "content": c}
    return mock


def make_chat_response(content: str) -> MagicMock:
    """Build a minimal ChatResponse mock with the given content."""
    r = MagicMock()
    r.content = content
    return r


def make_db_task(
    id=1,
    project_id=1,
    title="Test task",
    status="pending",
    priority=None,
    due_date=None,
    recurrence="none",
) -> MagicMock:
    """
    Build a dict-like mock that behaves like a sqlite3.Row,
    including support for row["key"] access and iteration.
    """
    data = dict(
        id=id,
        project_id=project_id,
        title=title,
        status=status,
        priority=priority,
        due_date=due_date,
        recurrence=recurrence,
    )
    mock = MagicMock()
    mock.__getitem__ = lambda self, k: data[k]
    mock.keys = lambda: list(data.keys())
    return mock


# =============================================================
# INTERNAL HELPERS
# =============================================================


class TestExtractJson:
    def test_plain_json_array(self):
        result = _extract_json('[{"a": 1}]')
        assert json.loads(result) == [{"a": 1}]

    def test_plain_json_object(self):
        result = _extract_json('{"a": 1}')
        assert json.loads(result) == {"a": 1}

    def test_strips_markdown_code_fence(self):
        text = '```json\n[{"a": 1}]\n```'
        result = _extract_json(text)
        assert json.loads(result) == [{"a": 1}]

    def test_strips_code_fence_without_language(self):
        text = '```\n[{"a": 1}]\n```'
        result = _extract_json(text)
        assert json.loads(result) == [{"a": 1}]

    def test_strips_preamble_text(self):
        text = 'Here is the result:\n[{"a": 1}]'
        result = _extract_json(text)
        assert json.loads(result) == [{"a": 1}]

    def test_nested_json_is_preserved(self):
        text = '[{"a": {"b": [1, 2, 3]}}]'
        result = _extract_json(text)
        assert json.loads(result) == [{"a": {"b": [1, 2, 3]}}]

    def test_returns_input_when_no_json_found(self):
        """
        If no brackets are found, return as-is and let
        json.loads raise a descriptive error in the caller.
        """
        text = "No JSON here at all."
        result = _extract_json(text)
        assert result == text


class TestTodayPlus:
    def test_zero_days_is_today(self):
        assert _today_plus(0) == date.today().isoformat()

    def test_positive_offset(self):
        expected = (date.today() + timedelta(days=7)).isoformat()
        assert _today_plus(7) == expected

    def test_returns_iso_format_string(self):
        result = _today_plus(1)
        date.fromisoformat(result)  # raises if format is wrong


class TestSerializeTasks:
    def test_includes_required_fields(self):
        task = make_db_task(
            id=1,
            title="T",
            status="pending",
            priority=2,
            due_date="2099-01-01",
            recurrence="weekly",
        )
        result = json.loads(_serialize_tasks([task]))
        assert result[0]["id"] == 1
        assert result[0]["title"] == "T"
        assert result[0]["status"] == "pending"
        assert result[0]["priority"] == 2
        assert result[0]["due_date"] == "2099-01-01"
        assert result[0]["recurrence"] == "weekly"

    def test_empty_list_returns_empty_array(self):
        assert json.loads(_serialize_tasks([])) == []

    def test_multiple_tasks_serialized(self):
        tasks = [make_db_task(id=i, title=f"Task {i}") for i in range(3)]
        result = json.loads(_serialize_tasks(tasks))
        assert len(result) == 3


# =============================================================
# ACTION: PARSE
# =============================================================


class TestParseTasks:
    def _model_response(self, items: list[dict]) -> str:
        return json.dumps(items)

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.create_task", return_value=42)
    def test_returns_list_of_parsed_tasks(self, mock_create, mock_log, client):
        payload = [
            {
                "title": "Review notes",
                "recurrence": "weekly",
                "suggested_due": "2099-06-01",
                "priority": 2,
            }
        ]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        results = parse_tasks("Review notes every week", project_id=1, client=client)

        assert len(results) == 1
        assert isinstance(results[0], ParsedTask)

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.create_task", return_value=42)
    def test_parsed_task_fields_correct(self, mock_create, mock_log, client):
        payload = [
            {
                "title": "Write report",
                "recurrence": "monthly",
                "suggested_due": "2099-06-30",
                "priority": 3,
            }
        ]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        result = parse_tasks("Write report monthly", project_id=1, client=client)[0]

        assert result.title == "Write report"
        assert result.recurrence == "monthly"
        assert result.suggested_due == "2099-06-30"
        assert result.priority == 3

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.create_task", return_value=7)
    def test_created_id_set_after_db_insert(self, mock_create, mock_log, client):
        payload = [{"title": "Task A", "recurrence": "none", "suggested_due": None, "priority": 1}]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        result = parse_tasks("Task A", project_id=1, client=client)[0]

        assert result.created_id == 7

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.create_task", return_value=1)
    def test_multiple_tasks_extracted(self, mock_create, mock_log, client):
        payload = [
            {"title": "Task A", "recurrence": "daily", "suggested_due": None, "priority": 1},
            {"title": "Task B", "recurrence": "weekly", "suggested_due": None, "priority": 2},
        ]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        results = parse_tasks("Task A daily, Task B weekly", project_id=1, client=client)

        assert len(results) == 2

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.create_task", return_value=1)
    def test_priority_clamped_to_valid_range(self, mock_create, mock_log, client):
        """Model returning priority=99 must be clamped to 3."""
        payload = [{"title": "T", "recurrence": "none", "suggested_due": None, "priority": 99}]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        result = parse_tasks("T", project_id=1, client=client)[0]

        assert result.priority == 3

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.create_task", return_value=1)
    def test_invalid_recurrence_defaults_to_none(self, mock_create, mock_log, client):
        """Model returning unknown recurrence must default to 'none'."""
        payload = [{"title": "T", "recurrence": "biweekly", "suggested_due": None, "priority": 1}]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        result = parse_tasks("T", project_id=1, client=client)[0]

        assert result.recurrence == "none"

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.create_task", return_value=1)
    def test_create_task_called_with_correct_args(self, mock_create, mock_log, client):
        payload = [
            {
                "title": "Deploy service",
                "recurrence": "none",
                "suggested_due": "2099-07-01",
                "priority": 3,
            }
        ]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        parse_tasks("Deploy service", project_id=5, client=client)

        mock_create.assert_called_once_with(
            project_id=5,
            title="Deploy service",
            priority=3,
            recurrence="none",
            due_date="2099-07-01",
        )

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.create_task", return_value=1)
    def test_log_ai_action_called_once(self, mock_create, mock_log, client):
        payload = [{"title": "T", "recurrence": "none", "suggested_due": None, "priority": 1}]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        parse_tasks("T", project_id=1, client=client)

        # One call for the parse action itself
        assert mock_log.call_count >= 1

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.create_task", return_value=1)
    def test_handles_markdown_wrapped_json(self, mock_create, mock_log, client):
        """Model sometimes wraps JSON in ```json fences."""
        payload = [{"title": "T", "recurrence": "none", "suggested_due": None, "priority": 1}]
        wrapped = f"```json\n{json.dumps(payload)}\n```"
        client.chat.return_value = make_chat_response(wrapped)

        results = parse_tasks("T", project_id=1, client=client)

        assert len(results) == 1


# =============================================================
# ACTION: SUGGEST
# =============================================================


class TestSuggestPriorities:
    def _model_response(self, items: list[dict]) -> str:
        return json.dumps(items)

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.update_task_priority")
    @patch("agent.actions.list_tasks")
    def test_returns_empty_list_when_no_tasks(self, mock_list, mock_update, mock_log, client):
        mock_list.return_value = []

        results = suggest_priorities(client)

        assert results == []

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.update_task_priority")
    @patch("agent.actions.list_tasks")
    def test_returns_priority_suggestions(self, mock_list, mock_update, mock_log, client):
        mock_list.return_value = [make_db_task(id=1, title="T", priority=None)]
        payload = [{"task_id": 1, "new_priority": 2, "reason": "No deadline, medium complexity."}]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        results = suggest_priorities(client)

        assert len(results) == 1
        assert isinstance(results[0], PrioritySuggestion)

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.update_task_priority")
    @patch("agent.actions.list_tasks")
    def test_suggestion_fields_correct(self, mock_list, mock_update, mock_log, client):
        mock_list.return_value = [make_db_task(id=3, title="Deploy", priority=1)]
        payload = [{"task_id": 3, "new_priority": 3, "reason": "Due tomorrow."}]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        result = suggest_priorities(client)[0]

        assert result.task_id == 3
        assert result.title == "Deploy"
        assert result.old_priority == 1
        assert result.new_priority == 3
        assert result.reason == "Due tomorrow."

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.update_task_priority")
    @patch("agent.actions.list_tasks")
    def test_not_applied_by_default(self, mock_list, mock_update, mock_log, client):
        mock_list.return_value = [make_db_task(id=1, priority=None)]
        payload = [{"task_id": 1, "new_priority": 2, "reason": "r"}]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        result = suggest_priorities(client)[0]

        assert result.applied is False
        mock_update.assert_not_called()

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.update_task_priority")
    @patch("agent.actions.list_tasks")
    def test_auto_apply_writes_to_db(self, mock_list, mock_update, mock_log, client):
        mock_list.return_value = [make_db_task(id=5, priority=None)]
        payload = [{"task_id": 5, "new_priority": 3, "reason": "Urgent."}]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        result = suggest_priorities(client, auto_apply=True)[0]

        assert result.applied is True
        mock_update.assert_called_once_with(5, 3)

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.update_task_priority")
    @patch("agent.actions.list_tasks")
    def test_hallucinated_task_id_skipped(self, mock_list, mock_update, mock_log, client):
        """Model returning a task_id that doesn't exist must be silently ignored."""
        mock_list.return_value = [make_db_task(id=1)]
        payload = [{"task_id": 9999, "new_priority": 3, "reason": "r"}]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        results = suggest_priorities(client)

        assert results == []

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.update_task_priority")
    @patch("agent.actions.list_tasks")
    def test_priority_clamped(self, mock_list, mock_update, mock_log, client):
        mock_list.return_value = [make_db_task(id=1, priority=None)]
        payload = [{"task_id": 1, "new_priority": 0, "reason": "r"}]
        client.chat.return_value = make_chat_response(self._model_response(payload))

        result = suggest_priorities(client)[0]

        assert result.new_priority == 1  # clamped from 0 → 1

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.update_task_priority")
    @patch("agent.actions.list_tasks")
    def test_filters_by_project_id(self, mock_list, mock_update, mock_log, client):
        mock_list.return_value = []
        client.chat.return_value = make_chat_response("[]")

        suggest_priorities(client, project_id=7)

        mock_list.assert_called_once_with(project_id=7, status="pending")


# =============================================================
# ACTION: WARN
# =============================================================


class TestWarnOverdue:
    def _today_minus(self, days: int) -> str:
        return (date.today() - timedelta(days=days)).isoformat()

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_overdue_tasks")
    def test_returns_empty_list_when_no_overdue(self, mock_get, mock_log, client):
        mock_get.return_value = []

        results = warn_overdue(client)

        assert results == []

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_overdue_tasks")
    def test_returns_overdue_warnings(self, mock_get, mock_log, client):
        mock_get.return_value = [
            make_db_task(id=1, title="Old task", due_date=self._today_minus(5))
        ]
        payload = [{"task_id": 1, "urgency": "medium", "suggestion": "Do it now."}]
        client.chat.return_value = make_chat_response(json.dumps(payload))

        results = warn_overdue(client)

        assert len(results) == 1
        assert isinstance(results[0], OverdueWarning)

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_overdue_tasks")
    def test_warning_fields_correct(self, mock_get, mock_log, client):
        due = self._today_minus(10)
        mock_get.return_value = [make_db_task(id=2, title="Stale task", due_date=due)]
        payload = [{"task_id": 2, "urgency": "high", "suggestion": "Reschedule immediately."}]
        client.chat.return_value = make_chat_response(json.dumps(payload))

        result = warn_overdue(client)[0]

        assert result.task_id == 2
        assert result.title == "Stale task"
        assert result.due_date == due
        assert result.days_overdue == 10
        assert result.urgency == "high"
        assert result.suggestion == "Reschedule immediately."

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_overdue_tasks")
    def test_days_overdue_computed_correctly(self, mock_get, mock_log, client):
        due = self._today_minus(3)
        mock_get.return_value = [make_db_task(id=1, due_date=due)]
        payload = [{"task_id": 1, "urgency": "low", "suggestion": "Do it."}]
        client.chat.return_value = make_chat_response(json.dumps(payload))

        result = warn_overdue(client)[0]

        assert result.days_overdue == 3

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_overdue_tasks")
    def test_invalid_urgency_defaults_to_medium(self, mock_get, mock_log, client):
        mock_get.return_value = [make_db_task(id=1, due_date=self._today_minus(2))]
        payload = [{"task_id": 1, "urgency": "critical", "suggestion": "s"}]
        client.chat.return_value = make_chat_response(json.dumps(payload))

        result = warn_overdue(client)[0]

        assert result.urgency == "medium"

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_overdue_tasks")
    def test_sorted_by_urgency_high_first(self, mock_get, mock_log, client):
        mock_get.return_value = [
            make_db_task(id=1, due_date=self._today_minus(1)),
            make_db_task(id=2, due_date=self._today_minus(9)),
        ]
        payload = [
            {"task_id": 1, "urgency": "low", "suggestion": "s"},
            {"task_id": 2, "urgency": "high", "suggestion": "s"},
        ]
        client.chat.return_value = make_chat_response(json.dumps(payload))

        results = warn_overdue(client)

        assert results[0].urgency == "high"
        assert results[1].urgency == "low"

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_overdue_tasks")
    def test_hallucinated_task_id_skipped(self, mock_get, mock_log, client):
        mock_get.return_value = [make_db_task(id=1, due_date=self._today_minus(1))]
        payload = [{"task_id": 9999, "urgency": "high", "suggestion": "s"}]
        client.chat.return_value = make_chat_response(json.dumps(payload))

        results = warn_overdue(client)

        assert results == []


# =============================================================
# ACTION: SUMMARIZE
# =============================================================


class TestSummarizeProgress:
    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_tasks_for_summary")
    def test_returns_progress_summary(self, mock_get, mock_log, client):
        mock_get.return_value = [make_db_task(status="done")]
        client.stream.return_value = iter(["Good ", "progress."])

        result = summarize_progress("week", client)

        assert isinstance(result, ProgressSummary)

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_tasks_for_summary")
    def test_empty_period_returns_no_activity_message(self, mock_get, mock_log, client):
        mock_get.return_value = []

        result = summarize_progress("day", client)

        assert "No tasks" in result.summary_text
        assert result.done_count == 0
        assert result.pending_count == 0
        assert result.in_progress_count == 0

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_tasks_for_summary")
    def test_counts_are_accurate(self, mock_get, mock_log, client):
        mock_get.return_value = [
            make_db_task(id=1, status="done"),
            make_db_task(id=2, status="done"),
            make_db_task(id=3, status="in_progress"),
            make_db_task(id=4, status="pending"),
        ]
        client.stream.return_value = iter(["Summary text."])

        result = summarize_progress("week", client)

        assert result.done_count == 2
        assert result.in_progress_count == 1
        assert result.pending_count == 1

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_tasks_for_summary")
    def test_summary_text_assembled_from_stream(self, mock_get, mock_log, client):
        mock_get.return_value = [make_db_task(status="done")]
        client.stream.return_value = iter(["Well ", "done ", "this week."])

        result = summarize_progress("week", client)

        assert result.summary_text == "Well done this week."

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_tasks_for_summary")
    def test_period_stored_in_result(self, mock_get, mock_log, client):
        mock_get.return_value = [make_db_task(status="pending")]
        client.stream.return_value = iter(["Text."])

        for period in ("day", "week", "month"):
            result = summarize_progress(period, client)
            assert result.period == period

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_tasks_for_summary")
    def test_filters_by_project_id(self, mock_get, mock_log, client):
        mock_get.return_value = [
            make_db_task(id=1, project_id=1, status="done"),
            make_db_task(id=2, project_id=2, status="done"),
        ]
        client.stream.return_value = iter(["Summary."])

        result = summarize_progress("week", client, project_id=1)

        # Only the task with project_id=1 should be counted
        assert result.done_count == 1

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_tasks_for_summary")
    def test_uses_streaming_for_output(self, mock_get, mock_log, client):
        """summarize_progress must call stream(), not chat()."""
        mock_get.return_value = [make_db_task(status="done")]
        client.stream.return_value = iter(["Text."])

        summarize_progress("week", client)

        client.stream.assert_called_once()
        client.chat.assert_not_called()

    @patch("agent.actions.log_ai_action")
    @patch("agent.actions.get_tasks_for_summary")
    def test_log_ai_action_called(self, mock_get, mock_log, client):
        mock_get.return_value = [make_db_task(status="done")]
        client.stream.return_value = iter(["Text."])

        summarize_progress("week", client)

        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args.kwargs
        assert call_kwargs["action"] == "summarize"
