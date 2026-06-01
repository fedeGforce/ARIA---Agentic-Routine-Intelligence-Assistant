# =============================================================
# aria/agent/actions.py
# =============================================================
# The four AI actions that ARIA can perform. This is where all
# three layers meet for the first time:
#
#   DB layer    → fetch task state as context for the model
#   Client layer → send prompts, receive structured responses
#   Models layer → parse raw DB rows into typed objects
#
# Every action follows the same five-step pattern:
#   1. Query DB  → get the relevant task state
#   2. Serialize → format that state as text for the prompt
#   3. Call LLM  → send prompt via OllamaClient
#   4. Parse     → extract structured data from the response
#   5. Persist   → write AI decision to ai_logs (+ optionally
#                  update tasks)
#
# Design rule: the model never touches the DB directly.
# It receives data as text, reasons about it, and returns
# structured JSON. Python decides what to do with that JSON.
# This is the core safety pattern in agentic systems.
# =============================================================

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta

from agent.client import (
    TEMPERATURE_BALANCED,
    TEMPERATURE_CREATIVE,
    TEMPERATURE_PRECISE,
    OllamaClient,
)
from db.database import (
    create_task,
    get_overdue_tasks,
    get_tasks_for_summary,
    list_tasks,
    log_ai_action,
    update_task_priority,
)
from models.task import AiAction, Recurrence

# =============================================================
# RESULT DATACLASSES
# Each action returns a typed result object — never a raw dict
# or string. The CLI layer only needs to know about these types,
# not about JSON, prompts, or DB internals.
# =============================================================


@dataclass
class ParsedTask:
    """
    One task extracted from a natural language string.
    The CLI will ask the user to confirm before persisting.
    """

    title: str
    recurrence: str  # Recurrence enum value
    suggested_due: str | None  # 'YYYY-MM-DD' or None
    priority: int  # 1 / 2 / 3
    created_id: int | None = None  # set after DB insert


@dataclass
class PrioritySuggestion:
    """One priority suggestion produced by the 'suggest' action."""

    task_id: int
    title: str
    old_priority: int | None
    new_priority: int
    reason: str
    applied: bool = False  # set to True after DB update


@dataclass
class OverdueWarning:
    """One overdue warning produced by the 'warn' action."""

    task_id: int
    title: str
    due_date: str
    days_overdue: int
    urgency: str  # 'low' | 'medium' | 'high'
    suggestion: str  # what the model recommends


@dataclass
class ProgressSummary:
    """Summary produced by the 'summarize' action."""

    period: str  # 'day' | 'week' | 'month'
    summary_text: str  # narrative written by the model
    done_count: int
    pending_count: int
    in_progress_count: int


# =============================================================
# INTERNAL HELPERS
# =============================================================


def _serialize_tasks(tasks) -> str:
    """
    Convert a list of sqlite3.Row / Task objects into a compact
    JSON string suitable for embedding in a prompt.

    We only include the fields the model actually needs to reason
    about — title, status, priority, due_date, recurrence.
    Sending the full row wastes tokens and can confuse the model.
    """
    serialized = []
    for t in tasks:
        serialized.append(
            {
                "id": t["id"],
                "title": t["title"],
                "status": t["status"],
                "priority": t["priority"],
                "due_date": t["due_date"],
                "recurrence": t["recurrence"],
            }
        )
    return json.dumps(serialized, indent=2)


def _extract_json(text: str) -> str:
    """
    Pull the first JSON array or object out of a model response.

    Models sometimes wrap JSON in markdown fences (```json ... ```)
    or add a preamble sentence before the JSON. This function
    strips that noise so json.loads() can parse the result cleanly.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find the first '[' or '{' and the matching closing bracket
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start != -1:
            # Walk forward to find the balanced closing bracket
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]

    return text  # Return as-is and let json.loads() raise a useful error


def _today_plus(days: int) -> str:
    """Return a YYYY-MM-DD date string offset from today."""
    return (date.today() + timedelta(days=days)).isoformat()


# =============================================================
# ACTION 1 — PARSE
# Convert a natural language string into one or more tasks.
# =============================================================

# System prompt for the parse action.
# Low temperature (PRECISE) is used: we need deterministic JSON,
# not creative variation.
_PARSE_SYSTEM = """You are a task extraction assistant for a personal productivity system.
Your job is to parse a natural language description and extract one or more tasks from it.

Rules:
- Return ONLY a valid JSON array. No preamble, no explanation, no markdown.
- Each task must have exactly these fields:
    "title"         : short, action-oriented task name (string)
    "recurrence"    : one of "none" | "daily" | "weekly" | "monthly"
    "suggested_due" : ISO date "YYYY-MM-DD" based on recurrence, or null if unclear
    "priority"      : integer 1 (low) | 2 (medium) | 3 (high)
- Infer recurrence from words like "every monday", "daily", "each month".
- Infer priority from urgency words like "urgent", "important", "whenever".
- Today's date is {today}. Use it to compute suggested_due dates.
- If multiple tasks are described, return one object per task.

Example output:
[
  {{"title": "Review Docker notes",
  "recurrence": "weekly",
  "suggested_due": "{next_monday}",
  "priority": 2}},
  {{"title": "Submit monthly report",
  "recurrence": "monthly",
  "suggested_due": "{end_of_month}",
  "priority": 3}}]"""


def parse_tasks(
    natural_language: str,
    project_id: int,
    client: OllamaClient,
) -> list[ParsedTask]:
    """
    Parse a natural language string into structured tasks and
    persist them to the DB.
    Args:
        natural_language: Free-form text describing one or more tasks.
        project_id:       The project these tasks will belong to.
        client:           An initialised OllamaClient.
    Returns:
        List of ParsedTask objects, each with created_id set after
        successful DB insertion.
    """
    today = date.today().isoformat()
    next_monday = _today_plus((7 - date.today().weekday()) % 7 or 7)
    end_of_month = date.today().replace(day=28).isoformat()  # safe proxy

    system_prompt = _PARSE_SYSTEM.format(
        today=today,
        next_monday=next_monday,
        end_of_month=end_of_month,
    )

    messages = [
        client.system(system_prompt),
        client.user(natural_language),
    ]

    response = client.chat(messages, temperature=TEMPERATURE_PRECISE)
    raw_json = _extract_json(response.content)

    # Log the interaction before attempting to parse — we always
    # want a record even if parsing later fails.
    log_ai_action(
        action=AiAction.CREATE,
        prompt=f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{natural_language}",
        response=response.content,
        model_used=client.model,
    )

    parsed = json.loads(raw_json)
    results: list[ParsedTask] = []

    for item in parsed:
        # Validate and clamp priority to 1–3
        raw_priority = int(item.get("priority", 2))
        priority = max(1, min(3, raw_priority))

        # Validate recurrence against allowed values
        raw_recurrence = item.get("recurrence", "none")
        recurrence = raw_recurrence if raw_recurrence in {r.value for r in Recurrence} else "none"

        task = ParsedTask(
            title=item["title"],
            recurrence=recurrence,
            suggested_due=item.get("suggested_due"),
            priority=priority,
        )

        # Persist to DB immediately
        task.created_id = create_task(
            project_id=project_id,
            title=task.title,
            priority=task.priority,
            recurrence=task.recurrence,
            due_date=task.suggested_due,
        )

        results.append(task)

    return results


# =============================================================
# ACTION 2 — SUGGEST
# Recommend priority levels for tasks that have none, or whose
# priority the model thinks is mis-calibrated.
# =============================================================

_SUGGEST_SYSTEM = """You are a priority assessment assistant for a personal task management system.
You will receive a JSON list of tasks. For each task that has no priority (null) or where
the priority seems wrong given the title and due date, suggest a new priority.

Rules:
- Return ONLY a valid JSON array. No preamble, no explanation, no markdown.
- Only include tasks where you are suggesting a change. Skip tasks that look correct.
- Each suggestion must have exactly these fields:
    "task_id"      : integer (copy from input)
    "new_priority" : integer 1 (low) | 2 (medium) | 3 (high)
    "reason"       : one concise sentence explaining why
- Today's date is {today}.
- Treat tasks due within 3 days as at least priority 2.
- Treat tasks due within 1 day or overdue as priority 3.
- Tasks with no due date and vague titles default to priority 1.
Example output:
[
  {{"task_id": 4, "new_priority": 3, "reason": "Due tomorrow with no priority set."}},
  {{"task_id": 7, "new_priority": 1, "reason": "Open-ended reference task with no deadline."}}]"""


def suggest_priorities(
    client: OllamaClient,
    project_id: int | None = None,
    auto_apply: bool = False,
) -> list[PrioritySuggestion]:
    """
    Ask the model to review pending tasks and suggest priorities.
    Args:
        client:     An initialised OllamaClient.
        project_id: If set, only consider tasks from this project.
                    If None, consider all pending tasks.
        auto_apply: If True, write the suggested priorities to the DB
                    immediately. If False, return suggestions only
                    and let the CLI ask the user to confirm.
    Returns:
        List of PrioritySuggestion objects.
    """
    tasks = list_tasks(project_id=project_id, status="pending")
    if not tasks:
        return []

    today = date.today().isoformat()
    tasks_json = _serialize_tasks(tasks)
    system_prompt = _SUGGEST_SYSTEM.format(today=today)

    messages = [
        client.system(system_prompt),
        client.user(f"Here are the current pending tasks:\n{tasks_json}"),
    ]

    response = client.chat(messages, temperature=TEMPERATURE_BALANCED)
    raw_json = _extract_json(response.content)

    log_ai_action(
        action=AiAction.SUGGEST,
        prompt=f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{tasks_json}",
        response=response.content,
        model_used=client.model,
    )

    parsed = json.loads(raw_json)

    # Build a quick lookup of task data by id
    task_by_id = {t["id"]: t for t in tasks}
    results: list[PrioritySuggestion] = []

    for item in parsed:
        task_id = int(item["task_id"])
        new_priority = max(1, min(3, int(item["new_priority"])))
        original = task_by_id.get(task_id)

        if original is None:
            continue  # model hallucinated a task_id — skip silently

        suggestion = PrioritySuggestion(
            task_id=task_id,
            title=original["title"],
            old_priority=original["priority"],
            new_priority=new_priority,
            reason=item.get("reason", ""),
        )

        if auto_apply:
            update_task_priority(task_id, new_priority)
            suggestion.applied = True

        # Log per-task so the audit trail links to specific tasks
        log_ai_action(
            action=AiAction.SUGGEST,
            prompt=f"task_id={task_id}",
            response=json.dumps(item),
            task_id=task_id,
            model_used=client.model,
        )

        results.append(suggestion)

    return results


# =============================================================
# ACTION 3 — WARN
# Detect overdue or neglected tasks and explain the urgency.
# =============================================================

_WARN_SYSTEM = """You are an accountability assistant for a personal task management system.
You will receive a JSON list of overdue tasks. For each task, assess the urgency and
provide a concrete recommendation.
Rules:
- Return ONLY a valid JSON array. No preamble, no explanation, no markdown.
- Each warning must have exactly these fields:
    "task_id"    : integer (copy from input)
    "urgency"    : one of "low" | "medium" | "high"
    "suggestion" : one concrete actionable sentence (what to do right now)
- Assign urgency based on days overdue:
    0–3 days  → "low"
    4–7 days  → "medium"
    8+ days   → "high"
- Keep suggestions specific and direct. Avoid vague advice like "address this soon".
- Today's date is {today}.
Example output:
[
  {{"task_id": 2, "urgency": "high",
  "suggestion": "Reschedule or cancel — this is 12 days overdue and blocking other work."}},
  {{"task_id": 5, "urgency": "low",
  "suggestion": "Complete today during the next available 30-minute block."}}]"""


def warn_overdue(client: OllamaClient) -> list[OverdueWarning]:
    """
    Find all overdue tasks and ask the model to assess urgency
    and suggest concrete next actions.
    Args:
        client: An initialised OllamaClient.
    Returns:
        List of OverdueWarning objects, sorted by days_overdue desc.
        Returns empty list if no overdue tasks exist.
    """
    tasks = get_overdue_tasks()
    if not tasks:
        return []

    today = date.today()
    tasks_json = _serialize_tasks(tasks)
    system_prompt = _WARN_SYSTEM.format(today=today.isoformat())

    messages = [
        client.system(system_prompt),
        client.user(f"These tasks are overdue:\n{tasks_json}"),
    ]

    response = client.chat(messages, temperature=TEMPERATURE_BALANCED)
    raw_json = _extract_json(response.content)

    log_ai_action(
        action=AiAction.WARN,
        prompt=f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{tasks_json}",
        response=response.content,
        model_used=client.model,
    )

    parsed = json.loads(raw_json)
    task_by_id = {t["id"]: t for t in tasks}
    results: list[OverdueWarning] = []

    for item in parsed:
        task_id = int(item["task_id"])
        original = task_by_id.get(task_id)

        if original is None:
            continue

        # Compute days_overdue from the actual due_date in the DB
        due = date.fromisoformat(original["due_date"])
        overdue = (today - due).days

        # Validate urgency — default to 'medium' if model returns garbage
        urgency = item.get("urgency", "medium")
        if urgency not in ("low", "medium", "high"):
            urgency = "medium"

        results.append(
            OverdueWarning(
                task_id=task_id,
                title=original["title"],
                due_date=original["due_date"],
                days_overdue=overdue,
                urgency=urgency,
                suggestion=item.get("suggestion", ""),
            )
        )

        log_ai_action(
            action=AiAction.WARN,
            prompt=f"task_id={task_id}",
            response=json.dumps(item),
            task_id=task_id,
            model_used=client.model,
        )

    # Sort by urgency so the most critical tasks appear first
    urgency_order = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda w: urgency_order.get(w.urgency, 1))

    return results


# =============================================================
# ACTION 4 — SUMMARIZE
# Generate a narrative progress summary for a time period.
# =============================================================

_SUMMARIZE_SYSTEM = """You are a productivity coach writing a progress summary for a personal
task management system. You will receive a JSON list of tasks that were active during the
requested period. Write a concise, honest, and encouraging summary of the work done.
Rules:
- Write in plain prose. No bullet points, no markdown headers.
- Be specific: mention actual task titles when relevant.
- Note what was completed, what is still in progress, and flag anything that looks stuck.
- Keep the tone constructive — acknowledge effort, note gaps without shame.
- Length: 3–5 sentences. No more.
- End with one concrete suggestion for the next period.
- Today's date is {today}. The summary covers the last {period}.
"""


def summarize_progress(
    period: str,
    client: OllamaClient,
    project_id: int | None = None,
) -> ProgressSummary:
    """
    Generate a narrative progress summary for the given period.
    Args:
        period:     'day' | 'week' | 'month'
        client:     An initialised OllamaClient.
        project_id: If set, summarize only tasks from this project.
    Returns:
        A ProgressSummary with the narrative text and task counts.
    """
    tasks = get_tasks_for_summary(period)

    # Filter by project if requested
    if project_id is not None:
        tasks = [t for t in tasks if t["project_id"] == project_id]

    # Compute counts before sending to model — these come from the
    # DB directly, so they are always accurate regardless of what
    # the model says.
    done_count = sum(1 for t in tasks if t["status"] == "done")
    pending_count = sum(1 for t in tasks if t["status"] == "pending")
    in_progress_count = sum(1 for t in tasks if t["status"] == "in_progress")

    if not tasks:
        return ProgressSummary(
            period=period,
            summary_text=f"No tasks were active during this {period}.",
            done_count=0,
            pending_count=0,
            in_progress_count=0,
        )

    today = date.today().isoformat()
    tasks_json = _serialize_tasks(tasks)
    system_prompt = _SUMMARIZE_SYSTEM.format(today=today, period=period)

    messages = [
        client.system(system_prompt),
        client.user(
            f"Here are the tasks active during this {period}:\n{tasks_json}\n\n"
            f"Completed: {done_count}  |  \
                In progress: {in_progress_count}  | \
                Pending: {pending_count}"
        ),
    ]

    # Use streaming for the summary — it's the most user-facing
    # response in the system and benefits from live output.
    summary_text = "".join(client.stream(messages, temperature=TEMPERATURE_CREATIVE))

    log_ai_action(
        action=AiAction.SUMMARIZE,
        prompt=f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{tasks_json}",
        response=summary_text,
        model_used=client.model,
    )

    return ProgressSummary(
        period=period,
        summary_text=summary_text.strip(),
        done_count=done_count,
        pending_count=pending_count,
        in_progress_count=in_progress_count,
    )
