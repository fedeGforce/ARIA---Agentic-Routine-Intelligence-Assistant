#!/usr/bin/env python3
# =============================================================
# aria/aria.py
# =============================================================
# The CLI entrypoint. Every console command lives here.
#
# Commands:
#   project add  <name> [--description] [--category]
#   project list [--status]
#   project archive <id>
#
#   task add     <title> --project <id> [--description]
#                [--priority] [--due] [--recurrence]
#   task list    [--project] [--status]
#   task done    <id>
#   task status  <id> <status>
#   task parse   <text> --project <id>
#
#   ai suggest   [--project] [--apply]
#   ai warn
#   ai summarize [--period] [--project]
#
# Usage:
#   python aria.py --help
#   python aria.py project add "Learn LangGraph" --category learning
#   python aria.py task parse "Review notes every monday" --project 1
#   python aria.py ai summarize --period week
# =============================================================

import sys

import click

from agent.actions import parse_tasks, suggest_priorities, summarize_progress, warn_overdue
from agent.client import OllamaClient, OllamaClientError
from db.database import (
    archive_project,
    create_project,
    create_task,
    get_project,
    get_task,
    init_db,
    list_projects,
    list_tasks,
    update_task_status,
)

# =============================================================
# ANSI COLOUR HELPERS
# Keep output readable without adding a heavy dependency.
# All colour calls go through these functions so disabling
# them later (e.g. for CI) is a one-line change.
# =============================================================


def _c(text: str, code: str) -> str:
    """Wrap text in an ANSI colour code."""
    return f"\033[{code}m{text}\033[0m"


def green(t):
    return _c(t, "32")


def yellow(t):
    return _c(t, "33")


def red(t):
    return _c(t, "31")


def bold(t):
    return _c(t, "1")


def dim(t):
    return _c(t, "2")


def cyan(t):
    return _c(t, "36")


# =============================================================
# PRIORITY / STATUS DISPLAY HELPERS
# =============================================================

_PRIORITY_LABEL = {
    1: dim("● low"),
    2: yellow("● med"),
    3: red("● high"),
    None: dim("○ none"),
}

_STATUS_LABEL = {
    "pending": dim("pending"),
    "in_progress": cyan("in progress"),
    "done": green("done"),
    "cancelled": dim("cancelled"),
}

_URGENCY_COLOUR = {
    "low": yellow,
    "medium": yellow,
    "high": red,
}


# =============================================================
# SHARED CLIENT FACTORY
# Passed as a Click context object so every subcommand that
# needs the LLM gets the same configured instance without
# re-initialising it.
# =============================================================


def _make_client() -> OllamaClient:
    return OllamaClient()


# =============================================================
# ROOT GROUP
# =============================================================


@click.group()
@click.pass_context
def cli(ctx):
    """
    ARIA — Agentic Routine & Intelligence Assistant

    A local-first task manager powered by a local LLM via Ollama.
    Run any command with --help for usage details.
    """
    # Initialise the DB on every invocation — safe because
    # init_db() uses CREATE IF NOT EXISTS and is a no-op on
    # an already-initialised database.
    init_db()
    ctx.ensure_object(dict)


# =============================================================
# PROJECT COMMANDS
# =============================================================


@cli.group()
def project():
    """Manage projects."""
    pass


@project.command("add")
@click.argument("name")
@click.option("--description", "-d", default="", help="Optional project description.")
@click.option(
    "--category",
    "-c",
    default="general",
    help="Category label: work | personal | learning | general (default).",
)
def project_add(name, description, category):
    """
    Create a new project.

    \b
    Examples:
      python aria.py project add "Learn LangGraph"
      python aria.py project add "Q3 Deliverables" --category work --description "All Q3 tasks"
    """
    pid = create_project(name, description, category)
    click.echo(f"{green('✓')} Project created  {bold(f'[{pid}]')} {name}  {dim(f'({category})')}")


@project.command("list")
@click.option(
    "--status",
    "-s",
    default="active",
    type=click.Choice(["active", "archived", "all"], case_sensitive=False),
    help="Filter by status (default: active).",
)
def project_list(status):
    """
    List projects.

    \b
    Examples:
      python aria.py project list
      python aria.py project list --status all
    """
    rows = list_projects(status=status)
    if not rows:
        click.echo(dim(f"No {status} projects found."))
        return

    click.echo(f"\n{bold('Projects')}  {dim(f'({status})')}\n")
    click.echo(f"  {dim('ID'):<6} {dim('NAME'):<30} {dim('CATEGORY'):<14} {dim('STATUS')}")
    click.echo(f"  {dim('─' * 58)}")
    for r in rows:
        status_label = green(r["status"]) if r["status"] == "active" else dim(r["status"])
        click.echo(
            f"  {bold(str(r['id'])):<6} {r['name']:<30} {dim(r['category']):<14} {status_label}"
        )
    click.echo()


@project.command("archive")
@click.argument("project_id", type=int)
def project_archive(project_id):
    """
    Archive a project (hides it from default views).

    \b
    Example:
      python aria.py project archive 3
    """
    row = get_project(project_id)
    if not row:
        click.echo(red(f"✗ Project {project_id} not found."))
        sys.exit(1)

    archive_project(project_id)
    click.echo(f"{yellow('◉')} Project archived  {bold(f'[{project_id}]')} {row['name']}")


# =============================================================
# TASK COMMANDS
# =============================================================


@cli.group()
def task():
    """Manage tasks."""
    pass


@task.command("add")
@click.argument("title")
@click.option("--project", "-p", "project_id", required=True, type=int, help="Project ID.")
@click.option("--description", "-d", default="", help="Optional task description.")
@click.option(
    "--priority",
    "-P",
    default=None,
    type=click.Choice(["1", "2", "3"]),
    help="Priority: 1=low  2=medium  3=high.",
)
@click.option("--due", "-D", default=None, help="Due date in YYYY-MM-DD format.")
@click.option(
    "--recurrence",
    "-r",
    default="none",
    type=click.Choice(["none", "daily", "weekly", "monthly"], case_sensitive=False),
    help="Recurrence pattern (default: none).",
)
def task_add(title, project_id, description, priority, due, recurrence):
    """
    Add a task to a project.

    \b
    Examples:
      python aria.py task add "Read LangGraph docs" --project 1
      python aria.py task add "Weekly review" --project 1 --priority 2 --recurrence weekly
      python aria.py task add "Deploy service" --project 1 --priority 3 --due 2025-06-15
    """
    proj = get_project(project_id)
    if not proj:
        click.echo(red(f"✗ Project {project_id} not found."))
        sys.exit(1)

    tid = create_task(
        project_id=project_id,
        title=title,
        description=description,
        priority=int(priority) if priority else None,
        recurrence=recurrence,
        due_date=due,
    )

    priority_label = _PRIORITY_LABEL.get(int(priority) if priority else None)
    click.echo(
        f"{green('✓')} Task created  {bold(f'[{tid}]')} {title}  "
        f"{priority_label}  "
        f"{dim('→ ' + proj['name'])}"
    )


@task.command("list")
@click.option("--project", "-p", "project_id", default=None, type=int, help="Filter by project ID.")
@click.option(
    "--status",
    "-s",
    default=None,
    type=click.Choice(["pending", "in_progress", "done", "cancelled"], case_sensitive=False),
    help="Filter by status.",
)
def task_list(project_id, status):
    """
    List tasks, optionally filtered by project and/or status.

    \b
    Examples:
      python aria.py task list
      python aria.py task list --project 1
      python aria.py task list --status pending
      python aria.py task list --project 1 --status in_progress
    """
    rows = list_tasks(project_id=project_id, status=status)
    if not rows:
        click.echo(dim("No tasks found."))
        return

    click.echo(f"\n{bold('Tasks')}\n")
    click.echo(
        f"  {dim('ID'):<6} {dim('PRI'):<12} {dim('STATUS'):<18} {dim('DUE'):<13} {dim('TITLE')}"
    )
    click.echo(f"  {dim('─' * 72)}")
    for r in rows:
        pri = _PRIORITY_LABEL.get(r["priority"], dim("○ none"))
        status = _STATUS_LABEL.get(r["status"], r["status"])
        due = r["due_date"] or dim("—")
        proj = dim(f'[{r["project_name"]}]')
        click.echo(
            f"  {bold(str(r['id'])):<6} {pri:<12} {status:<18} {due:<13} {r['title']}  {proj}"
        )
    click.echo()


@task.command("done")
@click.argument("task_id", type=int)
def task_done(task_id):
    """
    Mark a task as done.

    \b
    Example:
      python aria.py task done 5
    """
    row = get_task(task_id)
    if not row:
        click.echo(red(f"✗ Task {task_id} not found."))
        sys.exit(1)

    update_task_status(task_id, "done")
    click.echo(f"{green('✓')} Done  {bold(f'[{task_id}]')} {row['title']}")


@task.command("status")
@click.argument("task_id", type=int)
@click.argument(
    "new_status",
    type=click.Choice(["pending", "in_progress", "done", "cancelled"], case_sensitive=False),
)
def task_status(task_id, new_status):
    """
    Set a task's status explicitly.

    \b
    Example:
      python aria.py task status 3 in_progress
      python aria.py task status 3 cancelled
    """
    row = get_task(task_id)
    if not row:
        click.echo(red(f"✗ Task {task_id} not found."))
        sys.exit(1)

    update_task_status(task_id, new_status)
    label = _STATUS_LABEL.get(new_status, new_status)
    click.echo(f"{green('✓')} Status updated  {bold(f'[{task_id}]')} {row['title']}  →  {label}")


@task.command("parse")
@click.argument("text")
@click.option("--project", "-p", "project_id", required=True, type=int, help="Project ID.")
def task_parse(text, project_id):
    """
    Create tasks from natural language using the local LLM.

    The model extracts tasks, recurrence, due dates, and priorities
    from the free-form text and persists them to the database.

    \b
    Examples:
      python aria.py task parse "Review Docker notes every monday" --project 1
      python aria.py task parse "Submit report end of month and prep slides next friday" --project 2
    """
    proj = get_project(project_id)
    if not proj:
        click.echo(red(f"✗ Project {project_id} not found."))
        sys.exit(1)

    click.echo(f"\n{bold('Parsing tasks')}  {dim('→ ' + proj['name'])}\n")
    click.echo(dim(f'  "{text}"\n'))

    try:
        client = _make_client()
        results = parse_tasks(text, project_id=project_id, client=client)
    except OllamaClientError as e:
        click.echo(red(f"\n✗ Could not reach Ollama: {e}"))
        click.echo(dim("  Is Ollama running?  Try: ollama serve"))
        sys.exit(1)

    for t in results:
        pri = _PRIORITY_LABEL.get(t.priority)
        rec = dim(f"↻ {t.recurrence}") if t.recurrence != "none" else ""
        due = dim(f"due {t.suggested_due}") if t.suggested_due else ""
        click.echo(f"  {green('✓')} {bold(f'[{t.created_id}]')} {t.title}  " f"{pri}  {rec}  {due}")

    click.echo(f"\n  {bold(str(len(results)))} task(s) created.\n")


# =============================================================
# AI COMMANDS
# =============================================================


@cli.group()
def ai():
    """AI-powered task analysis and suggestions."""
    pass


@ai.command("suggest")
@click.option("--project", "-p", "project_id", default=None, type=int, help="Limit to one project.")
@click.option(
    "--apply",
    "-a",
    is_flag=True,
    default=False,
    help="Automatically apply suggestions to the database.",
)
def ai_suggest(project_id, apply):
    """
    Ask the LLM to suggest priorities for pending tasks.

    By default, suggestions are displayed only. Pass --apply to
    write them to the database immediately.

    \b
    Examples:
      python aria.py ai suggest
      python aria.py ai suggest --project 1 --apply
    """
    click.echo(f"\n{bold('Priority suggestions')}\n")

    try:
        client = _make_client()
        suggestions = suggest_priorities(client, project_id=project_id, auto_apply=apply)
    except OllamaClientError as e:
        click.echo(red(f"\n✗ Could not reach Ollama: {e}"))
        click.echo(dim("  Is Ollama running?  Try: ollama serve"))
        sys.exit(1)

    if not suggestions:
        click.echo(dim("  No suggestions — all pending tasks already look well-prioritised."))
        click.echo()
        return

    for s in suggestions:
        old_label = _PRIORITY_LABEL.get(s.old_priority)
        new_label = _PRIORITY_LABEL.get(s.new_priority)
        applied = green("applied") if s.applied else dim("preview")
        click.echo(
            f"  {bold(f'[{s.task_id}]')} {s.title}\n"
            f"       {old_label}  →  {new_label}  {dim('·')} \
                     {dim(s.reason)}  {dim(f'[{applied}]')}\n"
        )

    if not apply:
        click.echo(dim("  Re-run with --apply to write these changes to the database.\n"))


@ai.command("warn")
def ai_warn():
    """
    Detect overdue tasks and get concrete recommendations.

    \b
    Example:
      python aria.py ai warn
    """
    click.echo(f"\n{bold('Overdue task warnings')}\n")

    try:
        client = _make_client()
        warnings = warn_overdue(client)
    except OllamaClientError as e:
        click.echo(red(f"\n✗ Could not reach Ollama: {e}"))
        click.echo(dim("  Is Ollama running?  Try: ollama serve"))
        sys.exit(1)

    if not warnings:
        click.echo(green("  ✓ No overdue tasks. Good work.\n"))
        return

    for w in warnings:
        colour = _URGENCY_COLOUR.get(w.urgency, yellow)
        urgency = colour(f"[{w.urgency.upper()}]")
        click.echo(
            f"  {urgency} {bold(f'[{w.task_id}]')} {w.title}\n"
            f"  {dim(f'Due: {w.due_date}  ·  \
                {w.days_overdue} days overdue')}\n"
            f"  {w.suggestion}\n"
        )


@ai.command("summarize")
@click.option(
    "--period",
    "-p",
    default="week",
    type=click.Choice(["day", "week", "month"], case_sensitive=False),
    help="Time window for the summary (default: week).",
)
@click.option(
    "--project", "-proj", "project_id", default=None, type=int, help="Limit to one project."
)
def ai_summarize(period, project_id):
    """
    Generate a progress summary for a time period.

    The model streams the summary live to the terminal.

    \b
    Examples:
      python aria.py ai summarize
      python aria.py ai summarize --period day
      python aria.py ai summarize --period month --project 1
    """
    period_label = {"day": "Today", "week": "This week", "month": "This month"}[period]
    click.echo(f"\n{bold(period_label + ' — Progress Summary')}\n")

    try:
        client = _make_client()

        # We stream the summary text live rather than waiting for
        # the full response, so we bypass the normal action return
        # value and let the model print directly to stdout.
        # We still call summarize_progress to get the counts.
        result = summarize_progress(period=period, client=client, project_id=project_id)

    except OllamaClientError as e:
        click.echo(red(f"\n✗ Could not reach Ollama: {e}"))
        click.echo(dim("  Is Ollama running?  Try: ollama serve"))
        sys.exit(1)

    # Print narrative
    click.echo(f"  {result.summary_text}\n")

    # Print counts footer
    click.echo(
        f"  {dim('Done:')} {green(str(result.done_count))}  "
        f"{dim('In progress:')} {cyan(str(result.in_progress_count))}  "
        f"{dim('Pending:')} {yellow(str(result.pending_count))}\n"
    )


# =============================================================
# ENTRYPOINT
# =============================================================

if __name__ == "__main__":
    cli()
