-- =============================================================
-- ARIA — Agentic Routine & Intelligence Assistant
-- Database Schema
-- =============================================================
-- SQLite is used for local-first development.
-- All timestamps are stored in UTC ISO-8601 format.
-- Foreign keys are enforced — make sure to enable them at
-- connection time with: PRAGMA foreign_keys = ON;
-- =============================================================


-- -------------------------------------------------------------
-- TABLE: projects
-- A project is the top-level container for related tasks.
-- It belongs to a category (work, personal, learning, etc.)
-- and can be archived when no longer active.
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    description TEXT,

    -- Free-form category label: 'work', 'personal', 'learning', etc.
    -- Stored as text rather than an enum to keep the schema flexible.
    category    TEXT    NOT NULL DEFAULT 'general',

    -- 'active' | 'archived'
    -- Archived projects are hidden from default views but never deleted,
    -- preserving the full history of completed work.
    status      TEXT    NOT NULL DEFAULT 'active',

    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- -------------------------------------------------------------
-- TABLE: tasks
-- A task is the atomic unit of work. Every task belongs to
-- exactly one project. Recurrence describes how often the
-- task should reappear once completed.
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Every task must belong to a project.
    -- ON DELETE CASCADE: if a project is deleted, its tasks go with it.
    project_id  INTEGER NOT NULL,

    title       TEXT    NOT NULL,
    description TEXT,

    -- Priority: 1 = low, 2 = medium, 3 = high.
    -- Stored as integer so it can be sorted and compared numerically.
    -- The AI suggestion action will populate this if left null.
    priority    INTEGER CHECK(priority IN (1, 2, 3)),

    -- Recurrence pattern for repeating tasks.
    -- 'none'    → one-off task, no repetition
    -- 'daily'   → resets every day
    -- 'weekly'  → resets every week
    -- 'monthly' → resets every month
    recurrence  TEXT    NOT NULL DEFAULT 'none'
                CHECK(recurrence IN ('none', 'daily', 'weekly', 'monthly')),

    -- Target completion date. Null means no deadline (open-ended task).
    due_date    DATE,

    -- Lifecycle status of the task.
    -- 'pending'     → not started yet
    -- 'in_progress' → actively being worked on
    -- 'done'        → completed
    -- 'cancelled'   → dropped, not completed
    status      TEXT    NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'in_progress', 'done', 'cancelled')),

    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Enforce referential integrity: task must point to a real project.
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);


-- -------------------------------------------------------------
-- TABLE: ai_logs
-- Every AI action is recorded here: what context was sent to
-- the model, what it responded, and what action it triggered.
--
-- This is your agent's audit trail. It lets you:
--   - Debug unexpected AI behavior
--   - Review what the model suggested vs what you accepted
--   - Build a training dataset for future fine-tuning
--
-- task_id is nullable because some actions (e.g. summarize,
-- warn) operate across many tasks, not a single one.
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Nullable: links to a specific task when the action is task-scoped.
    -- NULL when the action spans multiple tasks (summarize, warn, etc.)
    task_id     INTEGER,

    -- The type of AI action that was performed.
    -- 'create'    → natural language → task creation
    -- 'suggest'   → AI suggested priority/deadline for existing tasks
    -- 'warn'      → AI flagged overdue or neglected tasks
    -- 'summarize' → AI generated a progress summary
    action      TEXT    NOT NULL
                CHECK(action IN ('create', 'suggest', 'warn', 'summarize')),

    -- The full prompt that was sent to the model.
    -- Storing this is critical for debugging — you can replay any
    -- interaction exactly as it happened.
    prompt      TEXT    NOT NULL,

    -- The raw response returned by the model.
    response    TEXT    NOT NULL,

    -- Which model was used (e.g. 'phi3.5', 'llama3.2:3b').
    -- Useful if you switch models over time and want to compare behavior.
    model_used  TEXT    NOT NULL DEFAULT 'phi3.5',

    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE SET NULL
);


-- =============================================================
-- INDEXES
-- SQLite doesn't auto-index foreign keys, so we add them
-- manually. These make filtering by project, status, and
-- due_date significantly faster as your task list grows.
-- =============================================================

-- Fast lookup of all tasks for a given project
CREATE INDEX IF NOT EXISTS idx_tasks_project_id
    ON tasks(project_id);

-- Fast filtering by status (e.g. all pending tasks)
CREATE INDEX IF NOT EXISTS idx_tasks_status
    ON tasks(status);

-- Fast filtering by due_date (used by the 'warn' action)
CREATE INDEX IF NOT EXISTS idx_tasks_due_date
    ON tasks(due_date);

-- Fast lookup of all AI logs for a given task
CREATE INDEX IF NOT EXISTS idx_ai_logs_task_id
    ON ai_logs(task_id);

-- Fast filtering of AI logs by action type
CREATE INDEX IF NOT EXISTS idx_ai_logs_action
    ON ai_logs(action);