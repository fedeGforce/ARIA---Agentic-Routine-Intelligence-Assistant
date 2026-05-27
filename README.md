# ARIA — Agentic Routine & Intelligence Assistant

ARIA is not just another task manager with an AI wrapper. It is a true **agentic system** designed to actively reason about your task state, make autonomous decisions (such as priority suggestions and deadline warnings), and execute structured actions (writing directly to a local database). Every feature is designed as a modular building block for advanced agentic workflows.

---

## 🧠 Core Concept: The Propose/Dispose Pattern

Unlike naive AI integrations, ARIA separates reasoning from execution to maintain strict safety:
* **The Model Proposes:** The AI receives data strictly as text, reasons about it, and returns clean, structured instructions.
* **The Code Disposes:** The underlying Python application evaluates the AI's proposal, validates it, and writes the changes deterministically to the database. 

> **Safety Architecture:** The language model *never* touches the database directly.

---

## 🗄️ Database Design

The system relies on three clean, normalized tables. A dedicated `ai_logs` table provides a full audit trail of every AI decision—essential for debugging agent behavior.

### 1. projects
Tracks high-level categories and initiatives.
* `id`: `INTEGER PRIMARY KEY AUTOINCREMENT`
* `name`: `TEXT NOT NULL`
* `description`: `TEXT`
* `category`: `TEXT` (e.g., `work`, `personal`, `learning`)
* `created_at`: `DATETIME DEFAULT CURRENT_TIMESTAMP`
* `status`: `TEXT DEFAULT 'active'` (`active` | `archived`)

### 2. tasks
Maintains individual items tied to projects.
* `id`: `INTEGER PRIMARY KEY AUTOINCREMENT`
* `project_id`: `INTEGER FK → projects.id`
* `title`: `TEXT NOT NULL`
* `description`: `TEXT`
* `priority`: `INTEGER` (`1` = low, `2` = medium, `3` = high)
* `recurrence`: `TEXT` (`none` | `daily` | `weekly` | `monthly`)
* `due_date`: `DATE`
* `status`: `TEXT DEFAULT 'pending'` (`pending` | `in_progress` | `done` | `cancelled`)
* `created_at`: `DATETIME DEFAULT CURRENT_TIMESTAMP`
* `updated_at`: `DATETIME DEFAULT CURRENT_TIMESTAMP`

### 3. ai_logs
An audit trail capturing what the model was told, what it responded, and what action it triggered.
* `id`: `INTEGER PRIMARY KEY AUTOINCREMENT`
* `task_id`: `INTEGER FK → tasks.id` (nullable)
* `action`: `TEXT` (`create` | `suggest` | `warn` | `summarize`)
* `prompt`: `TEXT`
* `response`: `TEXT`
* `created_at`: `DATETIME DEFAULT CURRENT_TIMESTAMP`

---

## 📂 Project Structure

The project layout is strictly modular. The agent layer handles LLM interactions, the database layer isolates storage operations, and the main CLI script wires them together. This isolation makes it trivial to swap a local Ollama model for a cloud-hosted API later.

```text
aria/
├── aria.py              # CLI entrypoint — all console commands live here
├── db/
│   ├── schema.sql       # Table definitions
│   └── database.py      # DB connection + query helpers
├── agent/
│   ├── client.py        # Ollama API wrapper (OpenAI-compatible)
│   └── actions.py       # AI actions: parse, suggest, warn, summarize
├── models/
│   └── task.py          # Task + Project dataclasses
├── requirements.txt
└── README.md
