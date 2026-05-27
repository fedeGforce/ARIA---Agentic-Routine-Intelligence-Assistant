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
```
---

## CLI Commands

### Project management
```console
python aria.py project add "Learn LangGraph" --category learning
python aria.py project list
```
### Task management — plain structured input
```console
python aria.py task add "Read LangGraph docs" --project 1 --due 2025-06-01 --recurrence weekly
```
### AI-powered task creation from natural language (uses phi3.5)
```console
python aria.py task parse "I need to review the Docker notes every monday and submit a report at the end of the month"
```
### AI actions
```console
python aria.py ai suggest          # Suggests priorities for pending tasks
python aria.py ai warn             # Flags overdue or neglected tasks
python aria.py ai summarize --period day   # Summarizes by 'day', 'week', or 'month'
```

## How the AI Fits In (The Agentic Part)
- Each ai command follows the exact same pattern—this is your first fundamental agent loop:

- Query DB: Get the relevant current task state.

- Build Context: Construct a structured prompt using that state as context.

- Inference: Call phi3.5 via the Ollama API.

- Parse Output: Extract and parse the structured response.

- Persist State: Write the AI's decision back to the database (ai_logs) + optionally update tasks.

- Report: Print the execution summary to the console.

⚠️ The Safety Pattern: The model never touches the database directly. It receives task data as text, reasons about it, and returns structured instructions that your Python code executes. The model proposes, the code disposes.

## API Payload Conventions
For the task parse command, the payload sent to Ollama looks like this. The model is strictly instructed to return clean JSON so your code can act on it deterministically:

### JSON
```console
{
  "model": "phi3.5",
  "messages": [
    {
      "role": "system",
      "content": "You are a task management assistant. Extract tasks from natural language and return ONLY a JSON array. Each task must have: title, recurrence (none/daily/weekly/monthly), suggested_due_date (YYYY-MM-DD or null), priority (1/2/3)."
    },
    {
      "role": "user",
      "content": "I need to review the Docker notes every monday and submit a report at the end of the month"
    }
  ],
  "temperature": 0.2
}
```

Note: Keeping a low temperature (0.2) on structured output tasks is highly intentional—you want deterministic, parseable responses, not creative ones.

## Build Order
To prevent dependencies from fracturing and ensure you always maintain a working system, implement features in this specific, iterative sequence:

1. db/schema.sql + db/database.py — Get the database working first; insert and verify a row manually.

2. models/task.py — Set up simple Python dataclasses (no logic yet).

3. agent/client.py — Build your Ollama API network client wrapper.

4. aria.py — Wire up base CLI CRUD features (project add/list and task add) completely devoid of AI elements.

5. agent/actions.py — Integrate the intelligent agent capabilities one piece at a time: start with parse, scale into suggest, implement warn, and finish with summarize.

This order guarantees you always have a working system at the end of a feature block. You never break everything at once.
