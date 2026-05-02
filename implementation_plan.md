# Refactor LangGraph Tutoring System into Clean MVP

The goal is to refactor `chat.py` into a streamlined, lightweight MVP for the Buddy AI Tutoring system, focusing on clarity, optimal context usage, buffering memory, answer evaluation, and reducing architectural complexity.

## User Review Required

> [!IMPORTANT]
> - **Caching & Tracking**: I propose completely removing the `ResponseCache` and `ProgressTracker` and `ChatExporter` classes to achieve maximum simplifications. Let me know if you prefer to keep a minimal dictionary for caching.
> - **Schema Assumption**: The plan assumes the `chapters` table in Supabase contains a `summary` column. If it doesn't, we can fall back to using `content[:MAX_CONTEXT_LENGTH]` dynamically. Is `summary` an available field in your database?
> - **Routing approach**: I will use a simple keyword-based intent matcher for parsing "summary", "test", "evaluate", and "doubt", as it's the most lightweight standard for MVP routing. Is this acceptable, or would you prefer an LLM to decide the route?

## Proposed Changes

### Core System

#### [MODIFY] [chat.py](file:///d:/Child-Friendly%20AI%20Mentor/chat.py)
- **Remove unnecessary abstractions:**
  - Delete `ResponseCache`, `ChatExporter`, `ProgressTracker`, `safety_check_node`.
- **Implement MemoryBuffer:**
  - Refactor `ConversationMemory` to track only the last 5 turns and yield a tightly compressed representation of past context for prompts.
- **Update LangGraph Architecture:**
  - Reduce Graph to sequential steps: `intent_detection` -> `context_fetch` -> `prompt_builder` -> `llm_call` -> `memory_update`.
  - Use conditional paths or unified logic in `prompt_builder` to support varied query objectives (Doubt, Summary, Test, Evaluate).
- **Implement Clean Routing & Context Strategy:**
  - **summary**: Extracted directly from `chapter["summary"]`.
  - **test**: Provide the chapter summary to the LLM to generate practice questions.
  - **doubt**: Top-3 vector search purely using SentenceTransformers embeddings.
  - **evaluate**: Extracts recent question from Memory + compares the student's current response. Instructs LLM to return structured JSON `{"is_correct", "score", "mistake", "explanation", "correct_answer"}`.
- **Keep Output Child-Friendly:**
  - Standardize system prompts with warm, encouraging instructions. Evaluate prompts will use gentle guidance for corrections.

## Open Questions

1. Should the structure for evaluating answers strictly be returned to the console as a formatted string, or do you want the raw JSON logged to the console?
2. Are there any other specific "evaluate" keyword triggers besides "evaluate", "check", or "my answer is"?

## Verification Plan

### Manual Verification
- Run the python script locally using `python chat.py`.
- Select a dataset and verify the flows:
  - Ask "Can you summarize the chapter?" -> Should route to `summary` and provide bullet points.
  - Ask "Can we take a test?" -> Should route to `test` using summary context.
  - Type an answer "I think the answer is..." -> Should route to `evaluate` and output the correctness evaluation format.
  - Ask a specific question about the content -> Should trigger vector RAG matching.
