# Almond

> *A memory-first AI companion that remembers your story, not just your prompts.*

Almond is an experimental local-first AI companion designed around one simple idea:

**Conversations shouldn't reset every time you open the app.**

Instead of treating every interaction as an isolated chat, Almond builds a persistent memory of what matters—your projects, goals, interests, preferences, promises, and personal journey—allowing every conversation to feel like a continuation rather than a fresh start.

---

## Why Almond?

Most AI assistants are incredibly capable.

Most are also incredibly forgetful.

Almond explores a different direction:

- Long-term memory instead of short-term context.
- Relationships instead of chat sessions.
- Continuity instead of repetition.
- Reflection instead of conversation history.

The goal isn't to replace human relationships.

The goal is to create a space where someone can think out loud without worrying about being forgotten.

---

## Current Features

- Persistent multi-layer memory architecture
- Memory retrieval during conversations
- Semantic memory search
- Timeline of remembered events
- Explorer for memories, projects and entities
- Reflection generation
- Local-first architecture
- LM Studio / OpenAI-compatible backend support
- Desktop application built with React + Tauri

---

## Architecture

```
Desktop (React + Tauri)
            │
            ▼
      FastAPI Server
            │
            ▼
      Almond Core
            │
 ┌──────────┼──────────┐
 │          │          │
 ▼          ▼          ▼
Memory   Retrieval   Oracle
Controller Engine    Context
            │
            ▼
     Local Language Model
(LM Studio / OpenAI-compatible)
```

---

## Project Goals

Almond is being built around four core principles.

- Memory should feel natural, not intrusive.
- Every conversation should feel like a continuation.
- The user always owns their data.
- AI should support reflection, not replace relationships.

---

## Tech Stack

- Python
- FastAPI
- SQLite
- React
- Tauri
- LM Studio
- Sentence Transformers
- FAISS (planned)
- Local LLMs

---

## Research

Almond is also the foundation of ongoing research into long-term conversational memory and semantic retrieval.

Current work focuses on evaluating memory quality using benchmark datasets such as LongMemEval and comparing baseline conversations against Oracle-enhanced retrieval.

---

## Roadmap

### v1
- Persistent memory
- Desktop application
- Timeline
- Explorer
- Reflections

### v2
- Better semantic retrieval
- Memory scoring improvements
- Reflection engine
- Improved Oracle retrieval

### Future

- Mobile client
- Multi-device sync
- Optional encrypted cloud backup
- Voice interaction
- Rich artifact storage
- Calendar and file understanding

---

## Philosophy

> "Speak like someone who values the person more than the conversation."

Almond isn't designed to be the smartest AI.

It's designed to be the one that remembers.

---

## Status

🚧 Active Research Project

Almond is currently under active development and the architecture is evolving rapidly.

---

## License

MIT License
