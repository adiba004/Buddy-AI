# Buddy AI — Product Decision Framework

> This document captures the prioritisation and strategic frameworks 
> used to define, scope, and build Buddy AI before a single line of 
> code was written.

---

## 1. North Star Metric

> **Number of students who resolved a doubt and passed a chapter test 
> in the same session**

### Why this metric
A student opening the app is not success.
A student chatting is not success.
Success is a student who was stuck, got unstuck via the AI tutor, 
and then validated their understanding by passing a test.
That is the full learning loop closing — and that is what Buddy AI exists for.

### Supporting metrics
| Metric | Why it matters |
|--------|---------------|
| Session completion rate | Are students finishing chapter sessions |
| Test attempt rate after chat | Are students using chat to prepare for tests |
| Weak topic resolution rate | Are flagged weak topics improving over retakes |
| Avg response latency | Is the experience fast enough to not break flow |

---

## 2. Jobs To Be Done (JTBD)

> Framework: When [situation], I want to [motivation], so I can [outcome]

| # | Job Statement | Priority |
|---|--------------|----------|
| 1 | When I am stuck on a concept at night with no tutor available, I want an instant explanation in my own language, so I can stop being blocked and keep studying | Critical |
| 2 | When I have finished reading a chapter, I want to test myself on it, so I can know if I actually understood it or just read it | Critical |
| 3 | When I fail a test, I want to know exactly which topics I am weak in, so I can go back and fix only those instead of re-reading the whole chapter | Critical |
| 4 | When I am preparing for exams, I want a quick chapter summary, so I can revise faster without reading everything again | High |
| 5 | When I ask a question in Hinglish, I want the AI to understand me naturally, so I do not have to think about how I am phrasing things | High |
| 6 | When my parent asks how I am doing, I want them to see my progress, so I can show I am studying seriously | Medium |

---

## 3. MoSCoW Prioritisation

### Must Have — MVP (Class 9 and 10, Science and Maths)
- [ ] Student signup and login with JWT auth
- [ ] Chapter selection by grade and subject
- [ ] Chat-based doubt solving per chapter (session scoped)
- [ ] Intent detection — doubt, summary, evaluate
- [ ] RAG-backed responses using chapter embeddings
- [ ] MCQ test generation per chapter
- [ ] Test submission with scoring
- [ ] Weak topic identification after test
- [ ] Hinglish query support
- [ ] Streaming AI responses via SSE

### Should Have — Phase 1 (Class 6 to 10)
- [ ] Expand to Class 6, 7, 8 content
- [ ] Add more subjects beyond Science and Maths
- [ ] Student progress dashboard
- [ ] Attempt history and score trends
- [ ] Parent view of progress
- [ ] Mobile native app (iOS and Android)

### Could Have — Phase 2
- [ ] Voice input and output
- [ ] PDF export of session notes
- [ ] A/B testing for prompt variants
- [ ] Redis caching layer for frequent queries
- [ ] Teacher-facing analytics dashboard
- [ ] Subscription and payment management
- [ ] Freemium model with premium chapter unlocks

### Will Not Have — Explicitly out of scope
- [ ] Live tutoring or video calls
- [ ] Non-NCERT syllabus content


---

## 4. RICE Scoring

> Formula: (Reach × Impact × Confidence) / Effort
> Scale: Reach 1-10 (students impacted), Impact 1-3, Confidence %, Effort in weeks

| Feature | Reach | Impact | Confidence | Effort | RICE Score | Decision |
|---------|-------|--------|------------|--------|------------|----------|
| Chat-based doubt solving | 10 | 3 | 90% | 3w | **90** | Build first |
| MCQ test generation | 10 | 3 | 85% | 2w | **127** | Build first |
| Weak topic identification | 9 | 3 | 80% | 1w | **216** | Build first |
| Hinglish support | 8 | 2 | 90% | 0.5w | **288** | Build first |
| Intent detection | 10 | 3 | 85% | 1w | **255** | Build first |
| Progress dashboard | 7 | 2 | 70% | 3w | **32** | Phase 1 |
| Voice input | 5 | 2 | 60% | 4w | **15** | Phase 2 |
| Parent view | 4 | 2 | 65% | 2w | **26** | Phase 1 |
| Redis caching | 10 | 1 | 80% | 2w | **40** | Phase 2 |
| PDF export | 3 | 1 | 75% | 2w | **11** | Phase 2 |

### Key insight from RICE
Weak topic identification scored highest despite low effort because it 
delivers direct learning value to every student who takes a test. 
This is why it was built into the MVP data model from day one rather 
than treated as a nice-to-have.

---

## 5. Build Decisions Log

| Decision | Metric used | Outcome |
|----------|------------|---------|
| Chose Ling-2.6-1T over GPT-4o | Cost: $0.30/1M vs $5/1M input tokens | 16x cheaper |
| Used local MiniLM embeddings | Speed: 4x faster than ada-002, cost: $0 vs $0.10/1K | Zero embedding cost |
| Capped context at 3000 chars | Token budget: keeps prompt under 1200 tokens total | Predictable latency |
| Limited history to 5 turns | Token budget: 10 messages ~ 400 tokens max | Lean prompts |
| Used summary for test gen | Context: summary ~500 tokens vs full chapter ~8000 tokens | 16x smaller context |
| Used RAG for chat | Relevance: only top chunks retrieved, not full chapter | Precise grounded answers |
| Scoped sessions to chapters | Accuracy: prevents hallucination across chapter boundaries | Higher answer quality |
| Intent detection before RAG | Efficiency: RAG skipped for summary requests | Reduced latency |
| Free tier architecture | Cost: OpenRouter + Supabase + HuggingFace = $0/month | $95/month saved |
| Projected 38M tokens/month | Scale: 1000 students × 10 chats × 5 tests modelled upfront | Cost ceiling known |

---

*Frameworks applied: North Star Metric, Jobs To Be Done, MoSCoW, RICE*

---

## System Architecture

![Buddy AI System Architecture](buddy_ai_system_architecture_v2%20(1).svg)

---

## User Journey

![Buddy AI User Journey](user_journey%20(1).svg)
