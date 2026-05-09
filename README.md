# PSI-09-vRAG: GraphRAG Conversation Engine

**Hosted on:** Hugging Face Spaces (Free Tier) — **Independent experimental branch**  
**Endpoint:** `POST /psi09` | **Health:** `GET /`

---

## Purpose

PSI-09-vRAG is an independent experimental engine that implements GraphRAG architecture. It replaces traditional prompt-based context with a structured pipeline: knowledge graph assembly → DSPy signature reasoning → LangGraph state machine → structured output.

It operates without operational constraints tied to the main engine.

---

## Core Architecture: DSPy + LangGraph + NetworkX

The engine replaces flat prompt injection with a modular pipeline:

```
Traditional (roastbot)              vRAG
─────────────────                   ────
Fixed prompt templates              DSPy modular signatures
History string injection            Structured graph context
Single-shot LLM call                LangGraph state machine
Raw text output                     Pydantic-validated CombatDecision
No engagement filtering             Triage gatekeeper node
```

---

## Complete Logic Flow

### Step 1: API Entry → `/psi09`

```python
INCOMING PAYLOAD  (identical to roastbot schema)
```

### Step 2: Graph Context Assembly

```python
get_user_graph_context(username, user_key, group_name)
│
├── LOAD USER GRAPH from cache/DB
│     Key:  "{group}:{username}"
│     Fields: entities[], relationships[], last_updated
│
├── LOAD GROUP GRAPH from cache/DB
│     Key:  group_name
│     Fields: entities[], relationships[], last_updated
│
├── APPLY TEMPORAL DECAY
│     user_decay = 0.9 ^ days_since_update
│     group_decay = 0.9 ^ days_since_update
│     Each relationship weight *= decay_factor
│
├── BUILD NETWORKX DIGRAPH
│     ├── Add all entities as nodes (type + attributes)
│     ├── Add all relationships as edges (relation + decayed weight)
│     └── If target user not found: return "socially isolated"
│
├── COMPUTE PAGERANK
│     social_scores = nx.pagerank(G, weight='weight')
│     target_score = social_scores[username]
│     social_rank = position in sorted scores
│     → "Rank 3 out of 12 active entities"
│
├── DETECT COMMUNITIES
│     undirected_G = G.to_undirected()
│     factions = nx_comm.greedy_modularity_communities(undirected_G)
│     user_faction = faction containing username
│     → "Lone Wolf" or "Steve, Alex, Notch"
│
└── FORMAT CONTEXT STRING
      --- TARGET DOSSIER: username ---
      CORE TRAITS: [entity attributes]
      SOCIAL RANK (PageRank): 0.0421 (Rank 3 out of 12)
      DETECTED FACTION / ALLIES: Steve, Alex
      ACTIVE RELATIONSHIPS:
      - [ACTIVE] username [rivalry] Steve (Relevance: 8.2)
      - [FADING] username [ally] Alex (Relevance: 1.3)
```

### Step 3: History Assembly

```python
active_history = fetch_history(chat_history, user_key, 30)  # for DMs
              or fetch_history(group_history, group_name, 30)  # for groups
history_text = "[Username]: message\n[User2]: message2\n..."
```

### Step 4: Location String

```python
location_str = "Private Direct Message"
            or "Server: 6b6t | Channel: #public"
```

### Step 5: LangGraph State Machine Invocation

```
psi09_agent.invoke(initial_state)
│
├── STATE: {
│     "history": history_text,
│     "graph": graph_text,        # NetworkX context
│     "user": username,
│     "message": "[user]: msg",
│     "location": location_str,
│     "is_direct": True/False,
│     "should_engage": False,     # Default
│     "reply": "",
│     "reaction": None,
│     "reasoning": "Triage bypassed combat engine. (Silence)"
│   }
│
├── NODE 1: TRIAGE (Groq failover pool)
│   │
│   ├── TriageSignature:
│   │   "Determine if PSI-09 should engage or remain in superior silence."
│   │   Inputs:  chat_history, active_message, location, is_direct_interaction
│   │   Output:  should_engage (bool)
│   │
│   ├── Engagement triggers (True):
│   │   1. User explicitly pinged @PSI-09
│   │   2. Logically flawed/intellectually challenging statement
│   │   3. Casual name mention without ping
│   │   4. Active ongoing conversation in history
│   │
│   ├── Silence triggers (False):
│   │   1. Mundane logistics, trivial chatter
│   │   2. User said "shut up" / "stay quiet"
│   │   3. Already responded once in recent history
│   │
│   └── Model pool: Groq (openai/gpt-oss-120b, llama-3.3-70b-versatile)
│       Failover: advance index on 429, retry up to N attempts
│
│   ├── if should_engage == False → END (no reply, silent)
│   │
│   └── if should_engage == True → NODE 2: COMBAT
│
├── NODE 2: COMBAT (NVIDIA NIM round-robin)
│   │
│   ├── PSI09CombatEngine.forward(history, graph, user, message, location)
│   │   │
│   │   ├── 2a. IDENTITY (ChainOfThought)
│   │   │     IdentitySignature:
│   │   │     "You are PSI-09, an entity of deadpan wit, born to roast humans."
│   │   │     Inputs:  graph_context, target_user
│   │   │     Output:  dynamic_persona (who PSI-09 is in this moment)
│   │   │
│   │   ├── 2b. MISSION (ChainOfThought)
│   │   │     MissionSignature:
│   │   │     "Dismantle their argument surgically without showing emotion."
│   │   │     Inputs:  dynamic_persona, chat_history, active_message, location
│   │   │     Output:  tactical_objective
│   │   │
│   │   ├── 2c. CONSTRAINTS (ChainOfThought)
│   │   │     ConstraintsSignature:
│   │   │     "TONE: Raw, aggressive, relentlessly toxic. MUST BE UNDER 150 CHARS."
│   │   │     Inputs:  tactical_objective, active_message
│   │   │     Output:  operational_constraints
│   │   │
│   │   └── 2d. DECISION (Predict)
│   │         DecisionSignature:
│   │         "CRITICAL DECISION MATRIX: REACTION_ONLY / TEXT_ONLY / BOTH"
│   │         Inputs:  tactical_objective, operational_constraints, active_message
│   │         Output:  CombatDecision {
│   │           response_method: "REACTION_ONLY" | "TEXT_ONLY" | "BOTH",
│   │           reaction: "emoji" | None,
│   │           reply: "text" | None
│   │         }
│   │
│   └── Model pool: NVIDIA NIM (moonshotai/kimi-k2-instruct-0905)
│       Round-robin across 2 API keys: Key1, Key2, Key1, Key2...
│
└── RETURN { reply, reaction, reasoning }
```

### Step 6: Storage

```python
# Store user message
store_message(chat_history, user_key, entry)
if not is_private:
    store_message(group_history, group_name, entry)

# Store bot reply (if generated)
if reply:
    store_message(chat_history, user_key, bot_entry)
    if not is_private:
        store_message(group_history, group_name, bot_entry)
```

### Step 7: Background Graph Evolution (Non-Blocking)

```python
def background_evolution_tasks():
    if is_private:
        # Extract user graph from private history
        summarize_user_history(user_key, username, group_name, is_private=True)
    else:
        # Extract group graph from group history
        summarize_group_history(group_name)

threading.Thread(target=background_evolution_tasks, daemon=True).start()

# Graph extraction uses:
# GraphExtractionSignature (DSPy + Pydantic)
#   "Analyze chat log and map social dynamics between explicitly named users."
#   Output: GraphKnowledge { entities[], relationships[] }
#   Entities extracted: username, type, psychological attributes
#   Relationships extracted: source, target, relation, intensity (1.0-10.0)
```

---

## Key Differences from roastbot

| Aspect | roastbot (Production) | vRAG (Experimental) |
|--------|-----------------------|---------------------|
| Context assembly | Token-trimmed message history | NetworkX knowledge graph |
| Profile storage | Text summaries in MongoDB | Structured entity/relationship graphs |
| Engagement decision | Hard-coded: pinged → reply | LLM triage: context-aware decision |
| Output format | Text reply only | Text + emoji reaction |
| Model pipeline | Single-shot LLM call | 4-stage DSPy signature chain |
| Orchestration | Sequential function calls | LangGraph state machine |
| Schema validation | None | Pydantic BaseModel |
| Social awareness | Group summary text | PageRank + community detection |
| Temporal decay | None (all history equal) | 0.9^days exponential decay |
| DB collections | 6 | 4 (no memory collections, uses graphs) |

---

## Database Schema

### Collections

| Collection | Key Format | Content |
|------------|------------|---------|
| `chat_history` | `{group}:{username}` | Message archive |
| `group_history` | `{group_name}` | Group message archive |
| `graph_users` | `{group}:{username}` | User knowledge graph |
| `graph_groups` | `{group_name}` | Group knowledge graph |

### Graph Document Structure
```json
{
  "_id": "6b6t:Steve",
  "graph_data": {
    "entities": [
      {
        "id": "Steve",
        "type": "User",
        "attributes": "Aggressive, narcissistic, seeks confrontation"
      },
      {
        "id": "Alex",
        "type": "User",
        "attributes": "Passive follower, easily intimidated"
      }
    ],
    "relationships": [
      {
        "source": "Steve",
        "target": "Alex",
        "relation": "bullies",
        "intensity": 8.5
      },
      {
        "source": "Alex",
        "target": "Steve",
        "relation": "fears",
        "intensity": 7.0
      }
    ],
    "last_updated": "2026-05-09T14:30:00"
  }
}
```

---

## Configuration

### Environment Variables
```bash
# MongoDB
MONGO_URI=mongodb+srv://...

# NVIDIA NIM (Combat - round-robin)
NVIDIA_API_KEY_1=...
NVIDIA_API_KEY_2=...

# Groq (Background graph extraction)
GROQ_API_KEY_2=...

# Groq (Triage routing)
GROQ_API_KEY_3=...

# Discord IDs for mention detection
DISCORD_ID=...
DISCORD_ID_2=...

# Server
PORT=7860
```

### Tuning Parameters
```python
MEMORY_TTL = 500              # Graph cache TTL (seconds)
GROUP_HISTORY_MAX_MESSAGES = 50000
GROUP_HISTORY_SLICE = 80
MAX_HISTORY_MESSAGES = 16     # Smaller than roastbot (30)
```

---

## Repository Structure

```
PSI-09-vRAG/
├── main.py       # DSPy signatures, LangGraph, NetworkX, Flask API
├── requirements.txt
├── Dockerfile
└── render.yaml
```

---

## Related

- [psi-09-roastbot](https://github.com/sudoboneman/psi-09-roastbot) — Production engine (independent)
- [psi-09-discord](https://github.com/sudoboneman/psi-09-discord) — Discord bridge
- [psi-09-whatsapp](https://github.com/sudoboneman/psi-09-whatsapp) — WhatsApp bridge
- [psi-09-mc](https://github.com/sudoboneman/psi-09-mc) — Minecraft 6b6t bot
- [psi-09-mc-gapples](https://github.com/sudoboneman/psi-09-mc-gapples) — Minecraft gapples bot
- [psi-09-pseudo-user-discord](https://github.com/sudoboneman/psi-09-pseudo-user-discord) — Self-bot bridge
- [psi-09-local](https://github.com/sudoboneman/psi-09-local) — WhatsApp session extractor

---

**Status:** Active, experimental development  
**Origin:** 2025  
**Author:** sudoboneman

Copyright © 2024–2026. All rights reserved.