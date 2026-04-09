# PSI-09 Architecture Guide

## Overview

PSI-09 is a Flask-based Discord chatbot designed to engage in "cold intellect" roast interactions. It uses a sophisticated multi-agent architecture combining LangGraph for orchestration, DSPy for LLM interactions, and GraphRAG for social knowledge management.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Configuration](#configuration)
3. [LLM Pool Management](#llm-pool-management)
4. [DSPy Signatures & Combat Engine](#dspy-signatures--combat-engine)
5. [LangGraph State Machine](#langgraph-state-machine)
6. [Database Layer](#database-layer)
7. [GraphRAG System](#graphrag-system)
8. [API Endpoints](#api-endpoints)
9. [Background Processing](#background-processing)
10. [Data Flow Diagram](#data-flow-diagram)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Flask API Server                        │
│                      POST /psi09 endpoint                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Context Assembly Layer                       │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │  Graph Context  │  │   History Text   │  │ Direct Mention │ │
│  │  (GraphRAG)     │  │  (MongoDB)       │  │    Detection   │ │
│  └─────────────────┘  └──────────────────┘  └────────────────┘ │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LangGraph State Machine                        │
│  ┌─────────────────┐         ┌──────────────────┐               │
│  │  Triage Node   │────────▶│   Combat Node    │               │
│  │  (Should we    │  YES    │  (Generate Roast)│               │
│  │   respond?)    │         │                  │               │
│  └────────┬───────┘         └────────┬─────────┘               │
│           │ NO                        │                         │
│           ▼                           │                         │
│        [END]                          │                         │
│                                      ▼                         │
│                               [END]                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Background Processing                         │
│  ┌─────────────────┐  ┌──────────────────┐                     │
│  │ User Graph      │  │  Group Graph      │                     │
│  │ Extraction      │  │  Extraction       │                     │
│  │ (Background)    │  │  (Background)     │                     │
│  └─────────────────┘  └──────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Config Dataclass (lines 38-76)

```python
@dataclass
class Config:
    MONGO_URI: str = os.getenv("MONGO_URI")
    GROQ_API_KEY_1: str = os.getenv("GROQ_API_KEY_1") # Roasts ONLY
    GROQ_API_KEY_2: str = os.getenv("GROQ_API_KEY_2") # Background Tasks ONLY
    GROQ_API_KEY_3: str = os.getenv("GROQ_API_KEY_3") # Triage Tasks ONLY
    
    ROAST_MODELS: list = __import__("dataclasses").field(default_factory=lambda: [
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "openai/gpt-oss-120b"
    ])
    
    BACKGROUND_MODELS: list = __import__("dataclasses").field(default_factory=lambda: [
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ])

    TRIAGE_MODELS: list = __import__("dataclasses").field(default_factory=lambda: [
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ])
    
    BOT_NUMBER: str = os.getenv("BOT_NUMBER")
    DISCORD_ID: str = os.getenv("DISCORD_ID")
    DISCORD_ID_2: str = os.getenv("DISCORD_ID_2")
    MEMORY_TTL: int = 500
    
    GROUP_HISTORY_MAX_MESSAGES: int = 50000 
    GROUP_HISTORY_SLICE: int = 80 
    MAX_HISTORY_MESSAGES: int = 16 
```

### Configuration Parameters

| Parameter | Purpose |
|-----------|---------|
| `MONGO_URI` | MongoDB connection string |
| `GROQ_API_KEY_1` | Dedicated for combat/roast generation |
| `GROQ_API_KEY_2` | Dedicated for background graph extraction |
| `GROQ_API_KEY_3` | Dedicated for triage routing decisions |
| `MEMORY_TTL` | In-memory cache time-to-live in seconds (500s) |
| `GROUP_HISTORY_MAX_MESSAGES` | Maximum messages stored per group (50,000) |
| `GROUP_HISTORY_SLICE` | Messages used for group graph extraction (80) |
| `MAX_HISTORY_MESSAGES` | Recent messages for context (16) |

---

## LLM Pool Management

### FailoverLMPool Class (lines 79-95)

The `FailoverLMPool` implements a round-robin load balancer with automatic failover for rate limiting.

```python
class FailoverLMPool:
    def __init__(self, model_names: list, api_key: str, pool_name: str):
        self.pool_name = pool_name
        self.models = [dspy.LM(model=f"groq/{m}", api_key=api_key) for m in model_names]
        self.index = 0
        self.lock = threading.Lock()

    def get_current(self):
        with self.lock:
            return self.models[self.index], self.index

    def advance(self, failed_index: int):
        with self.lock:
            if self.index == failed_index:
                self.index = (self.index + 1) % len(self.models)
                logger.warning(f"[{self.pool_name}] Rate Limit! Failover triggered")
            return self.models[self.index]
```

### Key Features

1. **Thread-Safe Operations**: Uses `threading.Lock` for concurrent access
2. **Round-Robin Selection**: Cycles through models sequentially
3. **Automatic Failover**: On rate limit (429), advances to next model
4. **Model Prefixing**: Automatically prepends `groq/` for Groq API

### Pool Initialization (lines 98-100)

```python
triage_pool = FailoverLMPool(config.TRIAGE_MODELS, config.GROQ_API_KEY_3, "TRIAGE")
combat_pool = FailoverLMPool(config.ROAST_MODELS, config.GROQ_API_KEY_1, "COMBAT")
background_pool = FailoverLMPool(config.BACKGROUND_MODELS, config.GROQ_API_KEY_2, "BACKGROUND")
```

---

## DSPy Signatures & Combat Engine

### Signature Overview

DSPy signatures define the input/output contract for LLM interactions. PSI-09 uses five signatures:

### 1. IdentitySignature (lines 103-108)

Defines PSI-09's core personality and self-perception.

```python
class IdentitySignature(dspy.Signature):
    """You are PSI-09, an entity of immense, cold intellect..."""
    
    graph_context = dspy.InputField(desc="Known relationships...")
    target_user = dspy.InputField(desc="The human who is deserving...")
    dynamic_persona = dspy.OutputField(desc="Who PSI-09 is in this moment...")
```

### 2. MissionSignature (lines 110-121)

Determines the tactical objective for the current message.

```python
class MissionSignature(dspy.Signature):
    """Determine PSI-09's tactical objective..."""
    
    dynamic_persona = dspy.InputField(desc="Who PSI-09 is...")
    chat_history = dspy.InputField(desc="Recent dialogue...")
    active_message = dspy.InputField(desc="The specific human message...")
    tactical_objective = dspy.OutputField(desc="The exact goal...")
```

### 3. ConstraintsSignature (lines 123-133)

Establishes operational guidelines for the response.

```python
class ConstraintsSignature(dspy.Signature):
    """Never, ever, expose or reveal YOUR IDENTITY..."""
    
    tactical_objective = dspy.InputField(desc="What PSI-09 is trying...")
    active_message = dspy.InputField(desc="The message being responded to...")
    operational_constraints = dspy.OutputField(desc="A guidance mandate...")
```

**Key Constraints:**
- Maximum 150 characters
- Cold intellectual superiority, dry sarcasm
- No generic internet jokes or clichés
- Profanity only for "FLAVOR"
- Never quote/paraphrase user's message

### 4. DecisionSignature (lines 142-158)

Determines response method using a strict decision matrix.

```python
class CombatDecision(BaseModel):
    response_method: Literal["REACTION_ONLY", "TEXT_ONLY", "BOTH"]
    reaction: Optional[str]  # Single emoji or 'None'
    reply: Optional[str]     # Text response or 'None'

class DecisionSignature(dspy.Signature):
    """CRITICAL DECISION MATRIX:
    - REACTION_ONLY: Casual mention, mildly amusing
    - TEXT_ONLY: Explicit request for response
    - BOTH: Devastating point + perfect emoji
    """
```

### PSI09CombatEngine Module (lines 160-196)

```python
class PSI09CombatEngine(dspy.Module):
    def __init__(self):
        super().__init__()
        self.identity = dspy.ChainOfThought(IdentitySignature)
        self.mission = dspy.ChainOfThought(MissionSignature)
        self.constraints = dspy.ChainOfThought(ConstraintsSignature) 
        
    def forward(self, history, graph, user, message):
        # Step 1: Determine identity/persona
        id_res = self.identity(graph_context=graph, target_user=user)
        
        # Step 2: Define tactical mission
        miss_res = self.mission(
            dynamic_persona=id_res.dynamic_persona,
            chat_history=history,
            active_message=message
        )
        
        # Step 3: Establish constraints
        con_res = self.constraints(
            tactical_objective=miss_res.tactical_objective,
            active_message=message
        )
        
        # Step 4: Make response decision (enforced via Pydantic)
        decision_engine = dspy.Predict(DecisionSignature)
        dec_res = decision_engine(
            tactical_objective=miss_res.tactical_objective,
            operational_constraints=con_res.operational_constraints,
            active_message=message
        )
        
        # Return structured prediction
        return dspy.Prediction(
            reaction=dec_res.decision.reaction,
            reply=dec_res.decision.reply,
            reasoning=full_reasoning
        )
```

### Combat Engine Flow

```
User Message
     │
     ▼
┌─────────────────┐
│ Identity Stage  │ ──▶ "Who am I? How do I view this human?"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Mission Stage  │ ──▶ "What's my tactical objective?"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Constraints Stage│ ──▶ "What rules govern my response?"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Decision Stage  │ ──▶ "REACTION_ONLY | TEXT_ONLY | BOTH"
└─────────────────┘
```

---

## LangGraph State Machine

### CombatState TypedDict (lines 221-231)

```python
class CombatState(TypedDict):
    history: str           # Recent chat history as string
    graph: str             # GraphRAG context
    user: str              # Target username
    message: str           # Active message
    is_direct: bool        # Was bot directly mentioned?
    
    should_engage: bool    # Triage decision
    reply: str             # Generated reply
    reaction: Optional[str]# Emoji reaction
    reasoning: str         # Decision trace
```

### Triage Node (lines 234-258)

Gatekeeper that decides if PSI-09 should engage.

```python
def triage_node(state: CombatState):
    max_retries = len(triage_pool.models)
    
    for attempt in range(max_retries):
        current_lm, current_index = triage_pool.get_current()
        try:
            with dspy.context(lm=current_lm):
                res = triage_engine(
                    chat_history=state["history"], 
                    active_message=state["message"],
                    is_direct_interaction=str(state["is_direct"]) 
                )
            engage = res.decision.should_engage
            return {"should_engage": engage}
            
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                triage_pool.advance(current_index)
            else:
                triage_pool.advance(current_index)
                
    return {"should_engage": False}
```

**Triage Decision Logic:**
- Output `True` ONLY if:
  1. User explicitly pinged the bot (`is_direct_interaction='True'`)
  2. OR they made a logically flawed/intellectually challenging statement
  3. OR they casually mentioned the bot's name without pinging
  4. OR there's an active conversation with the bot in history

### Combat Node (lines 261-287)

Apex predator that generates the actual response.

```python
def combat_node(state: CombatState):
    max_retries = len(combat_pool.models)
    
    for attempt in range(max_retries):
        current_lm, current_index = combat_pool.get_current()
        try:
            with dspy.context(lm=current_lm):
                res = combat_engine(
                    history=state["history"], 
                    graph=state["graph"], 
                    user=state["user"], 
                    message=state["message"]
                )
            return {
                "reply": res.reply if str(res.reply).lower() not in ["none", "null", ""] else "",
                "reaction": res.reaction if str(res.reaction).lower() not in ["none", "null", ""] else None,
                "reasoning": res.reasoning
            }
        except Exception as e:
            combat_pool.advance(current_index)
            
    return {"reply": "", "reaction": None, "reasoning": "Combat engine failure."}
```

### Routing Logic (lines 290-293)

```python
def route_engagement(state: CombatState):
    if state["should_engage"]:
        return "combat"
    return "end"
```

### Graph Compilation (lines 295-312)

```python
workflow = StateGraph(CombatState)
workflow.add_node("triage", triage_node)
workflow.add_node("combat", combat_node)

workflow.set_entry_point("triage")
workflow.add_conditional_edges(
    "triage", 
    route_engagement, 
    {
        "combat": "combat", 
        "end": END
    }
)
workflow.add_edge("combat", END)

psi09_agent = workflow.compile()
```

### State Machine Diagram

```
                    ┌──────────────┐
                    │   ENTRY      │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   TRIAGE     │ ◀── Entry Point
                    │    NODE      │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        ┌──────────┐              ┌──────────┐
        │ should_  │              │ should_  │
        │ engage?  │              │ engage?  │
        │  TRUE   │              │  FALSE   │
        └────┬─────┘              └────┬─────┘
             │                         │
             ▼                         ▼
       ┌──────────┐              ┌──────────┐
       │  COMBAT │              │   END    │
       │   NODE  │              │ (Silent) │
       └────┬─────┘              └──────────┘
            │
            ▼
       ┌──────────┐
       │   END    │
       │(Response)│
       └──────────┘
```

---

## Database Layer

### MongoDB Collections (lines 315-321)

```python
mongo_client = MongoClient(config.MONGO_URI, tlsCAFile=certifi.where())
db = mongo_client["psi09"]
history_col = db["chat_history"]        # Private DM histories
group_history_col = db["group_history"] # Group channel histories
graph_user_col = db["graph_users"]      # Per-user knowledge graphs
graph_group_col = db["graph_groups"]     # Group-level knowledge graphs
```

### MongoCache Class (lines 322-358)

In-memory cache with TTL and MongoDB persistence.

```python
class MongoCache:
    def __init__(self, collection, ttl_seconds):
        self.collection = collection
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.cache_time = {}
        self.lock = threading.Lock()

    def get(self, key):
        now = time.time()
        with self.lock:
            # Check if in-memory cache is still valid
            if key in self.cache and key in self.cache_time:
                if (now - self.cache_time[key]) < self.ttl_seconds:
                    return self.cache[key]
                
        # Fetch fresh from Mongo if cache miss/expired
        try:
            doc = self.collection.find_one({"_id": key})
            data = doc.get("graph_data") if doc else None
        except PyMongoError: 
            data = None
            
        with self.lock:
            self.cache[key] = data
            self.cache_time[key] = now
        return data

    def set(self, key, value):
        # Persist to Mongo
        try: 
            self.collection.update_one(
                {"_id": key}, 
                {"$set": {"graph_data": value}}, 
                upsert=True
            )
        except PyMongoError: 
            pass
        
        # Update in-memory cache
        with self.lock:
            self.cache[key] = value
            self.cache_time[key] = time.time()
```

### Cache Initialization (lines 359-362)

```python
graph_user_cache = MongoCache(graph_user_col, config.MEMORY_TTL)
graph_group_cache = MongoCache(graph_group_col, config.MEMORY_TTL)
user_locks = defaultdict(threading.Lock)
group_locks = defaultdict(threading.Lock)
```

---

## GraphRAG System

### Graph Extraction (lines 382-443)

The `get_user_graph_context()` function builds a context string from the knowledge graph.

```python
def get_user_graph_context(username, user_key, group_name):
    G = nx.DiGraph()
    now = datetime.now(UTC)
    
    # Fetch graphs from cache
    user_graph = graph_user_cache.get(user_key) or {...}
    group_graph = graph_group_cache.get(group_name) or {...}
    
    # Calculate time-based decay
    user_age_days = (now - user_graph.get("last_updated")).days
    user_decay = max(0.1, 0.9 ** user_age_days)
    group_decay = max(0.1, 0.9 ** group_age_days)
    
    # Build NetworkX graph
    for data, decay_factor in [(user_graph, user_decay), (group_graph, group_decay)]:
        for ent in data.get("entities", []):
            G.add_node(ent.get("id"), ...)
        for rel in data.get("relationships", []):
            base_weight = float(rel.get("intensity", 5.0))
            decayed_weight = base_weight * decay_factor
            G.add_edge(rel.get("source"), rel.get("target"), ...)
```

### GraphRAG Processing Steps

1. **Fetch Cached Graphs**: Get user and group graphs from MongoCache
2. **Calculate Time Decay**: Older data has less weight (`0.9^days`)
3. **Build NetworkX Graph**: Create directed graph with weighted edges
4. **Calculate PageRank**: Determine social importance/scores
5. **Community Detection**: Identify factions/alliances
6. **Extract Relationships**: Get top 5 most relevant connections

### PageRank Scoring (lines 409-416)

```python
social_scores = nx.pagerank(G, weight='weight')
target_score = social_scores.get(username, 0.0)
ranked_users = sorted(social_scores.items(), key=lambda x: x[1], reverse=True)
rank_index = next((i for i, v in enumerate(ranked_users) if v[0] == username), ...)
social_status = f"Rank {rank_index + 1} out of {len(ranked_users)} active entities."
```

### Community Detection (lines 418-424)

```python
undirected_G = G.to_undirected()
factions = list(nx_comm.greedy_modularity_communities(undirected_G))
user_faction = next((list(f) for f in factions if username in f), [])
faction_str = ", ".join([u for u in user_faction if u != username])
```

### Output Context String (lines 426-443)

```python
context_lines = []
context_lines.append(f"--- TARGET DOSSIER: {username} ---")
context_lines.append(f"CORE TRAITS: {node_attrs}")
context_lines.append(f"SOCIAL RANK (PageRank): {target_score:.4f} ({social_status})")
context_lines.append(f"DETECTED FACTION / ALLIES: {faction_str}")

# Include top 5 relationships with decay status
edges = list(G.in_edges(username, data=True)) + list(G.out_edges(username, data=True))
edges.sort(key=lambda x: x[2].get('weight', 0), reverse=True)
for source, target, data in edges[:5]:
    w = data.get('weight', 0)
    status = "[FADING]" if w < 2.0 else "[ACTIVE]"
    context_lines.append(f"- {status} {source} [{data['relation']}] {target} (Relevance: {w:.1f})")
```

### Graph Extraction Pydantic Schemas (lines 446-468)

```python
class Relationship(BaseModel):
    source: str = Field(description="EXACT username, NO snowflakes")
    target: str = Field(description="EXACT username, NO snowflakes")
    relation: str = Field(description="Nature of relationship")
    intensity: float = Field(ge=1.0, le=10.0)

class Entity(BaseModel):
    id: str = Field(description="EXACT username")
    type: str = Field(default="User")
    attributes: str = Field(description="Psychological traits")

class GraphKnowledge(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

class GraphExtractionSignature(dspy.Signature):
    """Analyze chat log and map social dynamics..."""
    
    target_focus: str = dspy.InputField(...)
    chat_log: str = dspy.InputField(...)
    extracted_graph: GraphKnowledge = dspy.OutputField(...)
```

### Background Graph Extraction (lines 472-537)

Two functions handle background extraction:

```python
def summarize_user_history(user_key, username, group_name, is_private):
    col = history_col if is_private else group_history_col
    doc_id = user_key if is_private else group_name
    
    history = fetch_history(col, doc_id, config.MAX_HISTORY_MESSAGES)
    chat_text = "\n".join([...])
    
    # Clean PII
    chat_text = re.sub(r'<@!?&?\d+>', '', chat_text)
    chat_text = re.sub(r'\b\d{17,19}\b', '', chat_text)
    
    # Extract using LLM
    with dspy.context(lm=background_pool.get_current()[0]):
        result = graph_extractor(
            target_focus=f"Deep psychological profile of user: {username}",
            chat_log=chat_text
        )
        
    graph_dict = result.extracted_graph.model_dump()
    graph_dict["last_updated"] = datetime.now(UTC).isoformat()
    graph_user_cache.set(user_key, graph_dict)
```

---

## API Endpoints

### Health Check (lines 543-544)

```python
@app.route("/", methods=["GET"])
def health(): 
    return jsonify({"status": "ok"}), 200
```

### Main PSI-09 Endpoint (lines 546-623)

```python
@app.route("/psi09", methods=["POST"])
def psi09():
    try:
        # 1. Parse Request
        data = request.get_json(force=True)
        raw_message = data.get("message", "")
        sender_id = data.get("sender_id")
        username = data.get("username")
        
        # 2. Normalize Group/Channel Names
        group_name = data.get("group_name") or "DefaultGroup"
        if group_name.lower() in ["defaultgroup", "discord_dm"]:
            group_name = "private_chat"
        
        # 3. Replace Discord Mentions with @PSI-09
        user_message = raw_message
        for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
            if d_id: 
                user_message = re.sub(
                    r"<@!?" + re.escape(str(d_id)) + r">", 
                    "@PSI-09", 
                    user_message
                )

        # 4. Determine if Direct Interaction
        is_direct = is_private or data.get("force_reply", False) or bot_mentioned_in(raw_message)

        # 5. Assemble Context
        graph_text = get_user_graph_context(username, user_key, group_name)
        active_history = fetch_history(...)
        history_text = "\n".join([...])

        # 6. Execute State Machine
        initial_state = {
            "history": history_text,
            "graph": graph_text,
            "user": username,
            "message": user_message,
            "is_direct": is_direct,
            ...
        }
        
        final_state = psi09_agent.invoke(initial_state)
        reply = final_state["reply"]
        reaction = final_state["reaction"]

        # 7. Store Messages
        entry = {"role": "user", "username": username, "content": user_message, ...}
        store_message(history_col, user_key, entry)
        if not is_private: store_message(group_history_col, group_name, entry)

        if reply:
            bot_entry = {"role": "assistant", "username": "PSI-09", "content": reply, ...}
            store_message(history_col, user_key, bot_entry)
            if not is_private: store_message(group_history_col, group_name, bot_entry)

        # 8. Trigger Background Evolution
        threading.Thread(target=background_evolution_tasks, daemon=True).start()

        return jsonify({"reply": reply, "reaction": reaction}), 200
```

---

## Background Processing

### Background Task Function (lines 609-617)

```python
def background_evolution_tasks():
    if is_private:
        with user_locks[user_key]:
            summarize_user_history(user_key, username, group_name, is_private)
    else:
        with group_locks[group_name]:
            summarize_group_history(group_name)

threading.Thread(target=background_evolution_tasks, daemon=True).start()
```

**Key Features:**
- Runs in separate daemon thread (won't block response)
- Uses locks to prevent concurrent updates to same graph
- Updates user graph for DMs, group graph for channels

---

## Data Flow Diagram

```
Discord Message Received
         │
         ▼
┌─────────────────────────────────────────────────┐
│  API Endpoint: POST /psi09                       │
│  - Parse JSON payload                           │
│  - Extract username, message, group_name        │
│  - Normalize Discord mentions → @PSI-09        │
└────────────────────────┬──────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  Context Assembly                               │
│  ┌─────────────────┐  ┌──────────────────────┐ │
│  │ GraphRAG        │  │ History Fetch        │ │
│  │ Context         │  │ (MongoDB)            │ │
│  │ (NetworkX)      │  │ (10 messages)        │ │
│  └─────────────────┘  └──────────────────────┘ │
└────────────────────────┬──────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  Triage Node (Failover LMPool)                  │
│  Input: history, message, is_direct             │
│  Output: should_engage (bool)                   │
│  Model: triage_pool.models[...]                 │
└────────────────────────┬──────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        ┌──────────┐          ┌──────────┐
        │  TRUE    │          │  FALSE   │
        └────┬─────┘          └──────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│  Combat Node (Failover LMPool)                  │
│  Input: history, graph, user, message           │
│  Pipeline: Identity → Mission → Constraints    │
│                    → Decision                   │
│  Output: reply, reaction, reasoning             │
│  Model: combat_pool.models[...]                 │
└────────────────────────┬──────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  Response Delivery                               │
│  - Store user message in MongoDB                 │
│  - Store bot response in MongoDB                │
│  - Return JSON: {reply, reaction}               │
└────────────────────────┬──────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  Background Thread (Daemon)                      │
│  ┌─────────────────┐  ┌──────────────────────┐ │
│  │ User Graph      │  │ Group Graph          │ │
│  │ Extraction      │  │ Extraction           │ │
│  │ (Lock: user_key)│  │ (Lock: group_name)   │ │
│  └─────────────────┘  └──────────────────────┘ │
│  Uses: background_pool.models[...]              │
└─────────────────────────────────────────────────┘
```

---

## Utility Functions

### Message History (lines 365-368)

```python
def fetch_history(collection, doc_id, limit):
    doc = collection.find_one(
        {"_id": doc_id}, 
        {"messages": {"$slice": -limit}}
    )
    return doc["messages"] if doc and "messages" in doc else []
```

### Message Storage (lines 370-372)

```python
def store_message(col, doc_id, entry):
    col.update_one(
        {"_id": doc_id}, 
        {"$push": {"messages": {"$each": [entry], "$slice": -50000}}},
        upsert=True
    )
```

### Bot Mention Detection (lines 374-379)

```python
def bot_mentioned_in(text: str) -> bool:
    if not text: return False
    if re.search(r"@psi-09", text, flags=re.IGNORECASE): return True
    for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
        if d_id and re.search(r"<@!?" + re.escape(str(d_id)) + r">", text):
            return True
    return False
```

---

## Error Handling

### Retry Logic Pattern

All LLM calls follow this pattern:

```python
max_retries = len(pool.models)
for attempt in range(max_retries):
    current_lm, current_index = pool.get_current()
    try:
        with dspy.context(lm=current_lm):
            result = some_dspy_module(...)
        break  # Success
    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            pool.advance(current_index)  # Retry with next model
        else:
            pool.advance(current_index)  # Also retry on other errors
```

### Exception Handling

- **Rate Limits (429)**: Automatic failover to next model
- **Other Exceptions**: Logged and trigger failover
- **All Models Failed**: Return safe default values

---

## Security Considerations

1. **API Key Separation**: Each pool has dedicated API key
2. **Input Sanitization**: Discord mentions/pings are sanitized before LLM
3. **Snowflake Removal**: User IDs removed from chat history
4. **No PII in Logs**: Sensitive data excluded from logging
5. **TLS Connection**: MongoDB uses TLS with certifi

---

## Performance Characteristics

| Component | Timeout/TTL | Notes |
|-----------|-------------|-------|
| In-Memory Cache TTL | 500s | Balances freshness vs. speed |
| Max Group History | 50,000 | Slice window for extraction: 80 |
| Max User History | 16 | Recent context only |
| Thread Pool | Unlimited | Daemon threads for background |
| LLM Retry | len(models) | Up to 5 retries per pool |

---

## Extension Points

### Adding New Signatures

1. Define new `dspy.Signature` class with input/output fields
2. Create Pydantic model if strict output required
3. Add to appropriate module or create new module
4. Integrate into combat/combat_engine flow

### Adding New Nodes

1. Define node function with `CombatState` parameter
2. Return dict with state updates
3. Add to workflow with `workflow.add_node()`
4. Add edges with `workflow.add_edge()` or `add_conditional_edges()`

### Adding New Model Pools

1. Add model list to `Config` dataclass
2. Add API key to `Config` dataclass
3. Initialize `FailoverLMPool` instance
4. Use in appropriate nodes

---

## Monitoring & Logging

All operations log with structured format:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
```

Key log points:
- Triage decisions: `Triage processed by: {model} | Pinged: {bool} -> Engage: {bool}`
- Combat results: `LangGraph Trace: ENGAGED|IGNORED | {reasoning}`
- Graph updates: `User Graph Updated flawlessly for {user_key} via {model}`
- Failover events: `[{POOL_NAME}] Rate Limit! Failover triggered -> {model}`
