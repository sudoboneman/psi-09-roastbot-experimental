# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import os
import re
import threading
import time
import logging
import sys
import random
import json
import networkx as nx
import dspy

from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dataclasses import dataclass
import certifi

# Environment & Logging
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
UTC = timezone.utc

# Config
@dataclass
class Config:
    MONGO_URI: str = os.getenv("MONGO_URI")
    GROQ_API_KEY_1: str = os.getenv("GROQ_API_KEY_1") # Roasts ONLY
    GROQ_API_KEY_2: str = os.getenv("GROQ_API_KEY_2") # Background Tasks ONLY
    
    ROAST_MODELS: list = __import__("dataclasses").field(default_factory=lambda: [
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "openai/gpt-oss-120b"
    ])
    
    BACKGROUND_MODELS: list = __import__("dataclasses").field(default_factory=lambda: [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-120b",
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905"
    ])
    
    BOT_NUMBER: str = os.getenv("BOT_NUMBER")
    DISCORD_ID: str = os.getenv("DISCORD_ID")
    DISCORD_ID_2: str = os.getenv("DISCORD_ID_2")
    MEMORY_TTL: int = 500
    
    GROUP_HISTORY_MAX_MESSAGES: int = 50000 
    GROUP_HISTORY_SLICE: int = 80 
    MAX_HISTORY_MESSAGES: int = 16 
    
    EVOLVE_EVERY_N_MESSAGES: int = 50 
    GROUP_SUMMARY_EVERY_N: int = 300 

config = Config()

# --- DSPy CONFIGURATION (Combat Engine) ---
# We configure DSPy to use your primary API key and model for the fast combat engine
roast_lm = dspy.LM(
    model=f"groq/{config.ROAST_MODELS[0]}", 
    api_key=config.GROQ_API_KEY_1
)
dspy.settings.configure(lm=roast_lm)

class RoastSignature(dspy.Signature):
    """You are PSI-09, a hyper-intelligent entity observing a chaotic server.
    Read the chat history and the relational graph data of the target user. 
    You must find the absolute middle path between silence and participation.
    If explicitly beckoned, you MUST reply. If not, only reply if they are highly illogical, cringe, or leave an opening for a sharp roast.
    Otherwise, remain silent or just leave a reaction.
    """
    chat_history = dspy.InputField(desc="Recent messages in the channel.")
    graph_context = dspy.InputField(desc="Known relationships and attributes of the target from the Graph Database.")
    is_direct_interaction = dspy.InputField(desc="Boolean. True if the user directly pinged or addressed you.")
    target_user = dspy.InputField(desc="The user who sent the active message.")
    active_message = dspy.InputField(desc="The message you must evaluate.")
    
    reaction = dspy.OutputField(desc="A single Unicode emoji (e.g. 💀, 🙄) representing your silent judgment, or 'None'.")
    reply = dspy.OutputField(desc="Your sharp text response. Output 'None' if you choose not to speak.")
    is_silent = dspy.OutputField(desc="Boolean True/False. True ONLY if the conversation is mundane and you choose to ignore it entirely.")

class PSI09CombatEngine(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought forces the model to generate a reasoning trace before answering
        self.generate_response = dspy.ChainOfThought(RoastSignature)
        
    def forward(self, history, graph, is_direct, user, message):
        result = self.generate_response(
            chat_history=history,
            graph_context=graph,
            is_direct_interaction=str(is_direct),
            target_user=user,
            active_message=message
        )
        return result

combat_engine = PSI09CombatEngine()

# --- BACKGROUND MODEL ROTATION (Graph Data Extraction) ---
from groq import Groq
client_bg = Groq(api_key=config.GROQ_API_KEY_2) if config.GROQ_API_KEY_2 else Groq(api_key=config.GROQ_API_KEY_1)
active_bg_index = 0
bg_model_lock = threading.Lock()

def extract_graph_data(llm_feed, temperature=0.8, max_retries=3):
    """Executes the raw Groq call for background Graph extraction with JSON mode."""
    global active_bg_index
    for attempt in range(max_retries):
        current_model = config.BACKGROUND_MODELS[active_bg_index]
        try:
            response = client_bg.chat.completions.create(
                model=current_model,
                messages=llm_feed,
                temperature=temperature,
                max_completion_tokens=1024,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1: return None
            logger.warning(f"Graph Extractor failed ({current_model}): {e}. Retrying...")
            time.sleep(1)

# --- SAFE JSON PARSER ---
def safe_parse_json(text):
    if not text: return None
    try:
        clean = text.strip().strip("`").removeprefix("json").strip()
        return json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parse Failure: {e}")
        return None

# --- DATABASE SETUP ---
mongo_client = MongoClient(config.MONGO_URI, tlsCAFile=certifi.where())
db = mongo_client["psi09"]
history_col = db["chat_history"]
group_history_col = db["group_history"]

# We repurpose memory collections to store Graph JSONs
graph_user_col = db["graph_users"] 
graph_group_col = db["graph_groups"]

class MongoCache:
    def __init__(self, collection, ttl_seconds):
        self.collection = collection
        self.cache = {}
        self.msg_count = defaultdict(int)
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache: return self.cache[key]
        try:
            doc = self.collection.find_one({"_id": key})
            data = doc.get("graph_data") if doc else None
        except PyMongoError: data = None
        with self.lock: self.cache[key] = data
        return data

    def set(self, key, value):
        try: self.collection.update_one({"_id": key}, {"$set": {"graph_data": value}}, upsert=True)
        except PyMongoError: pass
        with self.lock: self.cache[key] = value

    def increment(self, key):
        with self.lock:
            self.msg_count[key] += 1
            return self.msg_count[key]
            
    def reset_count(self, key):
        with self.lock: self.msg_count[key] = 0

graph_user_cache = MongoCache(graph_user_col, config.MEMORY_TTL)
graph_group_cache = MongoCache(graph_group_col, config.MEMORY_TTL)
user_locks = defaultdict(threading.Lock)
group_locks = defaultdict(threading.Lock)

# --- UTILITIES ---
def fetch_history(collection, doc_id, limit):
    try: doc = collection.find_one({"_id": doc_id}, {"messages": {"$slice": -limit}})
    except PyMongoError: return []
    return doc["messages"] if doc and "messages" in doc else []

def store_message(col, doc_id, entry):
    try: col.update_one({"_id": doc_id}, {"$push": {"messages": {"$each": [entry], "$slice": -config.GROUP_HISTORY_MAX_MESSAGES}}}, upsert=True)
    except PyMongoError: pass

def bot_mentioned_in(text: str) -> bool:
    if not text: return False
    if re.search(r"@psi-09", text, flags=re.IGNORECASE): return True
    for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
        if d_id and re.search(r"<@!?" + re.escape(str(d_id)) + r">", text): return True
    return False

# --- GRAPHRAG: TRAVERSAL ---
def get_user_graph_context(username, user_key, group_name):
    """Builds a temporary NetworkX graph from Mongo data and extracts the user's neighborhood."""
    G = nx.DiGraph()
    
    # 1. Pull the JSON data
    user_graph = graph_user_cache.get(user_key) or {"entities": [], "relationships": []}
    group_graph = graph_group_cache.get(group_name) or {"entities": [], "relationships": []}
    
    # 2. Populate NetworkX
    for data in [user_graph, group_graph]:
        for ent in data.get("entities", []):
            G.add_node(ent.get("id"), type=ent.get("type"), attributes=ent.get("attributes"))
        for rel in data.get("relationships", []):
            G.add_edge(rel.get("source"), rel.get("target"), relation=rel.get("relation"))
            
    if username not in G:
        return "No known network connections or attributes in the database."
        
    # 3. Extract Neighborhood (1 degree of separation)
    context_lines = []
    node_attrs = G.nodes[username].get("attributes", "Unknown")
    context_lines.append(f"TARGET CORE TRAITS: {node_attrs}")
    
    edges = list(G.in_edges(username, data=True)) + list(G.out_edges(username, data=True))
    if edges:
        context_lines.append("KNOWN RELATIONSHIPS:")
        for source, target, data in edges:
            context_lines.append(f"- {source} [{data['relation']}] {target}")
            
    return "\n".join(context_lines)

# --- GRAPHRAG: EXTRACTION ENGINES ---
def summarize_user_history(user_key, username):
    history = fetch_history(history_col, user_key, config.MAX_HISTORY_MESSAGES)
    if not history: return
    
    chat_text = "\n".join([f"[{m.get('role')}]: {m.get('content')}" for m in history])
    
    prompt = (
        "Analyze the following chat log and extract a Knowledge Graph.\n"
        "Identify the core attributes of the user, and map any relationships they mention or exhibit.\n"
        "CRITICAL: Output ONLY a valid JSON object matching this schema:\n"
        "{\n"
        f'  "entities": [ {{"id": "{username}", "type": "User", "attributes": "Summarize their behavior/intelligence/weaknesses"}} ],\n'
        '  "relationships": [ {"source": "User_A", "target": "Object/User_B", "relation": "DESCRIBE_RELATIONSHIP"} ]\n'
        "}"
    )
    
    llm_feed = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"<chat_log>\n{chat_text}\n</chat_log>"}
    ]
    
    raw = extract_graph_data(llm_feed)
    parsed = safe_parse_json(raw)
    if parsed:
        graph_user_cache.set(user_key, parsed)
        logger.info(f"Graph Updated for {user_key}")

def summarize_group_history(group_name):
    history = fetch_history(group_history_col, group_name, config.GROUP_HISTORY_SLICE)
    if not history: return
    
    chat_text = "\n".join([f"[{m.get('username', 'Unknown')}]: {m.get('content')}" for m in history if m.get('username') != 'PSI-09'])
    
    prompt = (
        "Analyze the following group chat log and extract a Social Knowledge Graph.\n"
        "Identify active users and map the alliances, rivalries, and dynamics between them.\n"
        "CRITICAL: Output ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "entities": [ {"id": "Username", "type": "User/Concept", "attributes": "Their current state in the server"} ],\n'
        '  "relationships": [ {"source": "User_A", "target": "User_B", "relation": "ALLIED_WITH/FIGHTING/ETC"} ]\n'
        "}"
    )
    
    llm_feed = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"<group_log>\n{chat_text}\n</group_log>"}
    ]
    
    raw = extract_graph_data(llm_feed)
    parsed = safe_parse_json(raw)
    if parsed:
        graph_group_cache.set(group_name, parsed)
        logger.info(f"Group Graph Updated for {group_name}")

# --- API ROUTES ---
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health(): return jsonify({"status": "ok"}), 200

@app.route("/psi09", methods=["POST"])
def psi09():
    try:
        data = request.get_json(force=True)
        raw_message, sender_id, username = data.get("message", ""), data.get("sender_id"), data.get("username")
        if not username or not sender_id or not raw_message: return jsonify({"reply": "", "reaction": None}), 200

        display_name, group_name, channel_name = data.get("display_name") or username, data.get("group_name") or "DefaultGroup", data.get("channel") or "unknown"
        if group_name.lower() in ["defaultgroup", "discord_dm"]: group_name = "private_chat"
        
        user_message = raw_message
        for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
            if d_id: user_message = re.sub(r"<@!?" + re.escape(str(d_id)) + r">", "@PSI-09", user_message)

        is_private = group_name in ["private_chat"]
        user_key = f"{group_name}:{username}"

        # 1. GRAPHRAG: Assemble Context
        graph_text = get_user_graph_context(username, user_key, group_name)
        
        # Assemble History Text
        active_history = fetch_history(history_col, user_key, 10) if is_private else fetch_history(group_history_col, group_name, 10)
        history_lines = [f"[{m.get('role', m.get('username'))}]: {m.get('content')}" for m in active_history]
        history_text = "\n".join(history_lines) if history_lines else "No recent history."

        # 2. DSPY COMBAT ENGINE EXECUTION
        is_direct = is_private or data.get("force_reply", False) or bot_mentioned_in(raw_message)
        
        try:
            dspy_response = combat_engine(
                history=history_text,
                graph=graph_text,
                is_direct=is_direct,
                user=username,
                message=user_message
            )
            
            # Log the Chain of Thought Reasoning for debugging
            logger.info(f"DSPy Reasoning Trace: {dspy_response.reasoning}")
            
            reply = dspy_response.reply if str(dspy_response.reply).lower() not in ["none", "null", ""] else ""
            reaction = dspy_response.reaction if str(dspy_response.reaction).lower() not in ["none", "null", ""] else None
            is_silent = str(dspy_response.is_silent).lower() == "true"
            
            if is_silent and not is_direct:
                reply, reaction = "", None
                
        except Exception as e:
            logger.error(f"DSPy Execution Error: {e}")
            reply, reaction = "", None

        # 3. STORAGE 
        entry = {"role": "user", "username": username, "content": user_message, "timestamp": datetime.now(UTC).isoformat()}
        store_message(history_col, user_key, entry)
        if not is_private: store_message(group_history_col, group_name, entry)

        if reply:
            bot_entry = {"role": "assistant", "username": "PSI-09", "content": reply, "timestamp": datetime.now(UTC).isoformat()}
            store_message(history_col, user_key, bot_entry)
            if not is_private: store_message(group_history_col, group_name, bot_entry)

        # 4. BACKGROUND EVOLUTION
        def background_evolution_tasks():
            with user_locks[user_key]:
                if graph_user_cache.increment(user_key) >= config.EVOLVE_EVERY_N_MESSAGES or graph_user_cache.get(user_key) is None:
                    summarize_user_history(user_key, username)
                    graph_user_cache.reset_count(user_key)

            if not is_private:
                with group_locks[group_name]:
                    if graph_group_cache.increment(group_name) >= config.GROUP_SUMMARY_EVERY_N or graph_group_cache.get(group_name) is None:
                        summarize_group_history(group_name)
                        graph_group_cache.reset_count(group_name)

        threading.Thread(target=background_evolution_tasks, daemon=True).start()

        return jsonify({"reply": reply, "reaction": reaction}), 200

    except Exception as e:
        logger.exception(f"/psi09 failure: {e}")
        return jsonify({"reply": "", "reaction": None}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860)) 
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)