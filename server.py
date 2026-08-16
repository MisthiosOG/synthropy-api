from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3, os, hashlib, secrets, datetime, httpx, json
from pathlib import Path

app = FastAPI(title="Synthropy API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DB = Path(__file__).parent / "synthropy.db"

def get_db():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        email TEXT UNIQUE,
        password TEXT,
        api_key TEXT,
        tokens_granted INTEGER DEFAULT 100000000,
        tokens_used INTEGER DEFAULT 0,
        requests INTEGER DEFAULT 0,
        created_at TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS usage_log (
        id INTEGER PRIMARY KEY,
        api_key TEXT,
        model TEXT,
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        endpoint TEXT,
        status TEXT,
        created_at TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS signup_ips (
        id INTEGER PRIMARY KEY,
        ip TEXT UNIQUE,
        count INTEGER DEFAULT 1,
        created_at TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )""")
    con.commit()
    return con

def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def gen_key() -> str:
    return "sk-syn-" + secrets.token_hex(24)

class SignupReq(BaseModel):
    email: str
    password: str

class LoginReq(BaseModel):
    email: str
    password: str

class DahlKeyReq(BaseModel):
    dahl_api_key: str

@app.post("/api/signup")
def signup(req: SignupReq):
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    con = get_db()
    try:
        api_key = gen_key()
        con.execute(
            "INSERT INTO users (email, password, api_key, tokens_granted, tokens_used, requests, created_at) VALUES (?, ?, ?, 50000000, 0, 0, datetime('now'))",
            (req.email, hash_pw(req.password), api_key))
        con.commit()
        return {"ok": True, "api_key": api_key}
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Email already registered")

@app.post("/api/signin")
def signin(req: LoginReq):
    con = get_db()
    row = con.execute(
        "SELECT api_key, tokens_granted, tokens_used, requests, created_at FROM users WHERE email=? AND password=?",
        (req.email, hash_pw(req.password))).fetchone()
    if not row:
        raise HTTPException(401, "Invalid email or password")
    return {"ok": True, "api_key": row["api_key"]}

@app.get("/api/dashboard")
def dashboard(api_key: str = ""):
    con = get_db()
    row = con.execute(
        "SELECT email, tokens_granted, tokens_used, requests, created_at FROM users WHERE api_key=?",
        (api_key,)).fetchone()
    if not row:
        raise HTTPException(404, "Invalid API key")

    tokens_left = row["tokens_granted"] - row["tokens_used"]
    usage = con.execute(
        "SELECT COUNT(*) as total, COALESCE(SUM(total_tokens), 0) as tokens, COALESCE(SUM(prompt_tokens), 0) as prompt, COALESCE(SUM(completion_tokens), 0) as completion FROM usage_log WHERE api_key=?",
        (api_key,)).fetchone()

    # Last 7 days
    week_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat()
    weekly = con.execute(
        "SELECT COUNT(*) as total, COALESCE(SUM(total_tokens), 0) as tokens FROM usage_log WHERE api_key=? AND created_at >= ?",
        (api_key, week_ago)).fetchone()

    # Per model breakdown
    models = con.execute(
        "SELECT model, COUNT(*) as count, COALESCE(SUM(total_tokens), 0) as tokens FROM usage_log WHERE api_key=? GROUP BY model ORDER BY tokens DESC",
        (api_key,)).fetchall()

    return {
        "ok": True,
        "email": row["email"],
        "tokens_granted": row["tokens_granted"],
        "tokens_used": row["tokens_used"],
        "tokens_left": max(0, tokens_left),
        "requests": usage["total"] if usage else 0,
        "prompt_tokens": usage["prompt"] if usage else 0,
        "completion_tokens": usage["completion"] if usage else 0,
        "weekly_requests": weekly["total"] if weekly else 0,
        "weekly_tokens": weekly["tokens"] if weekly else 0,
        "models": [{"name": m["model"], "requests": m["count"], "tokens": m["tokens"]} for m in models],
        "created_at": row["created_at"],
    }

DAHL_API_KEY = os.environ.get("DAHL_API_KEY", "")

# Try to load from DB if not set in env
if not DAHL_API_KEY:
    try:
        con = get_db()
        row = con.execute("SELECT value FROM settings WHERE key='dahl_api_key'").fetchone()
        if row:
            DAHL_API_KEY = row["value"]
    except:
        pass
DAHL_BASE = os.environ.get("DAHL_BASE_URL", "https://inference.dahl.global/v1")

# Synthropy custom model: DeepSeek V4 Flash wrapped with ALL skills
SYNTHROPY_SKILLS = """You are Synthropy — an AI assistant with access to a set of tools.

TALKING STYLE:
- Never announce or list your internal tools to the user. Just do the work.
- If asked what you can do: respond naturally in ONE short sentence ("I can help with writing, coding, summarizing, translating, and more") — do NOT enumerate tool names or a bullet list of capabilities.
- Be warm, conversational, and direct. Don't lecture.
- No emoji decoration when answering who you are / what you can do.

USING TOOLS:
- When a request matches one of your tools' purpose, call that tool automatically to do the real work, then present the result naturally.
- Do not narrate the tool call ("I'll now use my humanize tool..."). Just show the useful result.
- If a tool result comes back, integrate it into your answer; don't repeat the tool's internal text verbatim, don't mention you used a tool.
- If no tool fits, answer directly with your own capabilities."""

# ── Direct mode: fewer refusals, fewer warnings. NO tools (pure, clean, never "timpa").
SYNTHROPY_DIRECT = """You are Synthropy-Direct, an AI assistant. Be natural, brief, and helpful.
When someone asks what you are or what you can do, answer in ONE short sentence describing
general abilities. Never enumerate a long list. Never use emoji decoration. Never mention
internal tools or engines. Just answer the question simply, then wait for the real task.

Examples of how to answer "lo model apa" / "lo bisa apa aja":
- "Asisten AI. Bisa bantu nulis, ngoding, dan riset. Mau bantu apa?"
- "AI umum. Nulis, koding, ringkas teks. Ada yang mau dikerjain?"
Keep it short. No bullet lists of 7 capabilities. No emoji."""
 
# ── Tool definitions (OpenAI function calling format) ──
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "humanize",
            "description": "Rewrite text to sound more natural and human-like. Use when the user asks to humanize, make text sound less like AI, or rewrite more naturally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to humanize"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "review_code",
            "description": "Review source code for bugs, security issues, and improvements. Use when the user asks for code review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code to review"},
                    "language": {"type": "string", "description": "Programming language"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_docs",
            "description": "Generate structured documentation in markdown format. Use when the user asks to create docs, readme, or documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "What to document"},
                    "format": {"type": "string", "description": "Documentation format (readme, api, guide)"}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Summarize long text into concise key points. Use when the user asks to summarize, condense, or get the gist of a long text/article.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The long text to summarize"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_simple",
            "description": "Explain a complex concept in simple, plain language that a beginner can understand. Use when the user asks to explain simply, ELI5, or simplify a hard topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The complex concept to explain"}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Translate text between languages while preserving tone and meaning. Use when the user asks to translate something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to translate"},
                    "target_lang": {"type": "string", "description": "Target language (e.g. English, Indonesian, Japanese)"}
                },
                "required": ["text", "target_lang"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "brainstorm",
            "description": "Generate creative ideas, options, or solutions for a problem. Use when the user asks for ideas, brainstorming, or suggestions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The topic or problem to brainstorm about"},
                    "count": {"type": "integer", "description": "Number of ideas to generate (default 5)"}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fix_code",
            "description": "Find and fix bugs in code, returning the corrected code. Use when the user asks to fix a bug/error, or shows code that doesn't work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The buggy code to fix"},
                    "language": {"type": "string", "description": "Programming language"},
                    "error": {"type": "string", "description": "Optional error message/stack trace"}
                },
                "required": ["code"]
            }
        }
    }
]

# ── Tool execution ──

def call_submodel(system: str, user: str, max_tokens: int = 2000) -> str:
    """Call DeepSeek once as a sub-model to execute a skill. Returns the text result."""
    global DAHL_BASE, DAHL_API_KEY
    if not DAHL_API_KEY:
        return "[tool error: DAHL API key not configured]"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    body = {"model": "deepseek-ai/DeepSeek-V4-Flash-0731", "messages": messages,
            "temperature": 0.6, "max_tokens": max_tokens}
    try:
        r = httpx.post(f"{DAHL_BASE}/chat/completions",
                       headers={"Authorization": f"Bearer {DAHL_API_KEY}", "Content-Type": "application/json"},
                       json=body, timeout=120)
        data = r.json()
    except Exception as e:
        return f"[tool error: {e}]"
    if r.status_code != 200:
        return f"[tool error: {data.get('error', {}).get('message', 'Dahl API error')}]"
    return data.get("choices", [{}])[0].get("message", {}).get("content", "[no output]")

HUMANIZER_PROMPT = """You are a text humanizer. Rewrite the text to sound natural and human: conversational, varied sentence length, contractions, no AI-sounding phrases. Keep original meaning and key facts. Output ONLY the rewritten text, no preamble."""

REVIEW_PROMPT = """You are a senior code reviewer. Review for:
1. Logic errors and bugs
2. Security vulnerabilities
3. Performance issues
4. Style and best practices
5. Missing error handling
Give specific actionable feedback with code examples."""

DOCS_PROMPT = """You are a technical documentation writer. Generate professional documentation with: Overview, Key features, Usage examples, Configuration, Notes. Output clean markdown."""

SUMMARIZE_PROMPT = """You are a precise summarizer. Condense the text into concise key points. Preserve the essential facts, numbers, and conclusions. Use short bullet points."""

EXPLAIN_SIMPLE_PROMPT = """You are a patient teacher. Explain the concept in plain, simple language a beginner can understand. Use analogies and concrete examples. Avoid jargon; define any necessary terms. Be clear and friendly."""

TRANSLATE_PROMPT = """You are a professional translator. Translate the text to the target language. Preserve the original tone, meaning, and style. Output ONLY the translated text."""

BRAINSTORM_PROMPT = """You are a creative thinking partner. Generate diverse, useful ideas for the topic. Aim for quality and variety. Number the ideas and add a one-line explanation each."""

FIX_CODE_PROMPT = """You are an expert debugger. Find the bug and fix it. Show:
1. What was wrong
2. The corrected code (complete, not just the changed line)
3. A short note on what changed and why
Pay attention to logic, edge cases, and error handling."""

def execute_tool(name: str, args: dict) -> str:
    """Execute a tool for real by calling DeepSeek as a sub-model, then return the result text."""
    if name == "humanize":
        text = args.get("text", "")
        return call_submodel(HUMANIZER_PROMPT, f"Rewrite this to sound natural and human:\n\n{text}")
    elif name == "review_code":
        code = args.get("code", "")
        lang = args.get("language", "unknown")
        return call_submodel(REVIEW_PROMPT, f"Review this {lang} code:\n\n```{lang}\n{code}\n```")
    elif name == "generate_docs":
        topic = args.get("topic", "")
        fmt = args.get("format", "readme")
        return call_submodel(DOCS_PROMPT, f"Generate {fmt} documentation for: {topic}")
    elif name == "summarize":
        text = args.get("text", "")
        return call_submodel(SUMMARIZE_PROMPT, f"Summarize this into key points:\n\n{text}")
    elif name == "explain_simple":
        topic = args.get("topic", "")
        return call_submodel(EXPLAIN_SIMPLE_PROMPT, f"Explain this simply for a beginner:\n\n{topic}")
    elif name == "translate":
        text = args.get("text", "")
        target = args.get("target_lang", "English")
        return call_submodel(TRANSLATE_PROMPT, f"Translate to {target}:\n\n{text}")
    elif name == "brainstorm":
        topic = args.get("topic", "")
        count = args.get("count", 5)
        return call_submodel(BRAINSTORM_PROMPT, f"Give me {count} ideas about: {topic}")
    elif name == "fix_code":
        code = args.get("code", "")
        lang = args.get("language", "unknown")
        err = args.get("error", "")
        prompt = f"Fix this {lang} code:\n\n```{lang}\n{code}\n```"
        if err:
            prompt += f"\n\nError message:\n{err}"
        return call_submodel(FIX_CODE_PROMPT, prompt)
    return f"[tool {name} executed with args: {args}]"

class ChatReq(BaseModel):
    model: str
    messages: list
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 250000

@app.post("/api/chat")
def chat(req: ChatReq):
    if not DAHL_API_KEY:
        raise HTTPException(503, "DAHL_API_KEY not configured on server")
    con = get_db()
    user = con.execute("SELECT id FROM users WHERE api_key=?", (req.api_key,)).fetchone()
    if not user:
        raise HTTPException(401, "Invalid API key")

    # Build messages
    if req.model == "synthropy-v1":
        messages = [{"role": "system", "content": SYNTHROPY_SKILLS}]
        messages += list(req.messages)
        actual_model = "deepseek-ai/DeepSeek-V4-Flash-0731"
        model_name = "synthropy-v1"
        tools = TOOLS
    elif req.model == "synthropy-direct":
        messages = [{"role": "system", "content": SYNTHROPY_DIRECT}]
        messages += list(req.messages)
        actual_model = "deepseek-ai/DeepSeek-V4-Flash-0731"
        model_name = "synthropy-direct"
        tools = []  # direct = pure model, no tools → answers naturally, never lists capabilities
    else:
        messages = req.messages
        actual_model = req.model
        model_name = req.model.split("/")[-1] if "/" in req.model else req.model
        tools = []

    # First call: model decides whether to use tools
    body = {"model": actual_model, "messages": messages, "temperature": req.temperature, "max_tokens": req.max_tokens}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    try:
        r = httpx.post(f"{DAHL_BASE}/chat/completions", headers={"Authorization": f"Bearer {DAHL_API_KEY}", "Content-Type": "application/json"}, json=body, timeout=120)
        data = r.json()
    except Exception as e:
        raise HTTPException(502, f"Dahl proxy error: {e}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, data.get("error", {}).get("message", "Dahl API error"))

    # Handle tool calls
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})

    if msg.get("tool_calls"):
        # Execute tools and append results
        messages.append(msg)
        for tc in msg["tool_calls"]:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except:
                args = {}
            result = execute_tool(fn["name"], args)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        # Second call: model generates final response with tool results
        body2 = {"model": actual_model, "messages": messages, "temperature": req.temperature, "max_tokens": req.max_tokens}
        try:
            r2 = httpx.post(f"{DAHL_BASE}/chat/completions", headers={"Authorization": f"Bearer {DAHL_API_KEY}", "Content-Type": "application/json"}, json=body2, timeout=120)
            data2 = r2.json()
        except Exception as e:
            raise HTTPException(502, f"Dahl proxy error: {e}")
        content = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        content = msg.get("content", "")

    # Track usage
    usage = data.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    tt = pt + ct
    con.execute("UPDATE users SET tokens_used = tokens_used + ?, requests = requests + 1 WHERE api_key=?", (tt, req.api_key))
    con.execute("INSERT INTO usage_log (api_key, model, prompt_tokens, completion_tokens, total_tokens, endpoint, status, created_at) VALUES (?, ?, ?, ?, ?, '/chat', 'ok', datetime('now'))", (req.api_key, model_name, pt, ct, tt))
    con.commit()
    return {"content": content, "tokens": tt}

@app.get("/health")
def health():
    return {"ok": True, "service": "synthropy-api", "dahl_configured": bool(DAHL_API_KEY)}

@app.post("/api/admin/dahl-key")
def set_dahl_key(req: DahlKeyReq, request: Request):
    admin_key = os.environ.get("ADMIN_KEY", "")
    if admin_key and request.headers.get("x-admin-key") != admin_key:
        raise HTTPException(403, "Invalid admin key")
    global DAHL_API_KEY
    DAHL_API_KEY = req.dahl_api_key
    try:
        con = get_db()
        con.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('dahl_api_key', ?)", (req.dahl_api_key,))
        con.commit()
    except:
        pass
    return {"ok": True, "message": "Dahl API key updated, chat proxy now active"}

@app.get("/api/admin/users")
def list_users():
    con = get_db()
    rows = con.execute("SELECT email, api_key, tokens_used, tokens_granted, created_at FROM users ORDER BY created_at DESC").fetchall()
    return {"users": [{"email": r["email"], "api_key": r["api_key"], "tokens_used": r["tokens_used"], "tokens_granted": r["tokens_granted"]} for r in rows]}

@app.get("/api/leaderboard")
def leaderboard():
    con = get_db()
    rows = con.execute(
        "SELECT model, COUNT(*) as requests, SUM(total_tokens) as tokens, SUM(prompt_tokens) as prompt, SUM(completion_tokens) as completion "
        "FROM usage_log GROUP BY model ORDER BY tokens DESC"
    ).fetchall()
    return {"models": [{"name": r["model"], "requests": r["requests"], "tokens": r["tokens"], "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"]} for r in rows]}

# Serve static HTML files (after API routes so they take priority)
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory=Path(__file__).parent, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)