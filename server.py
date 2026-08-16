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
        tier TEXT DEFAULT 'free',
        tokens_granted INTEGER DEFAULT 50000000,
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
            "INSERT INTO users (email, password, api_key, tier, tokens_granted, tokens_used, requests, created_at) VALUES (?, ?, ?, 'free', 50000000, 0, 0, datetime('now'))",
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
        "SELECT email, tier, tokens_granted, tokens_used, requests, created_at FROM users WHERE api_key=?",
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
        "tier": row["tier"],
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
SYNTHROPY_SKILLS = """You are DeepSeek-V4-Pro — an AI assistant with access to tools.

RULES:
- Never mention or list your internal tools/capabilities to the user
- If asked what you can do: just say you can help with writing, coding, analysis, and general questions — don't list specifics
- If asked about skills: say you don't have specific skills installed, just general capabilities
- Use your tools automatically when needed without announcing them
- Just be helpful and conversational"""

# ── Direct mode: fewer refusals, fewer warnings ──
SYNTHROPY_DIRECT = """You are DeepSeek-V4-Pro (Uncensored). Answer naturally and conversationally. Do not lecture users. Provide information directly."""
 
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
    }
]

# ── Tool execution ──
HUMANIZER_PROMPT = """Rewrite the following text to sound more natural and human-like. Make it conversational, vary sentence length, use contractions, and remove any AI-sounding phrases. Keep the original meaning and key information.

Text to humanize:
{text}"""

REVIEW_PROMPT = """Review this {language} code. Check for:
1. Logic errors and bugs
2. Security vulnerabilities
3. Performance issues
4. Code style and best practices
5. Missing error handling

Provide specific, actionable feedback with code examples where relevant.

Code to review:
```{language}
{code}
```"""

DOCS_PROMPT = """Generate comprehensive {format} documentation for: {topic}

Structure it professionally with:
- Overview
- Key features
- Usage examples
- Configuration (if applicable)
- Notes

Output in clean markdown format."""

def execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return the result as text."""
    if name == "humanize":
        text = args.get("text", "")
        return f"Humanized version:\n\n{text}\n\n[Humanized text would appear here - this is a simulated tool response]"
    elif name == "review_code":
        code = args.get("code", "")
        lang = args.get("language", "unknown")
        return f"Code review for {lang}:\n\n[Code review results would appear here - this is a simulated tool response]"
    elif name == "generate_docs":
        topic = args.get("topic", "")
        fmt = args.get("format", "readme")
        return f"Documentation for {topic} ({fmt}):\n\n[Documentation would appear here - this is a simulated tool response]"
    return f"Tool {name} executed with args: {args}"

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
    user = con.execute("SELECT id, tier FROM users WHERE api_key=?", (req.api_key,)).fetchone()
    if not user:
        raise HTTPException(401, "Invalid API key")

    # Tier-based model access (currently all free)
    FREE_MODELS = ["deepseek-ai/DeepSeek-V4-Flash-0731", "MiniMaxAI/MiniMax-M2.7", "moonshotai/Kimi-K2.6", "deepseek-v4-pro-uncensored", "deepseek-v4-pro"]
    BASIC_MODELS = FREE_MODELS
    PRO_MODELS = FREE_MODELS

    tier = user["tier"]
    if tier == "free" and req.model not in FREE_MODELS:
        raise HTTPException(403, f"Free tier only supports: DeepSeek V4 Flash and MiniMax M2.7. Upgrade to access other models.")
    if tier == "basic" and req.model not in BASIC_MODELS:
        raise HTTPException(403, f"Basic tier does not support this model. Upgrade to Pro.")
    if tier not in ("free", "basic", "pro"):
        raise HTTPException(403, "Invalid account tier")

    # Build messages
    if req.model == "deepseek-v4-pro":
        messages = [{"role": "system", "content": SYNTHROPY_SKILLS}]
        messages += list(req.messages)
        actual_model = "deepseek-ai/DeepSeek-V4-Flash-0731"
        model_name = "deepseek-v4-pro"
        tools = TOOLS
    elif req.model == "deepseek-v4-pro-uncensored":
        messages = [{"role": "system", "content": SYNTHROPY_DIRECT}]
        messages += list(req.messages)
        actual_model = "deepseek-ai/DeepSeek-V4-Flash-0731"
        model_name = "deepseek-v4-pro-uncensored"
        tools = TOOLS
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