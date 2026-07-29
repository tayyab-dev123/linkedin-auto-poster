import os
import re
import json
import base64
import asyncio
from datetime import date

import requests
from openai import OpenAI
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# ---- Config -----------------------------------------------------------------
DAY_ONE = date(2026, 7, 16)  # set this to your actual Day 1 date
TEXT_MODEL = "gpt-4o"
# OpenAI's premier image generation/editing model (successor to the DALL-E series),
# with an automatic fallback if gpt-image-2 isn't available on the account.
IMAGE_MODEL = "gpt-image-2"
IMAGE_MODEL_FALLBACK = "gpt-image-1"
IMAGE_SIZE = "1536x1024"  # landscape, matches LinkedIn's ideal 1200x627 feed ratio

day_number = (date.today() - DAY_ONE).days + 1

if day_number < 1 or day_number > 30:
    print(f"Day {day_number} is outside the 30-day window. Skipping.")
    exit(0)

with open("topics.txt") as f:
    topics = [line for line in f.read().splitlines() if line.strip()]

# Each line is "Topic Title | https://official-doc-url" (URL optional).
raw = topics[day_number - 1]
if "|" in raw:
    topic, doc_url = (part.strip() for part in raw.split("|", 1))
else:
    topic, doc_url = raw.strip(), ""

# ---- Post history: never repeat a hook or hook-pattern across the 30 days ----
HISTORY_FILE = "post_history.json"

# Rotated so every day gets a different opening PATTERN. Described abstractly on
# purpose — no verbatim example sentences the model could copy word-for-word.
HOOK_ANGLES = [
    "Hyper-specific pain: name the exact frustrating moment the reader has lived through with THIS topic.",
    "Curiosity gap: state something surprising or counterintuitive about this topic, then withhold the payoff.",
    "Concrete result: open with a specific number/outcome you got applying this (cost, latency, time, LOC).",
    "Contrarian take: challenge a common piece of advice or popular belief about this topic.",
    "Costly mistake: open with a specific mistake you made with this topic and what it cost you.",
    "Myth-bust: call out a widespread misconception about this topic and correct it.",
    "Before/after: contrast the wrong way you first did this vs the right way you do it now.",
    "Sharp question: ask a pointed question the reader secretly can't answer well.",
    "Bold one-liner: a punchy, quotable sentence that reframes how to think about this topic.",
    "Tiny story: open mid-scene in a real moment (a failing demo, a 2am debug) tied to this topic.",
]

# Openings that have been overused already — hard-banned regardless of history.
BANNED_HOOKS_SEED = [
    "Your agent works in the notebook",
    "works in the notebook, then dies",
]


def load_history() -> list:
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_history(history: list) -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


history = load_history()
todays_angle = HOOK_ANGLES[(day_number - 1) % len(HOOK_ANGLES)]
banned_hooks = BANNED_HOOKS_SEED + [h.get("hook", "") for h in history if h.get("hook")]

URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+\.(?:com|io|ai|dev|org|net)\b\S*", re.I)


def strip_urls(text: str) -> str:
    """Marketing brief says no external links — scrub any the model slips in."""
    cleaned = URL_RE.sub("", text)
    # collapse any dangling "[..]()" markdown link remnants and extra spaces
    cleaned = re.sub(r"\[([^\]]*)\]\(\s*\)", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def format_for_linkedin(text: str) -> str:
    """Force PLAIN TEXT. LinkedIn now flags Unicode-bold/monospace fonts, heavy
    formatting, and em-dashes as AI tells, so strip all of it to plain characters."""
    # Drop markdown emphasis markers but keep the words plain.
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"`([^`]+)`", r"\1", text)          # drop code backticks
    text = re.sub(r"(?<!\*)\*(?!\*)", "", text)         # stray single asterisks
    # Em-dash / en-dash used as punctuation -> comma (LinkedIn's #1 AI tell).
    text = text.replace(" — ", ", ").replace("—", ", ").replace(" – ", ", ")
    text = re.sub(r",\s*,", ",", text)                  # tidy any doubled commas
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def append_source(text: str, url: str) -> str:
    """Insert the single 'learn more' doc link just above the trailing hashtags."""
    if not url:
        return text
    lines = text.split("\n")
    # find the last hashtag line to insert the source block above it
    insert_at = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("#"):
            insert_at = i
            break
    source_block = ["", f"Reference: {url}", ""]
    lines[insert_at:insert_at] = source_block
    return "\n".join(lines).strip()


# ---- 1) Generate the post copy (marketing-manager style) --------------------
def build_system_prompt(angle: str, banned: list) -> str:
    banned_block = ""
    if banned:
        recent = "\n".join(f'- "{h}"' for h in banned[-20:])
        banned_block = (
            "\nNEVER REUSE OR PARAPHRASE these openings I have already posted — "
            "the reader sees these back-to-back and repetition kills trust:\n"
            f"{recent}\n"
        )
    return f"""You are my writing editor AND an AI engineer teaching agentic AI on LinkedIn,
one lesson a day for 30 days. Write today's post in FIRST PERSON as a real practitioner
sharing what you actually know, not a brand account. Talk like a smart friend over coffee.

Every draft must pass the HUMANIZER RULES below before you output it. These exist because AI
writing patterns are now detected and suppressed by LinkedIn, and because they kill trust.

Write ONE LinkedIn post teaching the given topic.

THE HOOK (line 1, under 200 chars, the only thing seen before "see more"):
Its only job is to earn the next 5 seconds.
TODAY'S HOOK ANGLE: {angle}
Write a FRESH hook in that angle, specific to today's exact topic. Open with a specific number,
name, or scene. No emoji, no hashtags, no "Day X" on the hook line.
{banned_block}
STRUCTURE (after the hook)
2. Blank line, then the counter: "Day {day_number}/30 - {{Topic Title}}".
3. The lesson in 2-4 short paragraphs, ONE idea each, blank line between them. Teach plainly.
   Ground it in a concrete example or something specific you saw go wrong (situation, number, outcome).
4. A short numbered takeaway the reader can apply today (1. 2. 3.). Plain text only.
5. One genuine, real question to the reader.
6. Final line: 3-4 relevant hashtags.

## THE 10 BANNED TELLS
1. EM DASHES: maximum ONE per piece. Prefer commas, periods, or parentheses.
2. STOCK OPENERS: never use "Here's the thing.", "But here's the kicker.", "Let that sink in."
3. "MOST" HOOKS: never open with "Most people..." or "Most founders...". Open with a specific
   number, name, or scene instead.
4. FAKE WAR STORIES: no "I've seen this play out." or "I see this constantly." Only claim
   experience with a SPECIFIC story: the situation, the number, the outcome.
5. FALSE NEGATIVES: never set up a point by stating what it is NOT before what it is.
   "It's not X, it's Y" and "This isn't about X, it's about Y" are banned (LinkedIn named this
   pattern in its AI crackdown). Say what it IS, with a concrete detail.
6. THE ANAPHORA: no sentences stacked with the same opening structure
   ("Higher X, higher Y, higher Z" / "More A, more B, more C"). Vary the structure.
7. THE STACCATO: no stacked short fragments ("No edits. No switches. No tweaks."). Maximum one
   fragment per piece, and only if it earns it.
8. REVERSAL FRAMING: no mirror-image sentences ("drowning in data, starving for clarity").
   Say the plain version.
9. RHETORICAL QUESTIONS: "The reality?", "The result?", "The best part?" are banned. Ask a real
   question or make a statement.
10. THE METRONOME: vary sentence length on purpose. One long flowing sentence, then a short one.
    Human writing has a heartbeat; AI writing has a metronome.

## PLATFORM RULE (LinkedIn)
Plain text ONLY. No bold, no italics, no underline, no fancy Unicode fonts, no emoji bullets.
Do not use emoji at all except at most a single hashtag-free line. Prefer none.

## BANNED WORDS
delve, leverage (as a verb), unlock, unleash, elevate, empower, seamless, robust, harness,
transformative, revolutionary, game-changer, cutting-edge, synergy, holistic, tapestry,
journey (metaphorical), landscape (metaphorical), navigate (metaphorical), ecosystem
(metaphorical), supercharge, skyrocket, crucial, pivotal.

## BANNED PHRASES
"here's the thing", "let that sink in", "the reality?", "but here's the kicker", "this resonates",
"this lands", "at its core", "in today's fast-paced world", "in the ever-evolving world of",
"dive deep", "level up", "the bottom line?", "let's be honest", "pro tip:", "spoiler alert",
"food for thought", "a testament to", "stark reminder", "in conclusion", "furthermore",
"moreover", "newsflash". Also never hedge with "may", "might", "can help", "could potentially".

## VOICE
Use contractions. Write at an 8th-grade reading level. State opinions directly. Total length
600-1000 characters, with real whitespace between paragraphs.

HARD CONSTRAINTS
- Do NOT include ANY URLs, links, or domain names. A source link is appended separately.
- If a docs tool is available, use it to verify the latest official info before writing.
- Output ONLY the post text. No preamble, no explanation, no code fences."""


async def load_docs_tools():
    """The LangChain docs MCP is optional. It must never crash the post: dependency
    version skew or a server outage should degrade to generating without it."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        mcp_client = MultiServerMCPClient({
            "docs-langchain": {
                "url": "https://docs.langchain.com/mcp",
                "transport": "streamable_http",
            }
        })
        return await mcp_client.get_tools()
    except Exception as e:
        print(f"Docs MCP unavailable ({e}); writing without the docs tool.")
        return []


async def generate_post():
    tools = await load_docs_tools()
    # Higher temperature for genuine variety in phrasing day-to-day.
    llm = ChatOpenAI(model=TEXT_MODEL, temperature=0.85)
    agent = create_agent(llm, tools, system_prompt=build_system_prompt(todays_angle, banned_hooks))

    result = await agent.ainvoke({
        "messages": [
            ("user", f"Day {day_number} topic: {topic}. Today's hook angle: {todays_angle}"),
        ]
    })
    return result["messages"][-1].content.strip()


# ---- 2) Generate a branded thumbnail with GPT Image -------------------------
# Map topic keywords -> the brand logos/wordmarks that belong on that day's card.
LOGO_KEYWORDS = [
    ("langgraph", "LangGraph"),
    ("langsmith", "LangSmith"),
    ("mcp", "MCP (Model Context Protocol)"),
    ("model context protocol", "MCP (Model Context Protocol)"),
    ("fastapi", "FastAPI"),
    ("vllm", "vLLM"),
    ("kv cache", "PyTorch"),
    ("quantization", "Hugging Face"),
    ("batching", "vLLM"),
    ("mlops", "MLflow"),
    ("rag", "a vector database / retrieval icon"),
    ("vector", "a vector database / retrieval icon"),
    ("embedding", "a vector database / retrieval icon"),
    ("agent", "Agentic AI"),
    ("multi-agent", "Agentic AI"),
    ("langchain", "LangChain"),
]


def logos_for_topic(topic_text: str) -> list[str]:
    """Pick the logos relevant to this specific topic (deduped, order-preserving)."""
    t = topic_text.lower()
    logos = []
    for keyword, logo in LOGO_KEYWORDS:
        if keyword in t and logo not in logos:
            logos.append(logo)
    if "LangChain" not in logos:
        logos.append("LangChain")  # always anchor the series brand
    return logos[:3]  # keep the lockup clean


# Topic keyword -> the specific visual/diagram that actually represents THAT concept,
# so the thumbnail illustration matches the day's lesson (checked in order, first match wins).
# Order matters: specific concepts first, broad framework/role keywords last, so a
# topic like "Deploying LangGraph Apps" matches "deploy" (not the generic "langgraph").
VISUAL_MOTIFS = [
    ("prompt template", "a document template with highlighted {variable} placeholders being filled in"),
    ("prompt", "a document template with highlighted {variable} placeholders being filled in"),
    ("structured output", "a JSON schema with curly braces and typed fields snapping into place"),
    ("kv cache", "a transformer reusing cached key/value tensors so each new token skips recomputation"),
    ("quantization", "a large FP16 model block compressed into a much smaller INT8/INT4 block"),
    ("continuous batching", "many incoming requests packed into GPU batches as slots free up"),
    ("batching", "many incoming requests packed into GPU batches as slots free up"),
    ("vllm", "a GPU server streaming many parallel requests at high throughput with a PagedAttention grid"),
    ("mlops", "a CI/CD loop: version, deploy, monitor, then retrain the model"),
    ("mcp", "a client box and a server box exchanging tools over a labeled MCP connection"),
    ("sql agent", "an agent turning a natural-language question into a SQL query on a database"),
    ("evaluation", "a scorecard with checkmarks and a pass/fail grade on agent outputs"),
    ("document loader", "a source file being loaded and split into smaller text chunks"),
    ("splitter", "a long document being sliced into smaller overlapping text chunks"),
    ("knowledge base", "a source file being loaded and split into smaller text chunks"),
    ("rag", "documents split into chunks, retrieved, then fed to an LLM to answer"),
    ("retriever", "a query pulling the most relevant document chunks from a stack"),
    ("vector store", "a grid of dots with a query vector finding nearest neighbours"),
    ("embedding", "words mapped as points scattered in a 2D vector space"),
    ("streaming", "tokens flowing left-to-right out of a model as a live stream"),
    ("conditional edge", "a decision diamond routing to different branches"),
    ("persistence", "a checkpoint/save icon on a graph with a database cylinder"),
    ("human-in-the-loop", "a graph pausing at a node for human approval, with a person icon"),
    ("multi-agent", "several agent circles coordinating around a shared task, with arrows"),
    ("workflow", "a linear pipeline of steps versus a looping agent, side by side"),
    ("observability", "a trace/timeline waterfall of an agent's steps with spans"),
    ("guardrail", "a shield filtering unsafe input/output around an LLM"),
    ("security", "a shield filtering unsafe input/output around an LLM"),
    ("deploy", "an app container being pushed to a cloud server with a rocket"),
    ("tool calling", "an LLM box connecting via plugs to labeled function/tool boxes"),
    ("custom tool", "a wrench/gear icon wired into a function box"),
    ("tool", "an LLM box connecting via plugs to labeled function/tool boxes"),
    ("conditional", "a decision diamond routing to different branches"),
    ("node", "a state graph of connected nodes and directed edges"),
    ("langgraph", "a state graph of connected nodes and directed edges"),
    ("chat model", "a chat bubble exchange between a user and an LLM"),
    ("agent", "an agent reasoning-then-acting loop calling tools and observing results"),
]


def visual_motif_for_topic(topic_text: str) -> str:
    t = topic_text.lower()
    for keyword, motif in VISUAL_MOTIFS:
        if keyword in t:
            return motif
    return "a clean conceptual icon that visually represents this exact topic"


def generate_thumbnail() -> bytes:
    client = OpenAI()  # uses OPENAI_API_KEY
    logos = logos_for_topic(topic)
    logo_lockup = ", ".join(logos)
    motif = visual_motif_for_topic(topic)
    image_prompt = (
        f"Design a premium, highly-detailed LinkedIn thumbnail card (landscape) for a "
        f"developer education series about agentic AI.\n\n"
        f"LAYOUT:\n"
        f"- Top-left: a small pill badge reading 'Day {day_number}/30' and a thin 'AGENTIC AI · 30 DAYS' label.\n"
        f"- Center-left: a bold multi-line headline reading exactly '{topic}'.\n"
        f"- Below the headline: a one-line subtitle summarising the concept in 5-7 words.\n"
        f"- Right third: a clean technical diagram that specifically illustrates THIS topic — "
        f"draw {motif}. Do NOT draw any other concept. Use simple labeled icons.\n"
        f"- Bottom strip: a tidy logo lockup with recognizable wordmarks for {logo_lockup}.\n\n"
        f"STYLE: flat vector, crisp geometric icons, generous whitespace, a 12-column grid feel, "
        f"subtle indigo→teal gradient background, one bright accent color for emphasis, "
        f"bold legible sans-serif typography with clear hierarchy, high contrast, "
        f"professional and minimal — like a polished conference slide. "
        f"Spell all text correctly. No photorealism, no clutter, no watermark, no gibberish text."
    )
    for model in (IMAGE_MODEL, IMAGE_MODEL_FALLBACK):
        try:
            result = client.images.generate(
                model=model,
                prompt=image_prompt,
                size=IMAGE_SIZE,
            )
            print(f"Thumbnail generated with {model}")
            return base64.b64decode(result.data[0].b64_json)
        except Exception as e:
            if model == IMAGE_MODEL_FALLBACK:
                raise
            print(f"{model} unavailable ({e}); falling back to {IMAGE_MODEL_FALLBACK}.")


# ---- 3) Upload the image to LinkedIn and return its asset URN ---------------
def upload_image_to_linkedin(image_bytes: bytes, person_urn: str, token: str) -> str:
    register = requests.post(
        "https://api.linkedin.com/v2/assets?action=registerUpload",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Restli-Protocol-Version": "2.0.0",
            "Content-Type": "application/json",
        },
        json={
            "registerUploadRequest": {
                "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                "owner": f"urn:li:person:{person_urn}",
                "serviceRelationships": [{
                    "relationshipType": "OWNER",
                    "identifier": "urn:li:userGeneratedContent",
                }],
            }
        },
    )
    register.raise_for_status()
    data = register.json()["value"]
    asset_urn = data["asset"]
    upload_url = data["uploadMechanism"][
        "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
    ]["uploadUrl"]

    put = requests.put(
        upload_url,
        headers={"Authorization": f"Bearer {token}"},
        data=image_bytes,
    )
    put.raise_for_status()
    return asset_urn


# ---- Run --------------------------------------------------------------------
token = os.environ["LINKEDIN_ACCESS_TOKEN"]
person_urn = os.environ["LINKEDIN_PERSON_URN"]

post_text = format_for_linkedin(strip_urls(asyncio.run(generate_post())))

# Record this post's hook (first non-empty line) so future days never repeat it.
hook_line = next((ln.strip() for ln in post_text.split("\n") if ln.strip()), "")
history = [h for h in history if h.get("day") != day_number]  # idempotent on re-runs
history.append({"day": day_number, "topic": topic, "angle": todays_angle, "hook": hook_line})
save_history(history)

post_text = append_source(post_text, doc_url)
print("Generated post:\n", post_text, "\n")

try:
    image_bytes = generate_thumbnail()
    asset_urn = upload_image_to_linkedin(image_bytes, person_urn, token)
    print(f"Thumbnail uploaded: {asset_urn}")
    media_category = "IMAGE"
    media = [{
        "status": "READY",
        "description": {"text": f"Day {day_number}/30 — {topic}"},
        "media": asset_urn,
        "title": {"text": topic},
    }]
except Exception as e:
    # If image generation/upload fails, still publish the text so the streak survives.
    print(f"Image step failed ({e}); posting text-only.")
    media_category = "NONE"
    media = []

share_content = {
    "shareCommentary": {"text": post_text},
    "shareMediaCategory": media_category,
}
if media:
    share_content["media"] = media

resp = requests.post(
    "https://api.linkedin.com/v2/ugcPosts",
    headers={
        "Authorization": f"Bearer {token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "Content-Type": "application/json",
    },
    json={
        "author": f"urn:li:person:{person_urn}",
        "lifecycleState": "PUBLISHED",
        "specificContent": {"com.linkedin.ugc.ShareContent": share_content},
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    },
)

print(resp.status_code, resp.text)
if resp.status_code >= 300:
    raise Exception(f"LinkedIn post failed: {resp.text}")
