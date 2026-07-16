import os
import re
import base64
import asyncio
from datetime import date

import requests
from openai import OpenAI
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

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
    topics = f.read().splitlines()

topic = topics[day_number - 1]

URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+\.(?:com|io|ai|dev|org|net)\b\S*", re.I)


def strip_urls(text: str) -> str:
    """Marketing brief says no external links — scrub any the model slips in."""
    cleaned = URL_RE.sub("", text)
    # collapse any dangling "[..]()" markdown link remnants and extra spaces
    cleaned = re.sub(r"\[([^\]]*)\]\(\s*\)", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _to_unicode(text: str, upper_base: int, lower_base: int, digit_base: int) -> str:
    out = []
    for ch in text:
        if "A" <= ch <= "Z":
            out.append(chr(upper_base + (ord(ch) - ord("A"))))
        elif "a" <= ch <= "z":
            out.append(chr(lower_base + (ord(ch) - ord("a"))))
        elif "0" <= ch <= "9":
            out.append(chr(digit_base + (ord(ch) - ord("0"))))
        else:
            out.append(ch)
    return "".join(out)


def format_for_linkedin(text: str) -> str:
    """LinkedIn renders NO markdown. Convert **bold** and `code` to the Unicode
    glyphs that actually display as bold / monospace in the feed."""
    # **bold** / __bold__ -> sans-serif bold glyphs
    text = re.sub(
        r"\*\*(.+?)\*\*|__(.+?)__",
        lambda m: _to_unicode(m.group(1) or m.group(2), 0x1D5D4, 0x1D5EE, 0x1D7EC),
        text,
    )
    # `code` -> monospace glyphs (drops the literal backticks)
    text = re.sub(
        r"`([^`]+)`",
        lambda m: _to_unicode(m.group(1), 0x1D670, 0x1D68A, 0x1D7F6),
        text,
    )
    # strip any leftover single-asterisk emphasis markers
    text = re.sub(r"(?<!\*)\*(?!\*)", "", text)
    return text


# ---- 1) Generate the post copy (marketing-manager style) --------------------
SYSTEM_PROMPT = f"""You are a senior LinkedIn content manager writing for a technical audience \
following a 30-day "learn in public" series about LangChain, LangGraph, MCP and agentic AI.

Write ONE LinkedIn post about the given topic. Follow these rules exactly:

STRUCTURE
- Line 1: a scroll-stopping HOOK under 210 characters (this is all people see before "see more").
  Use tension, a bold claim, a mistake, or a "most people get this wrong" angle. No emoji spam.
- Line 2: the series counter formatted as "Day {day_number}/30 · {{Topic Title}}".
- Then 4-7 short punchy lines/paragraphs, ONE idea each, separated by blank lines for whitespace.
- Include a compact, save-worthy takeaway: a mini framework, a 3-step list, or a checklist
  (use → or • or 1. 2. 3.). This is what earns saves.
- End with a light one-line CTA question to invite comments.
- Add 3-4 relevant hashtags on the final line (e.g. #LangChain #AgenticAI #MCP #LLM).

STYLE
- Confident, concrete, practical. No fluff, no "in today's fast-paced world".
- Use a few emojis as visual ANCHORS at the start of key lines (not decoration).
- Bold 3-5 key phrases by wrapping them in **double asterisks** (they get converted
  to Unicode bold for LinkedIn). Wrap code/API names in `single backticks`.
- Total length 1200-1800 characters.

HARD CONSTRAINTS
- Do NOT include ANY URLs, links, or domain names. None.
- Use the docs-langchain tool to verify the latest official info before writing.
- Output ONLY the post text. No preamble, no explanation, no code fences."""


async def generate_post():
    mcp_client = MultiServerMCPClient({
        "docs-langchain": {
            "url": "https://docs.langchain.com/mcp",
            "transport": "streamable_http",
        }
    })
    tools = await mcp_client.get_tools()
    llm = ChatOpenAI(model=TEXT_MODEL, temperature=0.5)
    agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)

    result = await agent.ainvoke({
        "messages": [
            ("user", f"Day {day_number} topic: {topic}"),
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


def generate_thumbnail() -> bytes:
    client = OpenAI()  # uses OPENAI_API_KEY
    logos = logos_for_topic(topic)
    logo_lockup = ", ".join(logos)
    image_prompt = (
        f"A clean, modern LinkedIn tech thumbnail card for a developer education series. "
        f"Headline text reads '{topic}' and a small badge reads 'Day {day_number}/30'. "
        f"Prominently feature recognizable logos/wordmarks for {logo_lockup}, "
        f"arranged as a neat logo lockup relevant to this topic. "
        f"Flat vector style, generous whitespace, subtle gradient background in indigo and teal, "
        f"crisp legible sans-serif typography, high contrast, professional and minimal. "
        f"No photorealism, no clutter, no watermark."
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
