import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search


load_dotenv(Path(__file__).resolve().parent / ".env")

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY was not found. Check your .env file.")


MODEL = "gemini-3.8-flash"   # keep the model name that works for you

# Gemini 3.x models "think" before answering, which adds seconds and tokens.
# Each step below gets only as much thinking as it needs.
# If you see an error mentioning thinking_level, run:
#     pip install -U langchain-google-genai
# or set TUNE_THINKING = False to switch this off.
TUNE_THINKING = True


def make_llm(thinking_level: str):
    kwargs = dict(model=MODEL, temperature=0, max_retries=3)
    if TUNE_THINKING:
        try:
            return ChatGoogleGenerativeAI(thinking_level=thinking_level, **kwargs)
        except Exception as error:
            print(f"thinking_level not accepted ({error}); using model defaults.")
    return ChatGoogleGenerativeAI(**kwargs)


planner_llm = make_llm("minimal")   # tiny output, no reasoning needed
agent_llm = make_llm("low")         # deciding which tool to call
writer_llm = make_llm("low")        # writing from supplied research
critic_llm = make_llm("low")        # comparing claims to sources


# ---------- search agent (used only in Thorough mode) ----------
# The agent decides which queries to run, so it is a real agent.
# It costs several LLM calls, so Fast and Balanced modes search in plain code.

SEARCH_PROMPT = """You are a web research assistant.
Given a topic, run 2-4 differently worded searches.
Return ONLY a bullet list of findings. Every bullet must end with the URL
it came from. Do not add facts that are not in the search results."""


def build_search_agent():
    return create_agent(model=agent_llm, tools=[web_search], system_prompt=SEARCH_PROMPT)


def last_text(result: dict) -> str:
    """Pull the final answer text out of an agent result.

    With Gemini the message content can be a list of blocks, not a string.
    """
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content


# ---------- query planner (one small call) ----------

class SearchPlan(BaseModel):
    queries: list[str] = Field(
        description="2 short web search queries that cover different angles"
    )


planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You write web search queries. Be concise."),
    (
        "human",
        """Topic: {topic}

Write 2 short web search queries that cover different angles of this topic
(for example: recent developments, and evidence or data).
Do not repeat the topic word for word."""
    ),
])

planner_chain = planner_prompt | planner_llm.with_structured_output(SearchPlan)


# ---------- writer ----------
# Output length is the biggest driver of latency, so the prompt sets a target.

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Use ONLY the research provided. "
        "Never add facts that are not in it."
    ),
    (
        "human",
        """Write a report on the topic below. Aim for about 600-800 words.

Topic: {topic}

Research gathered (pages are labelled [Source N]):
{research}

Previous draft (only when revising):
{previous_draft}

Critic feedback to fix (only when revising):
{feedback}

Structure:
- Introduction
- Key Findings (as many as the evidence supports; do not pad)
- Limitations (what the research did not cover)
- Conclusion
- Sources (numbered; cite them inline as [1], [2])

If the research is thin, say so in Limitations instead of filling gaps."""
    )
])

writer_chain = writer_prompt | writer_llm | StrOutputParser()


# ---------- critic ----------
# Structured output, so the pipeline can act on the verdict.

class Critique(BaseModel):
    score: int = Field(ge=1, le=10, description="Overall quality, 1 (poor) to 10 (excellent)")
    strengths: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the report that the research does not support",
    )
    improvements: list[str] = Field(default_factory=list)
    verdict: str = Field(description="One-line verdict")
    needs_revision: bool


critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict research critic. Be specific and honest."),
    (
        "human",
        """Check the report against the research it was written from.

Research:
{research}

Report:
{report}

Judge on: every claim supported by the research, inline citations present,
no padding, clear structure. Set needs_revision to true if the score is
below 8 or any claim is unsupported."""
    )
])

critic_chain = critic_prompt | critic_llm.with_structured_output(Critique)