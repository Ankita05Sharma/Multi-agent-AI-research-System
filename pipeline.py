import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from agents import (
    build_search_agent,
    critic_chain,
    last_text,
    planner_chain,
    writer_chain,
)
from tools import scrape_url, tavily

# ---------- speed modes ----------
# search "direct": plain code calls Tavily (0 LLM calls, queries run in parallel)
# search "agent":  an LLM agent chooses the searches (2-5 LLM calls)
# plan:            one small LLM call writes extra search queries
# critic/revisions: review the draft; optionally rewrite it
MODES = {
    "fast":     {"search": "direct", "plan": False, "critic": False, "revisions": 0},
    "balanced": {"search": "direct", "plan": True,  "critic": True,  "revisions": 0},
    "thorough": {"search": "agent",  "plan": False, "critic": True,  "revisions": 2},
}
DEFAULT_MODE = "balanced"

MAX_RESULTS = 5        # results kept per Tavily query
FULL_PAGES = 3         # pages whose full text goes to the writer
PAGE_CHARS = 3500      # text kept per page (smaller = faster, less context)
MIN_PAGE_CHARS = 400   # below this, Tavily's text is too thin: fetch the page
URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
FAILED_PREFIXES = ("Could not", "Refused", "Skipped", "Page had no")


def banner(text: str) -> None:
    print("\n" + "=" * 50 + f"\n{text}\n" + "=" * 50)


# ---------- searching in plain code ----------

@lru_cache(maxsize=64)
def _tavily_search(query: str) -> tuple:
    """One Tavily call. Cached, so re-running a topic costs no search credits."""
    response = tavily.search(
        query=query, max_results=MAX_RESULTS, include_raw_content=True
    )
    return tuple(response.get("results", []))


def plan_queries(topic: str, usage: dict) -> list:
    """The topic itself plus up to 2 angles from one small LLM call."""
    queries = [topic]
    try:
        plan = planner_chain.invoke({"topic": topic})
        usage["llm_calls"] += 1
        for query in plan.queries:
            query = query.strip()
            if query and query.lower() not in (q.lower() for q in queries):
                queries.append(query)
    except Exception as error:   # planning is optional: fall back to the topic
        print(f"Query planning failed ({error}); using the topic only.")
    return queries[:3]


def direct_search(queries: list) -> list:
    """Run all queries in parallel, merge, drop duplicate URLs, best first."""
    errors = []

    def one(query):
        try:
            return _tavily_search(query)
        except Exception as error:
            errors.append(str(error))
            return ()

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        batches = list(pool.map(one, queries))

    merged = {}
    for batch in batches:
        for result in batch:
            url = result.get("url")
            if url and (url not in merged or result.get("score", 0) > merged[url].get("score", 0)):
                merged[url] = result

    results = sorted(merged.values(), key=lambda r: r.get("score", 0), reverse=True)
    if not results:
        detail = f" ({errors[0]})" if errors else ""
        raise RuntimeError(f"The web search returned no results{detail}.")
    return results


def format_results(results: list) -> str:
    return "\n\n".join(
        f"- {r.get('title', '')}\n  {r['url']}\n  {r.get('content', '')[:300]}"
        for r in results
    )


def gather_pages(results: list) -> str:
    """Page text for the best sources.

    Tavily already returns cleaned page text, so most pages need no download.
    Only thin results are fetched with scrape_url, in parallel.
    """
    top = results[:FULL_PAGES]
    if not top:
        return ""

    def text_for(result):
        raw = (result.get("raw_content") or "").strip()
        if len(raw) >= MIN_PAGE_CHARS:
            return raw
        scraped = scrape_url.invoke({"url": result["url"]})
        if scraped.startswith(FAILED_PREFIXES):
            return result.get("content", "")
        return scraped

    with ThreadPoolExecutor(max_workers=len(top)) as pool:
        texts = list(pool.map(text_for, top))

    blocks = []
    for index, (result, text) in enumerate(zip(top, texts), start=1):
        text = re.sub(r"\s+", " ", text).strip()[:PAGE_CHARS]
        if text:
            blocks.append(f"[Source {index}] {result['url']}\n{text}")
    return "\n\n".join(blocks)


# ---------- searching with an agent (Thorough mode) ----------

def extract_urls(text: str, limit: int = FULL_PAGES) -> list:
    seen, urls = set(), []
    for url in URL_RE.findall(text):
        url = url.rstrip(".,;")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls[:limit]


def read_sources(urls: list) -> str:
    """Scrape several URLs in parallel and label each page [Source N]."""
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        pages = list(pool.map(lambda u: scrape_url.invoke({"url": u}), urls))
    good = [
        f"[Source {i}] {url}\n{page[:PAGE_CHARS]}"
        for i, (url, page) in enumerate(zip(urls, pages), start=1)
        if not page.startswith(FAILED_PREFIXES)
    ]
    return "\n\n".join(good)


# ---------- the pipeline ----------

def run_research_pipeline(topic: str, on_step=None, mode: str = DEFAULT_MODE) -> dict:
    """Search -> read -> write -> (critique -> revise).

    mode is "fast", "balanced" or "thorough" (see MODES above).

    on_step is optional. The Streamlit UI passes a function that is called as
    on_step(stage, event, data) with stage in search/read/write/critique and
    event in start/done/skip, so it can show live progress. In the terminal it
    stays None and nothing changes.
    """

    def notify(stage, event, data=None):
        if on_step:
            on_step(stage, event, data)

    topic = topic.strip()
    if not topic:
        raise ValueError("Topic cannot be empty.")
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Use one of: {', '.join(MODES)}.")
    config = MODES[mode]

    usage = {"llm_calls": 0}
    state = {"topic": topic, "mode": mode, "history": [], "timings": {}}

    # Step 1: search
    banner(f"Step 1 - Searching ({mode} mode)")
    began = time.perf_counter()
    notify("search", "start")
    results = []
    if config["search"] == "agent":
        result = build_search_agent().invoke(
            {"messages": [("user",
                f"Find recent, reliable, detailed information about: {topic}")]},
            config={"recursion_limit": 12},
        )
        # every AI message in the agent's history was one LLM call
        usage["llm_calls"] += sum(
            1 for m in result["messages"] if getattr(m, "type", "") == "ai"
        )
        state["search_results"] = last_text(result)
        state["urls"] = extract_urls(state["search_results"])
    else:
        queries = plan_queries(topic, usage) if config["plan"] else [topic]
        state["queries"] = queries
        print(f"Queries: {queries}")
        results = direct_search(queries)
        state["search_results"] = format_results(results)
        state["urls"] = [r["url"] for r in results]
    print(state["search_results"])
    state["timings"]["search"] = time.perf_counter() - began
    notify("search", "done", state["search_results"])

    # Step 2: page text (plain code, not an agent)
    banner("Step 2 - Getting page text")
    began = time.perf_counter()
    notify("read", "start")
    if config["search"] == "agent":
        state["scraped_content"] = read_sources(state["urls"]) if state["urls"] else ""
    else:
        state["scraped_content"] = gather_pages(results)
    print(f"URLs found: {state['urls']}")
    state["timings"]["read"] = time.perf_counter() - began
    notify("read", "done", state["scraped_content"])

    research = (
        f"SEARCH FINDINGS:\n{state['search_results']}\n\n"
        f"FULL PAGE CONTENT:\n{state['scraped_content'] or '(no pages could be read)'}"
    )
    state["research"] = research

    # Steps 3-4: write -> critique -> (revise), capped so it always ends
    previous_draft, feedback = "", ""
    state["timings"]["write"] = 0.0
    state["timings"]["critique"] = 0.0

    for round_no in range(1, config["revisions"] + 2):
        banner(f"Step 3 - Writing (draft {round_no})")
        began = time.perf_counter()
        notify("write", "start", {"round": round_no})
        report = writer_chain.invoke({
            "topic": topic,
            "research": research,
            "previous_draft": previous_draft or "(none: this is the first draft)",
            "feedback": feedback or "(none)",
        })
        usage["llm_calls"] += 1
        state["report"] = report
        state["rounds"] = round_no
        state["timings"]["write"] += time.perf_counter() - began
        notify("write", "done", report)

        if not config["critic"]:
            state["critique"] = None
            notify("critique", "skip")
            break

        banner(f"Step 4 - Critiquing (draft {round_no})")
        began = time.perf_counter()
        notify("critique", "start", {"round": round_no})
        try:
            critique = critic_chain.invoke({"research": research, "report": report})
            usage["llm_calls"] += 1
        except Exception as error:   # keep the report even if the critic fails
            print(f"Critic failed: {error}")
            critique = None
        state["critique"] = critique
        state["timings"]["critique"] += time.perf_counter() - began
        notify("critique", "done", critique)

        if critique is None:
            break
        print(critique.model_dump_json(indent=2))
        state["history"].append({"round": round_no, "score": critique.score})

        # Only revise while the cap allows it
        if not critique.needs_revision or round_no > config["revisions"]:
            break
        previous_draft = report
        feedback = "\n".join(critique.unsupported_claims + critique.improvements)

    state["llm_calls"] = usage["llm_calls"]
    banner(
        f"Done: {usage['llm_calls']} LLM calls. Seconds per stage: "
        + ", ".join(f"{k} {v:.1f}" for k, v in state["timings"].items())
    )
    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    chosen = input(" Mode (fast / balanced / thorough) [balanced] : ").strip().lower()
    final = run_research_pipeline(topic, mode=chosen or DEFAULT_MODE)
    print("\nFINAL REPORT:\n", final["report"])