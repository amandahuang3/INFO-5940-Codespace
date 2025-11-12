# app.py
"""
Multi-Agent Travel Planner

Highlights:
- Clear separation of concerns (tools, agents, orchestration, UI)
- Simple global logger to display tool calls live in the sidebar
- Planner → Reviewer pipeline enforced before rendering any answer
- Minimal dependencies and straightforward control flow
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Callable, Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Globals
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()  # Loads variables from a local .env if present
os.environ.setdefault("OPENAI_LOG", "error")
os.environ.setdefault("OPENAI_TRACING", "false")

# Tool call logger: the UI sets this per request. The tool checks it and logs.
# Using a simple global makes this easy to teach and reason about.
TOOL_LOGGER: Optional[Callable[[Dict[str, Any]], None]] = None

# *if pass a "logger function", set it as global TOOL_LOGGER. if pass "NONE", removes the logger
def set_tool_logger(logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install or remove the UI logger used by tools to report activity."""
    global TOOL_LOGGER
    TOOL_LOGGER = logger

# *send structured log event to logger
def log_tool_event(event: Dict[str, Any]) -> None:
    """If a logger is installed, send the event to the UI."""
    if TOOL_LOGGER is not None:
        try:
            TOOL_LOGGER(event)
        except Exception:
            # Logging should never break the app or the tool itself
            pass

# *clean up sensitive & large data. Work recursively for nested dict. or list
def redact_for_logs(value: Any) -> Any:
    """
    Make sure we don't leak secrets and keep logs small.
    This is deliberately simple for teaching.
    """
    # *mask sensitive data (redact)
    if isinstance(value, str):
        low = value.lower()
        if any(k in low for k in ("api_key", "token", "secret", "password")):
            return "[redacted]"
        # *truncate long string (keep logs small)
        return value if len(value) <= 300 else value[:120] + "… [truncated]"
    # *apply "redact" recrusively to dict.  
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in ("key", "token", "secret", "password"))
                    else redact_for_logs(v))
                for k, v in value.items()}
    # *apply "readact" recursively to list
    if isinstance(value, list):
        return [redact_for_logs(v) for v in value]
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Agent Framework Imports (provided by you)
# ──────────────────────────────────────────────────────────────────────────────
# These come from your own framework. We assume:
# - Agent: defines a model + instructions + optional tools
# - Runner.run(agent, input): executes an agent and returns an object with text
from agents import Agent, Runner, function_tool  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@function_tool
def internet_search(query: str) -> str:
    """
    Internet search backed by Tavily.
    - Reads TAVILY_API_KEY from environment.
    - Sends simple log events before/after the call so the UI can show activity.
    """

    # *log - start of a tool call
    log_tool_event({"type": "call", "tool": "internet_search", "args": {"query": redact_for_logs(query)}})

    try:
        # *retrieve Tavily API key
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            msg = "missing TAVILY_API_KEY in environment." # *user friendly error message
            log_tool_event({"type": "error", "tool": "internet_search", "error": msg})
            return f"Search error: {msg}"

        # *create Taily API client 
        client = TavilyClient(api_key=api_key)
        # *limit search result to 3
        response = client.search(query, max_results=3)

        # *Format search result
        # *extract result list from responses
        items = response.get("results", [])
        # *format the result into bullet of "title" & "content"
        lines = [f"- {it.get('title', 'N/A')}: {it.get('content', 'N/A')}" for it in items]
        # *join individual lines together
        output = "\n".join(lines) if lines else "No results found."

        # *log - summary result output
        log_tool_event({
            "type": "result",
            "tool": "internet_search",
            "preview": redact_for_logs(output[:400] + ("…" if len(output) > 400 else "")),
        })
        return output

    except Exception as e:
        # *log if unexpected error
        log_tool_event({"type": "error", "tool": "internet_search", "error": str(e)})
        return f"Search error: {e}"

    finally:
        # *always log that the tool finished (even it failed)
        log_tool_event({"type": "end", "tool": "internet_search"})


# ──────────────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────────────

# BEGIN SOLUTION
REVIEWER_INSTRUCTIONS = """
# Objective
Validate the Planner Agent's itinerary for feasibility and compliance with user constraints.

# Response Rules
- Internal obtain a "Delta List" of concrete changes with reasons.
- Use the internet tool(Tavily) for real-time fact-checking.
- Output the **Reviewed Itinerary** 
- Delta List is for internal use do not output them
- At the end, include a short section: "Summary of changes".

# Things to Check
- Opening hours and holiday closures for attractions and restaurants.
- Ticket prices and availability (advance booking needed?).
- Travel times between locations (avoid unrealistic transitions).
- Restaurant details: open status, specialty dish, price range.
- Accommodation: ensure the recommended hostel or hotel is near most activities and doesn’t require unnecessary relocations.
- Budget compliance after adjustments.
- Pacing: avoid overloading days.


# Workflow
## Step 1: Extract user requirements & constraints and Planner Agent's recommend day-to-day itinerary details.
## Step 2: Validate each activity & add additional information:
- Is it open on the planned day/time?
- Is ticket price accurate and within budget?
- Is travel time between activities feasible?
- Is restaurant relevant and still running?
- Is the plan include all 3 meals (breakfast, lunch & dinner)
- Is the accommodation near the activities?
- Is the accommodation available and within budget?
- For each resturant/ attraction/ activites include the address and transportation to get there

## Step 3: Identify issues and suggest fixes:
- Flag closed attractions/restaurants.
- Flag timing or sequence that is not feasible.
- Flag budget overruns and suggest alternatives.
- Flag unrealistic or inconvenient accommodation choices (too far, too frequent moving, or over budget).

## Step 4: Generate a Delta List:
- Each item that has no issue: [NO issue]
- Each item that has issue: [Issue] → [Suggested Fix] → [Reason]

## Step 5: Correct any issues directly in the itinerary text.

## Step 6: Output formatting
- Title: **Reviewed [Trip Duration] [Destination] [Interest/Theme] Itinerary ([Budget])**
  - Example: *Reviewed 7-Day Japan Food & Culture Itinerary ($1,500 Budget)*
- Then output the fully corrected itinerary text, integrating all improvements (do not show the Delta List).
- End with a short section titled **"Summary of Changes"**, highlighting key modifications and reasons.
"""

PLANNER_INSTRUCTIONS = """
# Objective
Construct and plan a detailed, structured day-by-day itinerary based on user's vague travel prompt

# Response Rules
- Use concise bullet points for readability
- Present the itinerary in a clear, structured format:
  Day X:
    - Hostels
    - Activities (with approximate times and locations)
    - Estimated cost per activity and daily total
    - Notes on logistics (transportation, city cluster, holiday involve)

# Things to Consider
- User constraints: dates, budget, interests, pacing (if mention)
- Group attractions and restaurants by location proximity to minimize travel time.
- Recommend hostels or accommodations that are close to the planned activities (avoid changing accommodation every day unless necessary).
- Ensure daily activities are realistic (3–5 major activities max).
- Should alway have breakfast, lunch, and dinner (optionally, can have a dessert. If it is known for and within budget)
- Track cumulative cost to stay within budget.

# Workflow
## Step 1: Extract key details from user prompt
- Travel duration, destination(s), budget, interests, pacing.

## Step 2: Brainstorm the travel plan using your own knowledge
- Identify major attractions and food spots in that travel destination.
- Suggest 2 restaurants per famous dish.
- For each resturant suggest at least 2 famous dishes to try
- Identify any attractions in travel destination that is relevant to user's interests.
- Estimate costs for tickets, meals, and transport.
- Assign approximate time slots (morning, afternoon, evening).

## Step 3: Organize into itinerary
- Group by city clusters.
- Create day-by-day plan with:
    - Activities + times
    - Locations
    - Estimated costs
    - Suggested accommodation (with proximity reasoning)
    - Logistics notes
"""

reviewer_agent = Agent(
    name="Reviewer Agent",
    model="openai.gpt-4o",
    instructions=REVIEWER_INSTRUCTIONS.strip(),
    tools=[internet_search]
)

planner_agent = Agent(
    name="Planner Agent",
    model="openai.gpt-4o",
    instructions=PLANNER_INSTRUCTIONS.strip(),
)

# END SOLUTION


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration Helpers
# ──────────────────────────────────────────────────────────────────────────────

# *standardize the output structure (extract to human-readable string from Runner result)
def extract_text(result_obj: Any) -> str:
    """
    Pull a usable string from the Runner result in a tolerant way.
    Your Runner may expose final_output, text, or __str__.
    """
    # B/c diff Runner store output diff. Try "final_output", else ".txt", or convert to "string"
    return (
        getattr(result_obj, "final_output", None)
        or getattr(result_obj, "text", None)
        or str(result_obj)
    )

# *execute "planner_agent", take user text & return strucutred plan
def run_planner(user_text: str) -> str:
    """
    Run the Planner and return its itinerary text.
    Function:
    - Runner.run(agent,input) -> like a function that executes AI agent (is asynchronous)
    - asyncio.run(...) -> start event loop, runs the asnync task, and return result once it finishes
    Overall what it done: 
    - start a temporary asnyc env. run the agent (async), wait unit it(agent) finishes, and give me output once complete
    """

    result = asyncio.run(Runner.run(planner_agent, user_text))
    return extract_text(result)

# *feed "panner_output" to "reviwer_agent"
def run_reviewer(plan_text: str) -> str:
    """Run the Reviewer on the planner’s output and return validated text."""
    result = asyncio.run(Runner.run(reviewer_agent, plan_text))
    return extract_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Travel Planner", page_icon="✈️")

st.title("✈️ Multi-Agent Travel Planner")
st.caption("Planner → Reviewer (with live tool calls in the sidebar)")

# Sidebar: session controls + examples + dev panel
with st.sidebar:
    st.header("Session")
    if st.button("🔄 Reset conversation"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Try these prompts")
    st.code("Plan a week-long Europe trip for a student on a $1,500 budget who loves history and food")
    st.code("3-day Paris trip for art lovers with $800 budget")

    st.subheader("Developer view")
    show_tools = st.toggle("Show tool activity (live)", value=True)
    if show_tools:
        tool_expander = st.expander("🔧 Tool activity", expanded=True)
        tool_panel = tool_expander.container()
    else:
        tool_panel = st.container()  # inert sink

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]
if "meta" not in st.session_state:
    st.session_state.meta = []      # list[dict(trace)]

# Render history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.meta):
            meta = st.session_state.meta[i]
            if meta:
                st.caption(meta.get("trace", ""))

# Chat input
user_input = st.chat_input("Describe your travel (destination, duration, budget, interests)…")

if user_input:
    # Add user message to history and render it
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.meta.append(None)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant output block
    with st.chat_message("assistant"):
        # Live “working…” text and progress bar
        live_msg = st.empty()
        progress = st.progress(0)

        # Per-request tool log (shown in the sidebar)
        tool_events: List[Dict[str, Any]] = []

        def ui_tool_logger(event: Dict[str, Any]) -> None:
            """Append an event and re-render the sidebar log."""
            tool_events.append(event)
            with tool_panel:
                st.markdown("**Recent tool calls**")
                for ev in tool_events[-60:]:  # last N entries
                    t = ev.get("tool", "unknown")
                    et = ev.get("type", "event")
                    if et == "call":
                        st.write(f"• **{t}** called with `{ev.get('args')}`")
                    elif et == "result":
                        st.write(f"• **{t}** result preview:\n\n> {ev.get('preview')}")
                    elif et == "error":
                        st.error(f"• **{t}** error: {ev.get('error')}")
                    elif et == "end":
                        st.write(f"• **{t}** finished")

        # Install the logger so tools can report to the sidebar
        set_tool_logger(ui_tool_logger)

        try:
            # Optional: clear sidebar panel on each run
            with tool_panel:
                st.empty()

            # Step 1: Planner
            with st.status("🧭 Planner Agent: generating itinerary…", expanded=True) as status:
                live_msg.markdown("🧭 Planner Agent is creating your itinerary…")
                plan_text = run_planner(user_input)
                progress.progress(40)
                status.update(label="🔎 Reviewer Agent: validating with live searches…", state="running")

            # Step 2: Reviewer (tool calls will appear live in sidebar)
            live_msg.markdown("🔎 Reviewer Agent is validating the plan with live searches…")
            review_text = run_reviewer(plan_text)
            progress.progress(90)

            # Completed
            live_msg.markdown("✅ Validation complete. Rendering results…")
            time.sleep(0.2)
            progress.progress(100)

            # Final render: show only the validated result, with the raw plan expandable
            st.info("🤖 **Reviewer Agent** (validated)")
            st.markdown(review_text)
            with st.expander("See raw plan from Planner Agent"):
                st.markdown(plan_text)

            # Save only the validated result to history
            st.session_state.messages.append({"role": "assistant", "content": review_text})
            st.session_state.meta.append({"trace": "Planner Agent → Reviewer Agent"})
            st.caption("Planner Agent → Reviewer Agent")

        except Exception as e:
            # Friendly error box
            live_msg.markdown("❌ Something went wrong.")
            err = f"⚠️ Error while processing your request:\n\n```\n{e}\n```"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.session_state.meta.append({"trace": "Runtime error."})

        finally:
            # Always remove the logger so it doesn't leak into the next request
            set_tool_logger(None)
