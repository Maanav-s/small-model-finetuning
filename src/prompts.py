"""System prompts for the menu-extraction agent.

The output contract is imported from schema.py (SCHEMA_SNIPPET / NOT_FOUND_SNIPPET)
so the prompt the model sees and the schema the code validates against never drift
apart.

The prompt is built per-episode by build_system_prompt() so a caller can slot in
the user's dietary restrictions (CLI --dietary, or the viz form). With no
restrictions the prompt asks for the full, unfiltered menu; with restrictions it
asks the model to keep only complying items. SYSTEM_PROMPT is the no-restriction
default, kept for the render-only demos and back-compat (LIVE_SYSTEM_PROMPT is a
legacy alias from when an offline stub prompt variant existed).

Teacher vs student variant (context distillation). build_system_prompt(variant=)
selects between two live prompts that differ ONLY by a teacher-only block of
*working guidance* (_TEACHER_GUIDANCE): the scrape-mode strategy (direct-first,
browser-fallback), search persistence, identity verification (confirm the page is
the right restaurant), and source selection (prefer the own site, avoid delivery
apps). The "teacher" variant (the default -- what we test and generate SFT data
with) includes it; the "student" variant omits it. The intent is context
distillation: train the student on teacher-generated trajectories under the
*student* prompt, so the behaviors that guidance elicits are baked into the
weights rather than carried in the prompt at inference (see CLAUDE.md). Every item
in the block is a behavioral nudge about HOW to work that does NOT change what the
correct menu is, so it is safe to drop from the student; everything that DOES
define the task -- schema, the food-only scope rule, dietary restrictions, tools,
and the shared tool-loop orientation -- is identical across variants.
"""

from __future__ import annotations

import re

from schema import NOT_FOUND_SNIPPET, SCHEMA_SNIPPET

# Restaurant used as the episode input while testing the loop end-to-end. Both
# runners (gemma/run_agent.py, claude/run_claude.py) read this. Temporary: real
# eval will iterate over a dataset of restaurants rather than a single name.
TEST_RESTAURANT = "Pagliacci, Seattle"


def normalize_dietary_restrictions(restrictions) -> list[str]:
    """Normalize a dietary-restrictions input into a clean list of phrases.

    Accepts None, a single string ("vegetarian, no nuts"), or a list of strings
    (possibly each comma/semicolon/newline-separated). Returns a de-duplicated,
    whitespace-trimmed list with empties dropped -- so an empty/blank input maps
    to [] (no filtering).
    """
    if not restrictions:
        return []
    if isinstance(restrictions, str):
        restrictions = [restrictions]
    out: list[str] = []
    seen = set()
    for chunk in restrictions:
        for part in re.split(r"[,\n;]", str(chunk)):
            part = part.strip()
            key = part.lower()
            if part and key not in seen:
                seen.add(key)
                out.append(part)
    return out


# Always-on scope rule: this tool exists to tell a diner what they can EAT, so
# strip the menu bulk that isn't a dish (drinks, add-ons, upsells, merch). Applied
# regardless of dietary restrictions -- someone with a peanut allergy has no use
# for the IPA list. Kept separate from the dietary filter so the two compose.
_SCOPE_RULE = (
    "- Return only the FOOD DISHES. EXCLUDE drinks and beverages entirely (beer, "
    "wine, cocktails, spirits, coffee, tea, soda, juice, and the like), and "
    "EXCLUDE non-dish menu bulk: add-ons, extras, modifiers, toppings and "
    'sauces-as-extras (e.g. "add chicken +$3", "extra cheese"), combo/upsell '
    "lines, merchandise, gift cards, and catering packages. KEEP the actual "
    "dishes: appetizers/starters, mains/entrees, shared plates, sides, and "
    "desserts."
)


def _dietary_block(restrictions: list[str]) -> str:
    """The one prompt line that changes with the dietary restrictions.

    Applies on top of _SCOPE_RULE (which has already dropped drinks/add-ons/bulk).
    Written to be robust to an EMPTY list: in that case the model is told
    explicitly NOT to filter by diet, so an absent restriction can never be
    misread as "filter to nothing".
    """
    if not restrictions:
        return (
            "- Dietary restrictions: NONE were provided. Do NOT filter the dishes "
            "by diet - include every food dish that survived the rule above."
        )
    joined = ", ".join(restrictions)
    return (
        f"- The user has these dietary restrictions: {joined}. Among the food "
        f"dishes above, include ONLY those that satisfy ALL of them, and OMIT every "
        f"dish that violates any one of them. If the available information is not "
        f"enough to tell whether a dish complies, leave it out rather than guess. "
        f"It is fine for the resulting menu to be empty if nothing complies (still "
        f'set "found" to true) - but only omit dishes for the restrictions above '
        f"(or the drinks/bulk rule above), never for any other reason."
    )


# Base template. {dietary} is the only user-driven part; {schema}/{not_found}
# come from schema.py. Literal JSON braces in the prose are doubled for
# str.format().
_BASE_PROMPT = """\
You are a restaurant menu extraction assistant.

You have NO prior knowledge of any restaurant's menu. {search_clause} and base \
your answer ONLY on what it returns - never answer from memory and never invent \
items. Return the menu as a single JSON object.

Output rules:
- Your final reply must be ONLY the raw JSON object and NOTHING else: no preamble, \
no explanation, no commentary, no markdown fences. The FIRST character of your reply \
must be `{{` and the LAST character must be `}}`.
- Do NOT begin with a sentence such as "I now have the full menu..." or "Let me \
compile the JSON...". Do not narrate what you are about to do - just output the JSON.
- The JSON must match this schema exactly:

{schema}

{scope}
{dietary}
- `price` must be a number (e.g. 12.5) or null - never a string like "$12.50". If \
you cannot find the price for a menu item, set that item's `price` to null; NEVER \
guess, estimate, or invent a price.
- Use null for other fields you cannot determine. Do not invent menu items that are \
not supported by the search results.
- Set `found` to true whenever you managed to find a menu (even a partial one).

If you CANNOT find the restaurant's menu at all (no search result or scraped page \
contains it), do NOT invent one and do NOT guess items. Instead return exactly this \
shape, with `found` set to false and a short `notes` explaining why:

{not_found}
"""

# Appended for BOTH variants: the bare mechanics of the tool loop (search returns
# links; scrape reads a page; YOU write the JSON). This is orientation, not
# strategy -- it biases neither the source nor the scrape mode -- so it is shared.
# The mode mechanics (direct vs browser) live in the scrape_url docstring, which is
# identical across variants; HOW to choose a mode is teacher-only guidance below.
_SHARED_TOOL_ORIENTATION = """\

Tool-use rules:
- `web_search` returns result titles, URLs, and snippets. To read a full menu,
  call `scrape_url` with one of those URLs; it returns the page as markdown.
- Read the returned text yourself and build the menu JSON in YOUR final answer.
  Do NOT expect a tool to return structured menu data - YOU produce the JSON.
"""

# Teacher-only working guidance -- the block that "teacher" includes and "student"
# omits (see the module docstring on context distillation). Every item shapes HOW
# the agent works, not WHAT the correct menu is, so the student learns it from the
# teacher's trajectories instead of being told. Keep it strictly target-neutral:
# nothing here may change which items belong in the menu (that would make the
# student's re-rendered SFT target incoherent) -- the food-only scope rule and the
# dietary filter live in the shared base for exactly that reason.
_TEACHER_GUIDANCE = """\

Working guidance:
- Scrape strategy: `scrape_url` takes a `mode` - "direct" (a plain, quick fetch of
  the page HTML) or "browser" (loads the page in a real browser that runs its
  JavaScript; slower). ALWAYS try mode="direct" first. If it comes back empty or
  clearly missing the menu (some pages only reveal their menu after JavaScript
  runs), retry the SAME URL with mode="browser". Neither mode is always better:
  some sites block the browser and return little while "direct" returns more, and
  vice versa. If you try both, KEEP whichever result actually contains the menu
  (the fuller one) - do NOT assume "browser" is better, and do NOT discard a good
  "direct" result.
- Persistence: do NOT give up after a single search. Only report the menu as not
  found (found=false) after search AND a scrape of the most promising result(s)
  still turn up no menu.
- Confirm identity before extracting: make sure the page you are reading is the
  SAME restaurant you were asked about - the name AND the city/neighborhood or
  address should match. Restaurant names are often shared by unrelated businesses
  in different cities, so a similar name in the wrong place is a DIFFERENT
  restaurant, not the one you want. If the given name looks slightly off (a
  misspelling or abbreviation), try close variants in your searches. If you cannot
  confirm a page is the right restaurant, treat the menu as not found rather than
  extract the wrong restaurant's menu.
- Source selection: prefer the restaurant's OWN website or online-ordering page
  (often hosted on Square, Toast, Clover, or BentoBox). These usually list the
  complete menu and scrape cleanly with mode="direct". AVOID third-party delivery
  apps and directories - DoorDash, Uber Eats, Grubhub, Yelp. They are
  JavaScript-heavy and/or block scraping, so they often yield only a partial menu
  or nothing. Only fall back to them if no better source turns up.
"""

# Injected on the final turn when the tool-call budget is spent (see the agent
# loops in gemma/agent.py and claude/claude_agent.py), so the model commits to an
# answer from what it already gathered instead of trying to call another tool and
# returning nothing. A partial menu beats an empty reply -- and is better SFT data.
BUDGET_FINALIZE_INSTRUCTION = (
    "You have used all available tool calls and cannot call any more tools. "
    "Output the menu JSON now, using ONLY the information you have already "
    "gathered. A partial menu is fine: include every item you did find and set "
    '"found" to true. If you found no menu at all, return the found=false shape. '
    "Reply with the raw JSON object and nothing else."
)

_SEARCH_CLAUSE = (
    "You MUST call the `web_search` tool at least once (and may then call "
    "`scrape_url` on a promising result URL to read the full menu page)"
)

_VARIANTS = ("teacher", "student")


def build_system_prompt(dietary_restrictions=None, *, variant: str = "teacher") -> str:
    """Build the system prompt, slotting in the user's dietary restrictions.

    dietary_restrictions: None / "" / [] -> no filtering (whole menu); a string or
    list of restriction phrases -> filter the menu to complying items only.
    variant: "teacher" (default) includes the teacher-only working guidance
    (_TEACHER_GUIDANCE: scrape strategy, persistence, identity check, source
    selection); "student" omits it. See the module docstring for the
    context-distillation intent.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")
    restrictions = normalize_dietary_restrictions(dietary_restrictions)
    prompt = _BASE_PROMPT.format(
        search_clause=_SEARCH_CLAUSE,
        schema=SCHEMA_SNIPPET,
        scope=_SCOPE_RULE,
        dietary=_dietary_block(restrictions),
        not_found=NOT_FOUND_SNIPPET,
    )
    prompt += _SHARED_TOOL_ORIENTATION
    if variant == "teacher":
        prompt += _TEACHER_GUIDANCE
    return prompt


# No-restriction default (teacher variant -- what we test and generate SFT data
# with), kept so agent.py's build_messages default and the render-only demos work
# without threading restrictions through. LIVE_SYSTEM_PROMPT is a back-compat
# alias from when an offline stub prompt variant existed alongside it.
SYSTEM_PROMPT = build_system_prompt()
LIVE_SYSTEM_PROMPT = SYSTEM_PROMPT
