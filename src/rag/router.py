"""
src/rag/router.py — Dynamic Query Router
=========================================
Classifies incoming queries and decides which tool(s) to call.

Route types:
  ANALYTICAL   → pandas_analytics  (counts, rankings, stats)
  SEMANTIC     → vector_search     (themes, examples, qualitative)
  SUBSTITUTION → substitution      (who could replace X)
  GRAPH        → graph_query       (relationships, network, cross-department)
  HYBRID       → multiple tools    (needs stats + context)
  DIRECT       → no tool           (greetings, meta questions)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Route types
# ---------------------------------------------------------------------------

class RouteType(str, Enum):
    ANALYTICAL   = "analytical"
    SEMANTIC     = "semantic"
    SUBSTITUTION = "substitution"
    GRAPH        = "graph"
    HYBRID       = "hybrid"
    DIRECT       = "direct"


class Confidence(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


# ---------------------------------------------------------------------------
# Route decision
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    route:      RouteType
    tools:      list
    confidence: Confidence
    reasoning:  str
    use_crag:   bool = False
    rewrite:    Optional[str] = None


# ---------------------------------------------------------------------------
# Signal banks
# ---------------------------------------------------------------------------

_ANALYTICAL = [
    r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\bnumber of\b",
    r"\btop \d+\b", r"\bmost\b", r"\bleast\b", r"\brank\b",
    r"\bdistribution\b", r"\bbreakdown\b", r"\bpercentage\b",
    r"\bfrequency\b", r"\baverage\b", r"\bhighest\b", r"\blowest\b",
    r"\bcommon\b", r"\bpopular\b", r"\blist (all|the)\b",
    r"\bwho (is|are|has) the (most|top|best)\b",
    r"\bwhich (category|subcategory|role|award)\b",
    r"\bshow me (all|stats|how many)\b",
]

_SEMANTIC = [
    r"\bwhat (kind|type|sort) of\b", r"\bdescribe\b", r"\bsummariz\b",
    r"\btheme\b", r"\bpattern\b", r"\bexample\b", r"\bstyle\b",
    r"\btell me about\b", r"\bwhat does .+ do\b", r"\binsight\b",
    r"\bwhy (is|are|do|does)\b", r"\bhow do(es)?\b",
    r"\bqualit\b", r"\btone\b", r"\blanguage\b", r"\bmessage\b",
    r"\brecognition (look|sound|feel)\b",
]

_SUBSTITUTION = [
    r"\breplace\b", r"\bsubstitut\b", r"\bif .+ left\b",
    r"\bif .+ (quit|resign|leav)\b",
    r"\bwho (could|can|would) (replace|fill|cover|step in)\b",
    r"\bbackup for\b", r"\balternative to\b", r"\bsimilar to\b",
    r"\bwho else\b", r"\bsame (skills?|role|contribution)\b",
]

_GRAPH = [
    r"\brelationship\b", r"\bnetwork\b", r"\bconnect\b",
    r"\bflow\b", r"\bbetween .+ and\b", r"\bdepartment\b",
    r"\bwho recogni(ze|se)\b", r"\brecognize each other\b",
    r"\bhub\b", r"\bcentral\b", r"\binfluential\b",
    r"\bcross.?team\b", r"\bcross.?department\b",
    r"\bgiving\b", r"\bgiver\b", r"\breceiver\b",
    r"\bmost connected\b", r"\brecognition flow\b",
    r"\bwhich (team|group|department) (give|give|recogni)\b",
    r"\bbridge\b", r"\bspan\b",
]

_DIRECT = [
    r"^(hi|hello|hey)[!.,]?$",
    r"^(thanks?|thank you)[!.,]?$",
    r"^what (can you do|are you|is this)\??$",
    r"^(yes|no|ok|okay|sure)[!.,]?$",
]

_HYBRID = [
    r"\bcompare\b", r"\bboth\b", r"\band (also|then)\b",
    r"\bwho .+ and (what|how|why)\b",
    r"\b(number|count).{0,30}(why|because|reason)\b",
]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Rule-based query router with confidence scoring.

    Usage:
        router = QueryRouter()
        decision = router.route("Which departments recognize each other most?")
    """

    def route(self, query: str) -> RouteDecision:
        q = query.strip()

        direct_score = self._score(q, _DIRECT)
        sub_score    = self._score(q, _SUBSTITUTION)
        ana_score    = self._score(q, _ANALYTICAL)
        sem_score    = self._score(q, _SEMANTIC)
        hyb_score    = self._score(q, _HYBRID)
        graph_score  = self._score(q, _GRAPH)

        # 1. Direct — greetings, meta
        if direct_score >= 1 and ana_score == 0 and sem_score == 0 and graph_score == 0:
            return RouteDecision(
                route=RouteType.DIRECT,
                tools=[],
                confidence=Confidence.HIGH,
                reasoning="Greeting or meta question — no retrieval needed.",
            )

        # 2. Substitution — always wins when detected
        if sub_score >= 1:
            return RouteDecision(
                route=RouteType.SUBSTITUTION,
                tools=["substitution"],
                confidence=Confidence.HIGH,
                reasoning=f"Substitution intent detected ({sub_score} signals).",
            )

        # 3. Graph — relationship/network queries
        if graph_score >= 1:
            return RouteDecision(
                route=RouteType.GRAPH,
                tools=["graph_query"],
                confidence=Confidence.HIGH if graph_score >= 2 else Confidence.MEDIUM,
                reasoning=f"Graph/relationship intent detected ({graph_score} signals).",
            )

        # 4. Hybrid — explicit hybrid signals OR both ana + sem present
        if hyb_score >= 1 or (ana_score >= 1 and sem_score >= 1):
            return RouteDecision(
                route=RouteType.HYBRID,
                tools=["pandas_analytics", "vector_search"],
                confidence=Confidence.HIGH if hyb_score >= 2 else Confidence.MEDIUM,
                reasoning=f"Hybrid query — analytical: {ana_score}, semantic: {sem_score}, hybrid: {hyb_score}.",
                use_crag=True,
            )

        # 5. Pure analytical
        if ana_score > sem_score:
            return RouteDecision(
                route=RouteType.ANALYTICAL,
                tools=["pandas_analytics"],
                confidence=self._confidence(ana_score),
                reasoning=f"Analytical intent — {ana_score} signals detected.",
            )

        # 6. Pure semantic
        if sem_score > 0:
            return RouteDecision(
                route=RouteType.SEMANTIC,
                tools=["vector_search"],
                confidence=self._confidence(sem_score),
                reasoning=f"Semantic intent — {sem_score} signals detected.",
                use_crag=True,
            )

        # 7. Fallback — hybrid with low confidence
        return RouteDecision(
            route=RouteType.HYBRID,
            tools=["pandas_analytics", "vector_search"],
            confidence=Confidence.LOW,
            reasoning="No strong signals — falling back to hybrid search.",
            use_crag=True,
        )

    def _score(self, text: str, patterns: list) -> int:
        t = text.lower()
        return sum(1 for p in patterns if re.search(p, t))

    def _confidence(self, score: int) -> Confidence:
        if score >= 3:   return Confidence.HIGH
        elif score >= 1: return Confidence.MEDIUM
        return Confidence.LOW


# ---------------------------------------------------------------------------
# CRAG — Corrective RAG self-checker
# ---------------------------------------------------------------------------

class CRAGChecker:
    """
    After retrieval, scores how relevant the results are to the query.
    If score is too low, signals the agent to broaden or retry.
    """

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def check(self, query: str, retrieved_text: str) -> dict:
        score = self._overlap_score(query, retrieved_text)
        if score >= self.threshold:
            return {"score": score, "passed": True,  "action": "proceed"}
        elif score >= self.threshold * 0.5:
            return {"score": score, "passed": False, "action": "rerank"}
        else:
            return {"score": score, "passed": False, "action": "fallback"}

    def _overlap_score(self, query: str, text: str) -> float:
        if not text:
            return 0.0
        q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        t_words = set(re.findall(r"\b\w{3,}\b", text.lower()))
        if not q_words:
            return 0.0
        return len(q_words & t_words) / len(q_words)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    router = QueryRouter()

    test_queries = [
        "Hello!",
        "How many awards are in each category?",
        "What kind of messages do engineering teams write?",
        "Who could replace a Senior Director?",
        "Which departments recognize each other most?",
        "How connected is the leadership team in the recognition network?",
        "What is the recognition flow between engineering and product?",
        "Compare the top categories and show me example messages",
        "Who are the most central hubs in the recognition network?",
    ]

    print("=" * 60)
    print("  Query Router — Test Results")
    print("=" * 60)

    for q in test_queries:
        d = router.route(q)
        print(f"\nQuery:      {q}")
        print(f"Route:      {d.route.value.upper()}")
        print(f"Tools:      {d.tools}")
        print(f"Confidence: {d.confidence.value}")
        print(f"Reasoning:  {d.reasoning}")
        if d.use_crag:
            print(f"CRAG:       enabled")
