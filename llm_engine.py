"""
llm_engine.py — Production LLM Engine (GPT-4o / Claude 3.5 Sonnet / Local)

Uses LangChain for unified LLM abstraction across providers:
  • "openai"    → GPT-4o (ChatOpenAI)
  • "anthropic" → Claude 3.5 Sonnet (ChatAnthropic)
  • "local"     → Qwen2.5-1.5B-Instruct (HuggingFace, CPU/GPU)

Three roles:
  1. Query Expansion  — generate alternative search queries
  2. Answer Generation — grounded answer from context
  3. Self-Verification — confidence scoring + VERIFIED/HALLUCINATION verdict
"""

from __future__ import annotations

import re

import config
from utils import console, get_logger

logger = get_logger(__name__)


def _create_llm():
    """
    Factory: create the LangChain LLM based on config.LLM_PROVIDER.

    Returns a LangChain BaseChatModel instance.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        console.print(
            f"[cyan]⚡ LLM Provider:[/cyan] [bold]OpenAI ({config.OPENAI_MODEL})[/bold]"
        )
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        console.print(
            f"[cyan]⚡ LLM Provider:[/cyan] [bold]Anthropic ({config.ANTHROPIC_MODEL})[/bold]"
        )
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )

    elif provider == "local":
        # Replace Qwen with Groq API
        console.print(
            f"[cyan]⚡ LLM Provider:[/cyan] [bold]Groq (llama-3.3-70b-versatile)[/bold]"
        )
        import os
        from groq import Groq
        
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("GROQ_API_KEY environment variable is not set!")
            
        client = Groq(api_key=api_key)
        # We attach a marker so LLMEngine knows it's the Groq client, not LangChain
        client.is_groq = True
        return client

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'openai', 'anthropic', or 'local'.")


class LLMEngine:
    """
    Production LLM Engine with three roles:
    Query Expansion, Answer Generation, Self-Verification + Confidence Scoring.
    """

    def __init__(self) -> None:
        self.llm = _create_llm()
        console.print("  [green]✓[/green] LLM ready\n")

    def _invoke(self, system: str, user: str) -> str:
        """Send a system+user message pair and return the response text."""
        
        # Check if using the Groq client
        if getattr(self.llm, "is_groq", False):
            response = self.llm.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=config.LLM_TEMPERATURE,
                max_completion_tokens=config.LLM_MAX_TOKENS,
            )
            raw = response.choices[0].message.content.strip()
        else:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [SystemMessage(content=system), HumanMessage(content=user)]
            response = self.llm.invoke(messages)

            # LangChain returns AIMessage or string depending on provider
            if hasattr(response, "content"):
                raw = response.content.strip()
            else:
                raw = str(response).strip()

        # Guard: local HF models may still leak prompt fragments
        # Strip everything before the last "Answer" marker if present
        if "Answer (using only the context above):" in raw:
            raw = raw.split("Answer (using only the context above):")[-1].strip()
        elif "Answer:" in raw and raw.index("Answer:") < len(raw) // 2:
            raw = raw.split("Answer:")[-1].strip()

        # Strip "Assistant:" prefix leak (common with Qwen chat models)
        if "\nAssistant:" in raw:
            raw = raw.split("\nAssistant:")[0].strip()
        if raw.startswith("Assistant:"):
            raw = raw[len("Assistant:"):].strip()

        return raw

    # ── Role 1: Query Expansion ─────

    def _invoke_short(self, system: str, user: str) -> str:
        """
        Like _invoke but with a reduced max_tokens budget for local models.
        Used for query expansion where only a few lines are needed.
        """
        if getattr(self.llm, "is_groq", False):
            response = self.llm.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.0, # Deterministic for expansion
                max_completion_tokens=config.EXPAND_MAX_TOKENS,
            )
            raw = response.choices[0].message.content.strip()
            return raw

        from langchain_core.messages import SystemMessage, HumanMessage

        # For local HF pipeline, temporarily override max_new_tokens
        raw_pipe = getattr(self.llm, '_raw_pipe', None)
        had_max = False
        had_do_sample = False
        original_max = None
        original_do_sample = None
        if raw_pipe is not None:
            had_max = 'max_new_tokens' in raw_pipe._forward_params
            had_do_sample = 'do_sample' in raw_pipe._forward_params
            original_max = raw_pipe._forward_params.get('max_new_tokens')
            original_do_sample = raw_pipe._forward_params.get('do_sample')
            raw_pipe._forward_params['max_new_tokens'] = config.EXPAND_MAX_TOKENS
            raw_pipe._forward_params['do_sample'] = False  # deterministic for expansion

        try:
            messages = [SystemMessage(content=system), HumanMessage(content=user)]
            response = self.llm.invoke(messages)

            if hasattr(response, "content"):
                raw = response.content.strip()
            else:
                raw = str(response).strip()
        finally:
            # Restore original state: pop keys that weren't there before
            if raw_pipe is not None:
                if had_max:
                    raw_pipe._forward_params['max_new_tokens'] = original_max
                else:
                    raw_pipe._forward_params.pop('max_new_tokens', None)
                if had_do_sample:
                    raw_pipe._forward_params['do_sample'] = original_do_sample
                else:
                    raw_pipe._forward_params.pop('do_sample', None)

        return raw

    def expand_query(self, query: str) -> list[str]:
        """
        Generate alternative phrasings to improve retrieval recall.

        Uses a strict few-shot prompt that forces the model to preserve key
        technical terms and search intent rather than producing generic rewrites.
        """
        system = (
            "You rewrite academic search queries. Rules:\n"
            "1. Keep ALL technical terms exactly (e.g. multi-head attention, BERT, Transformer).\n"
            "2. Only rephrase connecting words. Do NOT generalize.\n"
            "3. Output ONLY the rewritten queries, one per line. No numbering, no prefixes.\n\n"
            "Example:\n"
            "Query: How does multi-head attention work?\n"
            "multi-head attention mechanism in transformers\n"
            "transformer multi-head attention architecture\n"
            "how parallel attention heads work in multi-head attention"
        )
        user = (
            f"Query: {query}\n"
        )

        raw   = self._invoke_short(system, user)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

        # Filter out prompt leakage, numbering prefixes, and garbage
        cleaned: list[str] = []
        for ln in lines:
            # Remove leading numbering like "1.", "1)", "- "
            ln = re.sub(r'^\d+[.):]\s*', '', ln).strip()
            ln = re.sub(r'^[-\u2022*]\s*', '', ln).strip()
            # Remove quotes wrapping the line
            ln = ln.strip('"').strip("'").strip()
            # Skip lines that look like prompt fragments
            if any(frag in ln.lower() for frag in (
                "system:", "human:", "assistant:", "you are",
                "original query", "generate", "alternative",
                "rephrase", "output only", "one per line",
                "query:", "write", "search queries", "example",
                "rewrite", "here are", "sure",
            )):
                continue
            if ln.lower() == query.lower() or len(ln) < 10:
                continue
            cleaned.append(ln)
        return cleaned[:config.NUM_EXPANDED_QUERIES]

    # ── Role 2: Answer Generation ──────────────────────────────────────────────

    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        """Generate a factual answer grounded in retrieved context."""
        context = "\n\n---\n\n".join(context_chunks)
        system  = (
            "You are a precise question-answering assistant. "
            "Answer the question using ONLY information from the provided context. "
            "If the context lacks sufficient information, respond with exactly: "
            "'I don't know based on the provided context.' "
            "Do NOT speculate or add knowledge beyond what is in the context."
        )
        user = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer (using only the context above):"
        )
        return self._invoke(system, user)

    # ── Role 3: Self-Verification + Grounding Check ───────────────────────────

    # Stop-words excluded from keyword overlap check
    _VERIFY_STOP = frozenset({
        "the", "a", "an", "is", "in", "on", "at", "to", "for", "of",
        "and", "or", "but", "with", "by", "from", "as", "it", "its",
        "was", "are", "be", "been", "this", "that", "these", "those",
        "which", "who", "what", "how", "when", "where", "why", "not",
        "also", "can", "may", "will", "would", "could", "should",
        "has", "have", "had", "do", "does", "did", "more", "most",
        "than", "then", "so", "if", "no", "yes", "all", "each",
        "both", "such", "into", "over", "only", "very", "just",
        "about", "between", "through", "during", "after", "before",
        "other", "some", "any", "new", "used", "using", "based",
        # Prompt leakage tokens from chat models
        "assistant", "user", "system", "human",
    })

    # Common English suffixes for rough stemming
    _SUFFIXES = (
        "ation", "tion", "sion", "ment", "ness", "ity", "ious",
        "eous", "ible", "able", "ally", "ively", "ting", "ning",
        "ling", "ful", "less", "ence", "ance", "ized", "ised",
        "ical", "onal", "ing", "ous", "ive", "ted", "ded",
        "ies", "ers", "est", "ent", "ant", "ism", "ist",
        "ary", "ory", "ely", "ial", "ual",
        "ly", "ed", "er", "en", "es", "or", "al",
    )

    @classmethod
    def _rough_stem(cls, word: str) -> str:
        """Strip common English suffixes for approximate matching.

        'concatenation' → 'concat'  (matches 'concatenated')
        'projection'    → 'project' (matches 'projected')
        'allowing'      → 'allow'   (matches 'allows')
        'reinforcement' → 'reinforc'(no match if not in context → flagged)
        """
        for suffix in cls._SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[:-len(suffix)]
        return word

    @classmethod
    def _word_in_context(cls, keyword: str, context: str) -> bool:
        """Check if keyword or its stem appears in context."""
        if keyword in context:
            return True
        stem = cls._rough_stem(keyword)
        return len(stem) >= 4 and stem in context

    @classmethod
    def _keyword_grounding_check(
        cls, answer: str, context_chunks: list[str],
    ) -> tuple[float, str]:
        """
        Fast keyword-overlap grounding check with novel-term penalty.

        Uses stem-matching to handle morphological variants:
          'concatenation' matches 'concatenated' (stem: 'concat')
          'projection' matches 'projected' (stem: 'project')

        Two-part check:
          1. What fraction of answer keywords appear in context? (recall)
          2. Does the answer introduce technical terms NOT in context? (fabrication)

        Returns (adjusted_score, verdict_text).
        """
        context_lower = " ".join(context_chunks).lower()

        # Extract meaningful keywords from answer
        answer_words = re.findall(r'[a-zA-Z]{4,}', answer.lower())
        keywords = [w for w in answer_words if w not in cls._VERIFY_STOP]

        if not keywords:
            return 0.5, "No meaningful keywords found in answer"

        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique_keywords.append(w)

        # Guard: answers with too few unique keywords can trivially match context
        # (e.g. "Vaswani Transformer" → 2 keywords → 100% match → false VERIFIED)
        min_keywords_for_full_confidence = 3
        few_keywords = len(unique_keywords) < min_keywords_for_full_confidence

        # ── Part 1: keyword overlap ratio (with stem matching) ───────────────
        hits = sum(1 for kw in unique_keywords if cls._word_in_context(kw, context_lower))
        missed = [kw for kw in unique_keywords if not cls._word_in_context(kw, context_lower)]
        base_ratio = hits / len(unique_keywords)

        # ── Part 2: novel technical term penalty ─────────────────────────────
        # Words ≥6 chars whose STEM also doesn't appear in context
        novel_technical = [w for w in missed if len(w) >= 6]
        novel_ratio = len(novel_technical) / len(unique_keywords) if unique_keywords else 0

        # Penalize: novel technical terms reduce confidence
        # Reduced from 1.5/0.4 — old penalty was crushing correct answers
        # (e.g. 'scaled dot-product attention' → 0.07 confidence)
        penalty = min(novel_ratio * 0.5, 0.20)  # gentler penalty
        adjusted = max(0.0, base_ratio - penalty)

        # Build verdict text
        verdict = (
            f"Keyword grounding: {hits}/{len(unique_keywords)} "
            f"({base_ratio:.0%}) answer keywords found in context"
        )
        if novel_technical:
            verdict += (
                f"\nNovel terms (not in context): {', '.join(novel_technical[:5])}"
                f"\nNovel term penalty: -{penalty:.2f}"
            )
        if missed and not novel_technical:
            verdict += f"\nMissing (short): {', '.join(missed[:5])}"

        # Cap confidence for answers with too few unique keywords
        if few_keywords:
            adjusted = min(adjusted, 0.50)
            verdict += (
                f"\nLow keyword count ({len(unique_keywords)}) — "
                f"confidence capped at 0.50"
            )

        return adjusted, verdict

    def verify_answer(
        self,
        query: str,
        context_chunks: list[str],
        answer: str,
    ) -> tuple[bool, str, float]:
        """
        Verify answer against context.

        Two modes (controlled by config.VERIFY_MODE):
          • "heuristic" — keyword overlap only (instant, 0 LLM calls)
          • "full"      — heuristic pre-filter + LLM claim-level check

        Returns:
            (is_verified, verdict_text, confidence_score)
        """
        # ── Keyword overlap check (always runs, instant) ─────────────────────
        h_ratio, h_verdict = self._keyword_grounding_check(answer, context_chunks)

        # ── Heuristic-only mode (local model: skip LLM call entirely) ────────
        if config.VERIFY_MODE == "heuristic":
            is_verified = h_ratio >= config.CONFIDENCE_THRESHOLD
            label = "VERIFIED" if is_verified else "HALLUCINATION"
            verdict = (
                f"{h_verdict}\n"
                f"VERDICT: {label}\n"
                f"CONFIDENCE: {h_ratio:.2f}\n"
                f"REASON: Keyword overlap {'sufficient' if is_verified else 'insufficient'}"
            )
            return is_verified, verdict, h_ratio

        # ── Full mode: fast-reject obvious hallucinations ────────────────────
        if h_ratio < 0.3:
            verdict = (
                f"HEURISTIC GROUNDING FAILED\n{h_verdict}\n"
                f"VERDICT: HALLUCINATION\nCONFIDENCE: {h_ratio:.2f}\n"
                f"REASON: Most claims lack textual grounding in context"
            )
            return False, verdict, h_ratio

        # ── Full mode: LLM claim-level verification ──────────────────────────
        context = "\n\n---\n\n".join(context_chunks)

        system = (
            "You are a strict grounding verifier for a question-answering system.\n"
            "Your job is to check whether EVERY claim in the Answer is EXPLICITLY "
            "supported by a specific quote from the Context.\n\n"
            "Instructions:\n"
            "1. Break the Answer down into atomic factual claims (prefix with CLAIM:)\n"
            "2. For each claim, you MUST extract the EXACT quote from the Context that proves it "
            "(prefix with EVIDENCE: \"<quote>\").\n"
            "3. If the context does NOT explicitly contain the claim, you MUST write exactly 'EVIDENCE: NONE'.\n\n"
            "If ANY claim has 'EVIDENCE: NONE', your final verdict MUST be HALLUCINATION.\n\n"
            "End with EXACTLY these three lines:\n"
            "VERDICT: VERIFIED (if ALL claims are supported) or HALLUCINATION "
            "(if ANY claim lacks evidence)\n"
            "CONFIDENCE: <score from 0.0 to 1.0>\n"
            "REASON: <one sentence>"
        )
        user = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer to verify: {answer}\n\n"
            f"Check each claim in the answer against the context above."
        )

        verdict_text = self._invoke(system, user)

        # Parse confidence score from LLM
        confidence = self._extract_confidence(verdict_text)

        # Count grounding evidence markers
        evidence_none_count = verdict_text.upper().count("EVIDENCE: NONE")
        evidence_count = verdict_text.upper().count("EVIDENCE:") - evidence_none_count

        # Penalise confidence if unsupported claims exist
        if evidence_none_count > 0 and evidence_count > 0:
            penalty = evidence_none_count / (evidence_count + evidence_none_count)
            confidence = max(0.0, confidence - penalty * 0.5)
        elif evidence_none_count > 0 and evidence_count == 0:
            confidence = min(confidence, 0.2)

        # Blend: if keyword heuristic says weak, cap confidence
        if h_ratio < 0.5:
            confidence = min(confidence, h_ratio + 0.1)

        is_verified = (
            "VERIFIED" in verdict_text.upper()
            and "HALLUCINATION" not in verdict_text.upper()
            and confidence >= config.CONFIDENCE_THRESHOLD
            and evidence_none_count == 0
        )

        return is_verified, verdict_text, confidence

    @staticmethod
    def _extract_confidence(text: str) -> float:
        """Extract confidence score from verification response."""
        match = re.search(r"CONFIDENCE:\s*([\d.]+)", text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)
            except ValueError:
                pass
        # Fallback heuristic: count grounding signals
        upper = text.upper()
        if "EVIDENCE: NONE" in upper:
            return 0.3
        if "VERIFIED" in upper and "HALLUCINATION" not in upper:
            return 0.75
        return 0.3
