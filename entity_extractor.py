"""
entity_extractor.py — Automatic Entity & Relation Extraction (v2)

Zero-dependency extractor that discovers entities and relationships from
document chunks using domain-specific patterns (regex + vocabulary).

v2 fixes:
  • Entity normalization: Transformer + transformer + the_transformer → Transformer
  • Garbage filtering: stop words, OCR artifacts, too-short entities
  • Co-occurrence cap: max 5 entities per chunk for co-occurrence (prevents O(n²) explosion)
  • Author dedup: merged with seed KG names via case-insensitive lookup

No SpaCy/NLTK needed — runs instantly on CPU.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from utils import get_logger

logger = get_logger(__name__)


# ── Domain Vocabulary ─────────────────────────────────────────────────────────

# Known model/architecture names — canonical forms (will be matched case-insensitive)
_MODEL_CANONICAL = {
    "transformer": "Transformer",
    "bert": "BERT",
    "gpt": "GPT",
    "gpt-2": "GPT-2",
    "gpt-3": "GPT-3",
    "gpt-4": "GPT-4",
    "elmo": "ELMo",
    "xlnet": "XLNet",
    "roberta": "RoBERTa",
    "albert": "ALBERT",
    "t5": "T5",
    "bart": "BART",
    "llama": "LLaMA",
    "word2vec": "Word2Vec",
    "glove": "GloVe",
}

# Known technique/method patterns → canonical name mapping
_TECHNIQUE_PATTERNS = [
    (r"self[- ]attention",                             "self_attention"),
    (r"multi[- ]head attention",                       "multi_head_attention"),
    (r"scaled dot[- ]product(?: attention)?",          "scaled_dot_product"),
    (r"cross[- ]attention",                            "cross_attention"),
    (r"masked (?:language )?model(?:ing)?",            "masked_language_modeling"),
    (r"next sentence prediction",                      "next_sentence_prediction"),
    (r"causal language model(?:ing)?",                 "causal_language_modeling"),
    (r"positional encod(?:ing|ings)",                  "positional_encoding"),
    (r"layer normali[sz]ation",                        "layer_normalization"),
    (r"batch normali[sz]ation",                        "batch_normalization"),
    (r"residual connect(?:ion|ions)",                  "residual_connection"),
    (r"skip connect(?:ion|ions)",                      "skip_connection"),
    (r"feed[- ]forward(?: network)?",                  "feed_forward_network"),
    (r"beam search",                                   "beam_search"),
    (r"label smoothing",                               "label_smoothing"),
    (r"attention (?:mechanism|weight|score|head)s?",   "attention_mechanism"),
    (r"encoder[- ]decoder",                            "encoder_decoder"),
    (r"sequence[- ]to[- ]sequence",                    "sequence_to_sequence"),
    (r"(?:fine[- ]tun(?:ing|ed))",                     "fine_tuning"),
    (r"(?:pre[- ]train(?:ing|ed))",                    "pre_training"),
    (r"transfer learning",                             "transfer_learning"),
    (r"few[- ]shot(?: learning)?",                     "few_shot_learning"),
    (r"zero[- ]shot(?: learning)?",                    "zero_shot_learning"),
    (r"in[- ]context learning",                        "in_context_learning"),
    (r"(?:knowledge )?distillation",                   "knowledge_distillation"),
    (r"(?:byte[- ]pair) (?:encoding|tokeniz\w+)",      "byte_pair_encoding"),
    (r"(?:word[- ]?piece) (?:encoding|tokeniz\w+)",    "wordpiece_tokenization"),
    (r"(?:sentence[- ]?piece) (?:encoding|tokeniz\w+)","sentencepiece_tokenization"),
    (r"alignment (?:model|score|function)",             "alignment_model"),
    (r"(?:context|fixed[- ]length) vector",             "context_vector"),
    (r"(?:bi-?directional) (?:encoding|transformer|rnn|lstm|gru)",
                                                        "bidirectional_encoding"),
    (r"retrieval[- ]augmented generation",              "RAG"),
    (r"reinforcement learning(?: from human feedback)?","reinforcement_learning"),
    (r"backpropagation",                                "backpropagation"),
]

# Known benchmark/dataset names — canonical forms
_BENCHMARK_CANONICAL = {
    "squad": "SQuAD",
    "glue": "GLUE",
    "superglue": "SuperGLUE",
    "mnli": "MNLI",
    "qqp": "QQP",
    "sst-2": "SST-2",
    "mrpc": "MRPC",
    "cola": "CoLA",
    "qnli": "QNLI",
    "imagenet": "ImageNet",
    "wmt": "WMT",
    "iwslt": "IWSLT",
    "penn treebank": "Penn_Treebank",
    "common crawl": "Common_Crawl",
    "wikipedia": "Wikipedia",
    "bookcorpus": "BookCorpus",
    "webtext": "WebText",
    "lambada": "LAMBADA",
    "triviaqa": "TriviaQA",
    "boolq": "BoolQ",
    "hellaswag": "HellaSwag",
    "natural questions": "Natural_Questions",
}

# Known metric names → canonical
_METRIC_CANONICAL = {
    "bleu": "BLEU",
    "rouge": "ROUGE",
    "f1": "F1_score",
    "accuracy": "accuracy",
    "perplexity": "perplexity",
}

# ── GARBAGE FILTER ────────────────────────────────────────────────────────────
# Generic academic words that should NEVER be entities
_ENTITY_STOP_WORDS = {
    # Generic nouns
    "model", "models", "paper", "papers", "result", "results",
    "method", "methods", "approach", "approaches", "system", "systems",
    "work", "works", "task", "tasks", "data", "dataset", "input", "output",
    "performance", "training", "evaluation", "experiment", "experiments",
    "figure", "table", "section", "chapter", "appendix", "abstract",
    "introduction", "conclusion", "related", "previous", "proposed",
    "architecture", "framework", "network", "networks", "module", "modules",
    "layer", "layers", "function", "functions", "parameter", "parameters",
    "weight", "weights", "value", "values", "key", "keys", "query", "queries",
    "token", "tokens", "word", "words", "sentence", "sentences",
    "example", "examples", "step", "steps", "process", "case", "state",
    # 2-char OCR artifacts
    "en", "de", "fr", "es", "mo", "yu", "qi", "ge", "et", "al",
    # Common false positives
    "the", "and", "for", "with", "from", "this", "that",
    "our", "their", "which", "each", "both", "also", "more",
    # Short noise
    "new", "use", "set", "one", "two", "may", "can", "will",
}

# Relation patterns: (regex_pattern, relation_type)
_RELATION_PATTERNS = [
    (r"(\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)*)\s+(?:is\s+)?based\s+on\s+(?:the\s+)?(\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)*)",
     "based_on"),
    (r"(\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)*)\s+(?:outperform|surpass|exceed|beat)\w*\s+(?:the\s+)?(\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)*)",
     "outperforms"),
    (r"(\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)*)\s+(?:was\s+)?(?:introduced|proposed|presented|developed)\s+(?:by|in)\s+(\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)*)",
     "introduced_by"),
    (r"(\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)*)\s+(?:extends?|builds?\s+(?:on|upon))\s+(?:the\s+)?(\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)*)",
     "extends"),
    (r"(\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)*)\s+is\s+a\s+(?:variant|extension|modification)\s+of\s+(\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)*)",
     "variant_of"),
    (r"(\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)*)\s+(?:replaces?|eliminates?)\s+(?:the\s+)?(\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)*)",
     "replaces"),
]

# Author name pattern
_AUTHOR_PATTERN = re.compile(
    r"\b([A-Z][a-z]{2,15})\s+"
    r"(?:[A-Z]\.\s*)?"
    r"([A-Z][a-z]{2,20})\b"
)

# False-positive first names
_FALSE_AUTHOR_FIRSTS = {
    "The", "This", "That", "These", "Those", "Each", "Every",
    "Figure", "Table", "Section", "Equation", "Chapter", "Algorithm",
    "Our", "We", "In", "On", "For", "From", "With", "Let",
    "Given", "Note", "See", "Since", "When", "Where", "While",
    "After", "Before", "During", "Between", "Through", "Over",
    "Under", "Above", "Below", "Unlike", "Similar", "Compared",
    "However", "Moreover", "Furthermore", "Therefore", "Thus",
    "Here", "Such", "Most", "Many", "Some", "Several", "First",
    "Second", "Third", "Finally", "Specifically", "Typically",
    "Clearly", "Recently", "Existing", "Previous", "Current",
}

# False-positive last names (common words that look like last names)
_FALSE_AUTHOR_LASTS = {
    "Model", "Models", "Learning", "Network", "Networks", "Language",
    "Attention", "Encoder", "Decoder", "Training", "Translation",
    "Representation", "Embedding", "Generation", "Classification",
    "Prediction", "Architecture", "Performance", "Sequence",
    "Information", "Computation", "Function", "Vector", "Matrix",
    "Layer", "Output", "Input", "Token", "Position", "Head",
}

# Co-occurrence limits
MAX_ENTITIES_PER_CHUNK_FOR_COOCCURRENCE = 5
MIN_FREQUENCY_FOR_COOCCURRENCE = 3


# ── Result Types ──────────────────────────────────────────────────────────────

@dataclass
class ExtractedEntity:
    """An entity discovered from text."""
    name: str
    entity_type: str
    frequency: int = 1
    source_chunks: list[int] = field(default_factory=list)

@dataclass
class ExtractedRelation:
    """A relationship discovered from text."""
    source: str
    target: str
    relation: str
    evidence: str = ""


# ── Extractor ─────────────────────────────────────────────────────────────────

class EntityRelationExtractor:
    """
    Extract entities and relations from document chunks using
    domain-specific regex patterns + vocabulary matching.

    v2: proper normalization, garbage filtering, co-occurrence caps.
    """

    def __init__(self) -> None:
        self._technique_res = [
            (re.compile(pat, re.IGNORECASE), canonical)
            for pat, canonical in _TECHNIQUE_PATTERNS
        ]

    def _is_garbage(self, name: str) -> bool:
        """Check if entity name is garbage / stop word."""
        lower = name.lower().replace("_", "")
        # Too short
        if len(lower) < 3:
            return True
        # Stop word
        if lower in _ENTITY_STOP_WORDS:
            return True
        # Pure numbers
        if re.match(r'^\d+$', lower):
            return True
        # OCR artifact patterns: two 2-letter fragments joined
        if re.match(r'^[a-z]{2}_[a-z]{2}$', name):
            return True
        # "the_X" prefix noise
        if name.startswith("the_"):
            return True
        return False

    def extract_entities_from_chunks(
        self, chunks: list[str],
    ) -> dict[str, ExtractedEntity]:
        """
        Extract all entities from a list of text chunks.
        Returns dict: canonical_name -> ExtractedEntity
        """
        entities: dict[str, ExtractedEntity] = {}

        def _add(key: str, etype: str, idx: int) -> None:
            """Add or update an entity, filtering garbage."""
            if self._is_garbage(key):
                return
            if key in entities:
                entities[key].frequency += 1
                if idx not in entities[key].source_chunks:
                    entities[key].source_chunks.append(idx)
            else:
                entities[key] = ExtractedEntity(
                    name=key, entity_type=etype, source_chunks=[idx],
                )

        for idx, chunk in enumerate(chunks):
            # ── Models (canonical names) ──────────────────────────────
            for pattern_lower, canonical in _MODEL_CANONICAL.items():
                pat = re.compile(r'\b' + re.escape(pattern_lower) + r'\b', re.IGNORECASE)
                if pat.search(chunk):
                    _add(canonical, "model", idx)

            # ── Techniques (canonical names) ──────────────────────────
            for pat_re, canonical in self._technique_res:
                if pat_re.search(chunk):
                    _add(canonical, "technique", idx)

            # ── Benchmarks (canonical names) ──────────────────────────
            for pattern_lower, canonical in _BENCHMARK_CANONICAL.items():
                pat = re.compile(r'\b' + re.escape(pattern_lower) + r'\b', re.IGNORECASE)
                if pat.search(chunk):
                    _add(canonical, "benchmark", idx)

            # ── Metrics ───────────────────────────────────────────────
            for pattern_lower, canonical in _METRIC_CANONICAL.items():
                pat = re.compile(
                    r'\b' + re.escape(pattern_lower) + r'\b',
                    re.IGNORECASE,
                )
                if pat.search(chunk):
                    _add(canonical, "metric", idx)

            # ── Author names (strict validation) ──────────────────────
            for m in _AUTHOR_PATTERN.finditer(chunk):
                first, last = m.group(1), m.group(2)
                if first in _FALSE_AUTHOR_FIRSTS:
                    continue
                if last in _FALSE_AUTHOR_LASTS:
                    continue
                # Must be near citation context
                ctx = chunk[max(0, m.start() - 30):m.end() + 40]
                if any(sig in ctx for sig in ("et al", "201", "202", "199",
                                               "(", ")", "[", "]")):
                    _add(last, "person", idx)

        # Filter: keep entities mentioned in ≥2 chunks (noise reduction)
        # Exception: known models/benchmarks always kept
        return {
            k: v for k, v in entities.items()
            if v.frequency >= 2 or v.entity_type in ("model", "benchmark")
        }

    def extract_relations_from_chunks(
        self, chunks: list[str],
        known_entities: set[str] | None = None,
        entity_freq: dict[str, int] | None = None,
    ) -> list[ExtractedRelation]:
        """
        Extract relations from chunks using pattern matching.
        Co-occurrence is capped to prevent edge explosion.
        """
        relations: list[ExtractedRelation] = []
        seen_rels: set[tuple[str, str, str]] = set()
        entity_freq = entity_freq or {}

        for idx, chunk in enumerate(chunks):
            # ── Pattern-based relations ───────────────────────────────
            for pat_str, rel_type in _RELATION_PATTERNS:
                for m in re.finditer(pat_str, chunk):
                    src = self._normalize_relation_arg(m.group(1))
                    tgt = self._normalize_relation_arg(m.group(2))
                    if src == tgt or len(src) < 3 or len(tgt) < 3:
                        continue
                    if self._is_garbage(src) or self._is_garbage(tgt):
                        continue
                    key = (src, tgt, rel_type)
                    if key not in seen_rels:
                        seen_rels.add(key)
                        start = max(0, m.start() - 50)
                        end = min(len(chunk), m.end() + 50)
                        evidence = chunk[start:end].strip()
                        relations.append(ExtractedRelation(
                            source=src, target=tgt,
                            relation=rel_type, evidence=evidence,
                        ))

            # ── Co-occurrence (CAPPED) ────────────────────────────────
            if known_entities:
                found_in_chunk = []
                chunk_lower = chunk.lower()
                for ent in known_entities:
                    ent_lower = ent.lower().replace("_", " ")
                    if ent_lower in chunk_lower or ent.lower() in chunk_lower:
                        # Only co-occur entities with sufficient frequency
                        freq = entity_freq.get(ent, 0)
                        if freq >= MIN_FREQUENCY_FOR_COOCCURRENCE:
                            found_in_chunk.append(ent)

                # Cap: only top N most frequent entities per chunk
                found_in_chunk = found_in_chunk[:MAX_ENTITIES_PER_CHUNK_FOR_COOCCURRENCE]

                for i, e1 in enumerate(found_in_chunk):
                    for e2 in found_in_chunk[i+1:]:
                        key = (e1, e2, "co_occurs_with")
                        rev_key = (e2, e1, "co_occurs_with")
                        if key not in seen_rels and rev_key not in seen_rels:
                            seen_rels.add(key)
                            relations.append(ExtractedRelation(
                                source=e1, target=e2,
                                relation="co_occurs_with",
                                evidence=f"Co-mentioned in chunk {idx}",
                            ))

        return relations

    def _normalize_relation_arg(self, text: str) -> str:
        """Normalize a relation argument to canonical form if possible."""
        text = text.strip().strip('.,;:!?()"\'')
        lower = text.lower().replace("-", "").replace(" ", "")

        # Check model canonical
        for pat, canon in _MODEL_CANONICAL.items():
            if pat.replace("-", "") == lower:
                return canon
        # Check benchmark canonical
        for pat, canon in _BENCHMARK_CANONICAL.items():
            if pat.replace("-", "").replace(" ", "") == lower:
                return canon

        # Default: underscore form
        normalized = re.sub(r'[\s\-]+', '_', text.lower())
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        return normalized

    @staticmethod
    def _is_garbage_static(name: str) -> bool:
        """Static version for use outside instances."""
        lower = name.lower().replace("_", "")
        if len(lower) < 3:
            return True
        if lower in _ENTITY_STOP_WORDS:
            return True
        if re.match(r'^\d+$', lower):
            return True
        if re.match(r'^[a-z]{2}_[a-z]{2}$', name):
            return True
        if name.startswith("the_"):
            return True
        return False


def extract_and_merge(
    chunks: list[str],
    kg,  # KnowledgeGraph instance
) -> tuple[int, int]:
    """
    Run entity + relation extraction on chunks and merge results
    into the existing KnowledgeGraph.

    Returns (new_nodes_added, new_edges_added).
    """
    extractor = EntityRelationExtractor()

    # Step 1: Extract entities (with canonical normalization)
    entities = extractor.extract_entities_from_chunks(chunks)
    logger.info(f"Extracted {len(entities)} unique entities from {len(chunks)} chunks")

    # Step 2: Build entity frequency map for co-occurrence filtering
    entity_freq = {name: ent.frequency for name, ent in entities.items()}
    # Include seed KG nodes with high frequency
    for node in kg.graph.nodes():
        if node not in entity_freq:
            entity_freq[node] = 10  # seed nodes are "always frequent"

    # Step 3: Merge entities into KG (add new nodes)
    new_nodes = 0
    for name, ent in entities.items():
        if name not in kg.graph:
            kg.add_entity(name, type=ent.entity_type, auto_extracted=True,
                          frequency=ent.frequency)
            new_nodes += 1

    # Step 4: Extract relations with all known entities
    all_entity_names = set(kg.graph.nodes())
    relations = extractor.extract_relations_from_chunks(
        chunks, all_entity_names, entity_freq,
    )

    # Step 5: Merge relations into KG (add new edges)
    new_edges = 0
    for rel in relations:
        src, tgt = rel.source, rel.target
        if src in kg.graph and tgt in kg.graph:
            if not kg.graph.has_edge(src, tgt):
                kg.add_relationship(src, tgt, rel.relation, context=rel.evidence)
                new_edges += 1

    logger.info(
        f"Auto-extraction complete: +{new_nodes} nodes, +{new_edges} edges "
        f"(total: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges)"
    )

 