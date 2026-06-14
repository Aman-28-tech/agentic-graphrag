"""
config.py — Central configuration for the Agentic GraphRAG System

All tunable parameters live here so callers never hard-code values.
Knowledge-graph seed data (nodes, edges, edge contexts) is defined here
and is aligned with the four research papers in docs/ai_papers/:

  • "Attention Is All You Need"          — Vaswani et al., 2017  (arXiv:1706.03762)
  • "BERT: Pre-training of Deep …"       — Devlin et al., 2018   (arXiv:1810.04805)
  • "Neural Machine Translation by …"   — Bahdanau et al., 2014 (arXiv:1409.0473)
  • "Language Models are Few-Shot …"     — Brown et al., 2020    (arXiv:2005.14165)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Hugging Face Offline Mode ─────────────────────────────────────────────────
# After models are downloaded once, skip the hundreds of HEAD requests to
# huggingface.co on every startup. Set HF_HUB_OFFLINE=0 if you need to
# download new models.
if os.getenv("HF_HUB_OFFLINE", "1") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── Paths ──────────────────────────────────────────────────────────────────────
#
#  docs/                       ← DOCS_DIR  (root; holds sample_doc.txt only)
#  └── ai_papers/              ← DOCS_SUBDIR  (ALL research PDFs go here)
#      ├── attention_is_all_you_need_vaswani2017.pdf
#      ├── bert_devlin2018.pdf
#      ├── neural_machine_translation_bahdanau2014.pdf
#      └── gpt3_paper.pdf   (downloaded by ingest.py --force-reindex)
#
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR         = os.path.join(BASE_DIR, "docs")               # root — txt/md files
DOCS_SUBDIR      = os.path.join(BASE_DIR, "docs", "ai_papers")  # PDFs only
QDRANT_PATH      = os.path.join(BASE_DIR, "qdrant_storage")
BM25_INDEX_PATH  = os.path.join(BASE_DIR, "bm25_index.pkl")
SCAN_SUBDIRS     = True   # loader walks DOCS_DIR recursively → picks up ai_papers/

# ── Embedding model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ── Qdrant ─────────────────────────────────────────────────────────────────────
QDRANT_MODE       = os.getenv("QDRANT_MODE", "local")
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "rag_documents"

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 800   # characters per chunk
CHUNK_OVERLAP   = 200   # increased: prevents info fragmentation across boundaries
MIN_CHUNK_LENGTH = 60   # discard chunks shorter than this

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER      = os.getenv("LLM_PROVIDER", "local")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
LOCAL_MODEL_ID    = os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
LLM_MAX_TOKENS    = 256
LLM_TEMPERATURE   = 0.1
EXPAND_MAX_TOKENS  = 64    # query expansion needs very few tokens

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_QDRANT = 15   # semantic search results (more candidates for cross-encoder)
TOP_K_BM25   = 15   # keyword search results (more candidates for cross-encoder)
TOP_K_GRAPH  = 10   # graph traversal results (was 5 — missing relationships)
TOP_K_RERANK = 8    # cross-encoder output (was 6 — more context for LLM)
RRF_K        = 60   # Reciprocal Rank Fusion constant
MAX_RETRIES  = 1    # reduced: retries waste 60s+ on CPU; correct answers were rejected

# KG slot reservation: always keep top N graph results in final context
# regardless of cross-encoder score. ms-marco was trained for web search
# and systematically under-scores KG edge context passages.
KG_RESERVED_SLOTS = 2

# Default retrieval strategy: "fusion_all" runs Graph + Vector + BM25 for every
# query, then fuses via RRF + cross-encoder. Best for small corpora (<1000 chunks)
# where retrieval cost is negligible vs LLM generation time.
# Other options: "graph", "hybrid", "bm25", "vector", "router" (auto-select)
DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", "fusion_all")

# ── Cross-Encoder Re-ranker ────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Query expansion ────────────────────────────────────────────────────────────
NUM_EXPANDED_QUERIES = 3
MIN_QUERY_WORDS_FOR_EXPANSION = 5  # skip LLM expansion for short entity queries
# Re-enabled: expansion adds ~24s on CPU but significantly improves recall.
# Override with ENABLE_QUERY_EXPANSION=false if latency is critical.
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"

# ── Caching ───────────────────────────────────────────────────────────────────
CACHE_ENABLED     = True
CACHE_TTL_SECONDS = 3600

# ── Self-Verification / Confidence ─────────────────────────────────────────────
# For local models: use fast keyword-overlap heuristic (0 LLM calls, instant)
# For API models (openai/anthropic): use full LLM claim-level verification
VERIFY_MODE = "heuristic" if LLM_PROVIDER.lower() == "local" else "full"
CONFIDENCE_THRESHOLD = 0.35  # balanced: works with gentler novel-term penalty

# ── FastAPI & Streamlit ────────────────────────────────────────────────────────
API_HOST       = os.getenv("API_HOST", "0.0.0.0")
API_PORT       = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL            = "INFO"   # DEBUG | INFO | WARNING | ERROR
SHOW_RETRIEVED_CHUNKS = True


# ══════════════════════════════════════════════════════════════════════════════
#  Knowledge Graph Seed Data
#  Aligned with the three papers in docs/:
#    [T]  Attention Is All You Need  (Vaswani 2017)
#    [B]  BERT                       (Devlin 2018)
#    [Ba] Bahdanau Attention         (Bahdanau 2014)
# ══════════════════════════════════════════════════════════════════════════════

KG_NODES = [
    # ── Models ────────────────────────────────────────────────────────────────
    ("Transformer",        {"type": "architecture", "paper": "1706.03762", "year": 2017}),
    ("BERT",               {"type": "model",        "paper": "1810.04805", "year": 2018}),
    ("GPT",                {"type": "model",        "creator": "OpenAI",   "year": 2018}),
    ("GPT-3",              {"type": "model",        "creator": "OpenAI",   "year": 2020,
                            "params": "175B"}),
    ("GPT-4",              {"type": "model",        "creator": "OpenAI",   "year": 2023}),

    # ── Transformer architecture components ───────────────────────────────────
    ("attention_mechanism",      {"type": "technique", "origin_paper": "1409.0473"}),
    ("self_attention",           {"type": "technique", "origin_paper": "1706.03762"}),
    ("multi_head_attention",     {"type": "technique", "origin_paper": "1706.03762"}),
    ("scaled_dot_product",       {"type": "technique", "origin_paper": "1706.03762"}),
    ("positional_encoding",      {"type": "technique", "origin_paper": "1706.03762"}),
    ("encoder_decoder",          {"type": "architecture_component"}),
    ("recurrent_network",        {"type": "architecture", "limitation": "sequential_computation"}),
    ("feed_forward_network",     {"type": "architecture_component", "origin_paper": "1706.03762"}),
    ("layer_normalization",      {"type": "technique", "origin_paper": "1706.03762"}),
    ("residual_connection",      {"type": "technique", "origin_paper": "1706.03762"}),
    ("dropout",                  {"type": "technique"}),
    ("beam_search",              {"type": "technique"}),
    ("label_smoothing",          {"type": "technique"}),

    # ── BERT techniques ──────────────────────────────────────────────────────
    ("bidirectional_encoding",   {"type": "technique", "origin_paper": "1810.04805"}),
    ("masked_language_modeling",  {"type": "technique", "origin_paper": "1810.04805",
                                  "alias": "MLM"}),
    ("next_sentence_prediction", {"type": "technique", "origin_paper": "1810.04805",
                                  "alias": "NSP"}),
    ("fine_tuning",              {"type": "task"}),
    ("WordPiece",                {"type": "technique", "origin_paper": "1810.04805"}),
    ("SQuAD",                    {"type": "benchmark"}),
    ("GLUE",                     {"type": "benchmark"}),

    # ── GPT-3 / scaling concepts ─────────────────────────────────────────────
    ("few_shot_learning",        {"type": "technique", "origin_paper": "2005.14165"}),
    ("in_context_learning",      {"type": "technique", "origin_paper": "2005.14165"}),
    ("zero_shot_learning",       {"type": "technique", "origin_paper": "2005.14165"}),
    ("scaling_laws",             {"type": "concept",   "origin_paper": "2005.14165"}),
    ("language_modeling",         {"type": "task"}),
    ("autoregressive",           {"type": "technique"}),

    # ── Bahdanau attention components ────────────────────────────────────────
    ("alignment_model",          {"type": "technique", "origin_paper": "1409.0473"}),
    ("context_vector",           {"type": "concept",   "origin_paper": "1409.0473"}),
    ("encoder_decoder_rnn",      {"type": "architecture", "origin_paper": "1409.0473"}),

    # ── Organizations / Authors ───────────────────────────────────────────────
    ("Google",            {"type": "organization"}),
    ("OpenAI",            {"type": "organization"}),
    ("University_Montreal", {"type": "organization"}),

    # Transformer authors (all 8)
    ("Vaswani",           {"type": "author", "full_name": "Ashish Vaswani",
                           "affiliation": "Google"}),
    ("Shazeer",           {"type": "author", "full_name": "Noam Shazeer",
                           "affiliation": "Google"}),
    ("Parmar",            {"type": "author", "full_name": "Niki Parmar",
                           "affiliation": "Google"}),
    ("Uszkoreit",         {"type": "author", "full_name": "Jakob Uszkoreit",
                           "affiliation": "Google"}),
    ("Jones",             {"type": "author", "full_name": "Llion Jones",
                           "affiliation": "Google"}),
    ("Gomez",             {"type": "author", "full_name": "Aidan N. Gomez",
                           "affiliation": "Google"}),
    ("Kaiser",            {"type": "author", "full_name": "Łukasz Kaiser",
                           "affiliation": "Google"}),
    ("Polosukhin",        {"type": "author", "full_name": "Illia Polosukhin",
                           "affiliation": "Google"}),

    # BERT authors
    ("Devlin",            {"type": "author", "full_name": "Jacob Devlin",
                           "affiliation": "Google"}),

    # Bahdanau attention authors
    ("Bahdanau",          {"type": "author", "full_name": "Dzmitry Bahdanau",
                           "affiliation": "University_Montreal"}),
    ("Cho",               {"type": "author", "full_name": "Kyunghyun Cho",
                           "affiliation": "University_Montreal"}),
    ("Bengio",            {"type": "author", "full_name": "Yoshua Bengio",
                           "affiliation": "University_Montreal"}),

    # GPT-3 authors
    ("Brown",             {"type": "author", "full_name": "Tom Brown",
                           "affiliation": "OpenAI"}),

    # ── Tasks ─────────────────────────────────────────────────────────────────
    ("machine_translation",  {"type": "task"}),
    ("question_answering",   {"type": "task"}),
    ("text_classification",  {"type": "task"}),
    ("RAG",                  {"type": "technique", "year": 2020}),
]


KG_EDGES = [
    # ── Transformer paper [T] ─────────────────────────────────────────────────
    ("Transformer",      "attention_mechanism",   "relies_on"),
    ("Transformer",      "self_attention",        "uses"),
    ("Transformer",      "multi_head_attention",  "uses"),
    ("Transformer",      "scaled_dot_product",    "uses"),
    ("Transformer",      "positional_encoding",   "uses"),
    ("Transformer",      "encoder_decoder",       "has_component"),
    ("Transformer",      "recurrent_network",     "replaces"),
    ("Transformer",      "machine_translation",   "applied_to"),
    ("Transformer",      "Google",                "introduced_by"),
    ("Transformer",      "feed_forward_network",  "has_component"),
    ("Transformer",      "layer_normalization",   "uses"),
    ("Transformer",      "residual_connection",   "uses"),
    ("Transformer",      "dropout",               "uses"),
    ("Transformer",      "beam_search",           "uses_for_decoding"),
    ("Transformer",      "label_smoothing",       "uses"),
    ("Vaswani",          "Transformer",           "authored"),
    ("Shazeer",          "Transformer",           "co_authored"),
    ("Parmar",           "Transformer",           "co_authored"),
    ("Uszkoreit",        "Transformer",           "co_authored"),
    ("Jones",            "Transformer",           "co_authored"),
    ("Gomez",            "Transformer",           "co_authored"),
    ("Kaiser",           "Transformer",           "co_authored"),
    ("Polosukhin",       "Transformer",           "co_authored"),

    # ── BERT paper [B] ───────────────────────────────────────────────────────
    ("BERT",             "Transformer",           "based_on"),
    ("BERT",             "bidirectional_encoding","uses"),
    ("BERT",             "masked_language_modeling", "pre_trained_with"),
    ("BERT",             "next_sentence_prediction", "pre_trained_with"),
    ("BERT",             "self_attention",        "uses"),
    ("BERT",             "fine_tuning",           "supports"),
    ("BERT",             "Google",                "developed_by"),
    ("BERT",             "WordPiece",             "tokenized_with"),
    ("BERT",             "SQuAD",                 "evaluated_on"),
    ("BERT",             "GLUE",                  "evaluated_on"),
    ("BERT",             "question_answering",    "applied_to"),
    ("BERT",             "text_classification",   "applied_to"),
    ("Devlin",           "BERT",                  "authored"),

    # ── GPT-3 paper [G] ─────────────────────────────────────────────────────
    ("GPT-3",            "Transformer",           "based_on"),
    ("GPT-3",            "few_shot_learning",     "demonstrates"),
    ("GPT-3",            "in_context_learning",   "introduces"),
    ("GPT-3",            "zero_shot_learning",    "demonstrates"),
    ("GPT-3",            "scaling_laws",          "validates"),
    ("GPT-3",            "language_modeling",      "trained_on"),
    ("GPT-3",            "autoregressive",        "uses"),
    ("GPT-3",            "OpenAI",                "developed_by"),
    ("Brown",            "GPT-3",                 "authored"),

    # ── Bahdanau attention [Ba] ───────────────────────────────────────────────
    ("attention_mechanism", "alignment_model",    "implemented_via"),
    ("attention_mechanism", "encoder_decoder",    "augments"),
    ("attention_mechanism", "machine_translation","improves"),
    ("attention_mechanism", "recurrent_network",  "augments"),
    ("attention_mechanism", "context_vector",     "produces"),
    ("alignment_model",  "context_vector",        "computes"),
    ("encoder_decoder_rnn", "attention_mechanism","enhanced_by"),
    ("Bahdanau",         "attention_mechanism",   "introduced"),
    ("Cho",              "attention_mechanism",   "co_authored"),
    ("Bengio",           "attention_mechanism",   "co_authored"),
    ("Bahdanau",         "University_Montreal",   "affiliated_with"),

    # ── Cross-paper lineage ───────────────────────────────────────────────────
    ("multi_head_attention",  "attention_mechanism", "extends"),
    ("self_attention",        "attention_mechanism", "variant_of"),
    ("scaled_dot_product",    "attention_mechanism", "implementation_of"),
    ("BERT",             "RAG",                   "used_in"),
    ("Transformer",      "GPT",                   "basis_for"),
    ("Transformer",      "BERT",                  "basis_for"),
    ("GPT",              "OpenAI",                "developed_by"),
    ("GPT",              "GPT-3",                 "predecessor_of"),
    ("GPT",              "autoregressive",        "uses"),
    ("few_shot_learning","in_context_learning",   "enabled_by"),
]


# Enriched prose context for key edges — handed verbatim to the LLM
KG_EDGE_CONTEXTS = {

    # ── Transformer ───────────────────────────────────────────────────────────
    ("Transformer", "attention_mechanism"): (
        "From 'Attention Is All You Need' (Vaswani et al., 2017): "
        "The Transformer model architecture eschews recurrence and instead relies entirely on "
        "an attention mechanism to draw global dependencies between input and output. "
        "This allows for significantly more parallelization and reaches state-of-the-art "
        "translation quality after training for as little as twelve hours on eight P100 GPUs."
    ),
    ("Transformer", "multi_head_attention"): (
        "Multi-head attention projects queries, keys, and values h times with different learned "
        "linear projections to dk, dk, and dv dimensions respectively. On each of these projected "
        "versions the attention function is performed in parallel, yielding dv-dimensional output "
        "values. These are concatenated and once again projected resulting in the final values. "
        "The Transformer uses h=8 parallel attention heads with dk=dv=64."
    ),
    ("Transformer", "positional_encoding"): (
        "Since the Transformer contains no recurrence and no convolution, positional encodings "
        "are injected into the input embeddings to give the model information about the relative "
        "or absolute position of tokens. The positional encodings use sine and cosine functions "
        "of different frequencies: PE(pos,2i)=sin(pos/10000^(2i/d_model))."
    ),
    ("Transformer", "recurrent_network"): (
        "The Transformer replaces recurrent neural networks (RNNs/LSTMs) entirely. "
        "RNNs process sequences token-by-token, making parallelization impossible within "
        "training examples. The Transformer's self-attention mechanism allows every position "
        "to attend to every other position in a single step."
    ),
    ("Vaswani", "Transformer"): (
        "The Transformer was introduced by Ashish Vaswani, Noam Shazeer, Niki Parmar, "
        "Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin "
        "from Google Brain and Google Research in the paper 'Attention Is All You Need' (2017)."
    ),

    # ── BERT ──────────────────────────────────────────────────────────────────
    ("BERT", "Transformer"): (
        "From 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding' "
        "(Devlin et al., 2018): BERT's model architecture is a multi-layer bidirectional "
        "Transformer encoder based on the original Transformer architecture. "
        "BERT-Base uses 12 layers, 768 hidden size, 12 attention heads (110M parameters). "
        "BERT-Large uses 24 layers, 1024 hidden size, 16 attention heads (340M parameters)."
    ),
    ("BERT", "bidirectional_encoding"): (
        "Unlike previous language representation models, BERT is designed to pre-train deep "
        "bidirectional representations from unlabeled text by jointly conditioning on both "
        "left and right context in all layers. As a result, the pre-trained BERT model can be "
        "fine-tuned with just one additional output layer to create state-of-the-art models "
        "for a wide range of tasks."
    ),
    ("BERT", "masked_language_modeling"): (
        "BERT uses Masked Language Modeling (MLM) as a pre-training objective: 15% of all "
        "WordPiece tokens in each sequence are masked at random. The model then predicts the "
        "masked tokens based on the surrounding context from both directions. "
        "This allows training a deep bidirectional Transformer."
    ),
    ("BERT", "next_sentence_prediction"): (
        "BERT is also pre-trained on Next Sentence Prediction (NSP): given two sentences A and B, "
        "50% of the time B is the actual next sentence; 50% of the time it is a random sentence. "
        "This task helps the model understand inter-sentence relationships, which is important "
        "for tasks like Question Answering and Natural Language Inference."
    ),
    ("BERT", "Google"): (
        "BERT was developed by Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova "
        "at Google AI Language. It was published in 2018 and achieved state-of-the-art results "
        "on eleven NLP benchmarks including GLUE, MultiNLI, SQuAD v1.1 and SQuAD v2.0."
    ),
    ("Devlin", "BERT"): (
        "Jacob Devlin (lead author), Ming-Wei Chang, Kenton Lee, and Kristina Toutanova from "
        "Google introduced BERT in the paper 'BERT: Pre-training of Deep Bidirectional "
        "Transformers for Language Understanding' (arXiv:1810.04805, 2018)."
    ),

    # ── Bahdanau attention ────────────────────────────────────────────────────
    ("attention_mechanism", "alignment_model"): (
        "From 'Neural Machine Translation by Jointly Learning to Align and Translate' "
        "(Bahdanau et al., 2014): Each time the proposed model generates a word in a translation, "
        "it soft-searches for a set of positions in a source sentence where the most relevant "
        "information is concentrated. The model then predicts a target word based on the context "
        "vectors associated with these source positions and all the previously generated target words."
    ),
    ("attention_mechanism", "machine_translation"): (
        "Bahdanau attention was introduced to overcome the fixed-length bottleneck of "
        "encoder-decoder RNNs for machine translation. By computing a context vector as a "
        "weighted sum of all encoder hidden states (weights = alignment scores), the decoder "
        "can focus on relevant parts of the source sentence at each decoding step, "
        "dramatically improving performance on long sentences."
    ),
    ("attention_mechanism", "recurrent_network"): (
        "The Bahdanau attention mechanism augments RNN-based encoder-decoder architectures. "
        "Instead of compressing the entire source sentence into a single fixed-length vector, "
        "the encoder produces a sequence of hidden states, and the attention mechanism "
        "produces a dynamic context vector as a weighted sum of those states at each decoding step."
    ),
    ("Bahdanau", "attention_mechanism"): (
        "Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio introduced the attention mechanism "
        "for neural machine translation in 2014 (arXiv:1409.0473). This was the first work "
        "to show that a neural network could selectively focus on relevant parts of input "
        "sequences, which became the foundation for the Transformer's self-attention."
    ),

    # ── Cross-paper lineage ───────────────────────────────────────────────────
    ("multi_head_attention", "attention_mechanism"): (
        "Multi-head attention (Vaswani et al., 2017) is a direct extension of the "
        "attention mechanism proposed by Bahdanau et al. (2014). Instead of a single "
        "attention function, multi-head attention runs h attention functions in parallel "
        "and concatenates their outputs, allowing the model to jointly attend to information "
        "from different representation subspaces."
    ),
    ("self_attention", "attention_mechanism"): (
        "Self-attention (also called intra-attention) is a variant where queries, keys, "
        "and values all come from the same sequence. Unlike Bahdanau attention which relates "
        "encoder and decoder states, self-attention relates different positions within a "
        "single sequence to compute a representation of that sequence."
    ),
    ("scaled_dot_product", "attention_mechanism"): (
        "Scaled dot-product attention computes: Attention(Q,K,V) = softmax(QK^T / √dk) V. "
        "The scaling factor 1/√dk counteracts the problem of dot products growing large in "
        "magnitude for high-dimensional keys, which pushes softmax into regions with "
        "extremely small gradients."
    ),
    ("Transformer", "BERT"): (
        "The Transformer architecture (Vaswani et al., 2017) is the direct architectural "
        "predecessor of BERT. While the Transformer was designed for sequence-to-sequence "
        "tasks, BERT adapts only the encoder stack and pre-trains it bidirectionally "
        "on unlabeled text corpora."
    ),
    ("Transformer", "GPT"): (
        "GPT (Radford et al., OpenAI 2018) uses only the decoder portion of the "
        "Transformer architecture for unidirectional (left-to-right) language modeling, "
        "while BERT uses only the encoder for bidirectional representation learning."
    ),

    # ── New Transformer architecture contexts ─────────────────────────────────
    ("Transformer", "feed_forward_network"): (
        "Each layer of the Transformer contains a position-wise feed-forward network "
        "consisting of two linear transformations with a ReLU activation: "
        "FFN(x) = max(0, xW1 + b1)W2 + b2. The inner layer dimensionality is "
        "d_ff = 2048, while the model dimensionality is d_model = 512."
    ),
    ("Transformer", "layer_normalization"): (
        "The Transformer applies layer normalization after each sub-layer (self-attention "
        "and feed-forward). Combined with residual connections, this forms the pattern: "
        "LayerNorm(x + Sublayer(x)), which stabilizes training of deep networks."
    ),
    ("Transformer", "residual_connection"): (
        "Residual connections are employed around each of the two sub-layers in the "
        "Transformer (multi-head attention and feed-forward network). The output of each "
        "sub-layer is LayerNorm(x + Sublayer(x)), enabling effective training of "
        "deep models by allowing gradients to flow directly through skip connections."
    ),
    ("Shazeer", "Transformer"): (
        "Noam Shazeer co-authored 'Attention Is All You Need' with Ashish Vaswani, "
        "Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, "
        "and Illia Polosukhin at Google Brain and Google Research (2017)."
    ),

    # ── New BERT contexts ────────────────────────────────────────────────────
    ("BERT", "WordPiece"): (
        "BERT uses WordPiece tokenization with a 30,000 token vocabulary. Input sequences "
        "always start with a special [CLS] token, and sentence pairs are separated by [SEP]. "
        "The WordPiece embeddings are combined with segment and position embeddings."
    ),
    ("BERT", "SQuAD"): (
        "BERT achieved state-of-the-art results on SQuAD v1.1 (F1: 93.2) and SQuAD v2.0 "
        "(F1: 83.1), surpassing human performance on the reading comprehension benchmark. "
        "For SQuAD, BERT learns start and end token pointers during fine-tuning."
    ),
    ("BERT", "GLUE"): (
        "BERT-Large achieved a GLUE benchmark score of 80.5, a 7.7 point absolute improvement "
        "over the prior state of the art. GLUE evaluates models on 9 NLU tasks including "
        "sentiment analysis (SST-2), textual entailment (MNLI), and similarity (STS-B)."
    ),

    # ── GPT-3 contexts ───────────────────────────────────────────────────────
    ("GPT-3", "Transformer"): (
        "GPT-3 is a 175 billion parameter autoregressive language model based on the "
        "Transformer decoder architecture. It uses 96 layers, 96 attention heads, "
        "and a context window of 2048 tokens. GPT-3 is 10x larger than any previous "
        "non-sparse language model."
    ),
    ("GPT-3", "few_shot_learning"): (
        "From 'Language Models are Few-Shot Learners' (Brown et al., 2020): GPT-3 achieves "
        "strong performance on many NLP tasks using only a few demonstration examples "
        "provided in the prompt (few-shot), without any gradient updates or fine-tuning. "
        "This ability emerges at scale and improves log-linearly with model size."
    ),
    ("GPT-3", "in_context_learning"): (
        "GPT-3 introduced in-context learning: the model conditions on a natural language "
        "task description and/or a few examples at inference time, and generates the answer "
        "without updating any model parameters. This differs from traditional fine-tuning "
        "which requires gradient descent on labeled data."
    ),
    ("GPT-3", "zero_shot_learning"): (
        "GPT-3 demonstrates zero-shot capabilities where the model performs tasks given only "
        "a natural language instruction and no examples. While less accurate than few-shot, "
        "zero-shot performance improves dramatically with model scale."
    ),
    ("GPT-3", "scaling_laws"): (
        "GPT-3 validates neural scaling laws: performance on language tasks improves "
        "as a smooth power law with model size, dataset size, and compute. GPT-3 was "
        "trained on 300 billion tokens from filtered Common Crawl, WebText2, Books, "
        "and Wikipedia corpora."
    ),
    ("Brown", "GPT-3"): (
        "Tom Brown and 30+ co-authors at OpenAI published 'Language Models are Few-Shot "
        "Learners' (arXiv:2005.14165, 2020), introducing GPT-3 with 175 billion parameters."
    ),

    # ── New Bahdanau contexts ────────────────────────────────────────────────
    ("attention_mechanism", "context_vector"): (
        "The attention mechanism produces a context vector as a weighted sum of encoder "
        "hidden states: c_i = Σ_j α_ij · h_j, where α_ij are attention weights (alignment "
        "scores) computed by the alignment model. This dynamic context vector replaces the "
        "fixed-length bottleneck of traditional encoder-decoder architectures."
    ),
    ("encoder_decoder_rnn", "attention_mechanism"): (
        "The encoder-decoder RNN architecture for machine translation was enhanced by "
        "Bahdanau's attention mechanism. The encoder (bidirectional GRU/LSTM) produces "
        "hidden state annotations, and the attention mechanism allows the decoder to "
        "selectively focus on different source positions at each generation step."
    ),

    # ── Cross-model lineage contexts ─────────────────────────────────────────
    ("GPT", "GPT-3"): (
        "GPT (2018, 117M params) established the paradigm of unsupervised pre-training "
        "followed by supervised fine-tuning. GPT-3 (2020, 175B params) scaled this approach "
        "by 1000x and showed that fine-tuning becomes unnecessary at sufficient scale, "
        "as the model can learn tasks purely from in-context examples."
    ),
    ("few_shot_learning", "in_context_learning"): (
        "Few-shot learning in GPT-3 is enabled by in-context learning: the model uses "
        "demonstration examples in the prompt to understand the task format and generate "
        "appropriate responses. Both rely on the model's ability to perform pattern "
        "completion over the prompt context without weight updates."
    ),
}
