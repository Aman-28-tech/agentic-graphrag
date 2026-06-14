"""
knowledge_graph.py — Phase 4: NetworkX Knowledge Graph

The graph stores NLP/AI domain entities as nodes and their named
relationships as directed edges.  Each edge is enriched with a ``context``
string — actual prose from source documents — so graph retrieval can hand
the LLM real evidence, not just labels like "BERT --uses--> Transformer".

Key capabilities:
  • Entity detection in free-text queries (substring match on node names).
  • 1-hop neighbourhood collection.
  • Multi-hop shortest-path traversal between entity pairs.
  • Runtime graph extension (add_entity / add_relationship).
"""

from __future__ import annotations

from typing import Optional
import networkx as nx
import json
import os
from networkx.readwrite import json_graph

import config
from utils import console, get_logger

logger = get_logger(__name__)


class KnowledgeGraph:
    """
    Directed knowledge graph built on NetworkX.

    Graph schema
    ------------
    Node  attributes : type, creator, year, …  (domain-specific)
    Edge  attributes : relation (str), context (str)

    The ``context`` field on each edge stores a passage of text that explains
    the relationship in prose, giving the LLM grounded evidence when
    constructing answers from graph retrieval results.
    """

    def __init__(self, chunks: list[str] | None = None) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.save_path = os.path.join("docs", "knowledge_graph.json")

        # If not ingesting new chunks and a saved graph exists, load it
        if not chunks and os.path.exists(self.save_path):
            self._load()
            return

        self._build()

        seed_nodes = self.graph.number_of_nodes()
        seed_edges = self.graph.number_of_edges()

        # Auto-extract entities/relations from document chunks (if provided)
        auto_nodes, auto_edges = 0, 0
        if chunks:
            try:
                from entity_extractor import extract_and_merge
                extract_and_merge(chunks, self)
                # Count what was added by comparing to seed counts
                auto_nodes = self.graph.number_of_nodes() - seed_nodes
                auto_edges = self.graph.number_of_edges() - seed_edges
            except Exception as e:
                logger.warning(f"Auto-extraction failed (non-fatal): {e}")
                # Still report whatever was added before the error
                auto_nodes = max(0, self.graph.number_of_nodes() - seed_nodes)
                auto_edges = max(0, self.graph.number_of_edges() - seed_edges)

        console.print(
            f"  [green]✓[/green] Knowledge Graph: "
            f"[bold]{self.graph.number_of_nodes()}[/bold] nodes "
            f"({seed_nodes} seed + {auto_nodes} extracted), "
            f"[bold]{self.graph.number_of_edges()}[/bold] edges "
            f"({seed_edges} seed + {auto_edges} extracted)\n"
        )

    # ── Construction ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        """Populate graph from config.KG_NODES, KG_EDGES, KG_EDGE_CONTEXTS."""
        console.print("[cyan]⚡ Building Knowledge Graph…[/cyan]")

        for node_id, attrs in config.KG_NODES:
            self.graph.add_node(node_id, **attrs)

        for src, dst, relation in config.KG_EDGES:
            fwd_ctx = config.KG_EDGE_CONTEXTS.get((src, dst), "")
            self.graph.add_edge(src, dst, relation=relation, context=fwd_ctx)

            # Mirror edge for bidirectional traversal (only if not already declared)
            if not self.graph.has_edge(dst, src):
                rev_ctx = config.KG_EDGE_CONTEXTS.get((dst, src), fwd_ctx)
                self.graph.add_edge(
                    dst, src,
                    relation=f"inverse_{relation}",
                    context=rev_ctx,
                )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def _find_entities(self, query: str) -> list[str]:
        """
        Return node names that appear in *query* (case-insensitive).

        Matching strategies (broadest to narrowest):
          1. Exact substring:         'BERT' in query
          2. Underscore/hyphen norm:  'multi_head_attention' ↔ 'multi-head attention'
          3. Alias from node attrs:   node has alias='MLM' → matches 'MLM' in query
          4. Individual word match:   node 'scaled_dot_product' matches if all of
                                      ['scaled', 'dot', 'product'] appear in query
        """
        # Normalize query: lowercase, replace hyphens with spaces
        q_lower = query.lower()
        q_normalized = q_lower.replace("-", " ").replace("_", " ")
        q_words = set(q_normalized.split())

        matched = []
        for node in self.graph.nodes():
            node_lower = node.lower()
            node_spaced = node_lower.replace("_", " ")

            # Strategy 1: exact substring
            if node_lower in q_lower:
                matched.append(node)
                continue

            # Strategy 2: underscore/hyphen normalization
            if node_spaced in q_normalized:
                matched.append(node)
                continue

            # Strategy 3: alias match (e.g. MLM, NSP)
            attrs = self.graph.nodes[node]
            alias = attrs.get("alias", "").lower()
            if alias and len(alias) >= 2 and alias in q_lower:
                matched.append(node)
                continue

            # Strategy 4: all parts of a multi-word node name appear in query
            # Only for nodes with 2+ parts (prevents matching single common words)
            parts = node_lower.split("_")
            if len(parts) >= 2 and all(
                part in q_words for part in parts if len(part) >= 3
            ):
                matched.append(node)
                continue

        return matched

    def _edge_text(self, src: str, dst: str) -> str:
        """Format an edge as a readable labelled context string."""
        data     = self.graph[src][dst]
        relation = data.get("relation", "related_to")
        context  = data.get("context", "").strip()
        header   = f"[{src} --{relation}--> {dst}]"
        return f"{header}: {context}" if context else header

    def _path_to_context(self, path: list[str]) -> str:
        """Collect edge context strings along a node path."""
        return "\n".join(
            self._edge_text(path[i], path[i + 1])
            for i in range(len(path) - 1)
            if self.graph.has_edge(path[i], path[i + 1])
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> list[str]:
        """
        Graph-based retrieval.

        Algorithm
        ---------
        1. Detect entities in *query*.
        2. Collect context from all 1-hop edges incident to each entity.
        3. For every ordered pair of entities, find the shortest directed path
           and collect edge context along that path (multi-hop reasoning).
        4. Deduplicate and return up to *top_k* context strings.

        Args:
            query:  Raw user query.
            top_k:  Max results; defaults to ``config.TOP_K_GRAPH``.

        Returns:
            List of context strings, each describing one or more relationships.
        """
        top_k    = top_k or config.TOP_K_GRAPH
        entities = self._find_entities(query)

        if not entities:
            console.print("  [dim]Graph: no entities detected in query[/dim]")
            return []

        console.print(f"  [dim]Graph entities: {entities}[/dim]")

        seen:    set[str]  = set()
        results: list[str] = []

        def _add(ctx: str) -> None:
            if ctx and ctx not in seen:
                seen.add(ctx)
                results.append(ctx)

        # 1-hop neighbourhood
        for entity in entities:
            for neighbour in list(self.graph.successors(entity)) + list(self.graph.predecessors(entity)):
                if self.graph.has_edge(entity, neighbour):
                    _add(self._edge_text(entity, neighbour))

        # Multi-hop paths between entity pairs
        for i, src in enumerate(entities):
            for dst in entities[i + 1:]:
                for u, v in [(src, dst), (dst, src)]:
                    try:
                        path = nx.shortest_path(self.graph, u, v)
                        if len(path) > 1:
                            _add(self._path_to_context(path))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

        return results[:top_k]

    def add_entity(self, entity: str, **attrs) -> None:
        """Add or update a node (e.g. when ingesting new documents at runtime)."""
        self.graph.add_node(entity, **attrs)

    def add_relationship(
        self,
        src: str,
        dst: str,
        relation: str,
        context: str = "",
    ) -> None:
        """Add a directed relationship edge between two entities."""
        self.graph.add_edge(src, dst, relation=relation, context=context)

    def describe(self, entity: str) -> Optional[str]:
        """Return a textual summary of an entity's attributes and edges."""
        if entity not in self.graph:
            return None
        lines = [f"Entity : {entity}"]
        attrs = dict(self.graph.nodes[entity])
        if attrs:
            lines.append("Attrs  : " + ", ".join(f"{k}={v}" for k, v in attrs.items()))
        out = [(_, d, dat) for _, d, dat in self.graph.out_edges(entity, data=True)]
        if out:
            lines.append("Out    : " + ", ".join(f"--{d.get('relation')}→ {n}" for _, n, d in out))
        inc = [(s, _, dat) for s, _, dat in self.graph.in_edges(entity, data=True)]
        if inc:
            lines.append("In     : " + ", ".join(f"{s} --{d.get('relation')}→" for s, _, d in inc))
        return "\n".join(lines)

    def save(self) -> None:
        """Persist the graph to disk."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        data = json_graph.node_link_data(self.graph)
        with open(self.save_path, "w") as f:
            json.dump(data, f)
        logger.info(f"Knowledge Graph saved to {self.save_path}")

    def _load(self) -> None:
        """Load the graph from disk."""
        with open(self.save_path, "r") as f:
            data = json.load(f)
        self.graph = json_graph.node_link_graph(data)
        console.print(f"  [green]✓[/green] Loaded Knowledge Graph from disk: [bold]{self.graph.number_of_nodes()}[/bold] nodes, [bold]{self.graph.number_of_edges()}[/bold] edges\n")

    def export_visual(self, filename: str = "graph_visualization.html") -> None:
        """Export the knowledge graph to an interactive HTML visualization using PyVis."""
        try:
            from pyvis.network import Network
            
            # Create a PyVis network
            net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)
            
            # PyVis takes NetworkX graph directly, but doing it manually allows finer styling
            for node, attrs in self.graph.nodes(data=True):
                title = "<br>".join([f"{k}: {v}" for k, v in attrs.items()]) if attrs else node
                net.add_node(node, label=node, title=title, color="#00ff1e")
                
            for source, target, data in self.graph.edges(data=True):
                relation = data.get("relation", "")
                title = data.get("context", relation)
                net.add_edge(source, target, title=title, label=relation, color="#97c2fc")
                
            # Use Barnes-Hut layout
            net.barnes_hut()
            
            # Save the file
            net.save_graph(filename)
            logger.info(f"Visual graph exported to {filename}")
        except ImportError:
            logger.error("PyVis is not installed. Run: pip install pyvis")
