"""Local, evidence-preserving hybrid retrieval for multimodal memories.

This module uses no network calls or language model. Independent BM25 views
(original text, structured semantics, and visual/OCR evidence) and compatible
embedding similarities contribute reciprocal ranks. Small entity and session
neighborhoods can then supply evidence that a direct search would miss.

The neighborhood and coverage heuristics are inspired by graph-based memory
retrieval; they are deliberately lightweight, not reproductions of any paper.
Returned scores are relative retrieval scores, not calibrated probabilities.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any


_WORD_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+", re.I)
_CJK_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")
_STOP_WORDS = frozenset(
    "a an the is are was were be been being i me my you your we our it its this that "
    "these those of to for from in on at and or with about what which who when where "
    "how please tell show find search memory memories information latest now current "
    "previous before after compare comparison earlier formerly currently today".split()
)
_CJK_STOP = frozenset("的了吗呢啊呀是在有和与及我你他她它这那请问什么个把被从到为于着地得将了")
_QUERY_FILLER_RE = re.compile(
    "请问|告诉我|帮我|查一下|查询|查找|搜索|是什么|有什么|哪些|什么|我的|我们的|"
    "现在|当前|目前|最近|最新|以前|之前|之后|以后|曾经|过去|比较|对比|相关|有关|信息|记忆"
)
# Pin this version with extraction/embedding identity before reusing an index.
RETRIEVAL_STRATEGY_VERSION = "aurel-candidate-retrieval-v1"


def tokenize(text: str) -> list[str]:
    """Retain OCR model numbers and both CJK characters and local bigrams."""
    cleaned = _QUERY_FILLER_RE.sub(" ", str(text).lower())
    result: list[str] = []
    for match in _WORD_RE.finditer(cleaned):
        token = match.group(0)
        if _CJK_RE.fullmatch(token):
            result.extend(char for char in token if char not in _CJK_STOP)
            result.extend(
                token[i : i + 2]
                for i in range(len(token) - 1)
                if not any(char in _CJK_STOP for char in token[i : i + 2])
            )
        elif token not in _STOP_WORDS:
            result.append(token)
            if re.search(r"[-_./]", token):
                result.extend(piece for piece in re.split(r"[-_./]", token) if len(piece) > 1 and piece not in _STOP_WORDS)
    return result


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(str(key) + " " + _flatten(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(item) for item in value)
    return str(value)


def _views(record: dict[str, Any]) -> tuple[str, str, str]:
    structured = record.get("structured")
    structured = structured if isinstance(structured, dict) else {}
    semantic = " ".join(
        _flatten(structured.get(key))
        for key in ("summary", "facts", "entities", "relations", "event_time", "state_updates")
    )
    visual = " ".join(
        _flatten(structured.get(key))
        for key in ("objects", "ocr", "caption", "image_caption", "visual_description", "visual_facts")
    )
    return str(record.get("raw_text") or ""), semantic, visual


def _term_weight(term: str) -> float:
    return 0.30 if len(term) == 1 and _CJK_RE.fullmatch(term) else 1.0


def _bm25(documents: list[list[str]], query: list[str]) -> dict[int, float]:
    if not query or not documents:
        return {}
    term_counts = [Counter(document) for document in documents]
    frequencies: Counter[str] = Counter()
    for counts in term_counts:
        frequencies.update(counts.keys())
    average_length = sum(map(len, documents)) / max(1, len(documents)) or 1.0
    scores: dict[int, float] = {}
    query_terms = set(query)
    query_cjk_characters = {term for term in query_terms if len(term) == 1 and _CJK_RE.fullmatch(term)}
    for index, counts in enumerate(term_counts):
        matching = query_terms & counts.keys()
        # One incidental Chinese character (e.g. 机 in 咖啡机 versus 量子计算机)
        # is too weak to seed a neighborhood for a longer Chinese question.
        if len(query_cjk_characters) > 1 and len(matching) == 1 and next(iter(matching)) in query_cjk_characters:
            continue
        score = 0.0
        for term in query_terms:
            count = counts.get(term, 0)
            if not count:
                continue
            idf = math.log(1 + (len(documents) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            denominator = count + 1.2 * (0.25 + 0.75 * len(documents[index]) / average_length)
            score += _term_weight(term) * idf * count * 2.2 / denominator
        if score > 0:
            scores[index] = score
    return scores


def _vector(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    norm = math.sqrt(sum(item * item for item in values))
    if not math.isfinite(norm) or norm <= 0:
        return None
    return tuple(item / norm for item in values)


def _canonical_entity(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", "", value.lower())


def _entity_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if 1 < len(value.strip()) <= 100 else []
    if isinstance(value, (list, tuple)):
        return [name for item in value for name in _entity_values(item)]
    if isinstance(value, dict):
        names = [value[key] for key in ("name", "text", "value", "label", "entity", "canonical_name") if key in value]
        if names:
            return [name for item in names for name in _entity_values(item)]
        return [name for key, item in value.items() if key not in {"type", "confidence", "id"} for name in _entity_values(item)]
    return []


def _entities(record: dict[str, Any]) -> set[str]:
    structured = record.get("structured")
    if not isinstance(structured, dict):
        return set()
    values = _entity_values(structured.get("entities", []))
    relations = structured.get("relations", [])
    relations = [relations] if isinstance(relations, dict) else relations
    if isinstance(relations, list):
        for relation in relations:
            if isinstance(relation, dict):
                for key in ("subject", "object", "source", "target", "head", "tail"):
                    values.extend(_entity_values(relation.get(key)))
    return {normalized for value in values if (normalized := _canonical_entity(value))}


def _message_time(record: dict[str, Any]) -> float | None:
    """Source-message seconds for local adjacency; event/ingestion time is not a substitute."""
    value = record.get("timestamp_ms")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            numeric = float(value)
        except (ValueError, OverflowError):
            return None
        return numeric / 1000 if math.isfinite(numeric) else None
    return None


def _identity(record: dict[str, Any], index: int) -> tuple[str, int]:
    return str(record.get("id", "")), index


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def rank_memories(
    records: list[dict],
    query_text: str,
    query_vector: list[float] | None,
    top_k: int,
    *,
    query_embedding_model: str | None = None,
    neighbor_hops: int = 1,
    graph_expansion: bool = True,
    diversity: bool = True,
) -> list[tuple[dict, float]]:
    """Return up to ``top_k`` evidence records and decreasing relative scores.

    Input records are never mutated. Empty/no-match queries return no records.
    Dense ranks require equal dimensions and, when supplied, an exact embedding
    model identifier match. Positive cosine scores below 0.25 are not seeds.
    Graph hops are capped at three and never cross a user boundary. Callers
    should still filter records to the authenticated user before invoking this
    function: retrieval is not an authorization layer.
    """
    if not records or top_k <= 0:
        return []
    top_k = int(top_k)
    if top_k <= 0:
        return []
    query_text = str(query_text or "")
    query_tokens = tokenize(query_text)
    vector = _vector(query_vector)
    if not query_tokens and vector is None:
        return []
    view_texts = [_views(record) for record in records]
    views = [[tokenize(texts[view]) for texts in view_texts] for view in range(3)]
    channels = [(_bm25(documents, query_tokens), weight) for documents, weight in zip(views, (1.0, 1.10, 1.15))]
    dense: dict[int, float] = {}
    if vector is not None:
        for index, record in enumerate(records):
            if query_embedding_model is not None and record.get("embedding_model") != query_embedding_model:
                continue
            candidate = _vector(record.get("embedding"))
            if candidate is None or len(candidate) != len(vector):
                continue
            similarity = sum(a * b for a, b in zip(vector, candidate))
            if similarity >= 0.25:
                dense[index] = min(1.0, similarity)
    channels.append((dense, 1.20))
    fused: dict[int, float] = defaultdict(float)
    # Capping each ranking also bounds graph expansion on large local stores.
    channel_limit = min(256, max(32, top_k * 8))
    for scores, weight in channels:
        ranked = sorted(scores, key=lambda index: (-scores[index], _identity(records[index], index)))[:channel_limit]
        for rank, index in enumerate(ranked, 1):
            fused[index] += weight / (60 + rank)
    if not fused:
        return []

    entities = [_entities(record) for record in records]
    message_times = [_message_time(record) for record in records]
    seed_limit = min(64, max(12, top_k * 3))
    seeds = sorted(fused, key=lambda index: (-fused[index], _identity(records[index], index)))[:seed_limit]
    hops = max(0, min(3, int(neighbor_hops)))
    if graph_expansion and hops:
        entity_index: dict[tuple[str, str], list[int]] = defaultdict(list)
        sessions: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            owner = str(record.get("user_id") or "")
            for entity in entities[index]:
                entity_index[(owner, entity)].append(index)
            if record.get("session_id") not in (None, ""):
                sessions[(owner, str(record["session_id"]))].append(index)
        adjacency: dict[int, list[int]] = defaultdict(list)
        for members in sessions.values():
            def session_key(index: int) -> tuple:
                record = records[index]
                try:
                    position = int(record.get("position", 0))
                except (ValueError, TypeError):
                    position = 0
                # Position records the conversation order. Event time can refer
                # to an old photograph and request IDs need not be sortable.
                return position, index

            ordered = sorted(members, key=session_key)
            for position, index in enumerate(ordered):
                adjacency[index] = ordered[max(0, position - 1) : position] + ordered[position + 1 : position + 2]
        frontier = seeds
        visited = set(seeds)
        maximum_candidates = min(512, max(64, top_k * 12))
        for _ in range(hops):
            upcoming: set[int] = set()
            for source in frontier:
                owner = str(records[source].get("user_id") or "")
                neighbors: dict[int, float] = {}
                for entity in sorted(entities[source]):
                    matches = entity_index[(owner, entity)]
                    # A ubiquitous tag is not a useful graph edge.
                    if len(matches) > 48:
                        continue
                    ordered = sorted(matches, key=lambda index: (-fused.get(index, 0), _identity(records[index], index)))[:16]
                    for target in ordered:
                        neighbors[target] = 0.72
                for target in adjacency[source]:
                    source_record, target_record = records[source], records[target]
                    same_request = bool(source_record.get("request_id")) and source_record.get("request_id") == target_record.get("request_id")
                    time_distance = abs(message_times[source] - message_times[target]) if message_times[source] is not None and message_times[target] is not None else 0
                    # Compare source-message times, never a photograph's event
                    # date. Missing times retain position adjacency.
                    if not same_request and time_distance > 1800:
                        continue
                    cross_modal = bool(source_record.get("has_image")) != bool(target_record.get("has_image"))
                    weight = 0.68 if same_request else (0.60 if cross_modal else 0.48)
                    neighbors[target] = max(neighbors.get(target, 0), weight)
                for target, weight in neighbors.items():
                    if target == source:
                        continue
                    propagated = fused[source] * weight
                    if target not in fused and len(fused) >= maximum_candidates:
                        continue
                    fused[target] = max(fused.get(target, 0), propagated)
                    if target not in visited:
                        upcoming.add(target)
            visited.update(upcoming)
            frontier = sorted(upcoming, key=lambda index: (-fused[index], _identity(records[index], index)))[:seed_limit]
            if not frontier:
                break

    # A candidate-wide latest date is not a per-entity/per-attribute version
    # chain. Preserve dates as evidence for Answer; do not infer temporal rank.
    base_scores = dict(fused)
    all_tokens = [set(views[0][index] + views[1][index] + views[2][index]) for index in range(len(records))]
    # Include structured facts: two updates with equal raw text but different
    # extracted facts are distinct evidence, even if their image is identical.
    signatures = ["\n".join(re.sub(r"\s+", " ", text.lower()).strip() for text in texts) for texts in view_texts]
    canonical_query = _canonical_entity(query_text)
    requested_entities = {entity for group in entities for entity in group if entity in canonical_query}
    selected: list[tuple[int, float]] = []
    covered: set[str] = set()
    remaining = set(fused)
    while remaining and len(selected) < top_k:
        adjusted: dict[int, float] = {}
        for index in remaining:
            # Repeated appearances of the same evidence can themselves matter
            # (different images, or the same image at different times). Only a
            # repeated *record ID* is hard-deduplicated; content is penalized.
            repeated_id = records[index].get("id") is not None and any(
                records[index].get("id") == records[chosen].get("id") for chosen, _ in selected
            )
            if repeated_id:
                continue
            score = base_scores[index]
            if diversity and selected:
                redundancy = max(_jaccard(all_tokens[index], all_tokens[chosen]) for chosen, _ in selected)
                if any(signatures[index] == signatures[chosen] and signatures[index].strip() for chosen, _ in selected):
                    redundancy = 1.0
                new_requested = (entities[index] & requested_entities) - covered
                coverage = len(new_requested) / max(1, len(requested_entities))
                new_entities = entities[index] - covered
                score *= 1.0 + 0.32 * coverage + min(0.10, 0.025 * len(new_entities))
                # Soft suppression keeps contradictory and historical evidence.
                score *= 1.0 - 0.55 * redundancy
            adjusted[index] = score
        if not adjusted:
            break
        chosen = min(adjusted, key=lambda index: (-adjusted[index], _identity(records[index], index)))
        selected.append((chosen, adjusted[chosen]))
        covered.update(entities[chosen])
        remaining.remove(chosen)
    # Greedy coverage scores can increase as a chain is explored; this final
    # ordering makes the public score contract explicit and deterministic.
    selected.sort(key=lambda pair: (-pair[1], _identity(records[pair[0]], pair[0])))
    return [(records[index], float(score)) for index, score in selected]
