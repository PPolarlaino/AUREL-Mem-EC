"""Local request attribution. These fields never enter provider payloads.

Contexts must be established inside each worker. ContextVar state isolates
threads/async contexts but is intentionally not implicitly copied to new threads.
Only opaque/control-plane identifiers belong here, never prompts or credentials.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import json
import math
import re


_CONTEXT = ContextVar('m2a_api_context', default=None)
FIELDS = frozenset({'run_id', 'stage', 'scope_hash', 'qid'})


class TelemetryError(ValueError):
    pass


def validate_context(value):
    if not isinstance(value, dict) or set(value) - FIELDS:
        raise TelemetryError('Invalid API attribution fields')
    result = {}
    for key, item in value.items():
        if item is None:
            continue
        if not isinstance(item, str) or not item or item.lower().startswith(('sk-', 'bearer')):
            raise TelemetryError('Attribution must contain public identifiers only')
        if key == 'scope_hash':
            valid = re.fullmatch(r'[0-9a-f]{64}', item)
        elif key == 'run_id':
            valid = re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', item)
        elif key == 'stage':
            valid = re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', item)
        else:
            valid = re.fullmatch(r'[\w:./@+-]{1,512}', item)
        if not valid:
            raise TelemetryError('Invalid API attribution identifier')
        result[key] = item
    return result


def current_api_context():
    return copy.deepcopy(_CONTEXT.get() or {})


@contextmanager
def api_context(**fields):
    """Inherit outer identifiers; an explicit None removes that identifier."""
    if set(fields) - FIELDS:
        raise TelemetryError('Invalid API attribution fields')
    merged = current_api_context()
    for key, value in fields.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    token = _CONTEXT.set(validate_context(merged))
    try:
        yield current_api_context()
    finally:
        _CONTEXT.reset(token)


def validate_elapsed(value):
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                             and math.isfinite(value) and value >= 0)


def aggregate_attempt_rows(rows):
    """Aggregate ledger rows only. Missing usage/timing never becomes zero cost."""
    rows = list(rows)
    usage, outcomes, statuses = {}, Counter(), Counter()
    with_usage = 0
    for row in rows:
        parsed = json.loads(row['usage_json'])
        with_usage += bool(parsed)
        for key, value in parsed.items():
            usage[key] = usage.get(key, 0) + value
        outcomes[row['outcome']] += 1
        statuses[str(row['http_status']) if row['http_status'] is not None else 'unknown'] += 1
    timed = [r for r in rows if r.get('elapsed_seconds') is not None]
    http_timed = [r for r in rows if r.get('http_elapsed_seconds') is not None]
    finished = [r for r in rows if r.get('finished') is not None]
    # End-to-end pipeline work before the first reservation or after final finish
    # is outside this span; overlapping attempts must not be summed as wall time.
    span = max(r['finished'] for r in finished) - min(r['started'] for r in rows) if rows and finished else None
    return {
        'requests_used': len(rows),
        'active_requests': sum(r['active'] for r in rows),
        'reserved_output_tokens': sum(r['reserved_output_tokens'] for r in rows),
        'reported_usage': usage,
        'attempts_with_reported_usage': with_usage,
        'attempts_without_reported_usage': len(rows) - with_usage,
        'legacy_attempts': sum(r.get('context_attempt_id') is None for r in rows),
        'unattributed_requests': sum(r.get('run_id') is None for r in rows),
        'logical_calls': len({r['call_id'] for r in rows if r.get('call_id')}),
        'retry_attempts': sum(r.get('retry_index') is not None and r['retry_index'] > 0 for r in rows),
        'attempts_without_retry_attribution': sum(r.get('retry_index') is None for r in rows),
        'outcomes': dict(outcomes), 'http_status_counts': dict(statuses),
        'elapsed': {
            'attempt_seconds_sum': sum(r['elapsed_seconds'] for r in timed),
            'http_seconds_sum': sum(r['http_elapsed_seconds'] for r in http_timed),
            'attempts_with_elapsed': len(timed), 'attempts_without_elapsed': len(rows) - len(timed),
            'http_attempts_with_elapsed': len(http_timed), 'http_attempts_without_elapsed': len(rows) - len(http_timed),
            'observed_span_seconds': max(0.0, span) if span is not None else None,
            'span_complete': bool(rows) and len(finished) == len(rows) and not any(r['active'] for r in rows),
        },
    }
