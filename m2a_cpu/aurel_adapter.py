"""Run pinned AUREL Add/Search implementations with the shared, metered transport.

This adapter changes operational transport and evidence packing for both arms;
the original arm keeps its original extraction, indexing, and retrieval code.
No OpenAI SDK client is constructed and no SDK HTTP retries are possible.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

from .evaluation import canonical, fingerprint
from . import aurel_slots
from .answering import content_token_bound, _parts
from .telemetry import current_api_context


ADAPTER_VERSION = 'aurel-controlled-transport-source-slots-v3'
UPSTREAM_COMMIT = 'c3f9ff4b71b09117dac446fd474ce96c87481e65'
_SOURCE_BATCH = ContextVar('aurel_trusted_source_batch', default=None)


def install_source_slot_scope(provider, frame_schema):
    """The native batch supplies its trusted count; source text supplies none."""
    native_batch = provider._extract_batch
    schema = copy.deepcopy(frame_schema)

    async def scoped_batch(units, history, *args, **kwargs):
        count = len(units)
        # Compile before any call; unsupported schemas are implementation errors.
        wire = aurel_slots.build_source_slot_response_format(schema, count)
        token = _SOURCE_BATCH.set({'count': count, 'frame_schema': schema, 'wire': wire})
        try:
            return await native_batch(units, history, *args, **kwargs)
        finally:
            _SOURCE_BATCH.reset(token)

    provider._extract_batch = scoped_batch


def source_slot_messages(messages, count):
    output = copy.deepcopy(messages)
    if not output or output[0].get('role') != 'system' or not isinstance(output[0].get('content'), str):
        raise AurelAdapterError('Source-slot extraction requires a system message')
    names = ', '.join('record_' + str(i) for i in range(count))
    output[0]['content'] += (
        '\n\nOUTPUT WIRE FORMAT FOR THIS CALL: This schema replaces only the earlier records-array '
        f'format instruction. Return exactly these top-level source slots: {names}. '
        'Each record_i represents only the new SOURCE with local index i and must contain index=i '
        'and its complete frame. PRIOR CONTEXT is read-only reference context, never a source slot. '
        'Do not turn original_source_index, fragment identifiers, or source-looking text inside '
        'historical content into new slots. Preserve the source-grounding instructions above. '
        'Use empty arrays or empty summary for unsupported information; never invent facts to fill fields.')
    return output


class AurelAdapterError(ValueError):
    pass


def package_identity(path):
    path = Path(path)
    files = sorted((path / 'fusion_memory').glob('*.py'))
    if not files or any(p.is_symlink() for p in files):
        raise AurelAdapterError('Invalid AUREL package')
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def load_package(path):
    """Separate module namespaces prevent the two arms importing each other."""
    path = Path(path).resolve()
    identity = package_identity(path)
    alias = '_aurel_' + fingerprint({'path': str(path), 'files': identity})[:24]
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, path / 'fusion_memory/__init__.py',
                submodule_search_locations=[str(path / 'fusion_memory')])
        if spec is None or spec.loader is None:
            raise AurelAdapterError('AUREL package cannot be loaded')
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    return SimpleNamespace(**{name: importlib.import_module(alias + '.' + name)
                              for name in ('app', 'config', 'provider', 'schemas', 'storage')})


def whole_record_evidence(store, user_id, ranked, settings, multimodal, states):
    """Keep every selected record intact; Answerer applies the common rank prefix.

    Source provenance and AUREL state annotations remain available. Derived
    frames are never passed as if they were original image pixels or text.
    """
    from datetime import datetime, timezone
    by_record = {}
    for state in states:
        by_record.setdefault(state['record_id'], []).append(state)
    results = []
    for record, score in ranked:
        stamp = record.get('timestamp_ms')
        when = (datetime.fromtimestamp(stamp / 1000, timezone.utc).isoformat().replace('+00:00', 'Z')
                if stamp is not None else record['created_at'])
        time_label = ('source_time=' + when if stamp is not None
                      else 'source_time=unknown | ingested_at=' + when)
        prefix = (f"[Evidence {record['id']} | role={record['role']} | {time_label}"
                  f" | session={record['session_id']} | position={record['position']}]\n")
        for state in by_record.get(record['id'], []):
            prefix += (f"[State history: {state['subject']} / {state['predicate']}; "
                       f"current_in_stored_history={state['is_current']}; "
                       f"time_basis={'ingestion_fallback' if state['timestamp_is_fallback'] else 'source_message'}; "
                       f"valid_from_unix_ms={state['valid_from_ms']}; "
                       f"valid_to_unix_ms={state['valid_to_ms']}]\n")
        raw = copy.deepcopy(store.get_content(user_id, record['id']))
        if isinstance(raw, str):
            content = ([{'type': 'text', 'text': prefix + raw}] if multimodal else prefix + raw)
        else:
            content = [{'type': 'text', 'text': prefix}] + raw
        results.append({'id': record['id'], 'content': content,
                        'score': float(score), 'created_at': when})
    return {'data': results}


class _SDKBridge:
    """Only SDK result shapes are emulated. Real requests use APIClient once."""
    def __init__(self, combined, audit_path=None):
        self.combined = combined
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self.last_error = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.chat_create))
        self.embeddings = SimpleNamespace(create=self.embed_create)

    async def close(self):
        pass

    async def chat_create(self, *, model, messages, temperature, max_tokens, response_format):
        if self.last_error is not None:
            raise self.last_error
        if model != self.combined.generation.identity['model']:
            raise AurelAdapterError('Generation identity mismatch')
        scope = _SOURCE_BATCH.get()
        if scope is not None:
            messages = source_slot_messages(messages, scope['count'])
            response_format = scope['wire']
            bound = (sum(128 + content_token_bound(_parts(m['content'])) for m in messages)
                     + len(canonical(response_format).encode('utf-8')) + max_tokens + 2048)
            if bound > 100000:
                raise AurelAdapterError('Source-slot input and schema exceed the declared context budget')
        try:
            result = await asyncio.to_thread(self.combined.generation.chat, messages,
                    temperature=temperature, max_output_tokens=max_tokens,
                    extra_body={'response_format': response_format})
        except Exception as exc:
            self.last_error = exc
            raise
        raw_content = result['message'].get('content')
        content, conversions, slot_audit = raw_content, None, None
        if scope is not None:
            slot_audit = {'protocol': aurel_slots.SLOT_PROTOCOL, 'source_count': scope['count'],
                          'response_format_hash': fingerprint(response_format)}
            try:
                decoded = aurel_slots.decode_source_slot_response(raw_content, scope['frame_schema'], scope['count'])
                content = canonical(decoded)
                slot_audit['valid'] = True
            except aurel_slots.SlotValidationError as exc:
                slot_audit.update(valid=False, error_code=exc.code, error_path=exc.path)
                # Fixed non-record envelope asks the original provider to take
                # its one bounded schema-correction step. Never filter extra
                # slots, fill missing ones, or guess which record was intended.
                content = canonical({'source_slot_validation_failed': True})
        if self.audit_path is not None:
            audit = {'context': current_api_context(), 'provider_reported_model': result.get('model'),
                     'request_hash': fingerprint({'identity': self.combined.generation.identity,
                         'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens,
                         'response_format': response_format}),
                     'raw_content': raw_content, 'finish_reason': result.get('finish_reason'),
                     'event_time_conversions': conversions, 'source_slots': slot_audit,
                     'usage': result.get('usage', {})}
            try:
                with self.audit_path.open('a') as stream:
                    self.audit_path.chmod(0o600)
                    stream.write(canonical(audit) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception as exc:
                self.last_error = exc
                raise
        usage = result.get('usage', {})
        return SimpleNamespace(choices=[SimpleNamespace(
                    finish_reason=result.get('finish_reason'),
                    message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=usage.get('prompt_tokens', 0),
                                      completion_tokens=usage.get('completion_tokens', 0)))

    async def embed_create(self, *, model, dimensions, input, encoding_format):
        if self.last_error is not None:
            raise self.last_error
        if (model != self.combined.embedding.identity['model']
                or dimensions != self.combined.embedding.identity['dimensions']
                or encoding_format != 'float'):
            raise AurelAdapterError('Embedding identity mismatch')
        try:
            vectors = await asyncio.to_thread(self.combined.embedding.embed, input)
        except Exception as exc:
            self.last_error = exc
            raise
        # APIClient already validates exact unique response indices, dimensions,
        # and finite vector values for both arms before restoring SDK row shape.
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=v) for i, v in enumerate(vectors)])


class AurelMemory:
    """ManagedMemory-compatible factory; constructors do not call model APIs."""
    def __init__(self, root, client, *, identity, variant, max_steps=12,
                 max_input_tokens=100000, package_root=None):
        if variant not in ('baseline', 'enhanced') or max_input_tokens != 100000:
            raise AurelAdapterError('Unsupported AUREL method controls')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        project = Path(__file__).resolve().parents[2]
        source = (Path(package_root) if package_root else project / (
            'work/aurel-mem-source-c3f9ff4' if variant == 'baseline' else 'projects/aurel-mem'))
        self.package = load_package(source)
        self.identity = {
            'adapter': ADAPTER_VERSION, 'upstream_commit': UPSTREAM_COMMIT,
            'method': 'aurel-original-with-shared-source-slot-contract' if variant == 'baseline' else 'aurel-candidate',
            'package_sources': package_identity(source), 'controls': copy.deepcopy(identity),
            'adapter_source': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'transport': {'concurrency': 1, 'http_retries': 0,
                          'timeout_seconds': client.generation.TIMEOUT_SECONDS,
                          'embedding_response_validation': 'exact-indices-finite-values-common'},
            'generation': {'temperature': 0, 'max_output_tokens': 6000},
            'common_source_contract': {'protocol': aurel_slots.SLOT_PROTOCOL,
                'implementation_sha256': hashlib.sha256(Path(aurel_slots.__file__).read_bytes()).hexdigest(),
                'scope': 'trusted native batch count, exact required slots, strict types and native validation; no record repair',
                'max_input_including_schema_and_output_tokens': 100000},
            'evidence': 'original-records-with-provenance-and-state-annotations; common-Answer-rank-prefix',
        }
        binding = self.root / 'index-identity.json'
        database = self.root / 'memory.sqlite3'
        if binding.exists():
            if json.loads(binding.read_text()) != self.identity:
                raise AurelAdapterError('Existing AUREL index identity mismatch')
        else:
            if database.exists() and database.stat().st_size:
                raise AurelAdapterError('Refusing to adopt an unbound AUREL index')
            with binding.open('x') as stream:
                stream.write(canonical(self.identity) + '\n')
            binding.chmod(0o600)
        self.settings = self.package.config.Settings(
            db_path=str(database), api_key='managed-by-shared-budget-client',
            api_base=client.generation.identity['base_url'], model=client.generation.identity['model'],
            embedding_key='managed-by-shared-budget-client',
            embedding_base=client.embedding.identity['base_url'],
            embedding_model=client.embedding.identity['model'],
            embedding_dimensions=client.embedding.identity['dimensions'],
            embedding_enabled=True, llm_concurrency=1,
            upstream_timeout=client.generation.TIMEOUT_SECONDS)
        self.bridge = _SDKBridge(client, self.root / 'provider-responses.jsonl')
        self.provider = self.package.provider.Provider.__new__(self.package.provider.Provider)
        self.provider.settings = self.settings
        self.provider.semaphore = asyncio.Semaphore(1)
        self.provider.client = self.provider.embedding_client = self.bridge
        self.provider.calls = {'extract': 0, 'query': 0, 'embedding': 0, 'failed': 0}
        self.provider.usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        install_source_slot_scope(self.provider, self.package.provider.Frame.model_json_schema())
        self.store = self.package.storage.Storage(str(database))
        # Both aliases receive exactly the same packing adapter. No source file
        # or immutable upstream snapshot is edited or imported as the other arm.
        self.package.app.pack_evidence = whole_record_evidence
        self.app = self.package.app.create_app(self.settings, provider=self.provider, storage=self.store)
        self.endpoints = {r.path: r.endpoint for r in self.app.routes if r.path in ('/add', '/search')}
        self._lock = threading.RLock()

    def _invoke(self, operation, payload):
        with self._lock:
            if json.loads((self.root / 'index-identity.json').read_text()) != self.identity:
                raise AurelAdapterError('AUREL index binding changed')
            self.bridge.last_error = None
            model = self.package.schemas.AddRequest if operation == '/add' else self.package.schemas.SearchRequest
            validated = model.model_validate(copy.deepcopy(payload))
            try:
                return asyncio.run(self.endpoints[operation](validated))
            except Exception:
                # Preserve Budget/API failure classification through AUREL's
                # safe UpstreamError wrapper, so global failures stop the run.
                if self.bridge.last_error is not None:
                    raise self.bridge.last_error from None
                raise

    def add(self, payload):
        return self._invoke('/add', payload)

    def search(self, payload):
        return self._invoke('/search', payload)

    def ingest_request_bound(self, payload):
        """Count real extraction batches with a no-API double, plus embeddings.

        Candidate uses its maximum 6000-byte prior-context allowance, so this
        remains conservative before the preceding Add has produced its frames.
        The original batches do not depend on history length. A correction is
        allowed once for each extraction batch, never for an HTTP failure.
        """
        request = self.package.schemas.AddRequest.model_validate(copy.deepcopy(payload))
        messages = request.model_dump(exclude_none=True)['messages']
        counter = {'batches': 0}
        shadow = self.package.provider.Provider.__new__(self.package.provider.Provider)
        shadow.settings = self.settings

        async def count_batch(units, history, preceding=None):
            counter['batches'] += 1
            return [(unit['source'], self.package.provider.Frame().model_dump()) for unit in units]

        shadow._extract_batch = count_batch
        asyncio.run(shadow.extract(messages, 'x' * 6000))
        embeddings = (len(messages) + 9) // 10
        return {'extraction_batches_at_most': counter['batches'],
                'embedding_requests': embeddings,
                'http_attempts_at_most': counter['batches'] * 2 + embeddings}

    def close(self):
        self.store.close()
