"""Work-only Search ordering prototype: preserve native prefix, then fill holes.

No model client/configuration, Gold, source changes, caching or network calls.
All sizing uses the current production Answerer and its public conservative
estimators. The Answer prompt, records, images and scores remain unchanged.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

from m2a_cpu import answering as a

VERSION = 'search-preserve-prefix-fill-whole-items-v1'
ANSWERING_SHA256 = hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest()
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class _NoModelClient:
    model = a.ANSWER_MODEL

    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return {'message': {'content': 'SYNTHETIC_PACKING_CHECK_ONLY'}, 'usage': {}}


def _pack(query, options, evidence, config):
    client = _NoModelClient()
    answerer = a.Answerer(client, config)
    result = answerer(query, options, evidence)
    if client.calls != 1:
        raise RuntimeError('Native fake packing call count changed')
    return result['packing']


def item_cost(item, rank):
    """Exact native header ordering and generated rank at the proposed position."""
    if not isinstance(item, dict) or set(item)-{'id','content','score','created_at'} or 'content' not in item:
        raise a.AnswerError('Evidence has unsupported fields')
    ident=item.get('id','rank:'+str(rank))
    if not isinstance(ident,str):
        raise a.AnswerError('Evidence ID must be a string')
    header={'rank':rank,'id':ident}
    if item.get('created_at') is not None:
        header['created_at']=item['created_at']
    wrapped=[a._text('[Evidence item] '+json.dumps(header,ensure_ascii=False,allow_nan=False)),
             *a._parts(item['content']),a._text('[End evidence item]')]
    return a.content_token_bound(wrapped)


def identity(answer_config):
    answerer=a.Answerer(_NoModelClient(),answer_config)
    return {'protocol':VERSION,'implementation_sha256':IMPLEMENTATION_SHA256,'answering_sha256':ANSWERING_SHA256,
            'answer_protocol':a.ANSWER_PROTOCOL,'answer_controls':{'model':a.ANSWER_MODEL,'dataset':answerer.dataset,
            'context_window':answerer.context_window,'max_output_tokens':answerer.max_output_tokens,
            'reserved_output_tokens':answerer.reserved_output_tokens,'safety_tokens':answerer.safety_tokens,
            'effective_input_limit':answerer.input_limit}}


def reorder_evidence(query, options, evidence, answer_config):
    """Return a reordered copy and content-free diagnostics; every item appears once.

    The old retained prefix has exactly the same positions/bytes. Scan the
    remaining items in original order, append each whole item that fits at its
    NEW rank, then append all deferred items in their original relative order.
    A real native Answerer with an in-memory fake response validates both sides.
    """
    if (hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest()!=ANSWERING_SHA256 or
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=IMPLEMENTATION_SHA256):
        raise RuntimeError('Frozen packing sources changed after prototype import')
    before=_pack(query,options,evidence,answer_config)
    prefix=len(before['retained_evidence'])
    selected=list(range(prefix));deferred=[]
    used=before['input_token_bound'];limit=before['input_limit']
    for original_rank in range(prefix,len(evidence)):
        cost=item_cost(evidence[original_rank],len(selected))
        if used+cost<=limit:
            selected.append(original_rank);used+=cost
        else:
            deferred.append(original_rank)
    order=selected+deferred
    if len(order)!=len(evidence) or sorted(order)!=list(range(len(evidence))) or order[:prefix]!=list(range(prefix)):
        raise RuntimeError('Permutation/prefix invariant failed')
    result=[copy.deepcopy(evidence[i]) for i in order]
    after=_pack(query,options,result,answer_config)
    if (after['retained_evidence'][:prefix]!=before['retained_evidence'] or
            len(after['retained_evidence'])!=len(selected) or after['input_token_bound']!=used or used>limit):
        raise RuntimeError('Proposed order disagrees with the production Answerer')
    return result, {'version':VERSION,'answering_sha256':ANSWERING_SHA256,'model_calls':0,
                    'original_prefix_count':prefix,'retained_count':len(selected),'added_items':len(selected)-prefix,
                    'evidence_count':len(evidence),'before_token_bound':before['input_token_bound'],
                    'after_token_bound':used,'input_limit':limit,'order_original_indices':order,
                    'selected_original_indices':selected,'deferred_original_indices':deferred,
                    'content_changed':False,'score_changed':False,'source_prefix_preserved':True,
                    'fake_native_packing_calls':2}
