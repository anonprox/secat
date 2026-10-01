"""Generic accuracy-first semantic diagnostics for OCA.

These checks use only the user question, the proposed plan, and runtime evidence.
They never inspect task IDs, benchmark gold answers, reference routes, or evaluator
metadata.  They are deliberately conservative: their purpose is to trigger a
bounded second solving attempt when the current route is clearly risky, not to
manufacture an answer.
"""
from __future__ import annotations

import json
import re
from typing import Any


def _cf(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _derivations(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [x for x in (plan or {}).get("derivations") or [] if isinstance(x, dict)]


def _steps(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [x for x in (plan or {}).get("steps") or [] if isinstance(x, dict)]


def _step_map(plan):
    return {str(x.get("id") or ""): x for x in _steps(plan)}




def _label_tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", _cf(value))


def _canonical_search_tokens(value: Any, resource: str) -> tuple[str, ...]:
    """Normalize only low-risk search label decorations.

    Some APIs canonicalize collection/company names by adding the resource noun
    and/or a leading article.  Treat that decoration as equivalent only for
    resource families where the noun is an entity-type label rather than a
    likely title word.  The caller still requires one unique observed match.
    """
    tokens = _label_tokens(value)
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    resource = _cf(resource).strip("/")
    if resource in {"collection", "company"} and tokens:
        endings = {resource, resource + "s"}
        if tokens[-1] in endings:
            tokens = tokens[:-1]
    return tuple(tokens)


def repair_observed_canonical_search_labels(plan: dict[str, Any] | None, ledger) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Repair search selection from already-observed uniquely matching labels.

    Two generic cases are handled without any benchmark knowledge:

    1. An exact equality filter uses the user's literal label, while the API
       returns one uniquely equivalent canonical label (e.g. a collection or
       company type decoration).  The filter is rewritten to that observed
       canonical label.
    2. A planner blindly selects endpoint rank 0, while the already-returned
       search candidates contain exactly one exact/canonical match for the query.
       The selector is moved to that candidate's observed endpoint position.

    The repair is deliberately conservative.  Exact text equality is allowed for
    every search resource.  Type-decoration canonicalization is limited to the
    collection/company resources implemented by ``_canonical_search_tokens`` and
    must identify exactly one returned candidate.  Ambiguity means no repair.
    """
    import copy
    out = copy.deepcopy(plan or {})
    steps = {str(x.get("id") or ""): x for x in _steps(out)}
    specs = {str(x.get("step_id") or ""): x for x in out.get("observation_specs") or []}
    repairs: list[dict[str, Any]] = []

    def rewrite_filter(blob: Any, old: str, new: str) -> bool:
        changed = False
        if not isinstance(blob, dict):
            return False
        for key, value in list(blob.items()):
            if isinstance(value, dict):
                op = _cf(value.get("op"))
                if op in {"eq", "eq_ci"} and _cf(value.get("value")) == _cf(old):
                    value["value"] = new
                    changed = True
                elif rewrite_filter(value, old, new):
                    changed = True
            elif isinstance(value, str) and _cf(value) == _cf(old):
                blob[key] = new
                changed = True
        return changed

    for sid, step in steps.items():
        endpoint = _cf(step.get("endpoint"))
        m = re.search(r"/search/([^/?{}]+)", endpoint)
        if not m:
            continue
        resource = m.group(1)
        literals = step.get("query_literals") or {}
        query = next((str(literals.get(k)).strip() for k in ("query", "q", "name", "title")
                      if literals.get(k) not in (None, "")), "")
        if not query:
            continue
        query_key = _canonical_search_tokens(query, resource)
        if not query_key:
            continue

        observed: list[tuple[int, str, str]] = []
        for obs in getattr(ledger, "observations", []) or []:
            if str(obs.get("plan_step_id") or "") != sid:
                continue
            fields = obs.get("fields") if isinstance(obs.get("fields"), dict) else {}
            for key in ("name", "title", "label", "display_name"):
                value = fields.get(key)
                if isinstance(value, str) and value.strip():
                    try:
                        pos = int(obs.get("position") or 0)
                    except Exception:
                        pos = 0
                    observed.append((pos, key, value.strip()))
                    break
        if not observed:
            continue

        def unique_rows(rows):
            unique = []
            seen = set()
            for item in sorted(rows, key=lambda x: x[0]):
                key = (_cf(item[2]), int(item[0]))
                if key not in seen:
                    seen.add(key)
                    unique.append(item)
            return unique

        exact = unique_rows([x for x in observed if _cf(x[2]) == _cf(query)])
        canonical = unique_rows([
            x for x in observed
            if _cf(x[2]) != _cf(query)
            and _canonical_search_tokens(x[2], resource) == query_key
        ])
        target = exact[0] if len(exact) == 1 else (canonical[0] if not exact and len(canonical) == 1 else None)
        if target is None:
            continue
        pos, field_name, target_label = target

        changed = False
        deriv_ids: list[str] = []
        filter_repaired = False
        for deriv in _derivations(out):
            if sid not in {str(x) for x in deriv.get("source_steps") or []}:
                continue
            local = False
            # Only canonical (non-exact) labels require equality rewrite.
            if _cf(target_label) != _cf(query):
                local = rewrite_filter(deriv.get("filter") or {}, query, target_label)
                if deriv.get("comparison_literal") is not None and _cf(deriv.get("comparison_literal")) == _cf(query):
                    deriv["comparison_literal"] = target_label
                    local = True
                filter_repaired = filter_repaired or local

            # Blind endpoint rank over an unfiltered search may be corrected to
            # the observed unique exact/canonical entity.  If a filter is present,
            # selection rank remains relative to the filtered set and must stay 0.
            op = _cf(deriv.get("operator"))
            filt = deriv.get("filter") if isinstance(deriv.get("filter"), dict) else {}
            if op in {"endpoint_rank", "first", "nth"} and not filt:
                current_rank = int(deriv.get("rank") or 0)
                if current_rank != int(pos):
                    deriv["operator"] = "endpoint_rank"
                    deriv["rank"] = int(pos)
                    local = True
            if local:
                changed = True
                deriv_ids.append(str(deriv.get("id") or ""))

        spec = specs.get(sid) or {}
        spec_filters = list(spec.get("filters") or [])
        spec_filter_repaired = False
        if _cf(target_label) != _cf(query):
            for filt in spec_filters:
                if _cf(filt.get("op")) in {"eq", "eq_ci"} and _cf(filt.get("value")) == _cf(query):
                    filt["value"] = target_label
                    spec_filter_repaired = True
                    changed = True
        if not spec_filters:
            select = spec.get("select") if isinstance(spec.get("select"), dict) else {}
            mode = _cf(select.get("mode"))
            current_index = int(select.get("index") or 0)
            if mode in {"head", "nth"} and current_index != int(pos):
                spec["select"] = {"mode": "nth", "limit": 1, "index": int(pos)}
                changed = True

        if changed:
            kind = "exact" if _cf(target_label) == _cf(query) else "canonical"
            warning = (f"step {sid}: repaired search selection to unique observed {kind} label "
                       f"{target_label!r} at endpoint rank {pos}")
            warnings = list(out.get("validation_warnings") or [])
            if warning not in warnings:
                warnings.append(warning)
            out["validation_warnings"] = warnings
            repairs.append({
                "step_id": sid, "query": query, "canonical_label": target_label,
                "match_kind": kind, "resource": resource, "position": int(pos),
                "field": field_name, "derivation_ids": list(dict.fromkeys(deriv_ids)),
                "filter_repaired": bool(filter_repaired or spec_filter_repaired),
            })
    return out, repairs


def repair_observed_actor_relation_siblings(question: str, plan: dict[str, Any] | None,
                                            ledger) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retarget an empty actor-like relation to one uniquely observed sibling relation.

    Some credit responses expose parallel actor-bearing collections (for example
    regular cast and guest stars).  If the plan chose one documented actor-like
    collection but that relation is empty while exactly one sibling actor-like
    relation is present in the *same already-fetched response*, reuse the response
    and rewrite only relation paths/bindings.  No new entity or answer value is
    guessed.
    """
    import copy
    q=_cf(question)
    if not re.search(r"\b(?:actor|cast|starred|starring|co[- ]?star)\b",q):
        return copy.deepcopy(plan or {}), []
    out=copy.deepcopy(plan or {}); repairs=[]
    actor_relations={"cast","guest_stars","guest-stars","actors","stars"}
    observations=list(getattr(ledger,"observations",[]) or [])
    for spec in out.get("observation_specs") or []:
        sid=str(spec.get("step_id") or "")
        root=str(spec.get("record_path") or "")
        m=re.match(r"^([A-Za-z0-9_-]+)\[\*\]$",root)
        if not m or _cf(m.group(1)) not in actor_relations:
            continue
        old=m.group(1)
        old_rows=[o for o in observations if str(o.get("plan_step_id") or "")==sid and _cf(o.get("relation"))==_cf(old)]
        if old_rows:
            continue
        present=[]
        for rel in actor_relations:
            if _cf(rel)==_cf(old): continue
            rows=[o for o in observations if str(o.get("plan_step_id") or "")==sid and _cf(o.get("relation"))==_cf(rel)]
            if rows and any(isinstance(o.get("fields"),dict) and o["fields"].get("id") not in (None,"") for o in rows):
                present.append((rel,rows))
        if len(present)!=1:
            continue
        new=present[0][0]
        spec["record_path"]=f"{new}[*]"
        def rewrite_path(value):
            text=str(value or "")
            text=re.sub(rf"^{re.escape(old)}(?:\[\*\])?\.",f"{new}.",text)
            if _cf(text)==_cf(old): text=new
            return text
        spec["project_paths"]=[rewrite_path(x) for x in spec.get("project_paths") or []]
        for f in spec.get("filters") or []:
            if "path" in f: f["path"]=rewrite_path(f.get("path"))
        for b in spec.get("bindings") or []:
            if "path" in b: b["path"]=rewrite_path(b.get("path"))
        for step in out.get("steps") or []:
            if str(step.get("id") or "")!=sid: continue
            bp=dict(step.get("binding_paths") or {})
            step["binding_paths"]={k:rewrite_path(v) for k,v in bp.items()}
        for d in out.get("derivations") or []:
            if sid not in {str(x) for x in d.get("source_steps") or []}: continue
            if d.get("field") is not None: d["field"]=rewrite_path(d.get("field"))
            filt=dict(d.get("filter") or {}); d["filter"]={rewrite_path(k):v for k,v in filt.items()}
        warning=f"step {sid}: observed actor relation {old!r} empty; retargeted to unique non-empty sibling {new!r} from the same response"
        out.setdefault("validation_warnings",[]).append(warning)
        repairs.append({"step_id":sid,"old_relation":old,"new_relation":new})
    return out, repairs



def repair_observed_ordered_extrema(question: str, plan: dict[str, Any] | None,
                                     ledger, *, min_records: int = 8) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Collapse a page-local extremum to endpoint rank only when observed order proves it.

    This is an evidence-dependent optimization for endpoints whose documentation is
    insufficiently explicit about ordering.  If at least ``min_records`` records
    from the *same first-page call* are strictly monotone by the exact field used by
    a declared argmax/argmin, the returned endpoint order itself is usable evidence
    for the winner on that observed population.  We then rewrite the selector to
    rank 0 and remove pagination-requiring projection sort semantics.

    No provider names, benchmark routes, or answer values are used.  A non-monotone
    page, too few records, multiple calls, missing values, or a direction mismatch
    means no repair.
    """
    import copy
    from datetime import datetime
    out=copy.deepcopy(plan or {})
    specs={str(x.get('step_id') or ''):x for x in out.get('observation_specs') or []}
    repairs=[]

    def scalar(v):
        if isinstance(v,(int,float)) and not isinstance(v,bool): return float(v)
        if isinstance(v,str):
            t=v.strip()
            try: return float(t)
            except Exception: pass
            try: return datetime.fromisoformat(t.replace('Z','+00:00')).timestamp()
            except Exception: return None
        return None

    for d in out.get('derivations') or []:
        op=_cf(d.get('operator'))
        if op not in {'argmax','argmin'}: continue
        src=[str(x) for x in d.get('source_steps') or [] if str(x)]
        if len(src)!=1: continue
        sid=src[0]; spec=specs.get(sid) or {}
        field=re.sub(r'\[(?:\*|\d*)\]','',str(d.get('field') or '')).strip('.')
        if not field: continue
        # Compiler-facing fields are often record-relative after normalization.
        leaf=field.split('.')[-1]
        by_call={}
        for obs in getattr(ledger,'observations',[]) or []:
            if str(obs.get('plan_step_id') or '')!=sid or obs.get('position') is None: continue
            f=obs.get('fields') if isinstance(obs.get('fields'),dict) else {}
            val=f.get(leaf)
            num=scalar(val)
            if num is None: continue
            cid=str(obs.get('call_id') or '')
            by_call.setdefault(cid,[]).append((int(obs.get('position')),num,obs.get('obs_id')))
        qualifying=[]
        for cid,rows in by_call.items():
            rows=sorted(rows,key=lambda x:x[0])
            if len(rows)<int(min_records) or rows[0][0]!=0: continue
            # Require contiguous observed ranks so projection/truncation cannot
            # manufacture apparent monotonicity from a sparse sample.
            if [r[0] for r in rows] != list(range(len(rows))): continue
            vals=[r[1] for r in rows]
            desc=all(vals[i] > vals[i+1] for i in range(len(vals)-1))
            asc=all(vals[i] < vals[i+1] for i in range(len(vals)-1))
            if (op=='argmax' and desc) or (op=='argmin' and asc):
                qualifying.append((cid,rows,'descending' if desc else 'ascending'))
        if len(qualifying)!=1: continue
        cid,rows,direction=qualifying[0]
        record_path=str(spec.get('record_path') or '')
        root=re.sub(r'\[\*\]$','',record_path).strip('.')
        d['operator']='endpoint_rank'; d['rank']=0
        if root and root!='$': d['field']=root
        else: d['field']=None
        d['comparison']=''; d['comparison_literal']=None
        spec['sort']=[]
        spec['select']={'mode':'head','limit':1,'index':0}
        spec['completeness']='observed endpoint-ranked head records'
        warning=(f"derivation {d.get('id')}: observed first-page {leaf} values were strictly "
                 f"{direction} across {len(rows)} contiguous records; replaced {op} with endpoint rank 0")
        out.setdefault('validation_warnings',[]).append(warning)
        repairs.append({'step_id':sid,'derivation_id':str(d.get('id') or ''),'field':leaf,
                        'direction':direction,'records':len(rows),'call_id':cid})
    if repairs:
        out['validation_warnings']=list(dict.fromkeys(out.get('validation_warnings') or []))
    return out,repairs


def observed_season_episode_relation_variant(question: str, plan: dict[str, Any] | None,
                                             ledger, tools: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Expand an empty aggregate season-credit role into episode-credit fan-out.

    Some APIs expose season-level aggregate credits that omit episodic roles even
    though the corresponding per-episode credits contain them.  When a plan asks
    for an explicit crew role, the aggregate season-credit call succeeded, and no
    observed crew row matches that role, build a read-only hierarchical variant:
    season details -> all episode numbers -> episode credits.  Existing non-season
    branches (e.g. a movie director used for comparison) are preserved.
    """
    import copy
    base=plan or {};steps=_step_map(base)
    get_paths={str(t.get('path') or '') for t in tools or []
               if str(t.get('method') or 'GET').upper() in {'GET','HEAD'}}
    for sid,st in steps.items():
        ep=str(st.get('endpoint') or '')
        if not re.search(r'/tv/\{[^{}]+\}/season/\{[^{}]+\}/credits$',ep): continue
        wanted=None
        for d in _derivations(base):
            if sid not in {str(x) for x in d.get('source_steps') or []}: continue
            filt=d.get('filter') if isinstance(d.get('filter'),dict) else {}
            for key,val in filt.items():
                if str(key).split('.')[-1].casefold()!='job' or not isinstance(val,dict): continue
                if _cf(val.get('op')) in {'eq','eq_ci'} and isinstance(val.get('value'),str):
                    wanted=val.get('value').strip();break
            if wanted: break
        if not wanted: continue
        saw_call=any(str(c.get('plan_step_id') or '')==sid and int(c.get('status_code') or 0)<400
                     for c in getattr(ledger,'api_calls',[]) or [])
        if not saw_call: continue
        matches=[]
        for obs in getattr(ledger,'observations',[]) or []:
            if str(obs.get('plan_step_id') or '')!=sid: continue
            fields=obs.get('fields') if isinstance(obs.get('fields'),dict) else {}
            if _cf(fields.get('job'))==_cf(wanted): matches.append(obs)
        if matches: continue
        season_detail=re.sub(r'/credits$','',ep)
        if season_detail not in get_paths: continue
        episode_credits=next((p for p in get_paths
            if re.search(r'/tv/\{[^{}]+\}/season/\{[^{}]+\}/episode/\{[^{}]+\}/credits$',p)),None)
        if not episode_credits: continue
        out=copy.deepcopy(base);vsteps={str(x.get('id') or ''):x for x in out.get('steps') or []}
        season=vsteps[sid]; original_pb=dict(season.get('path_bindings') or {}); original_pl=dict(season.get('path_literals') or {})
        season['endpoint']=season_detail;season['request_parameters']=[]
        season['binds']=['episode_number'];season['binding_paths']={'episode_number':'episodes.episode_number'}
        ep_sid=f'{sid}_episode_credits';n=2
        while ep_sid in vsteps: ep_sid=f'{sid}_episode_credits_{n}';n+=1
        old_ph=re.findall(r'\{([^{}]+)\}',ep); season_ph=re.findall(r'\{([^{}]+)\}',season_detail)
        # Retarget placeholder keys while retaining the original binding/literal values.
        new_pb={};new_pl={}
        for i,ph in enumerate(season_ph):
            if i<len(old_ph) and old_ph[i] in original_pb: new_pb[ph]=original_pb[old_ph[i]]
            elif i<len(old_ph) and old_ph[i] in original_pl: new_pl[ph]=original_pl[old_ph[i]]
        season['path_bindings']=new_pb;season['path_literals']=new_pl
        eph=re.findall(r'\{([^{}]+)\}',episode_credits)
        ep_pb={};ep_pl={}
        # Map series and season from the same original values; episode from season details.
        for i,ph in enumerate(eph):
            if i < len(old_ph):
                old=old_ph[i]
                if old in original_pb: ep_pb[ph]=original_pb[old]
                elif old in original_pl: ep_pl[ph]=original_pl[old]
            else: ep_pb[ph]='episode_number'
        episode_step={'id':ep_sid,'method':'GET','endpoint':episode_credits,
                      'purpose':f'Fetch per-episode credits because aggregate season credits contained no {wanted} rows.',
                      'depends_on':[sid],'binds':[],'binding_paths':{},'path_literals':ep_pl,'path_bindings':ep_pb,
                      'query_literals':{},'query_bindings':{},'body_literals':{},'body_bindings':{},
                      'answer_source':True,'request_parameters':[],'request_body_required':False,
                      'request_body_required_fields':[],'request_body_leaf_paths':[]}
        idx=next(i for i,x in enumerate(out['steps']) if str(x.get('id') or '')==sid)
        out['steps'].insert(idx+1,episode_step)
        # Add explicit all-episode selection so the binding is finite and replayable.
        pop_id=f'{sid}_episodes';existing={str(d.get('id') or '') for d in out.get('derivations') or []};nn=2
        while pop_id in existing: pop_id=f'{sid}_episodes_{nn}';nn+=1
        pop={'id':pop_id,'operator':'filter','source_steps':[sid],'source_derivations':[],
             'label_steps':[],'label_fields':[],'field':'episodes','comparison':'','comparison_literal':None,
             'unit':'raw','distinct_field':'episode_number','rank':0,'filter':{},'top_k':20,
             'purpose':'Select the bounded episode population for role recovery.'}
        # Identify the complete derivation subgraph rooted in the aggregate
        # crew relation so record-relative identities (e.g. field=name after a
        # crew/job filter) move with their producer too.
        original_derivs=[dict(d) for d in out.get('derivations') or []]
        relation_ids=set()
        for d in original_derivs:
            if sid not in {str(x) for x in d.get('source_steps') or []}: continue
            field=_cf(d.get('field'));filt=json.dumps(d.get('filter') or {},sort_keys=True).casefold()
            if 'crew' in field or 'job' in filt or _cf(d.get('operator'))=='membership':
                relation_ids.add(str(d.get('id') or ''))
        changed=True
        while changed:
            changed=False
            for d in original_derivs:
                did=str(d.get('id') or '')
                if did in relation_ids: continue
                if {str(x) for x in d.get('source_derivations') or []} & relation_ids:
                    relation_ids.add(did);changed=True
        derivs=[pop]
        for d in original_derivs:
            nd=dict(d);did=str(nd.get('id') or '')
            if did in relation_ids and sid in {str(x) for x in nd.get('source_steps') or []}:
                nd['source_steps']=[ep_sid if str(x)==sid else str(x) for x in nd.get('source_steps') or []]
            derivs.append(nd)
        out['derivations']=derivs
        out['answer_steps']=[ep_sid if str(x)==sid else str(x) for x in out.get('answer_steps') or []]
        out['observation_specs']=[]
        out.setdefault('validation_warnings',[]).append(
            f'observed aggregate season credits had no {wanted!r} crew rows; prepared season-detail -> episode-credit fan-out')
        return out,{'season_step_id':sid,'episode_step_id':ep_sid,'role':wanted,
                    'season_endpoint':season_detail,'episode_credits_endpoint':episode_credits}
    return None,{}

def observed_cross_resource_search_variants(question: str, plan: dict[str, Any] | None,
                                            ledger, tools: list[dict[str, Any]],
                                            *, max_variants: int = 3) -> list[dict[str, Any]]:
    """Build read-only sibling-search plan variants after a named work search misses.

    This is deliberately *acquisition* recovery, not answer guessing.  A variant is
    proposed only when the current free-text search has no exact/near returned
    label, a sibling ``/search/<resource>`` operation is documented, and every
    resource-bound descendant can be mapped to a documented sibling operation with
    the same suffix.  Runtime execution still has to retrieve evidence and satisfy
    the normal compiler/contract before a variant can be adopted.
    """
    import copy
    q=_cf(question); base=plan or {}; steps=_step_map(base)
    get_paths={str(t.get("path") or "") for t in tools or []
               if str(t.get("method") or "GET").upper() in {"GET","HEAD"}}
    variants=[]

    # Co-starring questions are often planned person-first.  If that route cannot
    # establish a shared work ID, prepare work-first movie/TV variants from the
    # same literal title and the already-declared person searches.  Both people
    # must be proven members of the selected work's cast before a Boolean can close.
    qco=_cf(question)
    co_starring=bool(re.search(r"\b(?:co[- ]?star(?:ring|red)?|star(?:ring|red)?\s+(?:together|in)|both\s+(?:star|appear|act))\b",qco))
    person_steps=[]
    if co_starring:
        for psid,pst in steps.items():
            if re.search(r"/search/person$",str(pst.get('endpoint') or '')):
                lits=pst.get('query_literals') or {}
                name=next((str(lits.get(k)).strip() for k in ('query','q','name') if lits.get(k) not in (None,'')),'')
                if name: person_steps.append((psid,name,pst))
    title_literals=[]
    if len(person_steps)>=2:
        title_first=[]; name_fallback=[]
        person_ids={x[0] for x in person_steps}
        for d in _derivations(base):
            filt=d.get('filter') if isinstance(d.get('filter'),dict) else {}
            sources={str(x) for x in d.get('source_steps') or []}
            for key,val in filt.items():
                leaf=str(key).split('.')[-1].casefold()
                if leaf not in {'title','name','original_title','original_name'}: continue
                if not (isinstance(val,dict) and _cf(val.get('op')) in {'eq','eq_ci'} and isinstance(val.get('value'),str)):
                    continue
                literal=val['value'].strip()
                if leaf in {'title','original_title'}: title_first.append(literal)
                elif not (sources & person_ids): name_fallback.append(literal)
        # Prefer title-bearing work relations over generic name filters so the two
        # person-search equality filters cannot be mistaken for the target work.
        title_literals=[x for x in dict.fromkeys(title_first or name_fallback) if x]
    if len(person_steps)>=2 and len(title_literals)==1:
        target_title=title_literals[0]
        for new in ('movie','tv'):
            search_path=next((p for p in get_paths if re.search(rf"/search/{new}$",p)),None)
            credit_path=next((p for p in get_paths if re.search(rf"/{new}/\{{[^{{}}]+\}}/credits$",p)),None)
            if not search_path or not credit_path: continue
            variant=copy.deepcopy(base)
            # Keep only the two identity searches so previously acquired person
            # evidence can be replayed; replace the unresolved person-credit chain.
            kept=[]
            keep_ids={person_steps[0][0],person_steps[1][0]}
            for st in variant.get('steps') or []:
                if str(st.get('id') or '') in keep_ids: kept.append(st)
            work_sid='co_work_search'; credit_sid='co_work_credits'
            existing={str(x.get('id') or '') for x in kept}
            n=2
            while work_sid in existing: work_sid=f'co_work_search_{n}';n+=1
            existing.add(work_sid);n=2
            while credit_sid in existing: credit_sid=f'co_work_credits_{n}';n+=1
            ph=(re.findall(r"\{([^{}]+)\}",credit_path) or [f'{new}_id'])[0]
            work_alias=f'{new}_id'
            kept += [
              {'id':work_sid,'method':'GET','endpoint':search_path,
               'purpose':f"Resolve the named work {target_title!r} before checking its cast.",
               'depends_on':[],'binds':[work_alias],'binding_paths':{work_alias:'id'},
               'path_literals':{},'path_bindings':{},'query_literals':{'query':target_title},
               'query_bindings':{},'body_literals':{},'body_bindings':{},'answer_source':False,
               'request_parameters':[],'request_body_required':False,'request_body_required_fields':[],
               'request_body_leaf_paths':[]},
              {'id':credit_sid,'method':'GET','endpoint':credit_path,
               'purpose':'Read the selected work cast and prove whether both named people appear.',
               'depends_on':[work_sid],'binds':[],'binding_paths':{},'path_literals':{},
               'path_bindings':{ph:work_alias},'query_literals':{},'query_bindings':{},
               'body_literals':{},'body_bindings':{},'answer_source':True,'request_parameters':[],
               'request_body_required':False,'request_body_required_fields':[],
               'request_body_leaf_paths':[]}
            ]
            variant['steps']=kept
            p1,p2=person_steps[:2]
            label_field='title' if new=='movie' else 'name'
            derivs=[
              {'id':'co_p1_pick','operator':'endpoint_rank','source_steps':[p1[0]],'source_derivations':[],
               'field':'results','rank':0,'filter':{},'top_k':10,'purpose':f"Select {p1[1]} from person search."},
              {'id':'co_p1_name','operator':'identity','source_steps':[p1[0]],'source_derivations':['co_p1_pick'],
               'field':'name','filter':{},'top_k':10,'purpose':f"Extract {p1[1]} name."},
              {'id':'co_p2_pick','operator':'endpoint_rank','source_steps':[p2[0]],'source_derivations':[],
               'field':'results','rank':0,'filter':{},'top_k':10,'purpose':f"Select {p2[1]} from person search."},
              {'id':'co_p2_name','operator':'identity','source_steps':[p2[0]],'source_derivations':['co_p2_pick'],
               'field':'name','filter':{},'top_k':10,'purpose':f"Extract {p2[1]} name."},
              {'id':'co_work_pick','operator':'endpoint_rank','source_steps':[work_sid],'source_derivations':[],
               'field':'results','rank':0,'filter':{},'top_k':10,'purpose':f"Select the top named-work search result for {target_title}."},
              {'id':'co_cast_population','operator':'filter','source_steps':[credit_sid],'source_derivations':['co_work_pick'],
               'field':'cast','filter':{},'top_k':50,'purpose':'Select the cast relation from the selected work credits.'},
              {'id':'co_cast_names','operator':'identity','source_steps':[credit_sid],'source_derivations':['co_cast_population'],
               'field':'name','filter':{},'top_k':50,'purpose':'Extract cast names from the selected work.'},
              {'id':'co_member_1','operator':'membership','source_steps':[p1[0],credit_sid],
               'source_derivations':['co_p1_name','co_cast_names'],'field':None,'filter':{},'top_k':50,
               'purpose':f"Check whether {p1[1]} is in the selected work cast."},
              {'id':'co_member_2','operator':'membership','source_steps':[p2[0],credit_sid],
               'source_derivations':['co_p2_name','co_cast_names'],'field':None,'filter':{},'top_k':50,
               'purpose':f"Check whether {p2[1]} is in the selected work cast."},
              {'id':'co_both','operator':'logical_and','source_steps':[],
               'source_derivations':['co_member_1','co_member_2'],'field':None,'filter':{},'top_k':10,
               'purpose':'Both named people must be present in the selected work cast.'},
            ]
            variant['derivations']=derivs;variant['answer_steps']=[credit_sid]
            variant['answer_derivations']=['co_both'];variant['answer_mode']='boolean'
            variant['answer_requirements']=[f"whether both named people co-star in {target_title}"]
            variant['observation_specs']=[]
            variant.setdefault('validation_warnings',[]).append(
                f"prepared work-first {new} co-starring recovery for named work {target_title!r}")
            variants.append({'plan':variant,'search_step_id':work_sid,'query':target_title,
                             'old_resource':'person-first','new_resource':new,'search_endpoint':search_path,
                             'mapped_descendants':[(credit_sid,'unresolved-person-first',credit_path)],
                             'co_starring_variant':True})
            if len(variants)>=max_variants: return variants

    def norm_label(v):
        toks=[x for x in re.findall(r"[a-z0-9]+",_cf(v)) if x not in {"a","an","the"}]
        return " ".join(toks)
    def descendants(root):
        out=set(); changed=True
        while changed:
            changed=False
            for sid,st in steps.items():
                if sid==root or sid in out: continue
                if set(str(x) for x in st.get("depends_on") or []) & ({root}|out):
                    out.add(sid); changed=True
        return out
    def sibling_resource_path(path, old_resource, new_resource):
        parts=[x for x in str(path).strip('/').split('/') if x]
        if old_resource not in parts: return None
        idx=parts.index(old_resource)
        cand=list(parts); cand[idx]=new_resource
        # Placeholder names are resource-specific. Match a documented path by
        # static segments and placeholder positions instead of guessing its name.
        matches=[]
        for p in get_paths:
            pp=[x for x in p.strip('/').split('/') if x]
            if len(pp)!=len(cand): continue
            ok=True
            for a,b in zip(cand,pp):
                aph=a.startswith('{') and a.endswith('}'); bph=b.startswith('{') and b.endswith('}')
                if aph or bph:
                    if not (aph and bph): ok=False; break
                elif a!=b: ok=False; break
            if ok: matches.append(p)
        return matches[0] if len(matches)==1 else None

    for sid,step in steps.items():
        path=str(step.get("endpoint") or ""); m=re.search(r"(.*/search/)([^/?{}]+)$",path)
        if not m or m.group(2) in {"person","people","company","collection"}: continue
        literals=step.get("query_literals") or {}
        query=next((str(literals.get(k)).strip() for k in ("query","q","name","title") if literals.get(k) not in (None,"")),"")
        if len(query)<2: continue
        labels=[]
        for obs in getattr(ledger,"observations",[]) or []:
            if str(obs.get("plan_step_id") or "")!=sid: continue
            f=obs.get("fields") if isinstance(obs.get("fields"),dict) else {}
            for k in ("name","title","original_name","original_title","label"):
                if isinstance(f.get(k),str) and f.get(k).strip(): labels.append(f[k].strip())
        qn=norm_label(query)
        if any(qn and (qn==norm_label(v) or qn in norm_label(v) or norm_label(v) in qn) for v in labels):
            continue
        prefix=m.group(1); old=m.group(2); dep_ids=descendants(sid)
        siblings=[]
        for p in sorted(get_paths):
            mm=re.search(re.escape(prefix)+r"([^/?{}]+)$",p)
            if mm and mm.group(1)!=old and mm.group(1) not in {"person","people","company","collection"}:
                siblings.append((mm.group(1),p))
        for new,new_search in siblings:
            variant=copy.deepcopy(base); vsteps={str(s.get("id") or ""):s for s in variant.get("steps") or []}
            vsearch=vsteps[sid]; vsearch["endpoint"]=new_search;vsearch["request_parameters"]=[]
            # Keep only response aliases that can safely survive a resource-family
            # switch (IDs). Human labels are re-derived from the sibling schema.
            bp=dict(vsearch.get("binding_paths") or {})
            keep={k:v for k,v in bp.items() if re.sub(r"\[(?:\*|\d*)\]", "",str(v)).split('.')[-1].casefold()=="id"}
            if keep:
                vsearch["binding_paths"]=keep;vsearch["binds"]=[x for x in vsearch.get("binds") or [] if x in keep]
            feasible=True; mapped=[]
            for did in dep_ids:
                st=vsteps[did]; ep=str(st.get("endpoint") or "")
                if f"/{old}/" not in ep: continue
                alt=sibling_resource_path(ep,old,new)
                if not alt:
                    feasible=False;break
                old_ph=re.findall(r"\{([^{}]+)\}",ep); new_ph=re.findall(r"\{([^{}]+)\}",alt)
                old_pb=dict(st.get("path_bindings") or {})
                st["endpoint"]=alt;st["request_parameters"]=[]
                if len(old_ph)==len(new_ph):
                    st["path_bindings"]={new_ph[i]: old_pb.get(old_ph[i], old_ph[i]) for i in range(len(new_ph))}
                mapped.append((did,ep,alt))
            if not feasible: continue
            # Observation specs are route/schema-specific and must be rebuilt.
            variant["observation_specs"]=[]
            variant.setdefault("validation_warnings",[]).append(
                f"observed named search {query!r} had no near match on {path}; prepared documented sibling search {new_search} with mapped descendants")
            variants.append({"plan":variant,"search_step_id":sid,"query":query,"old_resource":old,
                             "new_resource":new,"search_endpoint":new_search,"mapped_descendants":mapped})
            if len(variants)>=max_variants: return variants
    return variants

def plan_signature(plan: dict[str, Any] | None) -> str:
    """ID-insensitive semantic strategy fingerprint for no-progress detection.

    Step/derivation IDs are planner-local names, so they are normalized to stable
    ordinal references.  Binding sources and dependency topology *are* preserved:
    changing the owner/binding/selection path must count as real progress even if
    the endpoint set stays the same.
    """
    plan = plan or {}
    raw_steps = _steps(plan)
    raw_derivs = _derivations(plan)
    step_index = {str(s.get("id") or ""): i for i, s in enumerate(raw_steps)}
    deriv_index = {str(d.get("id") or ""): i for i, d in enumerate(raw_derivs)}

    def norm_ref(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): norm_ref(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, list):
            return [norm_ref(v) for v in value]
        if isinstance(value, tuple):
            return [norm_ref(v) for v in value]
        text = str(value) if isinstance(value, str) else value
        if isinstance(text, str) and text in step_index:
            return {"step_ref": step_index[text]}
        if isinstance(text, str) and text in deriv_index:
            return {"derivation_ref": deriv_index[text]}
        return value

    steps = []
    for s in raw_steps:
        steps.append({
            "method": str(s.get("method") or "GET").upper(),
            "endpoint": str(s.get("endpoint") or ""),
            "depends_on": [step_index.get(str(x), str(x)) for x in s.get("depends_on") or []],
            "path_literals": norm_ref(s.get("path_literals") or {}),
            "query_literals": norm_ref(s.get("query_literals") or {}),
            "body_literals": norm_ref(s.get("body_literals") or {}),
            "path_bindings": norm_ref(s.get("path_bindings") or {}),
            "query_bindings": norm_ref(s.get("query_bindings") or {}),
            "body_bindings": norm_ref(s.get("body_bindings") or {}),
            "binds": sorted(str(x) for x in s.get("binds") or []),
        })
    derivs = []
    for d in raw_derivs:
        derivs.append({
            "operator": _cf(d.get("operator")),
            "field": str(d.get("field") or ""),
            "comparison": _cf(d.get("comparison")),
            "rank": d.get("rank", 0),
            "filter": norm_ref(d.get("filter") or {}),
            "top_k": d.get("top_k"),
            "source_steps": [step_index.get(str(x), str(x)) for x in d.get("source_steps") or []],
            "source_derivations": [deriv_index.get(str(x), str(x)) for x in d.get("source_derivations") or []],
            "label_steps": [step_index.get(str(x), str(x)) for x in d.get("label_steps") or []],
            "label_fields": list(d.get("label_fields") or []),
        })
    payload = {
        "steps": steps,
        "derivations": derivs,
        "answer_steps": [step_index.get(str(x), str(x)) for x in plan.get("answer_steps") or []],
        "answer_derivations": [deriv_index.get(str(x), str(x)) for x in plan.get("answer_derivations") or []],
        "answer_mode": plan.get("answer_mode"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def is_read_only_plan(plan: dict[str, Any] | None) -> bool:
    steps = _steps(plan)
    return bool(steps) and all(str(x.get("method") or "GET").upper() in {"GET", "HEAD"}
                               for x in steps)


def plan_semantic_risks(question: str, plan: dict[str, Any] | None) -> list[str]:
    """High-confidence, API-agnostic question/plan mismatches."""
    q = _cf(question)
    steps = _steps(plan)
    derivs = _derivations(plan)
    by_step = _step_map(plan)
    risks: list[str] = []

    def source_text(d):
        return " ".join(
            _cf((by_step.get(str(sid)) or {}).get("endpoint")) + " " +
            _cf((by_step.get(str(sid)) or {}).get("purpose"))
            for sid in d.get("source_steps") or [])

    # Do not let an internal identifier become a certified user-facing value
    # unless the user actually asked for an ID/identifier. This is intentionally
    # generic and catches semantic leaks such as projecting ``genre_ids`` for a
    # request that asks for genre names.
    asks_identifier = bool(re.search(r"\b(?:id|ids|identifier|identifiers)\b", q))
    answer_ids = {str(x) for x in (plan or {}).get("answer_derivations") or []}
    if not asks_identifier and answer_ids:
        for d in derivs:
            if str(d.get("id") or "") not in answer_ids:
                continue
            field = str(d.get("field") or "").strip().casefold()
            terminal = field.rsplit(".", 1)[-1].replace("[*]", "")
            if terminal in {"id", "ids"} or terminal.endswith("_id") or terminal.endswith("_ids"):
                risks.append(
                    f"answer derivation {d.get('id')} projects identifier field {field!r} "
                    "although the question does not request an identifier")

    # A requested superlative cannot be satisfied by raw collection order unless
    # the selected route/purpose itself declares that criterion.
    criteria = []
    if re.search(r"\bmost\s+popular\b|\bhighest\s+popularity\b", q):
        criteria.append(("popularity", "max", {"popular", "popularity"}))
    if re.search(r"\b(?:highest|best|top)[ -]?rated\b|\bhighest\s+rating\b", q):
        criteria.append(("rating", "max", {"rating", "rated", "top rated", "top_rated", "vote_average", "vote average", "score"}))
    if re.search(r"\b(?:lowest|worst)[ -]?rated\b|\blowest\s+rating\b", q):
        criteria.append(("rating", "min", {"rating", "rated", "vote_average", "vote average", "score"}))
    for label, direction, aliases in criteria:
        represented = False
        suspicious = []
        for d in derivs:
            op = _cf(d.get("operator"))
            semantic = _cf(d.get("field")) + " " + _cf(d.get("purpose")) + " " + source_text(d)
            if op == ("argmax" if direction == "max" else "argmin") and any(a in semantic for a in aliases):
                represented = True
            if op in {"endpoint_rank", "first", "nth"}:
                if any(a in source_text(d) for a in aliases):
                    represented = True
                else:
                    suspicious.append(str(d.get("id") or op))
        if suspicious and not represented:
            risks.append(
                f"requested {label} superlative is resolved from raw endpoint/record order "
                f"({suspicious[:4]}) without an explicit {direction} selection or a criterion-aligned route")

    # Recency similarly needs a date/time extremum or an explicitly criterion-aligned
    # route. A generic "latest record" route is *not* enough when the user asks for
    # the latest *released* item: database insertion order and release chronology are
    # different semantics on many APIs.
    if re.search(r"\b(?:latest|newest|most recent|most recently)\b", q):
        release_recency = bool(re.search(
            r"\b(?:latest|newest|most recent)(?:\s+\w+){0,3}\s+released\b|"
            r"\b(?:latest|newest|most recent)\s+release\b|\bmost recently released\b", q))
        recency_ok = False
        rank_risky = False
        generic_latest_rank = False
        for d in derivs:
            op = _cf(d.get("operator")); semantic = _cf(d.get("field")) + " " + source_text(d)
            if op == "argmax" and any(t in semantic for t in ("date", "time", "year", "release")):
                recency_ok = True
            if op in {"endpoint_rank", "first", "nth"}:
                src = source_text(d)
                has_latest = "latest" in src or "recent" in src or "newest" in src
                has_release = "release" in src or "released" in src
                if release_recency:
                    if has_latest and has_release:
                        recency_ok = True
                    elif has_latest:
                        generic_latest_rank = True
                    else:
                        rank_risky = True
                elif has_latest:
                    recency_ok = True
                else:
                    rank_risky = True
        if release_recency and generic_latest_rank and not recency_ok:
            risks.append(
                "latest-released intent relies on a generic latest-record route without replayable release-date maximization")
        elif rank_risky and not recency_ok:
            risks.append("explicit latest/newest request relies on raw collection order without replayable recency semantics")

    # Preserve role restrictions through credit/people relations.
    if re.search(r"\b(?:directed by|director of|who directed|\bdirect\b)\b", q):
        credit_steps = {str(s.get("id") or "") for s in steps
                        if "credit" in _cf(s.get("endpoint")) or "crew" in _cf(s.get("endpoint"))}
        if credit_steps:
            plan_blob = _cf(json.dumps(plan or {}, ensure_ascii=False, default=str))
            if "director" not in plan_blob and '"job"' not in plan_blob:
                risks.append("director relation reaches credits/crew evidence without a terminal director/job restriction")

    # High-confidence media/resource-family conflicts.  This is vocabulary from
    # the question and route itself, not a benchmark-specific route table.
    # Infer the searched resource family from the terminal segment of generic
    # search-like routes instead of depending on any benchmark route table.
    tv_families = {"tv", "television", "series", "show", "shows"}
    movie_families = {"movie", "movies", "film", "films"}
    # Tie a resource-family check to the *searched literal*, not to unrelated nouns
    # elsewhere in a multi-hop question. A target resource mentioned later in the
    # relation chain must not relabel the source work that the plan is resolving.
    for s in steps:
        endpoint = _cf(s.get("endpoint")).split("?", 1)[0].rstrip("/")
        segments = [seg for seg in endpoint.split("/") if seg]
        if "search" not in segments or not segments:
            continue
        family = segments[-1].replace("-", "_").replace(" ", "_")
        literals = s.get("query_literals") or {}
        literal = next((str(literals.get(k)).strip() for k in ("query", "q", "name", "title")
                        if literals.get(k) not in (None, "")), "")
        if not literal:
            continue
        pos = q.find(_cf(literal))
        if pos < 0:
            continue
        prefix_words = re.findall(r"[a-z0-9]+", q[:pos])[-4:]
        suffix_words = re.findall(r"[a-z0-9]+", q[pos+len(literal):])[:2]
        local = " ".join(prefix_words + suffix_words)
        local_tv = bool(re.search(r"\b(?:tv|television|series|episode|show)\b", local))
        local_movie = bool(re.search(r"\b(?:movie|film)\b", local))
        if family in movie_families and local_tv and not local_movie:
            risks.append(
                f"named search literal {literal!r} is locally described as a TV/television work, "
                "but the plan searches only a movie work family")
        if family in tv_families and local_movie and not local_tv:
            risks.append(
                f"named search literal {literal!r} is locally described as a movie/film work, "
                "but the plan searches only a TV work family")

    return list(dict.fromkeys(risks))


def observed_semantic_risks(question: str, plan: dict[str, Any] | None, ledger) -> list[str]:
    """Use already retrieved evidence to spot clear entity-resolution mistakes."""
    q = _cf(question)
    if ledger is None:
        return []
    risks: list[str] = []
    steps = _step_map(plan)
    # If a free-text search query has an exact returned name/title at another
    # position, raw rank-0 selection is not the strongest deterministic resolution.
    for d in _derivations(plan):
        if _cf(d.get("operator")) != "endpoint_rank" or int(d.get("rank", 0) or 0) != 0:
            continue
        # A recovered plan may deliberately filter the search collection to the
        # exact requested label and then select rank 0 *within that filtered set*.
        # Treat that as real semantic progress rather than repeatedly challenging it.
        filt_blob = _cf(json.dumps(d.get("filter") or {}, ensure_ascii=False, default=str))
        for sid in d.get("source_steps") or []:
            step = steps.get(str(sid)) or {}
            if "/search/" not in _cf(step.get("endpoint")):
                continue
            literals = step.get("query_literals") or {}
            query = next((str(literals.get(k)).strip() for k in ("query", "q", "name", "title")
                          if literals.get(k) not in (None, "")), "")
            if len(query) < 2:
                continue
            if filt_blob and _cf(query) in filt_blob and any(
                    token in filt_blob for token in ("name", "title", "label", "display_name")):
                continue
            candidates = []
            for obs in getattr(ledger, "observations", []) or []:
                if str(obs.get("plan_step_id") or "") != str(sid):
                    continue
                if obs.get("position") is None:
                    continue
                fields = obs.get("fields") if isinstance(obs.get("fields"), dict) else {}
                for key in ("name", "title", "label", "display_name"):
                    value = fields.get(key)
                    if isinstance(value, str) and _cf(value) == _cf(query):
                        candidates.append((int(obs.get("position") or 0), key, value))
            if candidates and all(pos != 0 for pos, _, _ in candidates):
                pos, key, value = sorted(candidates)[0]
                risks.append(
                    f"search query {query!r} has an exact returned {key} match {value!r} at position {pos}, "
                    "but the plan selects raw rank 0")

    # An exact equality filter can itself be too strict when an API canonicalizes
    # a free-text label (for example by adding a resource-type suffix).  If the
    # search returned records but the requested literal has no exact label match,
    # report that evidence to the replanner rather than repeatedly issuing the
    # same search or silently treating the empty filtered set as absence.
    for d in _derivations(plan):
        if _cf(d.get("operator")) != "filter":
            continue
        for sid in d.get("source_steps") or []:
            step = steps.get(str(sid)) or {}
            if "/search/" not in _cf(step.get("endpoint")):
                continue
            literals = step.get("query_literals") or {}
            query = next((str(literals.get(k)).strip() for k in ("query", "q", "name", "title")
                          if literals.get(k) not in (None, "")), "")
            if not query:
                continue
            rows = []
            for obs in getattr(ledger, "observations", []) or []:
                if str(obs.get("plan_step_id") or "") != str(sid):
                    continue
                fields = obs.get("fields") if isinstance(obs.get("fields"), dict) else {}
                for key in ("name", "title", "label", "display_name"):
                    value = fields.get(key)
                    if isinstance(value, str):
                        rows.append((key, value))
            if not rows:
                continue
            filt_blob = _cf(json.dumps(d.get("filter") or {}, ensure_ascii=False, default=str))
            query_cf = _cf(query)
            if query_cf not in filt_blob:
                continue
            exact = [value for _key, value in rows if _cf(value) == query_cf]
            near = [value for _key, value in rows
                    if query_cf in _cf(value) or _cf(value) in query_cf]
            if not exact and near:
                risks.append(
                    f"exact search-label filter for {query!r} excludes all exact matches although "
                    f"returned canonical candidate labels include {near[:3]!r}; reconsider the selection "
                    "without repeating the same search")
    # If a named free-text search produced no plausible label match, do not let a
    # downstream empty collection become a confident negative immediately. A bounded
    # replan may try a documented sibling search/resource family. This is especially
    # important for APIs that expose the same named work under separate resource types.
    for sid, step in steps.items():
        if "/search/" not in _cf(step.get("endpoint")):
            continue
        literals = step.get("query_literals") or {}
        query = next((str(literals.get(k)).strip() for k in ("query", "q", "name", "title")
                      if literals.get(k) not in (None, "")), "")
        if len(query) < 2:
            continue
        labels: list[str] = []
        for obs in getattr(ledger, "observations", []) or []:
            if str(obs.get("plan_step_id") or "") != str(sid):
                continue
            fields = obs.get("fields") if isinstance(obs.get("fields"), dict) else {}
            for key in ("name", "title", "original_name", "original_title", "label", "display_name"):
                value = fields.get(key)
                if isinstance(value, str) and value.strip():
                    labels.append(value.strip())
        qn = " ".join(x for x in re.findall(r"[a-z0-9]+", _cf(query)) if x not in {"a","an","the"})
        near = []
        for value in labels:
            vn = " ".join(x for x in re.findall(r"[a-z0-9]+", _cf(value)) if x not in {"a","an","the"})
            if qn and vn and (qn == vn or qn in vn or vn in qn):
                near.append(value)
        if not labels or not near:
            risks.append(
                f"named search {query!r} on {step.get('endpoint')} produced no exact/near labeled candidate; "
                "before certifying absence, consider a documented sibling search/resource family using the same name")
    return list(dict.fromkeys(risks))


def answer_surface_risks(question: str, answer: str, contract: dict[str, Any] | None = None) -> list[str]:
    """Question-only checks for obviously polluted/shape-incompatible outputs."""
    q = _cf(question); ans = str(answer or "").strip()
    if not ans:
        return ["empty candidate answer"]
    parts = [x.strip() for x in re.split(r"\s*;\s*|\n[-*]\s*", ans) if x.strip()]
    risks: list[str] = []
    if re.match(r"^who\b", q):
        numeric = sum(bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s+(?:years?|days?|months?|points?|votes?))?", p, re.I))
                      for p in parts)
        named = sum(bool(re.search(r"[A-Za-z]", p)) for p in parts)
        numeric_requested = bool(re.search(
            r"\b(?:by how many|by how much|what (?:is|'s) the difference|difference (?:in|between)|"
            r"how many (?:years?|days?|months?|points?|votes?) (?:older|younger|more|less))\b", q))
        if numeric and named and len(parts) > 1 and not numeric_requested:
            risks.append("who-answer mixes likely support numbers/IDs with the requested identity")
    if re.search(r"\bgenres?\b", q):
        scalar_parts = [re.sub(r"^[^:]+:\s*", "", p).strip() for p in parts]
        if scalar_parts and all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", p) for p in scalar_parts):
            risks.append("genre answer exposes numeric identifiers rather than human-readable genre values")

    singular = bool(re.search(r"\b(?:one|a single|a keyword|an image|a photo|a poster|a cover)\b", q))
    if singular and len(parts) > 1:
        risks.append("explicit singular answer request produced multiple surface values")
    mode = _cf((contract or {}).get("answer_kind"))
    if mode == "asset":
        empty_asset = _cf(ans) in {"none", "null", "nil", "n/a", "na", "[]", "{}"}
        unavailable_asset = bool(re.match(
            r"^(?:no\s+(?:results?|images?|logos?|assets?)\b|no\s+.+\s+(?:available|found)\b|"
            r"(?:results?|images?|logos?|assets?)\s+(?:are|is)\s+not\s+available\b)", _cf(ans)))
        if empty_asset or unavailable_asset:
            risks.append("asset task candidate is an empty/null placeholder or unavailable-result message rather than a usable asset value")
    if mode == "boolean" and not re.match(r"^(?:yes|no|true|false)\b", _cf(ans)):
        risks.append("boolean task candidate does not surface a yes/no conclusion")
    return list(dict.fromkeys(risks))


def recovery_feedback(question: str, plan: dict[str, Any] | None, compiled: dict[str, Any] | None,
                      ledger=None, verification: dict[str, Any] | None = None) -> dict[str, Any]:
    warnings = list((compiled or {}).get("warnings") or [])[:12]
    observed_step_samples: dict[str, list[dict[str, Any]]] = {}
    if ledger is not None:
        relevant_steps = [step for step in _steps(plan)
                          if "/search/" in _cf(step.get("endpoint")) or step.get("answer_source")]
        for step in relevant_steps[:4]:
            sid = str(step.get("id") or "")
            samples = []
            for obs in getattr(ledger, "observations", []) or []:
                if str(obs.get("plan_step_id") or "") != sid:
                    continue
                fields = obs.get("fields") if isinstance(obs.get("fields"), dict) else {}
                compact = {str(k): v for k, v in list(fields.items())[:6]
                           if v is None or isinstance(v, (str, int, float, bool))}
                if compact:
                    samples.append({"position": obs.get("position"), "fields": compact})
                if len(samples) >= 3:
                    break
            if samples:
                observed_step_samples[sid] = samples

    semantic_risks = list(dict.fromkeys(
        plan_semantic_risks(question, plan) + observed_semantic_risks(question, plan, ledger)))[:16]
    warning_blob = " ".join(warnings).casefold()
    missing_slots = list((verification or {}).get("missing_slots") or [])[:12]
    validation_errors = [str(x) for x in (plan or {}).get("validation_errors") or []]

    # Some deterministic route validators report a semantic mismatch directly in
    # validation_errors.  Treat those as strategy failures rather than local plan
    # closure; otherwise a perfectly explicit "wrong population/route" diagnostic
    # could be mislabeled as a binding defect merely because plan.valid=False.
    semantic_validation_terms = (
        "semantic route invariant", "semantic incompatib", "relation invariant",
        "population mismatch", "unstated narrowing", "future/scheduled",
        "wrong resource", "requested relation",
    )
    semantic_validation_errors = [
        err for err in validation_errors
        if any(term in err.casefold() for term in semantic_validation_terms)
    ]
    if semantic_validation_errors:
        semantic_risks = list(dict.fromkeys(semantic_risks + semantic_validation_errors))[:16]

    # Failure-typed recovery prevents one generic "try another plan" instruction
    # from destroying useful evidence.  Local closure defects should preserve the
    # acquisition strategy; semantic/population failures should change it.
    if semantic_risks:
        failure_type = "semantic_strategy"
        instruction = (
            "Solve the same question with a genuinely different documented evidence strategy for the listed "
            "semantic risks. Preserve resource type, relation, cardinality, owner, selection criterion, and "
            "requested output fields. Reuse already observed evidence and do not repeat an identical search merely "
            "to obtain the same records. Do not merely rename steps or repeat the same failed route."
        )
    elif any(x in warning_blob for x in ("no candidates", "empty evidence set")):
        failure_type = "empty_or_wrong_population"
        instruction = (
            "The current acquisition produced no usable candidate population. Keep correct resolved entities and "
            "reuse the already observed evidence shown here: do not repeat an identical search merely to obtain the "
            "same records. If an exact free-text label filter returned no candidate but observations show a "
            "canonicalized label, use the strongest documented/observed discriminator. For empty dedicated asset "
            "results, try another plausible already-returned owner candidate rather than the same entity ID."
        )
    elif (plan or {}).get("valid") is False and (plan or {}).get("steps"):
        failure_type = "local_plan_closure"
        instruction = (
            "Preserve the current documented endpoint strategy unless a validation error proves a route is invalid. "
            "Repair only the unresolved local closure: producer bindings, selected record ownership, derivation "
            "lineage, answer derivations, or request/schema fields. Reuse existing evidence; do not refetch correct "
            "steps merely to make a new plan look different."
        )
    elif missing_slots and set(missing_slots).issubset(
            {"citations", "answer_values", "answer_lineage", "required_derivations",
             "terminal_entity_ownership", "plan_valid"}):
        failure_type = "certificate_or_lineage_closure"
        instruction = (
            "The acquisition is potentially usable; preserve correct routes and evidence. Repair deterministic "
            "answer lineage/citations/derivation closure first. Change the evidence strategy only if the listed "
            "verification notes demonstrate that the requested fact itself was not acquired."
        )
    else:
        failure_type = "evidence_strategy"
        instruction = (
            "Solve the same question with a documented evidence strategy that fills the remaining evidence gap. "
            "Preserve correct resource/relation semantics, reuse already observed evidence, and avoid repeating "
            "identical calls that cannot add information."
        )

    feedback = {
        "same_user_question": str(question or ""),
        "previous_plan_signature": plan_signature(plan),
        "previous_routes": [
            {"method": str(s.get("method") or "GET").upper(),
             "endpoint": str(s.get("endpoint") or ""),
             "purpose": str(s.get("purpose") or "")[:220]}
            for s in _steps(plan)
        ],
        "failure_type": failure_type,
        "semantic_risks": semantic_risks,
        "compiler_warnings": warnings,
        "plan_validation_errors": validation_errors[:16],
        "verification_status": str((verification or {}).get("verification_status") or ""),
        "verification_missing_slots": missing_slots,
        "verification_notes": list((verification or {}).get("notes") or [])[:12],
        "instruction": instruction,
        "observed_step_samples": observed_step_samples,
    }
    return feedback
