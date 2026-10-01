"""
log_analyzer.py
Parses execution output containing [S2], [S3], [S4] log lines
produced by the simplified log_injector.py.

New simple format:
  [S2] GET /endpoint
  [S3] status: 200 | response: {...}
  [S3] extracted: person_id = 1769
  [S4] final answer type: str | value: Christopher Nolan
"""

import re
from collections import defaultdict

SEVERITY = {
    "EMPTY_RESULTS":   "HIGH",
    "HTTP_ERROR":      "HIGH",
    "NONE_EXTRACTED":  "HIGH",
    "NULL_ANSWER":     "HIGH",
    "URL_AS_ANSWER":   "MEDIUM",
    "WRONG_TYPE":      "MEDIUM",
}


def analyze_logs(output: str) -> dict:
    """
    Parse [Sx] log lines and return structured findings + anomalies.
    """
    findings = {
        "s2_calls":     [],
        "s3_findings":  [],
        "s3_extracted": [],
        "s4_answer":    {},
        "anomalies":    [],
        "scope_summary": defaultdict(list),
    }

    for line in output.splitlines():
        line = line.strip()

        # ── S2: API call ──────────────────────────────────────────────
        if line.startswith("[S2:ToolInput]"):
            endpoint = line[len("[S2:ToolInput]"):].strip()
            findings["s2_calls"].append({"endpoint": endpoint})
            continue

        # ── S3: response / extracted variable ─────────────────────────
        # Injector emits: [S3:ResponseProcessing] varname = value
        # (both regex and LLM modes use this format)
        if line.startswith("[S3:ResponseProcessing]"):
            rest = line[len("[S3:ResponseProcessing]"):].strip()

            # Inline anomaly flagged by the injector
            if "ANOMALY" in rest:
                msg = rest.split("ANOMALY", 1)[-1].strip().lstrip(":]").strip()
                _add_anomaly(findings, "Scope3", "EMPTY_RESULTS", msg or rest, "HIGH")
                continue

            eq_match = re.match(r"(\w+)\s*=\s*(.+)", rest)
            if eq_match:
                vname = eq_match.group(1)
                value = eq_match.group(2).strip()
                findings["s3_extracted"].append({"variable": vname, "value": value})

                # None / empty extraction will corrupt downstream steps
                if value in ("None", "null", "[]", "{}"):
                    _add_anomaly(findings, "Scope3", "NONE_EXTRACTED",
                        f"'{vname}' extracted as {value} — will corrupt next steps", "HIGH")
                # Empty results list in a response preview
                elif "'results': []" in value or '"results": []' in value:
                    _add_anomaly(findings, "Scope3", "EMPTY_RESULTS",
                        f"'{vname}' holds an empty results list — agent may select wrong data", "HIGH")
            continue

        # ── S4: final answer ──────────────────────────────────────────
        if line.startswith("[S4:FinalAnswer]"):
            # Format: [S4] final answer type: str | value: Christopher Nolan
            type_match  = re.search(r"type:\s*(\w+)", line)
            value_match = re.search(r"value:\s*(.+)", line)

            if type_match:
                ans_type = type_match.group(1).strip()
                findings["s4_answer"]["type"] = ans_type
                if ans_type in ("list", "dict", "NoneType"):
                    _add_anomaly(findings, "Scope4", "WRONG_TYPE",
                        f"Final answer type is '{ans_type}' — not human-readable", "MEDIUM")

            if value_match:
                preview = value_match.group(1).strip()
                findings["s4_answer"]["preview"] = preview
                if preview.startswith("http") and "tmdb.org" in preview:
                    _add_anomaly(findings, "Scope4", "URL_AS_ANSWER",
                        f"Final answer is a URL: {preview[:60]}", "MEDIUM")
                if preview in ("None", "", "[]", "{}"):
                    _add_anomaly(findings, "Scope4", "NULL_ANSWER",
                        "Final answer is empty or null", "HIGH")
            continue

    findings["scope_summary"] = dict(findings["scope_summary"])
    return findings


def _add_anomaly(findings, scope, atype, message, severity):
    findings["anomalies"].append({
        "scope": scope, "type": atype,
        "message": message, "severity": severity,
    })
    findings["scope_summary"][scope].append(atype)


def summarize_findings(findings: dict) -> str:
    lines = []
    if findings["s2_calls"]:
        lines.append(f"  API calls ({len(findings['s2_calls'])}):")
        for c in findings["s2_calls"]:
            lines.append(f"    → {c['endpoint']}")
    if findings["s3_extracted"]:
        lines.append("  Extracted:")
        for e in findings["s3_extracted"]:
            lines.append(f"    {e['variable']} = {e['value'][:80]}")
    if findings["s4_answer"]:
        lines.append(f"  Final answer: [{findings['s4_answer'].get('type','?')}] "
                     f"{findings['s4_answer'].get('preview','?')[:80]}")
    if findings["anomalies"]:
        lines.append(f"  ANOMALIES ({len(findings['anomalies'])}):")
        for a in findings["anomalies"]:
            lines.append(f"    [{a['severity']}] {a['scope']}: {a['message'][:80]}")
    return "\n".join(lines) if lines else "  No findings."
