"""Evidence-grounded Root Cause Analysis agent, independent of anomaly checks."""
from src.ollama_agent import explain

CAUSES = {
    "5g_congestion.txt": ("5G cell congestion", ["Check PRB utilization", "Check active user count", "Check cell resource utilization"]),
    "5g_backhaul.txt": ("Backhaul congestion", ["Check backhaul utilization", "Check transport latency", "Check packet loss"]),
    "5g_retransmission.txt": ("Packet retransmissions", ["Check retransmission rate", "Check radio link quality", "Check packet loss"]),
    "5g_radio_conditions.txt": ("Poor radio conditions", ["Check SINR", "Check CQI", "Check RSRP/RSRQ"]),
    "5g_interference.txt": ("Radio interference", ["Check interference levels", "Check neighboring-cell configuration"]),
    "5g_qos.txt": ("QoS or queueing issue", ["Check QoS policy", "Check queue depth"]),
}

def _get_metric(anomaly: dict, metric_name: str):

    # First check the network KPI snapshot
    network_metrics = anomaly.get("network_metrics", {})

    if metric_name in network_metrics:
        value = network_metrics[metric_name]

        if value is not None:
            return float(value)

    # Fallback: check anomaly records
    for item in anomaly.get("anomalies", []):

        if item.get("metric") == metric_name:

            for key in [
                "value",
                "current_value",
                "actual_value",
                "metric_value"
            ]:
                if item.get(key) is not None:
                    return float(item[key])

    return None
def score_hypotheses(anomaly: dict):

    latency = _get_metric(anomaly, "latency_ms")
    throughput = _get_metric(anomaly, "throughput_mbps")
    jitter = _get_metric(anomaly, "jitter_ms")
    prb = _get_metric(anomaly, "prb_utilization")
    users = _get_metric(anomaly, "active_users")
    sinr = _get_metric(anomaly, "sinr_db")
    cqi = _get_metric(anomaly, "cqi")
    rsrp = _get_metric(anomaly, "rsrp_dbm")
    retransmission = _get_metric(
        anomaly,
        "retransmission_rate"
    )
    backhaul = _get_metric(
        anomaly,
        "backhaul_utilization"
    )

    results = {}

    # ==================================================
    # CELL CONGESTION
    # ==================================================

    score = 0
    evidence = []
    contradictions = []

    if prb is not None:

        if prb > 90:
            score += 3
            evidence.append(
                f"PRB utilization is very high ({prb:.1f}%)"
            )

        elif prb < 70:
            score -= 3
            contradictions.append(
                f"PRB utilization is not high ({prb:.1f}%)"
            )

    if users is not None:

        if users > 150:
            score += 3
            evidence.append(
                f"Active user count is high ({users:.0f})"
            )

    if throughput is not None and throughput < 250:

        score += 2
        evidence.append(
            f"Throughput is low ({throughput:.1f} Mbps)"
        )

    if latency is not None and latency > 50:

        score += 2
        evidence.append(
            f"Latency is high ({latency:.1f} ms)"
        )

    if jitter is not None and jitter > 10:

        score += 2
        evidence.append(
            f"Jitter is high ({jitter:.1f} ms)"
        )

    results["5g_congestion.txt"] = {
        "score": score,
        "evidence": evidence,
        "contradictions": contradictions
    }


    # ==================================================
    # BACKHAUL CONGESTION
    # ==================================================

    score = 0
    evidence = []
    contradictions = []

    if backhaul is not None:

        if backhaul > 90:

            score += 3
            evidence.append(
                f"Backhaul utilization is very high ({backhaul:.1f}%)"
            )

        else:

            score -= 3
            contradictions.append(
                f"Backhaul utilization is normal ({backhaul:.1f}%)"
            )

    if latency is not None and latency > 50:

        score += 2
        evidence.append(
            f"Latency is high ({latency:.1f} ms)"
        )

    if throughput is not None and throughput < 250:

        score += 2
        evidence.append(
            f"Throughput is low ({throughput:.1f} Mbps)"
        )

    results["5g_backhaul.txt"] = {
        "score": score,
        "evidence": evidence,
        "contradictions": contradictions
    }


    # ==================================================
    # RADIO CONDITIONS
    # ==================================================

    score = 0
    evidence = []
    contradictions = []

    if sinr is not None:

        if sinr < 10:

            score += 3
            evidence.append(
                f"SINR is poor ({sinr:.1f} dB)"
            )

        else:

            score -= 2
            contradictions.append(
                f"SINR is not severely degraded ({sinr:.1f} dB)"
            )

    if cqi is not None:

        if cqi < 7:

            score += 3
            evidence.append(
                f"CQI is poor ({cqi:.1f})"
            )

        else:

            score -= 2
            contradictions.append(
                f"CQI is not severely degraded ({cqi:.1f})"
            )

    if rsrp is not None and rsrp < -110:

        score += 2
        evidence.append(
            f"RSRP is weak ({rsrp:.1f} dBm)"
        )

    if retransmission is not None and retransmission > 5:

        score += 2
        evidence.append(
            f"Retransmission rate is high ({retransmission:.1f}%)"
        )

    if throughput is not None and throughput < 250:

        score += 1
        evidence.append(
            f"Throughput is low ({throughput:.1f} Mbps)"
        )

    results["5g_radio_conditions.txt"] = {
        "score": score,
        "evidence": evidence,
        "contradictions": contradictions
    }


    return results

def rank_hypotheses(results):

    return sorted(
        results.items(),
        key=lambda item: item[1]["score"],
        reverse=True
    )


def format_rca_report(rca: dict | None) -> str:
    if not rca:
        return "No RCA report available."

    lines = []
    lines.append("=" * 90)
    lines.append("5G ROOT CAUSE ANALYSIS REPORT")
    lines.append("=" * 90)
    lines.append(f"Status: {rca.get('status', 'UNKNOWN')}")
    if rca.get("severity"):
        lines.append(f"Severity: {rca['severity']}")
    if rca.get("status") == "INSUFFICIENT EVIDENCE":
        lines.append(f"Reason: {rca.get('reason', 'No supporting evidence found.')}")
        lines.append("=" * 90)
        return "\n".join(lines)

    causes = rca.get("root_causes", [])
    if not causes:
        lines.append("No root causes identified.")
        lines.append("=" * 90)
        return "\n".join(lines)

    lines.append("Possible root causes:")
    for index, cause in enumerate(causes, start=1):
        lines.append(f"\n{index}. {cause['cause']}")

        lines.append(
    f"   Evidence Score: {cause.get('evidence_score', 0)}"
        )

        lines.append(
            f"   RAG Similarity: {cause.get('rag_similarity', 0):.3f}"
        )

        lines.append(
            f"   Confidence: {cause.get('confidence', 0):.0%}"
        )

        lines.append(
            f"   Reason: {cause.get('reason', 'No reason provided.')}"
        )

        if cause.get("evidence"):
            lines.append(
                f"   Supporting Evidence: {', '.join(cause['evidence'])}"
            )

        if cause.get("contradictions"):
            lines.append(
                f"   Contradictions: {', '.join(cause['contradictions'])}"
            )

        if cause.get("diagnostic_checks"):
            lines.append(
                f"   Diagnostic Checks: {'; '.join(cause['diagnostic_checks'])}"
            )



    actions = rca.get("recommended_actions", [])
    lines.append("\nRecommended actions:")
    if actions:
        for action in actions:
            lines.append(f"- {action}")
    else:
        lines.append("- No actions recommended.")

    rag = rca.get("rag")
    if rag:
        lines.append("\nRAG retrieval:")
        lines.append(f"Query: {rag.get('query', 'N/A')}")
        for item in rag.get("results", []):
            lines.append(f"- {item.get('document', 'Unknown document')} | score={item.get('score', 0):.3f}")
            snippet = item.get("content", "")
            if snippet:
                lines.append(f"  {snippet[:220]}{'...' if len(snippet) > 220 else ''}")

    reasoning = rca.get("qwen_reasoning")
    if reasoning:
        lines.append("\nQwen reasoning:")
        lines.append(str(reasoning))

    lines.append("=" * 90)
    return "\n".join(lines)


class RootCauseAgent:
    def _query(self, anomaly: dict) -> str:
        kinds = sorted({item["metric"].replace("_ms", "") for item in anomaly.get("anomalies", [])})
        return "Possible causes of " + ", ".join(kinds) + " in a 5G network: congestion, backhaul, radio conditions, packet loss and retransmissions."

    def analyze(self, anomaly: dict, top_k: int = 5, use_ollama: bool = True) -> dict:

    # ==================================================
    # 1. CHECK ANOMALY
    # ==================================================

        if anomaly.get("status") != "ANOMALY":

            return {
                "status": "NORMAL",
                "root_causes": [],
                "recommended_actions": [],
                "rag": None
            }


        # ==================================================
        # 2. CREATE RAG QUERY
        # ==================================================

        query = self._query(anomaly)


        # ==================================================
        # 3. RAG RETRIEVAL
        # ==================================================

        try:

            from src.rag_retriever import search

            retrieval = search(query, top_k)

        except (
            FileNotFoundError,
            ImportError,
            ModuleNotFoundError
        ) as error:

            return {
                "status": "INSUFFICIENT EVIDENCE",
                "reason": str(error),
                "root_causes": [],
                "recommended_actions": [],
                "rag": {
                    "query": query,
                    "results": []
                }
            }


        # ==================================================
        # 4. SCORE ACTUAL NETWORK EVIDENCE
        # ==================================================

        scores = score_hypotheses(anomaly)


        # ==================================================
        # 5. RANK ROOT-CAUSE HYPOTHESES
        # ==================================================

        ranked_scores = rank_hypotheses(scores)


        # ==================================================
        # 6. MAP RAG RESULTS
        # ==================================================

        retrieval_map = {
            item["document"]: item
            for item in retrieval.get("results", [])
        }


        # ==================================================
        # 7. COMBINE KPI EVIDENCE + RAG
        # ==================================================

        ranked = []

        for document, result in ranked_scores:

            cause_info = CAUSES.get(document)

            if cause_info is None:
                continue

            cause, checks = cause_info

            rag_item = retrieval_map.get(document)


            # Ignore a cause when:
            # - it has no KPI evidence
            # - and RAG did not retrieve it

            if (
                result["score"] <= 0
                and rag_item is None
            ):
                continue


            ranked.append({

                "cause": cause,

                # IMPORTANT:
                # This is the actual evidence score,
                # NOT the FAISS similarity score.
                "evidence_score": result["score"],

                # Keep RAG similarity separately.
                "rag_similarity": (
                    round(rag_item["score"], 3)
                    if rag_item
                    else 0.0
                ),

                "reason": (
                    "This hypothesis was evaluated against "
                    "the observed network metrics and "
                    "retrieved 5G troubleshooting evidence."
                ),

                "evidence": result["evidence"],

                "contradictions": result["contradictions"],

                "diagnostic_checks": checks

            })


        # ==================================================
        # 8. NO SUPPORTED ROOT CAUSE
        # ==================================================

        if not ranked:

            return {
                "status": "INSUFFICIENT EVIDENCE",
                "root_causes": [],
                "recommended_actions": [],
                "rag": retrieval
            }


        # ==================================================
        # 9. CALCULATE RELATIVE CONFIDENCE
        # ==================================================

        positive_scores = [
            max(0, item["evidence_score"])
            for item in ranked
        ]

        total_score = sum(positive_scores)


        for item in ranked:

            if total_score > 0:

                item["confidence"] = round(
                    max(0, item["evidence_score"])
                    / total_score,
                    2
                )

            else:

                item["confidence"] = 0.0


        # ==================================================
        # 10. RECOMMENDED ACTIONS
        # ==================================================

        # Give priority to the highest-ranked cause.

        actions = ranked[0]["diagnostic_checks"]


        # ==================================================
        # 11. SEND RCA INFORMATION TO QWEN
        # ==================================================

        reasoning = explain(

            {
                "status": anomaly["status"],
                "severity": anomaly["severity"],
                "anomalies": anomaly["anomalies"]
            },

            {

                "anomaly_json": anomaly,

                "retrieved_evidence":
                    retrieval.get("results", []),

                "ranked_causes":
                    ranked,

                "instruction": """

    You are a 5G Root Cause Analysis Agent.

    IMPORTANT:

    An abnormal condition is NOT automatically
    the root cause.

    RAG provides possible causes and
    technical troubleshooting knowledge.

    The Evidence Scoring system evaluates
    the actual network metrics.

    Evaluate each candidate root cause using:

    1. Supporting evidence
    2. Contradicting evidence
    3. Actual KPI values
    4. RAG evidence
    5. Evidence score

    Do NOT choose a root cause simply because
    it has the highest RAG similarity score.

    Choose the hypothesis that is best supported
    by the available evidence.

    If the evidence is insufficient, say:

    INSUFFICIENT EVIDENCE

    Return:

    1. Most likely root cause
    2. Evidence score
    3. Confidence
    4. Supporting evidence
    5. Contradicting evidence
    6. Alternative root causes
    7. Missing evidence
    8. Recommended diagnostic checks
    9. Recommended actions

    Clearly distinguish between:

    ABNORMAL CONDITION

    and

    ROOT CAUSE.

    """
            }
        ) if use_ollama else "Ollama explanation disabled by the AI Harness."


        # ==================================================
        # 12. RETURN FINAL RCA
        # ==================================================

        return {

            "status": anomaly["status"],

            "severity": anomaly["severity"],

            "root_causes": ranked,

            "recommended_actions": actions,

            "rag": retrieval,

            "qwen_reasoning": reasoning

        }
