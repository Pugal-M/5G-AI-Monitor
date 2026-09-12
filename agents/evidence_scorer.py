def score_root_causes(metrics):

    results = {}

    # =========================================
    # 1. CELL CONGESTION
    # =========================================

    score = 0
    evidence = []
    contradictions = []

    if metrics.get("prb_utilization", 0) > 90:
        score += 3
        evidence.append(
            "PRB utilization is very high"
        )

    if metrics.get("active_users", 0) > 150:
        score += 3
        evidence.append(
            "Active user count is high"
        )

    if metrics.get("throughput_mbps", 0) < 250:
        score += 2
        evidence.append(
            "Throughput is low"
        )

    if metrics.get("latency_ms", 0) > 50:
        score += 2
        evidence.append(
            "Latency is high"
        )

    if metrics.get("jitter_ms", 0) > 10:
        score += 2
        evidence.append(
            "Jitter is high"
        )

    results["Cell Congestion"] = {
        "score": score,
        "evidence": evidence,
        "contradictions": contradictions
    }


    # =========================================
    # 2. BACKHAUL CONGESTION
    # =========================================

    score = 0
    evidence = []
    contradictions = []

    if metrics.get("backhaul_utilization", 0) > 90:
        score += 3
        evidence.append(
            "Backhaul utilization is very high"
        )
    else:
        score -= 3
        contradictions.append(
            "Backhaul utilization is normal"
        )

    if metrics.get("latency_ms", 0) > 50:
        score += 2
        evidence.append(
            "Latency is high"
        )

    if metrics.get("throughput_mbps", 0) < 250:
        score += 2
        evidence.append(
            "Throughput is low"
        )

    results["Backhaul Congestion"] = {
        "score": score,
        "evidence": evidence,
        "contradictions": contradictions
    }


    # =========================================
    # 3. RADIO INTERFERENCE
    # =========================================

    score = 0
    evidence = []
    contradictions = []

    if metrics.get("sinr_db", 100) < 10:
        score += 3
        evidence.append(
            "SINR is very low"
        )
    else:
        contradictions.append(
            "SINR does not indicate severe degradation"
        )

    if metrics.get("cqi", 100) < 7:
        score += 3
        evidence.append(
            "CQI is poor"
        )

    if metrics.get("retransmission_rate", 0) > 5:
        score += 2
        evidence.append(
            "Retransmission rate is high"
        )

    if metrics.get("throughput_mbps", 0) < 250:
        score += 2
        evidence.append(
            "Throughput is low"
        )

    results["Radio Interference"] = {
        "score": score,
        "evidence": evidence,
        "contradictions": contradictions
    }


    # =========================================
    # 4. HARDWARE / PROCESSING ISSUE
    # =========================================

    score = 0
    evidence = []
    contradictions = []

    if metrics.get("cpu_utilization", 0) > 90:
        score += 3
        evidence.append(
            "CPU utilization is very high"
        )
    else:
        contradictions.append(
            "CPU utilization is normal"
        )

    if metrics.get("hardware_alarm", False):
        score += 3
        evidence.append(
            "Hardware alarm detected"
        )
    else:
        contradictions.append(
            "No hardware alarm detected"
        )

    results["Hardware Issue"] = {
        "score": score,
        "evidence": evidence,
        "contradictions": contradictions
    }

    return results


def rank_root_causes(results):

    ranked = sorted(
        results.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    return ranked