from agents.evidence_scorer import (
    score_root_causes,
    rank_root_causes
)


metrics = {

    "latency_ms": 78,

    "throughput_mbps": 185,

    "jitter_ms": 16,

    "prb_utilization": 94,

    "active_users": 185,

    "sinr_db": 19,

    "cqi": 12,

    "retransmission_rate": 2.8,

    "backhaul_utilization": 54,

    "cpu_utilization": 60,

    "hardware_alarm": False
}


results = score_root_causes(metrics)

ranked = rank_root_causes(results)


print("\n========== RCA RANKING ==========")

for cause, result in ranked:

    print(
        f"\n{cause}"
    )

    print(
        f"Score: {result['score']}"
    )

    print(
        "Evidence:"
    )

    for item in result["evidence"]:
        print(f"  + {item}")

    print(
        "Contradictions:"
    )

    for item in result["contradictions"]:
        print(f"  - {item}")

print("\n=================================")