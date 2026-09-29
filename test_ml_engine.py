from backend.engine.threat_intelligence import ThreatIntelligencePredictor


def test_ml_quantile_brain():

    predictor = ThreatIntelligencePredictor()
    predictor.warmup()

    test_cases = [
        {
            "mode": "sea",
            "origin": "Shanghai",
            "dest": "Rotterdam",
            "nlp": 0.0,
            "desc": "Baseline Sea Transit",
        },
        {
            "mode": "sea",
            "origin": "Shanghai",
            "dest": "Rotterdam",
            "nlp": 0.95,
            "desc": "Suez Canal Blockage (Severe)",
        },
        {
            "mode": "air",
            "origin": "Mumbai",
            "dest": "Frankfurt",
            "nlp": 0.0,
            "desc": "Routine Air Freight",
        },
        {
            "mode": "air",
            "origin": "Dubai",
            "dest": "London",
            "nlp": 0.65,
            "desc": "Dubai Hub Surge (Air Congestion)",
        },
        {
            "mode": "rail",
            "origin": "Los Angeles",
            "dest": "Chicago",
            "nlp": 0.0,
            "desc": "Baseline Rail Corridor",
        },
        {
            "mode": "road",
            "origin": "Chennai",
            "dest": "Bengaluru",
            "nlp": 0.75,
            "desc": "Chennai Monsoon Flood (Road Block)",
        },
    ]

    print(
        f"\n{'Test Scenario':<32} | {'Mode':<5} | "
        f"{'p50 (h)':<7} | {'p85 (h)':<7} | {'p95 (h)':<7} | "
        f"{'Risk Tier':<9} | {'NLP Buffer':<10}"
    )
    print("-" * 92)

    for tc in test_cases:

        res = predictor.predict_worst_case_delay(
            origin=tc["origin"],
            destination=tc["dest"],
            transport_mode=tc["mode"],
            nlp_score=tc["nlp"],
        )

        # ---------------------------------------------------------
        # 1. Required output validation
        # ---------------------------------------------------------
        required_keys = [
            "p50_delay",
            "p85_delay",
            "p95_delay",
            "risk_tier",
            "explainability",
        ]

        for key in required_keys:
            assert key in res, f"Missing required output: {key}"

        assert "nlp_impact_hours" in res["explainability"], \
            "Missing NLP explainability output"

        # ---------------------------------------------------------
        # 2. Quantile ordering validation
        # ---------------------------------------------------------
        p50 = res["p50_delay"]
        p85 = res["p85_delay"]
        p95 = res["p95_delay"]

        assert p50 >= 0, f"p50 cannot be negative: {p50}"
        assert p85 >= 0, f"p85 cannot be negative: {p85}"
        assert p95 >= 0, f"p95 cannot be negative: {p95}"

        assert p50 <= p85 <= p95, (
            f"Invalid quantile ordering: "
            f"p50={p50}, p85={p85}, p95={p95}"
        )

        # ---------------------------------------------------------
        # 3. Explainability validation
        # ---------------------------------------------------------
        nlp_impact = res["explainability"]["nlp_impact_hours"]

        assert nlp_impact >= 0, (
            f"NLP impact cannot be negative: {nlp_impact}"
        )

        # ---------------------------------------------------------
        # 4. Threat-response validation
        # ---------------------------------------------------------
        if tc["nlp"] > 0:
            assert nlp_impact > 0, (
                f"Expected positive NLP impact for threat scenario "
                f"'{tc['desc']}', got {nlp_impact}"
            )

        nlp_buf = (
            f"+{nlp_impact}h"
            if nlp_impact > 0
            else "0.0h"
        )

        print(
            f"{tc['desc']:<32} | "
            f"{tc['mode'].upper():<5} | "
            f"{p50:<7.1f} | "
            f"{p85:<7.1f} | "
            f"{p95:<7.1f} | "
            f"{res['risk_tier']:<9} | "
            f"{nlp_buf:<10}"
        )

    print("-" * 92)
    print(" Quantile ordering validated: p50 <= p85 <= p95")
    print(" Non-negative delay predictions validated")
    print("NLP explainability validated")
    print(" Threat-response behavior validated")
    print(" Required ML outputs validated")
    print(" Validation Complete")


if __name__ == "__main__":
    test_ml_quantile_brain()