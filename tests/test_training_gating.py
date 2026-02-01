from app.training.gating import evaluate_publish_gate


def test_gate_passes_without_previous_metrics():
    decision = evaluate_publish_gate(
        rows_trainable=1000,
        new_metrics={
            "overall": {"mdape": 100.0},
            "by_realestate_type": {"enebolig": {"mdape": 200.0}},
        },
        prev_metrics=None,
    )
    assert decision.passed is True


def test_gate_blocks_on_min_rows():
    decision = evaluate_publish_gate(
        rows_trainable=10,
        new_metrics={
            "overall": {"mdape": 100.0},
            "by_realestate_type": {"enebolig": {"mdape": 200.0}},
        },
        prev_metrics=None,
    )
    assert decision.passed is False
    assert any("insufficient_rows_trainable" in r for r in decision.reasons)


def test_gate_blocks_on_overall_mae_degradation():
    prev = {"overall": {"mdape": 100.0}, "by_realestate_type": {"enebolig": {"mdape": 200.0}}}
    new = {"overall": {"mdape": 200.0}, "by_realestate_type": {"enebolig": {"mdape": 200.0}}}
    decision = evaluate_publish_gate(rows_trainable=1000, new_metrics=new, prev_metrics=prev)
    assert decision.passed is False
    assert any("degraded:overall.mdape" in r for r in decision.reasons)


def test_gate_blocks_on_enebolig_mae_degradation():
    prev = {"overall": {"mdape": 100.0}, "by_realestate_type": {"enebolig": {"mdape": 100.0}}}
    new = {"overall": {"mdape": 100.0}, "by_realestate_type": {"enebolig": {"mdape": 1000.0}}}
    decision = evaluate_publish_gate(rows_trainable=1000, new_metrics=new, prev_metrics=prev)
    assert decision.passed is False
    assert any("degraded:enebolig.mdape" in r for r in decision.reasons)
