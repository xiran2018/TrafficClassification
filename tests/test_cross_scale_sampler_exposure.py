from audit_cross_scale_sampler_exposure import batch_cross_scale_exposure


def row(flow, uid, label):
    return {"flow_id": flow, "packet_uid": uid, "label_id": label}


def test_alias_cannot_supply_cross_scale_context():
    rows = [
        row("singleton", "a", 0),
        row("other", "b", 1),
        row("other", "c", 1),
    ]
    result = batch_cross_scale_exposure(rows, [0, 0, 1, 2], {"a", "b", "c"})
    assert result["duplicate_rows"] == 1
    assert result["alias_only_false_context_anchors"] == 1
    assert result["factual_to_intervened_valid_anchors"] == 2
    assert result["intervened_to_factual_valid_anchors"] == 2


def test_missing_paired_view_is_directionally_availability_masked():
    rows = [
        row("left", "a", 0),
        row("left", "b", 0),
        row("right", "c", 1),
        row("right", "d", 1),
    ]
    result = batch_cross_scale_exposure(rows, [0, 1, 2, 3], {"a", "c", "d"})
    assert result["distinct_own_context_anchors"] == 4
    assert result["factual_to_intervened_valid_anchors"] == 3
    assert result["intervened_to_factual_valid_anchors"] == 3
    assert result["bidirectional_valid_anchors"] == 2


def test_same_class_other_flow_is_not_a_negative():
    rows = [
        row("left", "a", 0),
        row("left", "b", 0),
        row("same-class", "c", 0),
        row("same-class", "d", 0),
    ]
    result = batch_cross_scale_exposure(rows, [0, 1, 2, 3], {"a", "b", "c", "d"})
    assert result["distinct_own_context_anchors"] == 4
    assert result["factual_to_intervened_valid_anchors"] == 0
    assert result["intervened_to_factual_valid_anchors"] == 0
