# Paper Evidence Pack

Framework consistency: `True`

## Claims

| Dataset | Acc | Macro-F1 | Target | Point Gate | CI Gate | Group CI Gate | Claim | Acc 95% CI | Macro-F1 95% CI | Grouped Acc/F1 95% CI |
|---|---:|---:|---|---|---|---|---|---:|---:|---:|
| vpn-app | 0.7512 | 0.7522 | 0.7400/0.6500 | True | False | False | point_pass_ci_mixed | [0.7287, 0.7716] | [0.7181, 0.7830] | [0.7298, 0.7729]/[0.7171, 0.7822] |
| tls-120 | 0.8461 | 0.8292 | 0.7800/0.7000 | True | True | True | strong | [0.8391, 0.8526] | [0.8196, 0.8354] | [0.8397, 0.8525]/[0.8197, 0.8359] |

## Paper Audit Gates

| Unified Audit | Defaults Audit | Shared Core | Flow Manifests | Packet Manifests | Flow Defaults | Packet Defaults | Strict Reproduction |
|---|---|---|---:|---:|---:|---:|---|
| review | False | True | 0 | 0 | True/2 | False/6 | False |

Flow scope: `vpn-app, tls-120`

Packet scope: `vpn-app, tls-120, vpn-service, vpn-binary, ustc-app, ustc-binary`

## Unified Module Usage

| Dataset | Module family evidence | Selector decision |
|---|---|---|
| vpn-app | packet_embedding_backbone=active; flow_base_expert=active; validation_gated_selector=active; bootstrap_guard=inherited; target_shift_guard=inherited; expert_switch_or_fusion=active:cross_fold_consensus:vote_priority; class_bias_calibration_candidate=evaluated; trainable_multiview_gate=not_observed; cross_fold_consensus=active; selector_expert_slots=crossfold_consensus_identity_compatible:10;provided:0;identity:10;extra:0 | cross_fold_consensus auto_confidence->vote_priority, inputs=3, mean_conf=0.9345 |
| tls-120 | packet_embedding_backbone=active; flow_base_expert=active; validation_gated_selector=active; bootstrap_guard=inherited; target_shift_guard=inherited; expert_switch_or_fusion=active:cross_fold_consensus:log_mean; class_bias_calibration_candidate=evaluated; trainable_multiview_gate=not_observed; cross_fold_consensus=active; selector_expert_slots=crossfold_consensus_identity_compatible:10;provided:0;identity:10;extra:0 | cross_fold_consensus auto_confidence->log_mean, inputs=3, mean_conf=0.7305 |

## Unified Expert Slots

| Dataset | Mode | Slots | Provided | Identity-from-base | Extra |
|---|---|---|---|---|---|
| vpn-app | crossfold_consensus_identity_compatible | base, graph, seq, prior_base, emb_lr, emb_et, proto_emb, paired, slot_stacker, soft_gate |  | base, graph, seq, prior_base, emb_lr, emb_et, proto_emb, paired, slot_stacker, soft_gate |  |
| tls-120 | crossfold_consensus_identity_compatible | base, graph, seq, prior_base, emb_lr, emb_et, proto_emb, paired, slot_stacker, soft_gate |  | base, graph, seq, prior_base, emb_lr, emb_et, proto_emb, paired, slot_stacker, soft_gate |  |

## Ablation Effects

| Dataset | Stage | Delta Acc | Delta F1 | Delta Acc 95% CI | Delta F1 95% CI | Effect |
|---|---|---:|---:|---:|---:|---|
| vpn-app | base constrained ensemble | 0.0000 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | uncertain_or_neutral |
| vpn-app | unsafe reliability fusion | -0.0532 | -0.0924 | [-0.0676, -0.0395] | [-0.1125, -0.0735] | harmful |
| vpn-app | calibration-enabled selector | -0.0150 | -0.0317 | [-0.0206, -0.0090] | [-0.0449, -0.0182] | harmful |
| vpn-app | safe selector | 0.0000 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | uncertain_or_neutral |
| tls-120 | graph/seq base | 0.0000 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | uncertain_or_neutral |
| tls-120 | strict safe selector | 0.0000 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | uncertain_or_neutral |
| tls-120 | tolerant safe selector | 0.0000 | +0.0003 | [-0.0011, +0.0010] | [-0.0010, +0.0012] | uncertain_or_neutral |
| tls-120 | unified-slot stacker | +0.0082 | +0.0128 | [+0.0036, +0.0124] | [+0.0074, +0.0174] | helpful |
| tls-120 | guarded slot-stacker selector | +0.0036 | +0.0038 | [+0.0017, +0.0056] | [+0.0018, +0.0058] | helpful |
| tls-120 | soft expert gate | +0.0065 | +0.0073 | [+0.0040, +0.0091] | [+0.0048, +0.0098] | helpful |
| tls-120 | soft-gate calibrated selector | +0.0088 | +0.0100 | [+0.0060, +0.0117] | [+0.0071, +0.0134] | helpful |
| tls-120 | coverage-audited distill student fusion | -0.0472 | -0.0668 | [-0.0561, -0.0390] | [-0.0782, -0.0576] | harmful |
| tls-120 | coverage-audited distill student selector | +0.0063 | +0.0107 | [+0.0021, +0.0103] | [+0.0063, +0.0154] | helpful |

## Content-Unique Robustness

| Dataset | Original Acc/F1 | Content-Unique Acc/F1 | Delta Acc/F1 | Unique/Original Flows | Duplicate Groups | Unique Acc/F1 95% CI | Grouped Acc/F1 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| vpn-app | 0.7512/0.7522 | 0.7532/0.7570 | +0.0020/+0.0047 | 1645/1672 | 27 | [0.7325, 0.7745]/[0.7246, 0.7836] | [0.7298, 0.7729]/[0.7171, 0.7822] |
| tls-120 | 0.8461/0.8292 | 0.8461/0.8292 | 0.0000/0.0000 | 11542/11542 | 0 | [0.8397, 0.8525]/[0.8197, 0.8359] | [0.8397, 0.8525]/[0.8197, 0.8359] |

## Raw Best vs Paper-Safe Result

| Dataset | Raw Best Acc | Raw Best F1 | Paper-Safe Acc | Paper-Safe F1 | Raw-Paper Acc | Raw-Paper F1 | Same Path |
|---|---:|---:|---:|---:|---:|---:|---|
| vpn-app | 0.7512 | 0.7522 | 0.7512 | 0.7522 | 0.0000 | 0.0000 | True |
| tls-120 | 0.8475 | 0.8304 | 0.8461 | 0.8292 | +0.0014 | +0.0012 | False |

## Next-Step Recommendations

- vpn-app: Fresh flow-aware paired-view probes are also negative. Do not increase Tower-2 paired IP/port-randomization weight. Content-grouped robustness evidence is now the promotion gate; next prioritize coverage-audited consensus distillation into trainable graph/seq models. Keep native structural pretraining as a negative ablation unless its objective or gate is redesigned.
- tls-120: Paper-safe target is met; keep raw-best probes as ablations unless validation-gated selection and target-shift guards accept them into the framework result.

## Paper Positioning

Recommended claims:
- The method should be framed as a unified candidate-expert traffic classification framework with validation-gated safety controls.
- The strongest performance claim is supported on datasets whose point estimates and bootstrap lower bounds both pass target gates.
- Dataset-specific behavior should be described as automatic expert activation, gating, or identity fallback inside the same module family.

Risk controls:
- Use bootstrap and target-shift guards to prevent validation-favorable but target-unstable experts from overriding the base model.
- Report harmful expert candidates as negative ablations instead of hiding them; they motivate the gated-selector design.
- Separate strong performance claims from exploratory evidence when confidence intervals are wide.

Reviewer risks:
- Some datasets pass point targets but not bootstrap lower-bound targets; avoid overclaiming statistical dominance.
- Some datasets pass point targets but not content-grouped bootstrap lower-bound targets; present them as point-estimate gains until group-level stability improves.

Next experiments:
- Do not increase Tower-2 paired IP/port-randomization weight; fresh flow-aware paired-view probes are negative.
- Use content-grouped bootstrap lower bounds as the VPN promotion gate, and prioritize coverage-audited consensus distillation for group-level stability.
- Distill the cross-fold consensus back into trainable graph/seq models; keep native structural pretraining as a negative ablation until its objective or gate is redesigned.
- Keep per-packet-split datasets outside the flow-level main table unless a leakage-free per-flow split is released.
