# Paper Method Card

Method: **Unified Shortcut-Resistant Packet-to-Flow Framework**

Framework profile: `paper_unified`

Recommended main claim: Candidate unified shortcut-resistant packet-to-flow framework; historical packet-level scores and flow-level point gains remain non-headline evidence until strict_shared_core_v2 cross-task provenance, fixed-consensus, and CI gates pass.

Architecture rule: Every dataset and both tasks execute all shared representation modules; only learned parameters and gates differ. Supervised Packet-module weights are trained independently from each task's own split and are never transferred across tasks. Flow adds packet_to_flow_proj and aggregation after the shared packet representation.

## Scope

- Flow-level datasets: `vpn-app`, `tls-120`
- Packet-level datasets: `vpn-app`, `tls-120`
- Supplementary packet-only datasets (not cross-task evidence): `vpn-service`, `vpn-binary`, `ustc-app`, `ustc-binary`

## Shared Contract

Shared representation modules: `label_free_protocol_content_pretraining`, `field_aware_header_intervention`, `semantic_tower1_channel`, `current_packet_structural_encoder`, `shared_intervention_view_fusion`, `bounded_tri_channel_router`

Shared training/evaluation protocol: `per_flow_split_guard`, `content_group_empirical_risk`, `validation_only_selection`, `fixed_cross_fold_consensus`

Flow-only modules: `packet_to_window_flow_aggregator`, `flow_level_classifier`

Packet-only modules: `strict_current_packet_protocol`, `packet_level_classifier`

## Problems To Modules

| Reviewer-facing problem | Unified modules | Paper angle |
|---|---|---|
| Shortcut learning from endpoints, ports, and split artifacts | `per_flow_split_guard`, `field_aware_header_intervention` | Treat header intervention and split provenance as part of the model protocol, not cleanup. |
| Packet-level evidence does not automatically transfer to flow-level decisions | `label_free_protocol_content_pretraining`, `semantic_tower1_channel`, `current_packet_structural_encoder`, `packet_to_window_flow_aggregator` | Use a shared packet representation contract and a task-specific flow aggregator. |
| Semantic, packet-content, and structural evidence have split-dependent reliability | `shared_intervention_view_fusion`, `bounded_tri_channel_router` | Route every sample through the same three channels and learn bounded, data-dependent reliability weights. |
| Validation-selected recipes and duplicate content can inflate evidence | `content_group_empirical_risk`, `validation_only_selection`, `fixed_cross_fold_consensus` | Freeze one cross-dataset recipe on validation and report a fixed equal-weight consensus. |

## Contributions

1. **Field-aware shortcut intervention**: A paired factual/intervened view masks endpoint and port fields under the same policy for both tasks, so shortcut resistance is learned inside the shared representation rather than added at prediction time.
2. **Packet-to-flow dual representation contract**: Packet-level and flow-level classification reuse the same label-free protocol-content, Tower1 semantic, 13-dimensional packet-local structural encoders and bounded router under a strict current-packet input policy; supervised parameters are re-trained from each task's own training split, while sequence position and window context enter only through the Flow aggregator. The shared core may retain a direction cue inferred from the current packet alone, but never previous-packet IAT or whole-flow server-role inference.
3. **Counterfactual semantic-structural routing**: Factual and header-intervened semantic views are fused before one semantic-anchored bounded router combines semantic, protocol-content, and packet-local structural evidence for every dataset and both tasks.
4. **Validation-safe publication protocol**: One validation-frozen cross-dataset recipe, content-group empirical risk, fixed equal-weight log-mean, flow-cluster bootstrap, executed-policy/checkpoint audits, and reporting-only train-signature novelty strata distinguish publishable evidence from exploratory probes.

## Flow Evidence

| Dataset | Acc | Macro-F1 | Target | Point | CI | Group CI | Claim |
|---|---:|---:|---|---|---|---|---|
| vpn-app | 0.7512 | 0.7522 | 0.7400/0.6500 | True | False | False | point_pass_ci_mixed |
| tls-120 | 0.8461 | 0.8292 | 0.7800/0.7000 | True | True | True | strong |

## Packet Evidence

| Dataset | Acc | Macro-F1 | Publication | Path |
|---|---:|---:|---|---|
| vpn-app | 0.9066 | 0.8112 | pass | `reasoningDataset/packet-level/vpn-app/paper_default_result.json` |
| tls-120 | 0.8744 | 0.8479 | pass | `reasoningDataset/packet-level/tls-120/paper_default_result.json` |

## Supplementary Packet-Only Evidence

These datasets do not establish the unified Packet-to-Flow claim because a protocol-matched Flow task is not in scope.

| Dataset | Acc | Macro-F1 | Publication | Path |
|---|---:|---:|---|---|
| vpn-service | 0.9512 | 0.9435 | pass | `reasoningDataset/packet-level/vpn-service/paper_default_result.json` |
| vpn-binary | 0.9999 | 0.9999 | pass | `reasoningDataset/packet-level/vpn-binary/paper_default_result.json` |
| ustc-app | 0.9773 | 0.9849 | pass | `reasoningDataset/packet-level/ustc-app/paper_default_result.json` |
| ustc-binary | 1.0000 | 1.0000 | pass | `reasoningDataset/packet-level/ustc-binary/paper_default_result.json` |

## Ablation Positioning

- Total ablations: `13`
- Helpful: `5`
- Harmful: `3`
- Neutral or uncertain: `5`

Candidate modules remain controlled ablations unless they pass the same pre-registered paper_unified promotion rule on every in-scope dataset; harmful candidates motivate the shared guards rather than becoming dataset-specific branches.

## Content-Group Candidate Scan

| Dataset | Candidates | Best Acc | Best Macro-F1 | Best Group Acc/F1 Lower | Target | Best Path |
|---|---:|---:|---:|---:|---|---|
| vpn-app | 343/411 | 0.7512 | 0.7522 | 0.7318/0.7103 | False | `reasoningDataset/vpn-app/test_crossfold_consensus_auto_confidence.json` |
| tls-120 | 55/63 | 0.8475 | 0.8304 | 0.8401/0.8227 | True | `reasoningDataset/tls-120/test_crossfold_consensus_rgssi_auto_confidence.json` |

## Readiness

| Gate | Status |
|---|---|
| legacy protocol audit | False |
| strict shared-core v2 provenance | False |
| durable session-novelty evidence | False |
| SWEET end-to-end exceeded on all four tasks | False |
| paper_unified profile | False |
| flow point targets | True |
| flow bootstrap CI targets | False |
| flow content-grouped CI targets | False |
| packet publication defaults | True |
| CCF-A risk level | high |

## Reviewer Risks

- VPN/TLS packet/flow canonical results do not yet share complete strict_shared_core_v2 publication provenance.
- At least one current VPN/TLS task result does not exceed the protocol-matched SWEET end-to-end accuracy and macro-F1 pair.
- At least one flow dataset misses the content-grouped bootstrap lower-bound gate.
- At least one flow dataset passes point targets but not the ordinary bootstrap lower-bound gate.
- The legacy unified/defaults protocol audit is not fully passing.

## Next Paper-Grade Action

Complete the frozen exact-v2 VPN/TLS packet/flow cross-fold matrix, checkpoint-schema audits, content-group bootstrap, and matched ablations. Do not add another expert unless the same module is pre-registered for every dataset and both tasks.
