# Unified Framework Audit

Status: `review`
Exact common-reference v2: `not_ready`
Unified method v2: `not_ready`

Expected profile fingerprints:

- flow-level: `4e2a352938050900210e82ea7b94f03f21eac9eaa84d493eb697b3a561bb2548`
- packet-level: `d5dcbd2a487d8233947ab31a7a6c7e29d2d659f91f57be3a57d244b72680cf86`

## Shared Representation

- `label_free_protocol_content_pretraining`
- `field_aware_header_intervention`
- `semantic_tower1_channel`
- `current_packet_structural_encoder`
- `shared_intervention_view_fusion`
- `bounded_tri_channel_router`

## Shared Protocol Guards

- `per_flow_split_guard`
- `content_group_empirical_risk`
- `validation_only_selection`
- `fixed_cross_fold_consensus`

## Unified Candidate Experts

- `packet_tree_feature_expert`
- `packet_probability_fusion_expert`
- `graph_flow_expert`
- `multi_view_flow_expert`
- `flow_statistics_expert`
- `label_free_prior_calibration_expert`
- `embedding_space_experts`
- `paired_view_expert`
- `cross_fold_consensus_expert`

## Ablation-Only Modules

- `confidence_penalty`
- `vpn_specific_hierarchical_coarse_head`
- `vpn_specific_confusion_groups`
- `unconstrained_probability_stacker`
- `unsafe_target_tuned_prior`
- `manual_single_split_threshold_routing`
- `residual_fusion_grid_search`
- `slot_stacker_expert`
- `validation_probability_selector`
- `label_prior_residual_expert`

## Flow-Level Results

| Dataset | Accuracy | Macro-F1 | Metric Status | Publication Status | Canonical Path | Provenance | Path |
| --- | ---: | ---: | --- | --- | --- | ---: | --- |
| vpn-app | 0.7511961722488039 | 0.7522269064093092 | pass | needs_paper_unified_repro | True | 0 | `reasoningDataset/vpn-app/test_crossfold_consensus_auto_confidence.json` |
| tls-120 | 0.8461271876624502 | 0.8291911492056544 | pass | needs_paper_unified_repro | True | 0 | `reasoningDataset/tls-120/test_crossfold_consensus_auto_confidence.json` |

## Packet-Level Results

| Dataset | Accuracy | Macro-F1 | Metric Status | Publication Status | Canonical Path | Provenance | Path |
| --- | ---: | ---: | --- | --- | --- | ---: | --- |
| vpn-app | 0.9066244022994682 | 0.8112461556528934 | pass | needs_paper_unified_repro | True | 0 | `reasoningDataset/packet-level/vpn-app/paper_default_result.json` |
| tls-120 | 0.8744011667996404 | 0.8479274094708068 | pass | needs_paper_unified_repro | True | 0 | `reasoningDataset/packet-level/tls-120/paper_default_result.json` |
| vpn-service | 0.9512068098556138 | 0.9434656797061393 | pass | needs_paper_unified_repro | True | 0 | `reasoningDataset/packet-level/vpn-service/paper_default_result.json` |
| vpn-binary | 0.9999186212633597 | 0.9999186068245844 | pass | needs_paper_unified_repro | True | 0 | `reasoningDataset/packet-level/vpn-binary/paper_default_result.json` |
| ustc-app | 0.9773248211171217 | 0.9848701958816448 | pass | needs_paper_unified_repro | True | 0 | `reasoningDataset/packet-level/ustc-app/paper_default_result.json` |
| ustc-binary | 1.0 | 1.0 | pass | needs_paper_unified_repro | True | 0 | `reasoningDataset/packet-level/ustc-binary/paper_default_result.json` |

## Runner Manifests

| Task | Manifests | Passing | paper_unified Passing |
| --- | ---: | ---: | ---: |
| flow_level | 68 | 0 | 0 |
| packet_level | 24 | 0 | 0 |
