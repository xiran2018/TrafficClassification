# Paper Framework Defaults Audit

Overall status: `False`

Default framework profile: `paper_unified`

Shared core modules:

- `label_free_protocol_content_pretraining`
- `field_aware_header_intervention`
- `semantic_tower1_channel`
- `current_packet_structural_encoder`
- `shared_intervention_view_fusion`
- `bounded_tri_channel_router`
- `per_flow_split_guard`
- `content_group_empirical_risk`
- `validation_only_selection`
- `fixed_cross_fold_consensus`

Unified framework gate:

| Status | Shared Core Match | paper_unified Flow Manifests | paper_unified Packet Manifests |
|---|---|---:|---:|
| review | True | 0 | 0 |

## Flow-Level Defaults

| Dataset | Exists | Acc | Macro-F1 | Target | Target Met | Slot Mode | Slots Match | Errors |
|---|---|---:|---:|---|---|---|---|---|
| vpn-app | True | 0.7512 | 0.7522 | 0.7500/0.6500 | True | crossfold_consensus_identity_compatible | True | - |
| tls-120 | True | 0.8461 | 0.8292 | 0.7800/0.7000 | True | crossfold_consensus_identity_compatible | True | - |

## Packet-Level Defaults

| Dataset | Exists | Acc | Macro-F1 | Publication Status | Provenance | Path |
|---|---|---:|---:|---|---:|---|
| vpn-app | True | 0.9066 | 0.8112 | needs_paper_unified_repro | 0 | `reasoningDataset/packet-level/vpn-app/paper_default_result.json` |
| tls-120 | True | 0.8744 | 0.8479 | needs_paper_unified_repro | 0 | `reasoningDataset/packet-level/tls-120/paper_default_result.json` |
| vpn-service | True | 0.9512 | 0.9435 | needs_paper_unified_repro | 0 | `reasoningDataset/packet-level/vpn-service/paper_default_result.json` |
| vpn-binary | True | 0.9999 | 0.9999 | needs_paper_unified_repro | 0 | `reasoningDataset/packet-level/vpn-binary/paper_default_result.json` |
| ustc-app | True | 0.9773 | 0.9849 | needs_paper_unified_repro | 0 | `reasoningDataset/packet-level/ustc-app/paper_default_result.json` |
| ustc-binary | True | 1.0000 | 1.0000 | needs_paper_unified_repro | 0 | `reasoningDataset/packet-level/ustc-binary/paper_default_result.json` |
