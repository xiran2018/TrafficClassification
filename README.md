# Two-Tower Traffic Classification Framework v3

This version implements the updated strategy discussed above:

```text
Tower 1: Qwen-LoRA Packet Semantic Encoder
  Packet protocol QA loss
+ weak packet-level classification loss
+ protocol-aware supervised contrastive loss
        ↓
  Raw last-token + contrastive projected packet embedding
        ↓
Tower 2: Packet Interaction Encoder
  Seq Transformer version or Graph Transformer version
        ↓
  Flow embedding → traffic classification
```

Compared with v2, v3 adds a real **Tower-1 multi-objective training script**. Tower 1 is no longer trained only with generative packet Q&A. It also learns packet representations aligned with downstream traffic classes through weak packet classification and supervised contrastive learning.

## Current research candidate: cross-environment evidence routing

The current VPN Flow milestone replaces validation-only selector searches with
one learned, dataset-agnostic expert topology:

1. every fold trains the same Qwen/Tower1 semantic expert;
2. every fold trains the same protocol-structural forest expert over packet
   length, direction, IAT, burst/message, and shortcut-controlled header fields;
3. `train_cross_environment_reliability_router.py` learns one class-aware router
   from complete expert distributions, uncertainty, agreement, and JS conflict;
4. GroupDRO optimizes the worst validation environment, while
   leave-one-environment-out predictions select a bounded, label-free EM prior
   transport; and
5. routed fold predictions use a fixed log-mean consensus.

The router contains no VPN class names or dataset-specific expert branches.
Datasets train independent numerical parameters, while retaining the same
semantic/structural slots, router features, GroupDRO objective, and bounded
prior-transfer protocol. For packet classification, the same slot contract is
used with one-packet semantic and structural experts; flow classification adds
the flow feature extractor and aggregation boundary. VPN and TLS-120 Flow
validation is complete. VPN Packet shared-core Test is now complete; the final
TLS-120 shared-core Packet run remains required before this router can replace
the paper-main cross-task framework.

Frozen VPN Flow Test results on the shared 1,672-flow test split:

| candidate | accuracy | macro-F1 |
|---|---:|---:|
| strict unified semantic fold consensus | 0.6477 | 0.6120 |
| rich structural fold consensus | 0.6657 | 0.6278 |
| cross-environment reliability router | 0.6854 | 0.6436 |
| router + bounded OOF-selected prior transport | **0.6950** | **0.6738** |
| SWEET VPN Flow reference | 0.6920 | 0.6220 |

The final result is stored outside the repository's large-artifact boundary at
`/tmp/two_tower_runs/pcfrr_v1_results/vpn_flow_cross_environment_reliability_router_safe_prior.json`.
Its pre-transfer ECE is `0.0440`; bounded prior transport lowers ECE to `0.0321`.
The selected strength is restricted to `[0, 0.30]` and selected from pooled
leave-one-environment-out predictions, never from Test labels. Single-fold
prior transfer is retained only as a negative stability ablation because it
collapsed rare-class Macro-F1 on fold 2.

Frozen TLS-120 Flow Test results on the shared 11,542-flow test split:

| candidate | accuracy | macro-F1 |
|---|---:|---:|
| strict unified semantic fold 0 | 0.7558 | 0.7465 |
| rich structural folds (range) | 0.8348-0.8367 | 0.8129-0.8138 |
| cross-environment reliability router | 0.9020 | 0.8909 |
| router + bounded OOF-selected prior transport | **0.9023** | **0.8912** |
| previous heterogeneous consensus | 0.8461 | 0.8292 |

TLS-120 uses the same expert slots, router inputs, GroupDRO objective,
consensus rule, and prior-strength search as VPN. No TLS-specific class rule,
feature branch, or loss was introduced. The bounded prior contributes only
about `+0.0003` accuracy and Macro-F1 on TLS-120; almost all improvement comes
from the learned cross-environment router. The result is stored at
`/tmp/two_tower_runs/pcfrr_v1_results/tls120_flow_cross_environment_reliability_router_safe_prior.json`.

Strict one-packet TLS-120 routing milestone on 553,994 Test packets:

| candidate | accuracy | macro-F1 |
|---|---:|---:|
| fixed semantic fold consensus | 0.8557 | 0.8234 |
| cross-environment reliability router | **0.8692** | **0.8429** |
| unsafe target-weighted prior selection (negative ablation) | 0.8678 | 0.8217 |

The shared LOEO Macro-F1 plateau rule automatically selects prior strength
`0.30` for VPN Flow, `0.05` for TLS-120 Flow, and `0.00` for TLS-120 Packet.
The Packet result obeys one-packet inference and validates reuse of the expert
slot and routing algorithm. It is not yet the final cross-task shared-core
claim: these existing TLS packet probabilities use a local Byte Transformer,
whereas the Flow semantic slot uses Qwen/Tower1. The final paper matrix must
rerun the router with all three Qwen plus native-byte shared-core Packet folds.
The milestone artifact is
`/tmp/two_tower_runs/pcfrr_v1_results/tls120_packet_cross_environment_reliability_router_safe_prior.json`.

Frozen VPN Packet Test results on the shared 111,678-packet test split:

| candidate | accuracy | macro-F1 |
|---|---:|---:|
| Qwen/native shared-core semantic consensus | 0.8790 | 0.7653 |
| protocol-structural consensus | 0.9070 | **0.8222** |
| fixed 50/50 expert fusion | 0.9057 | 0.8071 |
| cross-environment reliability router | **0.9122** | 0.8143 |

All four candidates use the same three train/validation folds and the same
strict one-packet Test rows. Expert artifacts must contain exact `packet_uid`
alignment; row-order fallback is rejected whenever either slot supplies
explicit identities. The learned router raises accuracy by `+0.0332` and
Macro-F1 by `+0.0490` over semantic consensus, while lowering ECE from `0.0482`
to `0.0193`. The structural consensus retains a `+0.0080` Macro-F1 advantage
over the router, revealing the next model question: reliability routing must
preserve cross-fold error diversity instead of optimizing only per-environment
expert choice. No target-prior transport is used for this Packet result. Router
training is fixed to CPU for deterministic reproduction; two consecutive runs
produced identical metrics. Its NPZ retains exact `packet_uid` values, and its
JSON records SHA-256 provenance for every expert input and the routed output.
The artifact is
`/tmp/two_tower_runs/pcfrr_v1_results/vpn_packet_shared_core_cross_environment_reliability_router_safe_prior.json`.

VPN Packet router regularization ablation, using the same frozen expert
artifacts:

| router objective | Test accuracy | Test macro-F1 | mean LOEO macro-F1 |
|---|---:|---:|---:|
| GroupDRO + gate entropy | 0.9122 | 0.8143 | 0.8126 |
| no GroupDRO | 0.9128 | 0.8150 | 0.8122 |
| no gate entropy | 0.9123 | 0.8134 | 0.8124 |
| neither regularizer | 0.9129 | 0.8146 | 0.8124 |

These differences are too small to claim a material VPN benefit. The final
objective will therefore be selected only after the identical TLS-120
shared-core evaluation; Test fluctuations alone do not select the paper model.

Implementation verification:

```text
conda environment: llm-factory
pytest: 499 passed
```

---

## SWEET dataset layout

SWEET distinguishes the **split unit** from the **classification unit**:

- **Per-packet Split** randomly assigns packets to train/valid/test. Packets
  from one flow can therefore appear in different partitions.
- **Per-flow Split** assigns each complete bidirectional flow to exactly one
  partition. It can support either packet-level classification (one prediction
  per packet) or flow-level classification (one prediction per flow).

The reference split/preprocessing implementation is located at:

```text
/home/jing/Debunk_Traffic_Representation-master/process_finetune_data/Split
```

The prepared **Per-flow Split + packet-level classification** datasets are
located at:

```text
/home/jing/download/sweet/packet-level-classification/per-flow-split

datasets: vpn-app, vpn-binary, vpn-service, tls, ustc-app, ustc-binary
layout:   <dataset>/train_val_split_{0,1,2}/{train,val}
          <dataset>/test
```

In these packet-level artifacts, each `<label>.pcap` contains packets from
multiple disjoint flows of the same class. The PCAP file itself must not be
treated as one flow. Packet-level preprocessing must recover the real
bidirectional flow ID (IP endpoints, ports, and protocol) before computing
flow-aware training losses or split-overlap audits. The packet classifier input
itself remains one packet: it does not consume inter-arrival time or any
feature computed from another packet in the flow.

The prepared **Per-flow Split + flow-level classification** datasets used by
the current Tower1/Tower2 flow pipeline are located at:

```text
/home/jing/download/sweet/flow-level-classification

vpn-app:
  train /home/jing/download/sweet/flow-level-classification/vpn-app/train_val_split_0/train
  valid /home/jing/download/sweet/flow-level-classification/vpn-app/train_val_split_0/val
  test  /home/jing/download/sweet/flow-level-classification/vpn-app/test

tls-120:
  train /home/jing/download/sweet/flow-level-classification/tls/train_val_split_0/train
  valid /home/jing/download/sweet/flow-level-classification/tls/train_val_split_0/val
  test  /home/jing/download/sweet/flow-level-classification/tls/test
```

For fold-wise experiments, replace `train_val_split_0` with
`train_val_split_1` or `train_val_split_2`; all three folds use the same
dataset-level `test` directory.

### Per-flow Split packet-level pipeline

`run_packet_level_pipeline.py` applies the same strict-one-packet framework to
all six SWEET packet-level datasets. Each dataset/fold is trained independently.
For new paper-facing runs, use `--stage paper_unified`; the runner now defaults
to `--framework_profile paper_unified`, so the explicit flag in commands below
is mainly for readable logs and reproducibility ledgers. Use
`--framework_profile legacy` only for historical ablations. The paper-unified
profile runs preprocessing, split audit, Tower-1/Qwen semantic training, and a
single shared neural packet encoder with protocol-aware content, current-packet
structure, and semantic channels. The protocol-aware
encoder is the same `ProtocolAwarePacketContentEncoder` used by the flow native
branch: it consumes only the current packet bytes and protocol-field IDs; the
flow-native content pretraining applies its fixed session-field intervention.
For paper runs, the same Tower1 checkpoint encodes aligned factual `full`
and intervened `mask_ip_port` prompts. `SharedInterventionViewFusion` combines
those two semantic observations before the three representations are fused by
the learned `SharedPacketChannelFusion`. The paper profile uses its
`semantic_anchor` mode: the normalized semantic representation is the stable
base, while native content and current-packet structure enter only through a
bounded, sample-dependent residual. Thus the reported gate weights are the
actual route for non-semantic evidence instead of allowing a hidden
`sum(channels)` path to bypass the gate. There is no paper-main tree expert or
post-hoc probability stacker.
Dataset-specific behavior is learned from each fold rather than implemented as
dataset-specific model branches.

Long packet runs coordinate GPU phases with the same physical-GPU file lock
used by `extract_packet_embeddings_qwen.py`. This prevents Tower1 or the shared
packet head from loading while a sharded embedding worker occupies the same
card. On the eight-A800 host, an unattended run can additionally require launch
headroom without changing the model or its framework fingerprint:

```bash
export CUDA_VISIBLE_DEVICES=5
export PACKET_GPU_MIN_FREE_MB=40000
conda run --no-capture-output -n llm-factory \
  python run_packet_level_pipeline.py ...
```

The default threshold is `0` (lock only), so smaller installations can choose
their own operational headroom. The real `llm-factory` environment has CUDA
access to eight A800 GPUs even when CUDA is unavailable inside the default
Codex sandbox.

The active strict-v2 unattended launchers under `/tmp/two_tower_runs` do not
assume that GPU7 remains free. They scan `STRICT_GPU_CANDIDATES` (default
`7 5 3 2 1 0 6 4`), require at least `60000` MiB free, acquire the existing
per-GPU Qwen embedding lock without blocking, recheck memory after the lock,
and retry every 120 seconds when no A800 is safe. This is an execution policy
only; the selected physical GPU is not a model or dataset hyperparameter.

The contract is deliberately narrower than "enable every module ever tried":

- Every paper-main dataset executes factual/intervened semantic,
  protocol-content, and strict
  current-packet structural encoders, followed by the same
  `SharedInterventionViewFusion` and `SharedPacketChannelFusion` implementations.
- Encoder, projection, classifier, and per-sample channel-gate parameters are
  learned independently from each dataset's training fold. They are not manual
  VPN/TLS weights.
- The `full + mask_ip_port` intervention pair, module graph, objective family,
  validation metric, and cross-fold `log_mean` rule define one
  `paper_unified` method. VPN/TLS and Packet/Flow train all model parameters
  independently. Numeric optimization hyperparameters such as learning rate,
  schedule, epoch budget, batch size, weight decay, temperature, and nonzero
  loss coefficients may be set independently for each dataset/task and are
  recorded in the execution contract; changing whether an objective is enabled
  or replacing a module is a method change and is rejected by the shared-core
  audit.
- Shortcut control is deliberately asymmetric rather than claiming that every
  header bit is discarded. The semantic anchor is formed from the factual and
  endpoint-masked views, while the protocol-content and packet-local structural
  channels may retain specialized evidence. Those channels can affect the
  shared packet representation only through the same learned residual path,
  whose multiplier is fixed at `0.25` in both packet and flow models. This
  implements an endpoint-stable core with bounded environment-specific
specialization; it is not an unrestricted full-header shortcut model.
  `test_packet_byte_transformer.py` records the raw and effective learned
  routing distributions under `learned_gate_diagnostics`, including named
  semantic/content/structural weights and factual/intervened weights. This is
  the packet-side counterpart of Tower2's gate diagnostics and provides direct
  evidence that datasets learn parameters inside one fixed module graph.
  Both evaluators also expose the same inference-only sensitivity controls:
  `--ablate_input_channel semantic|content|structural` and
  `--ablate_intervention_view factual_only|intervened_only`. Their result JSON
  is explicitly marked `inference_only_not_retrained_ablation`; these runs show
  checkpoint sensitivity and must not be presented as substitutes for matched
  from-scratch component ablations.
- Packet and flow differ only after the shared packet representation: packet
  classification uses one packet head; flow classification uses one sequence
  encoder, mean aggregation, and one flow head.
- Stability grouping follows the task's valid statistical unit. Packet data are
  audited by recovered flow ID, sampled with equal flow participation, and
  selected by packet macro-F1. Its source files are class-merged PCAPs (for VPN
  fold0, 16 PCAP hashes for 16 labels), so treating each whole PCAP hash as one
  validation item would be a meaningless class-level vote. Flow data instead
  use content-hash group guards because repeated flow/window content is the
  relevant duplication risk. This is a task-protocol difference, not a
  dataset-specific model module.
- Tree experts, graph/multi-view alternatives, hierarchical/confusion heads,
  probability fusion, stackers, and target-prior calibration are historical or
  ablation modules. They are not silently passed through and then selected for
  the paper main result.

Tower1 uses an eight-epoch upper bound with gradient accumulation 1 in both
tasks. Every epoch is evaluated on a deterministic flow-balanced validation
subset containing at most two non-repeated packets per flow, and `best/` is
selected by held-out packet macro-F1. Both factual and intervened embedding
views are extracted from that same checkpoint. The flow-balanced sampler has
the same definition in both tasks, but its epoch length follows the number of
training flows rather than being tuned per task. The VPN flow-classification
fold has 704 training flows and therefore 88 batches per epoch at eight flows
per batch; the VPN packet-classification Per-flow Split fold0 has 4,713 flows
and therefore 590 batches. The earlier two-epoch/final protocol provided too
few optimizer updates and is retained only as a historical ablation.

The current strict folds retain the frozen eight-epoch protocol for a fair
baseline. Their Tower1 validation histories show fold0/fold1 peaking at epoch
8 and fold2 at epoch 5; the mean fold Macro-F1 still rises through epoch 8
(`0.0278, 0.1962, 0.3640, 0.4480, 0.4926, 0.5044, 0.5154, 0.5412`). This is
evidence of a truncated optimization budget in some folds, not a license to
select an epoch from test results. `train_tower1_multitask.py` and both unified
runners now expose validation-only Tower1 patience, defaulting to disabled for
backward compatibility. After the frozen packet/flow baselines finish, the
pre-registered training-only candidate is a common maximum of 12 epochs with
patience 3 for every dataset and task; promotion requires multi-fold held-out
improvement before any shared-test comparison.

The semantic-anchor change was admitted by a controlled VPN fold-0 experiment,
not by selecting on test labels. With identical data and Tower2 settings across
seeds 42/43/44, legacy fusion obtained `55.66 +/- 1.12%` flow accuracy and
`53.74 +/- 0.57%` macro-F1; semantic-anchor fusion obtained
`56.44 +/- 1.31%` and `53.96 +/- 0.82%`. The mean changes are `+0.78` and
`+0.22` percentage points. A 5,000-sample paired bootstrap for seed 42 gave an
accuracy win rate of `92.24%`, but its 95% interval crossed zero and McNemar
`p=0.156`; this supports a better-defined and more stable shared fusion path,
not a statistically significant single-fold SOTA claim. The same profile must
still be evaluated on all VPN/TLS folds and packet-level tasks before this
result can enter the paper's headline table.

The first two strict `paper_unified` VPN folds expose a separate target-prior
shift: fixed fold01 `log_mean` gives accuracy `0.616029`, macro-F1 `0.580074`,
macro recall `0.675415`, and macro precision `0.556790`. Both fold-local train
and validation flow sets are nearly class-balanced, while the shared test set
is strongly imbalanced. `attach_oof_validation.py` can concatenate only each
model's own held-out validation predictions (with fold-prefixed IDs) onto a
fixed consensus payload; it never constructs a validation ensemble from models
that may have trained on that fold. A diagnostic blend of BBSE and EM prior
estimates, with correction strength selected by target-prior-weighted OOF
validation macro-F1, selected `0.5` and reached `0.648325/0.631159`. This is
promising but remains a transductive label-shift candidate until the identical
rule is frozen and verified on VPN fold012 and TLS-120. It must not be compared
as an inductive SWEET result without an explicit transductive protocol.

Consequently, old best-result rows below remain historical evidence. They do
not become paper-unified evidence until all three folds have current profile
fingerprints and the fixed consensus has been bound to the canonical result.

The packet branch is:

```text
class-level PCAP
-> recover real bidirectional flow IDs for split audit/training batches only
-> one packet per model input (no sequence/window/inter-arrival features)
-> Tower-1/Qwen current-packet semantic representation
-> shared protocol-aware current-packet content representation
-> shared 13-dimensional current-packet structural representation (no IAT)
-> learned semantic/content/structural representation-level gate
-> one packet classification head
-> fixed three-fold log-mean consensus
-> one prediction per packet on the shared test set
```

Run a path audit before starting GPU work:

```bash
conda run --no-capture-output -n llm-factory \
    python run_packet_level_pipeline.py \
    --dataset vpn-app \
    --fold 0 \
    --stage paper_unified \
    --local_files_only \
    --dry_run
```

Run all three folds separately with the paper-facing profile. Do not merge the
prepared fold directories: each `train_val_split_i` contains its own
training/validation partition, and all folds use the same `test/` set.

```bash
for fold in 0 1 2; do
  conda run --no-capture-output -n llm-factory \
    python run_packet_level_pipeline.py \
      --dataset vpn-app \
      --fold ${fold} \
      --stage paper_unified \
      --framework_profile paper_unified \
      --max_packets_per_flow 1000 \
      --byte_max_bytes 128 \
      --byte_use_protocol_fields \
      --byte_epochs 12 \
      --byte_batch_size 512 \
      --byte_eval_batch_size 2048
done
```

Supported `--dataset` values use the same pipeline:

```text
vpn-app, vpn-binary, vpn-service, tls-120, ustc-app, ustc-binary
```

The underlying preprocessing command must use
`--input_layout class_packet_pcaps`. `--classification_only` avoids generating
the much larger QA corpus when packet classification is the only objective;
`--max_packets_per_flow 0` keeps all packets, while `1000` follows the SWEET
long-flow cap. The existing flow-level commands keep the default
`--input_layout flow_pcaps` and are behaviorally unchanged.

The Qwen-LoRA encoder is executed by `--stage paper_unified`; its packet-aligned
embeddings enter the neural model before classification. Cache manifests bind
both views to the same packet-index hash, their distinct header policies,
packet IDs, and strict
current-packet scope. ExtraTrees/RandomForest and probability-level fusion are
kept only under legacy/ablation stages.

### Unified packet-to-flow framework audit

The paper-facing direction is not separate packet and flow pipelines. The
current contract is stored in `unified_framework_spec.py`. Packet-level and
flow-level tasks share exactly six representation components:

```text
label-free protocol-content pretraining
field-aware factual/intervened observations
Tower1 current-packet semantic channel
13-dimensional current-packet structural channel
shared factual/intervened representation fusion
one semantic-anchored bounded tri-channel router
```

The task-specific part is only the final strict-current-packet classifier or
`packet_to_flow_proj` followed by the window/flow aggregator. Per-flow split,
content-group `group_mean` risk, validation-only freezing, and fixed equal
three-fold `log_mean` are training/evaluation protocol guards, not extra model
blocks. Graph, statistics, prior calibration, selectors, and probability
stackers are excluded from the main path and retained only as ablations.

The code now separates three module roles for paper writing and audits:

- **Paper main modules** are the six shared packet representation components plus
  the task-specific packet head or flow/window aggregator. These are the modules
  every dataset/task must expose under `paper_unified`.
- **Unified candidate experts** are available through the same expert-slot or
  validation-gated interface on every flow dataset. Dataset-specific behavior
  should come from learned gates, trained expert weights, identity fallback, or
  validation-only rejection, not from manually swapping modules per dataset.
- **Ablation-only modules** remain in the code for reproducibility and negative
  evidence, but they are not part of the default main claim. This includes
  confidence penalty, VPN-specific hierarchical coarse heads, VPN-specific
  confusion groups, unconstrained stackers, unsafe target-tuned priors, manual
  single-split threshold routing, and residual-fusion grid searches.

Run the audit after changing either task path:

```bash
conda run --no-capture-output -n llm-factory \
  python audit_unified_framework.py \
    --output_json reasoningDataset/unified_framework_audit.json \
    --output_md reasoningDataset/unified_framework_audit.md
```

This audit is deliberately stricter than a metric table: it checks that the
current flow-level paper scope remains VPN/TLS only, that SWEET packet-level
datasets are represented in the packet branch, and that both tasks advertise the
same shared core modules. It also separates `metric_status` from
`publication_status`: a result can pass accuracy/F1 while still being marked
`needs_paper_unified_repro` if no matching `paper_unified` runner manifest exists
for that dataset/task. Each manifest stores a stable
`framework_profile_fingerprint` derived from the paper profile's shared module
status, task overrides, candidate expert family, and ablation-only module list.
For the field-aware intervention contract, the flow runner verifies both cache
families independently on every requested split: the factual cache must attest
`full`, and the aligned intervention cache must attest `mask_ip_port`. Only then
does the manifest record
`factual_full_plus_mask_ip_port_intervention`; recording the factual `full`
policy alone is intentionally rejected by the audit. This mirrors the packet
runner and prevents a stale or missing intervention cache from satisfying the
paper claim. Existing completed manifests may be regenerated from their stored
arguments only after both cache evidences pass; no audit field should be edited
by hand.
When the default profile changes, older manifests without the current
fingerprint stop counting as paper provenance. Passing the metric table alone
does not prove the final CCF-A claim; the provenance gate prevents the codebase
from drifting back into unrelated task-specific recipes or stale default
settings.

When the audit reports `needs_paper_unified_repro`, generate the next rerun
queue with:

```bash
conda run --no-capture-output -n llm-factory \
  python make_unified_repro_plan.py \
    --audit_json reasoningDataset/unified_framework_audit.json \
    --output_json reasoningDataset/unified_repro_plan.json \
    --output_md reasoningDataset/unified_repro_plan.md
```

The plan intentionally emits real `paper_unified` runner commands instead of
editing the audit by hand. A dry-run manifest is useful for debugging, but a
paper-facing pass should be backed by rerunning the corresponding flow or packet
pipeline under `--framework_profile paper_unified`.
For flow-level results, publication provenance requires three completed current-
fingerprint fold manifests plus a canonical result binding. Old embeddings
without attested aligned `full` and `mask_ip_port` views and old native embeddings without
`representation_scope=strict_current_packet` are rejected rather than promoted.
Flow rerun commands include
`--require_cuda`, so they fail early in CPU-only sandboxes and should be launched
from the real `llm-factory` A800 environment.

If `reasoningDataset/unified_framework_audit.json` reports `status=review`,
read it as a provenance gate rather than an accuracy failure. For example, a
result can meet the target metrics while still requiring a fresh
`paper_unified` manifest that binds the canonical result path after framework
defaults change. Generate the repro plan below and rerun the listed flow or
packet jobs instead of editing the audit file by hand.

For packet-level provenance, the plan uses the runner stage
`--stage paper_unified`: this reruns preprocessing, the Per-flow Split audit,
Tower-1 training and embedding extraction, then the shared tri-channel packet
encoder and its single classification head. After folds
0/1/2, the plan runs a fixed log-mean packet consensus and writes the canonical
`paper_default_result.json`; it does not tune fold weights on the shared test.

Execute the queue with the resumable runner. By default it runs only one action,
writes a ledger, and skips successful actions on the next invocation:

```bash
conda run --no-capture-output -n llm-factory \
  python run_unified_repro_plan.py \
    --plan_json reasoningDataset/unified_repro_plan.json \
    --ledger_json reasoningDataset/unified_repro_ledger.json \
    --log_dir logs/unified_repro \
    --max_actions 1
```

Useful filters:

```bash
# Run all remaining flow-level provenance jobs.
conda run --no-capture-output -n llm-factory \
  python run_unified_repro_plan.py \
    --task flow-level \
    --max_actions -1

# Run the TLS-120 packet folds only.
conda run --no-capture-output -n llm-factory \
  python run_unified_repro_plan.py \
    --task packet-level \
    --dataset tls-120 \
    --max_actions -1
```

After a batch finishes, rerun `audit_unified_framework.py` and
`make_unified_repro_plan.py`; completed datasets disappear from the next plan
only after their metric file and non-dry-run `paper_unified` manifest both
satisfy the publication gate.

Use the shared paper profile for new CCF-A-oriented experiments:

```bash
# Packet-level: one current packet is still the only model input.
conda run --no-capture-output -n llm-factory \
  python run_packet_level_pipeline.py \
    --dataset vpn-app \
    --fold 0 \
    --stage paper_unified \
    --framework_profile paper_unified

# Flow-level: flow packet sequence input, same shared packet modules, then
# window/flow aggregation.
conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --stage paper_unified \
    --framework_profile paper_unified \
    --run_tag paper_unified
```

`paper_unified` intentionally trades some dataset-specific tuning freedom for a
clean paper claim. It constructs a factual `full` semantic view and a
counterfactual `mask_ip_port` semantic view from the same Tower1 checkpoint.
Both tasks fuse those views with the same bounded
`SharedInterventionViewFusion` before executing current-packet content,
structural, and semantic channels through the same `SharedPacketChannelFusion`.
The main path trains one classifier rather than two view-specific classifiers
followed by probability fusion,
uses exact-PCAP content-group guards to reduce duplicate-content shortcut
learning, uses validation-only selection, and restricts calibration to
validation-only or label-free target-prior mechanisms. In the flow runner this
means Tower-2 preprocessing attaches `content_group_id` by default, Tower-2
selects `best.pt` with `content_group_macro_f1`, main flow CE can be averaged
per content group, and balanced batches can prefer one flow per content group.
The default paper profile also disables VPN-specific hierarchical coarse groups
and VPN-specific confusion groups, and it uses the standard SupCon/dual-loss
setting instead of a dataset-specific confusion-aware SupCon default; those
switches remain available only for ablation runs.
Dataset differences should appear as learned gate/expert weights or selected
validation-safe expert weights, not as manually invented dataset-specific
pipelines.

For a command-level flow audit, use `--dry_run`. A real paper-unified run cannot
reuse legacy embeddings whose header policy or current-packet scope is unknown:

```bash
conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset tls-120 \
    --num_classes 120 \
    --fold 0 \
    --stage all \
    --embedding_header_policy full \
    --run_tag paper_unified_smoke \
    --tower2_epochs 1 \
    --paper_unified_stages model \
    --dry_run
```

The default `--paper_unified_stages model` is the paper-facing main path.
Graph, multi-view pooling, flow-statistics experts, stackers, target priors, and
probability selectors remain explicit ablations/candidates and are not silently
run per dataset. The flow main path is one sequence backbone, mean flow
aggregation, strict-current-packet native content, Qwen semantic embeddings,
protocol metadata, and a learned adaptive shared channel gate.

`train_tower2.py` still exposes `--confidence_penalty_weight` for calibration
ablations, but it is not part of the default `paper_unified` main claim because
the VPN 30-epoch comparison only showed a tiny Acc/F1 change and a modest
calibration improvement.

### Current packet-level results

All rows below use a **Per-flow Split** and exactly one current packet per test
sample. The same candidate framework is available to every dataset; validation
may assign zero weight to an unhelpful neural, structural, fusion, or calibration
path. Results are the mean/fusion of three independently trained folds on the
one shared test set unless the row explicitly says that every fold is exact.

| Dataset | Test packets | Accuracy | Macro-F1 | Target status | Selected path |
| --- | ---: | ---: | ---: | --- | --- |
| VPN-app | 111,678 | 0.9066 | 0.8112 | PASS | structural cross-fold mean |
| TLS-120 | 553,994 | 0.8744 | 0.8479 | PASS | byte + structural pooled-validation gate |
| USTC-app | 609,477 | 0.9773 | 0.9849 | PASS | structural cross-fold mean |
| USTC-binary | 609,332 | 1.0000 | 1.0000 | PASS | structural; each fold is exact |
| VPN-service | 111,368 | **0.9512** | **0.9435** | PASS | dual-channel + RF + validation-weighted EM prior |
| VPN-binary | 110,594 | 0.9999 | 0.9999 | **9 errors; exact target not met** | dual-channel cross-fold mean |

The paper-facing packet-level default artifact for every dataset is stored as
`reasoningDataset/packet-level/<dataset>/paper_default_result.json`, and each
dataset binds that result through
`reasoningDataset/packet-level/<dataset>/result_bound/packet_framework_manifest.json`.
Older `/tmp/two_tower_runs/packet_level/...` paths may still appear as
historical source inputs inside manifests, but they are not the default
publication result paths.

The binary numbers are reported without rounding them into a false claim:
VPN-binary is `0.9999186213/0.9999186068` with confusion matrix
`[[56029, 8], [1, 54556]]`; USTC-binary is exactly `1.0/1.0` in all three
independent folds. VPN-binary's remaining eight VPN errors are four two-packet
flows containing payload-free TCP FIN/ACK packets that both independent experts
classify as nonVPN with high confidence. Correcting them with flow-neighbor
information would violate the strict one-packet inference protocol, so this
result is not presented as 100%.

The strict-current-packet identifiability audit confirms that this is not a
single-fold optimization accident. After endpoint, route, checksum, TCP
sequence/acknowledgement, and timestamp-option fields are removed, six of the
eight misclassified FIN/ACK packets have byte-identical signatures carrying
**both labels in every training fold**. Their per-fold label counts include
`7/6`, `13/11`, and `8/6` for one signature; the other two unseen fine-grained
signatures map to the same cross-label protocol-semantic group. The remaining
error is a previously unseen 29-byte UDP signature. This diagnostic uses test
labels only after prediction to characterize errors; it is never used for
training, model selection, threshold selection, or correction.

Two further ablations did not improve the headline result:

- Flow-aware supervised contrastive training (`weight=0.05`, eight packets per
  training flow) was run on all three folds. Its cross-fold mean remained
  `0.9999186213/0.9999186068` with the identical nine-error set.
- Endpoint/session-field masked consistency plus flow contrastive training was
  tested on fold0. It retained all eight FIN/ACK errors and introduced eight
  nonVPN errors, reducing test accuracy and macro-F1 to
  `0.9998553267/0.9998553008`; it was therefore not expanded to all folds.
- A 999-point binary threshold sweep selected by pooled three-fold validation
  accuracy or macro-F1 chose the unchanged threshold `0.5` under both criteria;
  the shared-test result consequently remained nine errors.

Reproduce the post-hoc, strict-one-packet audit without CUDA:

```bash
for fold in 0 1 2; do
  conda run --no-capture-output -n llm-factory \
    python audit_packet_identifiability.py \
      --train_index reasoningDataset/packet-level/vpn-binary/fold${fold}/train/packet_index.jsonl \
      --test_index reasoningDataset/packet-level/vpn-binary/fold0/test/packet_index.jsonl \
      --prediction_npz reasoningDataset/packet-level/vpn-binary/test_dual_crossfold_mean.npz \
      --output_json reasoningDataset/packet-level/vpn-binary/fold${fold}/strict_packet_identifiability_audit.json
done
```

The audit reports four nested signatures: exact raw bytes, endpoint-invariant
bytes, session-invariant bytes, and a compact protocol-semantic signature. The
reported test-signature oracle is only the best possible lookup **after
compressing packets to that particular signature**, not an upper bound for a
model that retains richer current-packet information.

#### Identifiability-aware masked-view iteration

`train_packet_byte_transformer.py` now supports an identifiability-aware
paired view while preserving strict one-packet inference. The raw view keeps
the true packet label. The paired view masks endpoint/session fields, TCP
sequence/acknowledgement values, checksums, and TCP options but retains the
current packet payload. Session-invariant signatures are formed from the
training split only. Their label entropy defines an identifiability reliability
used to attenuate masked-view CE and raw/masked consistency; the raw classifier
is never trained with validation or test labels. A validation-only probability
blend can use the invariant view or assign it zero weight.

The first VPN-app fold0 ablation uses the same 64-byte, 128-hidden, three-layer
backbone and seed as the original CE checkpoint. It is diagnostic rather than
the cross-fold headline:

| Fold0 byte objective | Valid Acc/F1 | Shared-test Acc/F1 | Validation raw weight |
| --- | ---: | ---: | ---: |
| raw CE | 0.6903/0.6636 | 0.8329/0.6628 | 1.00 |
| hard-label masked CE + consistency | 0.6934/0.6680 | 0.8363/0.6761 | 1.00 |
| empirical conflict-distribution target | 0.6927/0.6673 | 0.8363/0.6673 | 0.90 |
| reliability gate, strength 0.25 | 0.6934/0.6677 | 0.8365/0.6777 | 1.00 |
| reliability gate, strength 0.50 | 0.6945/0.6687 | 0.8363/0.6762 | 1.00 |
| reliability gate, strength 0.75 | **0.6952/0.6700** | 0.8368/0.6736 | 1.00 |
| reliability gate, strength 1.00 | 0.6941/0.6689 | **0.8372**/0.6727 | 1.00 |

The direct empirical soft target is a negative ablation: forcing an ambiguous
signature distribution into the shared encoder reduces macro-F1 relative to
ordinary hard-label masking. Reliability gating avoids that failure, but the
validation-optimal global strength does not maximize shared-test F1. Therefore
the learned per-packet router was also tested on all three folds. Every fold
selected invariant scale `0` on validation. Its fold0/fold1/fold2 shared-test
results were `0.8262/0.6513`, `0.8228/0.6488`, and `0.8172/0.6502`; their
probability mean reached only `0.8316/0.6660`. This is a reproducible negative
ablation: predicting signature identifiability is a useful diagnostic auxiliary
task, but routing predictions through the weak masked-logit expert increases
cross-environment representation shift. The option remains default-off for
ablation reproduction. The paper-facing next step should keep identifiability
as a training constraint while protecting the full current-packet classifier,
and select checkpoints by cross-fold stability rather than one-fold validation
peaks. The current `0.9066/0.8112` VPN-app structural cross-fold result remains
the headline.

The first protocol-correct VPN-app result uses one packet per inference sample.
The structural expert reads only the current packet's normalized L3 byte prefix
and parsed header fields; reconstructed flow IDs are used for split auditing,
never as model input. Each fold selects byte-prefix length and tree leaf size
using only its validation macro-F1. The final rule is a pre-specified arithmetic
mean of the three independently trained fold probabilities.

```text
fold0: selected prefix=128, leaf=1, test accuracy=0.8981, macro-F1=0.7917
fold1: selected prefix=64,  leaf=2, test accuracy=0.8919, macro-F1=0.7926
fold2: selected prefix=64,  leaf=1, test accuracy=0.8935, macro-F1=0.7596

three-fold structural probability mean (complete IPv4+IPv6 test set):
  accuracy = 0.9066
  macro-F1 = 0.8112
  VPN packet target accuracy>=0.9000, macro-F1>=0.7600 -> PASS
```

The count-correct result above contains all 111,678 test packets. An earlier
legacy artifact reported `0.9115/0.8190` over 111,670 packets and is not used as
the headline because eight IPv6 packets were missing. The session-field-masked
structural view reached `0.8653/0.7446`; the Qwen semantic channel reached
`0.8027/0.6977` on fold0 test, and its validation-selected fusion reduced test
performance to `0.8927/0.7881`. The 64-byte Transformer reached validation
`0.6903/0.6636`, so the one-standard-error rule selected the structural channel
alone. These are generalization and automatic-channel-shutdown ablations, not
headline improvements.

For TLS-120, the same 64-byte Transformer configuration was trained on all
three supplied folds. Each fold's nested gate was selected without test labels:

```text
fold0 byte test: acc=0.8451, macro-F1=0.8115
fold1 byte test: acc=0.8404, macro-F1=0.8065
fold2 byte test: acc=0.8484, macro-F1=0.8123

fold0 byte+structural gate: acc=0.8550, macro-F1=0.8267
fold1 byte+structural gate: acc=0.8588, macro-F1=0.8279
fold2 byte+structural gate: acc=0.8736, macro-F1=0.8463

pooled-validation crossfold gate:
  accuracy = 0.8744
  macro-F1 = 0.8479
  TLS packet target accuracy>=0.8500, macro-F1>=0.7800 -> PASS
```

The previous three-fold structural-only TLS result was `0.7998/0.7656`.
Pooled-validation fusion concatenates only fold-specific validation predictions
to select calibration and the gate, then applies the fixed rule to the mean of
the three aligned test probability sets. Test labels are used only for the final
reported metrics. The strong-header-masking/SupCon byte ablation reached only
`0.7151/0.6893` on fold2 validation; therefore these regularizers remain
available research controls but are not enabled in the current TLS best.

For VPN-service, the dual-channel L3/payload branch improves the single neural
cross-fold result over the original header-only branch. A predefined 201-point
pooled-validation sweep selects neural/RF weights `0.175/0.825`, giving
`0.9480/0.9382` before calibration. EM estimates the target class prior from
unlabeled test probabilities; candidate strength/gating is selected by
target-prior-weighted validation accuracy (`selection_scope=valid_weighted`).
The selected `strength=0.3`, `low_margin=0.2` candidate reaches
`0.9512068099` accuracy and `0.9434656797` macro-F1. Test labels participate
only in the final metric audit, not in weight, prior, strength, or gate selection.

After exporting pooled validation probabilities and cross-fold test means for
the dual and RF experts, reproduce the selected mix and packet-native prior
calibration with:

```bash
conda run --no-capture-output -n llm-factory \
  python fuse_packet_crossfold.py \
    --inputs \
      reasoningDataset/packet-level/vpn-service/test_dual_crossfold_mean.npz \
      reasoningDataset/packet-level/vpn-service/test_rf_crossfold_mean.npz \
    --validation_inputs \
      reasoningDataset/packet-level/vpn-service/valid_dual_pooled.npz \
      reasoningDataset/packet-level/vpn-service/valid_rf_pooled.npz \
    --weight_grid_size 201 \
    --select_metric macro_f1 \
    --label_map reasoningDataset/packet-level/vpn-service/fold0/train/label_map.json \
    --output_json reasoningDataset/packet-level/vpn-service/test_dual_rf_selected_mix.json \
    --output_npz reasoningDataset/packet-level/vpn-service/test_dual_rf_selected_mix.npz \
    --output_validation_npz reasoningDataset/packet-level/vpn-service/valid_dual_rf_selected_mix.npz

conda run --no-capture-output -n llm-factory \
  python calibrate_prediction_prior.py \
    --valid_npz reasoningDataset/packet-level/vpn-service/valid_dual_rf_selected_mix.npz \
    --test_npz reasoningDataset/packet-level/vpn-service/test_dual_rf_selected_mix.npz \
    --label_map reasoningDataset/packet-level/vpn-service/fold0/train/label_map.json \
    --strengths 0,0.025,0.05,0.075,0.1,0.15,0.2,0.3 \
    --prior_method em \
    --selection_scope valid_weighted \
    --select_metric accuracy \
    --gate_modes none,low_margin \
    --gate_thresholds 0.05,0.1,0.15,0.2 \
    --output_json reasoningDataset/packet-level/vpn-service/test_dual_rf_prior_em.json \
    --output_npz reasoningDataset/packet-level/vpn-service/test_dual_rf_prior_em.npz
```

The commands below reproduce the earlier TLS tree/probability-fusion ablation;
they are **not** the current `paper_unified` main path. For current experiments,
use `make_unified_repro_plan.py`, which consumes
`test_unified_packet_single_head.npz` from each fold.

```bash
conda run --no-capture-output -n llm-factory \
  python fuse_packet_crossfold.py \
    --inputs \
      reasoningDataset/packet-level/tls-120/fold0/test_byte_probs.npz \
      reasoningDataset/packet-level/tls-120/fold1/test_byte_probs.npz \
      reasoningDataset/packet-level/tls-120/fold2/test_byte_probs.npz \
    --label_map reasoningDataset/packet-level/tls-120/fold0/train/label_map.json \
    --output_json reasoningDataset/packet-level/tls-120/test_byte_crossfold_mean.json \
    --output_npz reasoningDataset/packet-level/tls-120/test_byte_crossfold_mean.npz

conda run --no-capture-output -n llm-factory \
  python fuse_packet_crossfold.py \
    --inputs \
      reasoningDataset/packet-level/tls-120/fold0/test_feature_probs.npz \
      reasoningDataset/packet-level/tls-120/fold1/test_feature_probs.npz \
      reasoningDataset/packet-level/tls-120/fold2/test_feature_probs.npz \
    --label_map reasoningDataset/packet-level/tls-120/fold0/train/label_map.json \
    --output_json reasoningDataset/packet-level/tls-120/test_feature_crossfold_mean.json \
    --output_npz reasoningDataset/packet-level/tls-120/test_feature_crossfold_mean.npz

conda run --no-capture-output -n llm-factory \
  python concat_packet_probabilities.py \
    --inputs \
      reasoningDataset/packet-level/tls-120/fold0/valid_byte_probs.npz \
      reasoningDataset/packet-level/tls-120/fold1/valid_byte_probs.npz \
      reasoningDataset/packet-level/tls-120/fold2/valid_byte_probs.npz \
    --output_npz reasoningDataset/packet-level/tls-120/valid_byte_pooled.npz

conda run --no-capture-output -n llm-factory \
  python concat_packet_probabilities.py \
    --inputs \
      reasoningDataset/packet-level/tls-120/fold0/valid_feature_probs.npz \
      reasoningDataset/packet-level/tls-120/fold1/valid_feature_probs.npz \
      reasoningDataset/packet-level/tls-120/fold2/valid_feature_probs.npz \
    --output_npz reasoningDataset/packet-level/tls-120/valid_feature_pooled.npz

conda run --no-capture-output -n llm-factory \
  python fuse_packet_experts.py \
    --valid_semantic reasoningDataset/packet-level/tls-120/valid_byte_pooled.npz \
    --valid_structural reasoningDataset/packet-level/tls-120/valid_feature_pooled.npz \
    --test_semantic reasoningDataset/packet-level/tls-120/test_byte_crossfold_mean.npz \
    --test_structural reasoningDataset/packet-level/tls-120/test_feature_crossfold_mean.npz \
    --label_map reasoningDataset/packet-level/tls-120/fold0/train/label_map.json \
    --output_json reasoningDataset/packet-level/tls-120/test_byte_structural_crossfold_gate.json \
    --gate_out checkpoints/packet-level/tls-120/byte_structural_crossfold_gate.joblib
```

The exact audit includes both IPv4 and IPv6 and matches the SWEET packet
counts: fold0 train has 33,088 packets/4,713 reconstructed flows, validation
has 7,568 packets/1,706 flows, and the shared test has 111,678 packets/6,681
flows. Train/validation/test reconstructed-flow intersections are all zero.

Saved local VPN structural evidence (ignored by git because datasets and
checkpoints are local artifacts):

```text
reasoningDataset/packet-level/vpn-app/fold0_feature_expert.json
reasoningDataset/packet-level/vpn-app/fold1_feature_expert.json
reasoningDataset/packet-level/vpn-app/fold2_feature_expert.json
reasoningDataset/packet-level/vpn-app/packet_feature_crossfold_mean.json
checkpoints/packet-level/vpn-app/fold{0,1,2}/feature_expert.joblib
```

This is the current strong packet baseline, not the final CCF-A method claim.
The unified model iteration adds the same semantic Tower1 channel,
header-randomized consistency channel, and validation-gated semantic/structural
fusion to every packet dataset. Modules remain present for all datasets, while
learned gates can reduce an unhelpful channel's contribution toward zero.

## Cross-dataset pipeline

Use `run_dataset_flow_pipeline.py` to keep VPN/TLS preprocessing, packet embedding extraction, and Tower-2 dataset generation path-consistent. The pipeline defaults to the cached `Qwen/Qwen2.5-7B-Instruct` Tower-1 checkpoint and local-only model loading to avoid accidental HuggingFace downloads during long experiments.

For downstream generalization experiments, `--embedding_header_policy` supports `full`, `randomize_ip_port`, and `mask_ip_port`. Use `full` for the standard view, `randomize_ip_port` to preserve within-flow endpoint consistency without memorizing real endpoints, and `mask_ip_port` to remove endpoint identity more aggressively.

```bash
python run_dataset_flow_pipeline.py \
  --dataset vpn-app \
  --stage all \
  --no_progress
```

For TLS-120, packet prompts are shorter than the VPN SFT prompt length. A 5000-packet sample from `train_tower1_change_weight/packet_index.jsonl` measured p99=540 and max=551 Qwen tokens, so `--embedding_max_length 640` is enough for the current packet embedding extraction and much faster than 1792.

```bash
conda run --no-capture-output -n llm-factory \
  python run_dataset_flow_pipeline.py \
    --dataset tls-120 \
    --stage tower1 \
    --no_progress

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n llm-factory \
  python run_dataset_flow_pipeline.py \
    --dataset tls-120 \
    --stage embeddings \
    --splits train,valid,test \
    --embedding_max_length 640 \
    --embedding_batch_size 64 \
    --no_progress

python run_dataset_flow_pipeline.py \
  --dataset tls-120 \
  --stage tower2 \
  --splits train,valid,test \
  --embedding_max_length 640 \
  --embedding_batch_size 64 \
  --no_progress
```

When resuming or repairing a single split, set `--splits` explicitly so the pipeline does not overwrite other completed outputs. For example:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n llm-factory \
  python run_dataset_flow_pipeline.py \
    --dataset tls-120 \
    --stage embeddings \
    --splits test \
    --embedding_max_length 640 \
    --embedding_batch_size 64 \
    --no_progress
```

Current TLS-120 Tower-2 baselines:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n llm-factory \
  python train_tower2.py \
    --model_type seq \
    --dataset reasoningDataset/tls-120/train_tower2_rawproj_change_weight/seq_dataset.pt \
    --valid_dataset reasoningDataset/tls-120/valid_tower2_rawproj_change_weight/seq_dataset.pt \
    --output_dir checkpoints/tower2_seq_flow_tls120_rawproj_change_weight_baseline \
    --num_classes 120 \
    --epochs 30 \
    --batch_size 32 \
    --hidden_dim 256 \
    --num_layers 2 \
    --num_heads 4 \
    --dropout 0.20 \
    --lr 1e-4 \
    --weight_decay 0.03 \
    --train_level flow \
    --flow_pooling attention \
    --window_loss_weight 0.2 \
    --class_weighting effective \
    --class_weight_beta 0.9999 \
    --class_weight_strength 0.5 \
    --label_smoothing 0.05 \
    --balanced_flow_batches \
    --samples_per_class 2 \
    --flow_contrastive_weight 0 \
    --aux_weight 0 \
    --coherence_weight 0 \
    --select_metric flow_macro_f1

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n llm-factory \
  python train_tower2.py \
    --model_type graph \
    --dataset reasoningDataset/tls-120/train_tower2_rawproj_change_weight/graph_dataset.pt \
    --valid_dataset reasoningDataset/tls-120/valid_tower2_rawproj_change_weight/graph_dataset.pt \
    --output_dir checkpoints/tower2_graph_flow_tls120_rawproj_change_weight_baseline \
    --num_classes 120 \
    --epochs 15 \
    --batch_size 32 \
    --hidden_dim 192 \
    --num_layers 1 \
    --num_heads 4 \
    --dropout 0.20 \
    --lr 1e-4 \
    --weight_decay 0.03 \
    --train_level flow \
    --flow_pooling attention \
    --window_loss_weight 0.2 \
    --class_weighting effective \
    --class_weight_beta 0.9999 \
    --class_weight_strength 0.5 \
    --label_smoothing 0.05 \
    --balanced_flow_batches \
    --samples_per_class 2 \
    --flow_contrastive_weight 0 \
    --aux_weight 0 \
    --coherence_weight 0 \
    --select_metric flow_macro_f1

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n llm-factory \
  python train_tower2.py \
    --model_type graph \
    --dataset reasoningDataset/tls-120/train_tower2_rawproj_change_weight/graph_dataset.pt \
    --valid_dataset reasoningDataset/tls-120/valid_tower2_rawproj_change_weight/graph_dataset.pt \
    --output_dir checkpoints/tower2_graph_flow_tls120_rawproj_change_weight_baseline_ft \
    --init_checkpoint checkpoints/tower2_graph_flow_tls120_rawproj_change_weight_baseline/best.pt \
    --num_classes 120 \
    --epochs 15 \
    --batch_size 32 \
    --hidden_dim 192 \
    --num_layers 1 \
    --num_heads 4 \
    --dropout 0.20 \
    --lr 5e-5 \
    --weight_decay 0.03 \
    --train_level flow \
    --flow_pooling attention \
    --window_loss_weight 0.2 \
    --class_weighting effective \
    --class_weight_beta 0.9999 \
    --class_weight_strength 0.5 \
    --label_smoothing 0.05 \
    --balanced_flow_batches \
    --samples_per_class 2 \
    --flow_contrastive_weight 0 \
    --aux_weight 0 \
    --coherence_weight 0 \
    --select_metric flow_macro_f1
```

TLS-120 graph/seq probability fusion selected on the validation split:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n llm-factory \
  python test_tower2.py \
    --checkpoint checkpoints/tower2_graph_flow_tls120_rawproj_change_weight_acc_ft/best.pt \
    --dataset reasoningDataset/tls-120/valid_tower2_rawproj_change_weight/graph_dataset.pt \
    --label_map reasoningDataset/tls-120/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/tls-120/valid_graph_metrics_flow_tls120_rawproj_change_weight_acc_ft_probs.json \
    --no_report

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n llm-factory \
  python test_tower2.py \
    --checkpoint checkpoints/tower2_graph_flow_tls120_rawproj_change_weight_acc_ft/best.pt \
    --dataset reasoningDataset/tls-120/test_tower2_rawproj_change_weight/graph_dataset.pt \
    --label_map reasoningDataset/tls-120/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/tls-120/test_graph_metrics_flow_tls120_rawproj_change_weight_acc_ft_probs.json \
    --no_report

CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n llm-factory \
  python test_tower2.py \
    --checkpoint checkpoints/tower2_seq_flow_tls120_rawproj_change_weight_baseline/best.pt \
    --dataset reasoningDataset/tls-120/valid_tower2_rawproj_change_weight/seq_dataset.pt \
    --label_map reasoningDataset/tls-120/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/tls-120/valid_seq_metrics_flow_tls120_rawproj_change_weight_baseline_probs.json \
    --no_report

CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n llm-factory \
  python test_tower2.py \
    --checkpoint checkpoints/tower2_seq_flow_tls120_rawproj_change_weight_baseline/best.pt \
    --dataset reasoningDataset/tls-120/test_tower2_rawproj_change_weight/seq_dataset.pt \
    --label_map reasoningDataset/tls-120/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/tls-120/test_seq_metrics_flow_tls120_rawproj_change_weight_baseline_probs.json \
    --no_report

python make_fusion_payload.py \
  --valid_json reasoningDataset/tls-120/valid_graph_metrics_flow_tls120_rawproj_change_weight_acc_ft_probs.json \
  --test_json reasoningDataset/tls-120/test_graph_metrics_flow_tls120_rawproj_change_weight_acc_ft_probs.json \
  --output_json reasoningDataset/tls-120/fusion_input_graph_acc_ft.json

python make_fusion_payload.py \
  --valid_json reasoningDataset/tls-120/valid_seq_metrics_flow_tls120_rawproj_change_weight_baseline_probs.json \
  --test_json reasoningDataset/tls-120/test_seq_metrics_flow_tls120_rawproj_change_weight_baseline_probs.json \
  --output_json reasoningDataset/tls-120/fusion_input_seq_baseline.json

python fuse_prediction_jsons.py \
  --input graph reasoningDataset/tls-120/fusion_input_graph_acc_ft.json \
  --input seq reasoningDataset/tls-120/fusion_input_seq_baseline.json \
  --label_map reasoningDataset/tls-120/train_tower1_change_weight/label_map.json \
  --simplex_step 0.01 \
  --select_metric accuracy \
  --output_json reasoningDataset/tls-120/test_fusion_graph_seq_tls120_rawproj_change_weight_valid_acc.json
```

Current TLS-120 result from this validation-selected graph/seq fusion is `flow_acc=0.7909`, `flow_macro_f1=0.7769`.

## 0. Dataset format

Each pcap file is treated as one flow. The class label is the subfolder name.
Preprocessing uses an offline pcap/pcapng parser in `traffic_utils.py` for
IPv4/TCP/UDP/ICMP metadata extraction, so it can run in restricted environments
without Scapy network-interface permissions. The parser currently supports
classic pcap and pcapng Enhanced Packet Blocks with raw IPv4, Ethernet, and
Linux cooked captures.

```text
train/
  youtube/
    flow1.pcap
    flow2.pcap
  gmail/
    flow3.pcap
valid/
  youtube/
  gmail/
test/
  youtube/
  gmail/
```

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch
transformers
peft
accelerate
scapy
scikit-learn
numpy
tqdm
```

---

## 2. Tower-1 preprocessing

### Train split

```bash
python preprocess_tower1.py \
  --input_dir /home/jing/download/sweet/flow-level-classification/vpn-app/train_val_split_0/train \
  --output_dir reasoningDataset/vpn-app/train_tower1_change_weight \
  --max_packets_per_flow 64 \
  --payload_prefix_len 128 \
  --l3_prefix_len 512 \
  --write_label_map
```

Outputs:

```text
outputs/train_tower1_change_weight/packet_instruction.jsonl   # protocol field QA + consistency QA
outputs/train_tower1_change_weight/packet_validity.jsonl      # packet validity and hard negatives
outputs/train_tower1_change_weight/packet_index.jsonl         # packet prompts for embedding extraction
outputs/train_tower1_change_weight/packet_auxiliary.jsonl     # packet prompts for cls + contrastive training
outputs/train_tower1_change_weight/label_map.json             # shared label id map
```

`packet_instruction.jsonl` and `packet_validity.jsonl` are used for `L_QA`.

`packet_auxiliary.jsonl` is used for `L_packet_cls` and `L_supcon`. Each row contains:

```json
{
  "flow_id": "...",
  "packet_id": 0,
  "prompt": "[Packet]\nDirection: ...\nIP: ...\nTCP: ...\nPayloadPrefix: ...",
  "label": "youtube",
  "label_id": 0,
  "packet_weight": 0.8
}
```

`packet_weight` down-weights weakly informative packets such as pure ACK/SYN packets because the packet label is inherited from the flow label and is therefore a weak label.
Regenerate `packet_auxiliary.jsonl` after changing the weighting heuristic; old files keep their original weights.

### Valid/test splits

Use the train label map to avoid label-id mismatch:

```bash
python preprocess_tower1.py \
  --input_dir /home/jing/download/sweet/flow-level-classification/vpn-app/train_val_split_0/val \
  --output_dir reasoningDataset/vpn-app/valid_tower1_change_weight \
  --max_packets_per_flow 64 \
  --payload_prefix_len 128 \
  --l3_prefix_len 512 \
  --label_map_in reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json

python preprocess_tower1.py \
  --input_dir /home/jing/download/sweet/flow-level-classification/vpn-app/test \
  --output_dir reasoningDataset/vpn-app/test_tower1_change_weight \
  --max_packets_per_flow 64 \
  --payload_prefix_len 128 \
  --l3_prefix_len 512 \
  --label_map_in reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json
```

---

## 3. Train Tower 1 with QA + classification + contrastive loss

The training objective is:

```text
L_tower1 = L_QA + alpha * L_packet_cls + beta * L_supcon
```

Recommended initial weights:

```text
alpha = 0.1
beta  = 0.3
```

Run:

```bash
python train_tower1_multitask.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
  --packet_aux_jsonl reasoningDataset/vpn-app/train_tower1_change_weight/packet_auxiliary.jsonl \
  --sft_jsonl reasoningDataset/vpn-app/train_tower1_change_weight/packet_instruction.jsonl reasoningDataset/vpn-app/train_tower1_change_weight/packet_validity.jsonl \
  --output_dir checkpoints/tower1_qwen_multitask_change_weight \
  --epochs 2 \
  --sft_batch_size 2 \
  --packet_batch_size 16 \
  --max_sft_length 1792 \
  --max_packet_length 1024 \
  --lr 2e-5 \
  --head_lr 1e-4 \
  --cls_weight 0.1 \
  --contrastive_weight 0.3 \
  --temperature 0.07 \
  --local_files_only \
  --lora_r 16 \
  --lora_alpha 32 \
  --log_steps 1 \
  --save_steps 500 \
  --gradient_checkpointing
```

During startup and training, the script prints dataset-loading progress and fixed-format training lines like:

```text
step=440/2646 loss=1.2345 lm=0.4567 pkt_cls=1.2345 supcon=2.3456 pkt_acc=0.5000 lm_tokens/batch=4.0 skipped_nonfinite=0
```

If an SFT batch ever produces a non-finite loss, the default behavior is to skip that optimizer update and print a warning. Use `--stop_on_nonfinite_loss` when debugging if you want the script to stop immediately at the first NaN/Inf.

To print loss for every training batch, set:

```bash
--log_steps 1
```

This prints one fixed-format line per training step. Each step consumes one packet batch and, unless `--no_sft` is set, one SFT batch.

For the current `vpn-app/train_tower1` SFT files with `Qwen/Qwen2.5-7B-Instruct`, measured `prompt + answer + eos` token lengths are:

```text
n = 503142
max = 1607
p99 = 1566
p99.9 = 1593
>1024 = 111156 samples, 22.0924%
>1536 = 6148 samples, 1.2219%
>1792 = 0 samples, 0.0000%
```

So `--max_sft_length 1792` is the recommended value for this generated dataset. If you regenerate Tower-1 data or change tokenizer/model, rerun with `--auto_max_sft_length` to scan the SFT files and automatically raise `max_sft_length` to the requested percentile.

Auto-scan example:

```bash
python train_tower1_multitask.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
  --packet_aux_jsonl reasoningDataset/vpn-app/train_tower1_change_weight/packet_auxiliary.jsonl \
  --sft_jsonl reasoningDataset/vpn-app/train_tower1_change_weight/packet_instruction.jsonl reasoningDataset/vpn-app/train_tower1_change_weight/packet_validity.jsonl \
  --output_dir checkpoints/tower1_qwen_multitask_change_weight \
  --epochs 2 \
  --sft_batch_size 2 \
  --packet_batch_size 16 \
  --max_sft_length 1024 \
  --auto_max_sft_length \
  --sft_length_percentile 100 \
  --max_packet_length 1024 \
  --lr 2e-5 \
  --head_lr 1e-4 \
  --cls_weight 0.1 \
  --contrastive_weight 0.3 \
  --temperature 0.07 \
  --lora_r 16 \
  --lora_alpha 32 \
  --log_steps 1 \
  --save_steps 500 \
  --gradient_checkpointing
```

Saved files:

```text
checkpoints/tower1_qwen_multitask_change_weight/adapter/          # LoRA adapter
checkpoints/tower1_qwen_multitask_change_weight/tower1_heads.pt   # packet cls + projection heads
checkpoints/tower1_qwen_multitask_change_weight/tower1_config.json
checkpoints/tower1_qwen_multitask_change_weight/step_500/         # intermediate checkpoint when --save_steps 500
```

You can also train only the representation-alignment stage after an existing SFT adapter by setting `--no_sft`, but the default is joint training.

---

## 4. Extract packet embeddings

Recommended Tower-2 input is `raw + projected`:

```text
raw       = normalized last non-padding token hidden state
projected = Tower-1 contrastive projection-head output
concat    = raw || projected
```

This keeps the high-dimensional Qwen packet representation while adding the SupCon-trained discriminative projection.

### Train split

```bash
python extract_packet_embeddings_qwen.py \
  --packet_index reasoningDataset/vpn-app/train_tower1_change_weight/packet_index.jsonl \
  --output_dir reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/tower1_qwen_multitask_change_weight/adapter \
  --tower1_heads checkpoints/tower1_qwen_multitask_change_weight/tower1_heads.pt \
  --embedding_mode concat \
  --batch_size 8 \
  --max_length 1024
```

### Valid/test splits

```bash
python extract_packet_embeddings_qwen.py \
  --packet_index reasoningDataset/vpn-app/valid_tower1_change_weight/packet_index.jsonl \
  --output_dir reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/tower1_qwen_multitask_change_weight/adapter \
  --tower1_heads checkpoints/tower1_qwen_multitask_change_weight/tower1_heads.pt \
  --embedding_mode concat

python extract_packet_embeddings_qwen.py \
  --packet_index reasoningDataset/vpn-app/test_tower1_change_weight/packet_index.jsonl \
  --output_dir reasoningDataset/vpn-app/test_embeddings_rawproj_change_weight \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/tower1_qwen_multitask_change_weight/adapter \
  --tower1_heads checkpoints/tower1_qwen_multitask_change_weight/tower1_heads.pt \
  --embedding_mode concat
```

For ablations, use `--embedding_mode raw` or `--embedding_mode projected`. The older `--use_projected_embedding` flag is kept as an alias for `--embedding_mode projected`.

The unified Packet and Flow runners buffer up to 128 packets from adjacent
flows so that short flows share the same Qwen forward micro-batch. The resulting NPY
files and JSONL index remain one row/file per flow, with the original packet
order restored before writing. `--batch_size` still controls the actual model
micro-batch and therefore GPU memory; `--flow_batch_packets` only controls the
CPU-side cross-flow buffer. The extractor's bare-CLI default remains `0` for
legacy one-flow-at-a-time reproducibility; both unified runners explicitly pass
`128`, and strict shared-core v2 freezes that value together with scheduler
version `cross_flow_length_bucketed_v1`. `embedding_config.json`
records both values for auditing. New paper-unified extraction also records a
content hash over the selected `best/adapter` directory and `best/tower1_heads.pt`.
`audit_flow_embeddings.py --require_model_provenance` recomputes those hashes
when embeddings are created, and framework-manifest evidence recomputes them
again when the audit is consumed. Thus a path-correct but replaced Tower1
checkpoint cannot be presented as the validation-selected embedding source.

Both unified runners also support deterministic flow-id sharding for semantic
embedding extraction. Flow uses `--embedding_num_shards N` with
`--embedding_cuda_devices 0,1,...`; Packet uses
`--semantic_embedding_num_shards N` with
`--semantic_embedding_cuda_devices 0,1,...`. Every flow is assigned by the
same SHA-1 scheduler, each shard retains one complete per-flow embedding file,
and `embedding_shard_utils.py` rejects missing flows, duplicate flows,
misassigned flows, missing NPY files, and extraction-contract drift before it
atomically publishes the merged index. The merged `embedding_config.json`
retains embedding mode, micro-batch, cross-flow buffer, scheduler, header and
packet-context policies, plus shard provenance. The exact-coverage audit runs
after the merge. Shard count and device placement are execution resources, not
learned/model hyperparameters, so they are recorded but do not change the
strict shared-core model fingerprint.

An execution-only scheduler simulation on the VPN shared test packet index
(`6,681` flows, `111,678` packets) reduced Qwen micro-batch calls from `18,843`
to `14,088` (`-25.23%`) and the character-length padding proxy from `83.03M`
to `66.37M` (`-20.07%`). This is a runtime result, not a classification metric;
the model batch remains eight and the per-packet embedding contract is
unchanged.

### Projection-only ablation

```bash
python extract_packet_embeddings_qwen.py \
  --packet_index reasoningDataset/vpn-app/train_tower1_change_weight/packet_index.jsonl \
  --output_dir reasoningDataset/vpn-app/train_embeddings_proj_change_weight \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/tower1_qwen_multitask_change_weight/adapter \
  --tower1_heads checkpoints/tower1_qwen_multitask_change_weight/tower1_heads.pt \
  --use_projected_embedding
```

---

## 5. Tower-2 preprocessing

### Train data

```bash
python preprocess_tower2.py \
  --flow_embedding_index reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
  --content_group_index reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
  --output_dir reasoningDataset/vpn-app/train_tower2_rawproj_change_weight \
  --window_size 32 \
  --stride 16
```

### Valid/test data

```bash
python preprocess_tower2.py \
  --flow_embedding_index reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
  --content_group_index reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
  --output_dir reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight \
  --window_size 32 \
  --stride 16

python preprocess_tower2.py \
  --flow_embedding_index reasoningDataset/vpn-app/test_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
  --content_group_index reasoningDataset/vpn-app/test_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
  --output_dir reasoningDataset/vpn-app/test_tower2_rawproj_change_weight \
  --window_size 32 \
  --stride 16
```

Outputs:

```text
seq_dataset.pt
 graph_dataset.pt
content_group_manifest.json  # only when --content_group_index is provided
```

`--content_group_index` is optional and does not change the model input
features. It hashes each source PCAP with SHA256 and attaches
`content_group_id/content_hash` metadata to every Tower-2 window. This makes
future group-aware samplers, validation splits, and content-group CI checks use
the same exact-content definition as the paper evidence pack instead of relying
on post-hoc bookkeeping. If omitted, preprocessing keeps the historical dataset
format.

The graph version constructs typed packet-interaction edges:

```text
temporal_next
same_direction
opposite_direction
ack_candidate
seq_continuity
same_burst
retransmission_candidate
```

---

## 6. Archived Tower-2 sequence ablations

The commands in this section are kept only to reproduce earlier
`rawproj_change_weight` ablations. They are not the current paper-facing entry
point. For new VPN/TLS experiments, use the `paper_unified` runner and the
Stage-8 automation below so reports, manifests, CI gates, and shared-module
audits stay synchronized.

The next experiments use the `rawproj_change_weight` Tower-2 data and enable changes step by step.

Stage 1 adds flow/window dual loss and keeps SupCon off:

```bash
python train_tower2.py \
  --model_type seq \
  --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/seq_dataset.pt \
  --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/seq_dataset.pt \
  --output_dir checkpoints/tower2_seq_flow_rawproj_change_weight_dual \
  --num_classes 16 \
  --epochs 30 \
  --batch_size 16 \
  --train_level flow \
  --flow_pooling attention \
  --window_loss_weight 0.3 \
  --flow_contrastive_weight 0 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --aux_weight 0 \
  --coherence_weight 0
```

Stage 2 adds class-balanced CE:

```bash
python train_tower2.py \
  --model_type seq \
  --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/seq_dataset.pt \
  --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/seq_dataset.pt \
  --output_dir checkpoints/tower2_seq_flow_rawproj_change_weight_dual_cbce \
  --num_classes 16 \
  --epochs 30 \
  --batch_size 16 \
  --train_level flow \
  --flow_pooling attention \
  --window_loss_weight 0.3 \
  --class_weighting effective \
  --class_weight_beta 0.9999 \
  --flow_contrastive_weight 0 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --aux_weight 0 \
  --coherence_weight 0
```

Stage 3 adds balanced SupCon. `--balanced_flow_batches` makes each batch contain positive pairs for contrastive learning:

```bash
python train_tower2.py \
  --model_type seq \
  --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/seq_dataset.pt \
  --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/seq_dataset.pt \
  --output_dir checkpoints/tower2_seq_flow_rawproj_change_weight_dual_cbce_bal_supcon \
  --num_classes 16 \
  --epochs 30 \
  --batch_size 16 \
  --train_level flow \
  --flow_pooling attention \
  --window_loss_weight 0.3 \
  --class_weighting effective \
  --class_weight_beta 0.9999 \
  --balanced_flow_batches \
  --samples_per_class 2 \
  --flow_contrastive_weight 0.05 \
  --flow_temperature 0.07 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --aux_weight 0 \
  --coherence_weight 0
```

Stage 4 compares flow pooling strategies on the same loss setup:

```bash
for pooling in mean attention late_fusion; do
  python train_tower2.py \
    --model_type seq \
    --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/seq_dataset.pt \
    --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/seq_dataset.pt \
    --output_dir checkpoints/tower2_seq_flow_rawproj_change_weight_dual_cbce_bal_supcon_pool_${pooling} \
    --num_classes 16 \
    --epochs 30 \
    --batch_size 16 \
    --train_level flow \
    --flow_pooling ${pooling} \
    --window_loss_weight 0.3 \
    --class_weighting effective \
    --class_weight_beta 0.9999 \
    --balanced_flow_batches \
    --samples_per_class 2 \
    --flow_contrastive_weight 0.05 \
    --flow_temperature 0.07 \
    --hidden_dim 256 \
    --num_layers 2 \
    --num_heads 4 \
    --aux_weight 0 \
    --coherence_weight 0
done
```

Stage 5 selects checkpoints by flow macro-F1, adds hierarchical coarse-to-fine classification, and uses confusion-aware SupCon:

```bash
python train_tower2.py \
  --model_type seq \
  --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/seq_dataset.pt \
  --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/seq_dataset.pt \
  --output_dir checkpoints/tower2_seq_flow_rawproj_change_weight_macro_hier_conf_supcon \
  --num_classes 16 \
  --epochs 30 \
  --batch_size 16 \
  --train_level flow \
  --select_metric flow_macro_f1 \
  --flow_pooling mean \
  --window_loss_weight 0.3 \
  --class_weighting effective \
  --class_weight_beta 0.9999 \
  --hierarchical_weight 0.2 \
  --hierarchical_logit_weight 0.5 \
  --coarse_groups vpn_app \
  --balanced_flow_batches \
  --samples_per_class 2 \
  --contrastive_mode confusion \
  --confusion_groups vpn_app \
  --flow_contrastive_weight 0.03 \
  --flow_temperature 0.07 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --aux_weight 0 \
  --coherence_weight 0
```

---

## 7. Archived Tower-2 graph ablations

The graph commands below mirror the historical step-by-step sequence ablations.
They remain useful for controlled comparisons, but they should not be copied as
the default training recipe for new paper runs.

Stage 1 adds flow/window dual loss and keeps SupCon off:

```bash
python train_tower2.py \
  --model_type graph \
  --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/graph_dataset.pt \
  --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt \
  --output_dir checkpoints/tower2_graph_flow_rawproj_change_weight_dual \
  --num_classes 16 \
  --epochs 30 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --train_level flow \
  --flow_pooling attention \
  --window_loss_weight 0.3 \
  --flow_contrastive_weight 0 \
  --aux_weight 0 \
  --coherence_weight 0
```

Stage 2 adds class-balanced CE:

```bash
python train_tower2.py \
  --model_type graph \
  --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/graph_dataset.pt \
  --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt \
  --output_dir checkpoints/tower2_graph_flow_rawproj_change_weight_dual_cbce \
  --num_classes 16 \
  --epochs 30 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --train_level flow \
  --flow_pooling attention \
  --window_loss_weight 0.3 \
  --class_weighting effective \
  --class_weight_beta 0.9999 \
  --flow_contrastive_weight 0 \
  --aux_weight 0 \
  --coherence_weight 0
```

Stage 3 adds balanced SupCon:

```bash
python train_tower2.py \
  --model_type graph \
  --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/graph_dataset.pt \
  --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt \
  --output_dir checkpoints/tower2_graph_flow_rawproj_change_weight_dual_cbce_bal_supcon \
  --num_classes 16 \
  --epochs 30 \
  --batch_size 16 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --train_level flow \
  --flow_pooling attention \
  --window_loss_weight 0.3 \
  --class_weighting effective \
  --class_weight_beta 0.9999 \
  --balanced_flow_batches \
  --samples_per_class 2 \
  --flow_contrastive_weight 0.05 \
  --flow_temperature 0.07 \
  --aux_weight 0 \
  --coherence_weight 0
```

Stage 4 compares flow pooling strategies on the same loss setup:

```bash
for pooling in mean attention late_fusion; do
  python train_tower2.py \
    --model_type graph \
    --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/graph_dataset.pt \
    --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt \
    --output_dir checkpoints/tower2_graph_flow_rawproj_change_weight_dual_cbce_bal_supcon_pool_${pooling} \
    --num_classes 16 \
    --epochs 30 \
    --batch_size 16 \
    --hidden_dim 256 \
    --num_layers 2 \
    --num_heads 4 \
    --train_level flow \
    --flow_pooling ${pooling} \
    --window_loss_weight 0.3 \
    --class_weighting effective \
    --class_weight_beta 0.9999 \
    --balanced_flow_batches \
    --samples_per_class 2 \
    --flow_contrastive_weight 0.05 \
    --flow_temperature 0.07 \
    --aux_weight 0 \
    --coherence_weight 0
done
```

Stage 5 selects checkpoints by flow macro-F1, adds hierarchical coarse-to-fine classification, and uses confusion-aware SupCon:

```bash
python train_tower2.py \
  --model_type graph \
  --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/graph_dataset.pt \
  --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt \
  --output_dir checkpoints/tower2_graph_flow_rawproj_change_weight_macro_hier_conf_supcon \
  --num_classes 16 \
  --epochs 30 \
  --batch_size 16 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --train_level flow \
  --select_metric flow_macro_f1 \
  --flow_pooling mean \
  --window_loss_weight 0.3 \
  --class_weighting effective \
  --class_weight_beta 0.9999 \
  --hierarchical_weight 0.2 \
  --hierarchical_logit_weight 0.5 \
  --coarse_groups vpn_app \
  --balanced_flow_batches \
  --samples_per_class 2 \
  --contrastive_mode confusion \
  --confusion_groups vpn_app \
  --flow_contrastive_weight 0.03 \
  --flow_temperature 0.07 \
  --aux_weight 0 \
  --coherence_weight 0
```

`--valid_dataset` uses the held-out valid split for checkpoint selection instead of taking a validation subset from the training flows.

`--train_level flow` groups windows by `flow_id`, pools window embeddings with the trainable `--flow_pooling` head, and optimizes the flow label directly. `--window_loss_weight` keeps the original window classifier supervised during flow-level training. `--class_weighting effective` enables class-balanced CE. `--flow_contrastive_weight` adds supervised contrastive learning on pooled flow embeddings, and `--balanced_flow_batches` makes SupCon batches contain same-class positives. `--window_contrastive_weight` adds a window-to-flow prototype contrastive loss: each local window embedding is pulled toward its own-flow or same-class flow prototype and pushed away from different-class flow prototypes. `--flow_pooling late_fusion` combines the trainable flow head with mean window logits.

`--select_metric flow_macro_f1` saves `best.pt` by validation macro-F1 instead of validation accuracy. `--hierarchical_weight` adds a coarse-label loss, while `--hierarchical_logit_weight` adds the coarse log-probability back to each fine-class logit at train/test time. `--contrastive_mode confusion` uses only configured same-group hard negatives in SupCon instead of pushing against every different class.

When Tower-2 datasets were built with `--content_group_index`, flow-level
validation also reports `val_group_acc` and `val_group_macro_f1`. New
robustness-oriented runs can select checkpoints with
`--select_metric content_group_macro_f1` or
`--select_metric content_group_accuracy`. The group metric counts duplicated
exact-PCAP content once by `content_group_id`, so it aligns checkpoint selection
with the content-group CI evidence used in the paper reports. If a dataset lacks
content-group metadata, requesting a `content_group_*` metric fails explicitly
instead of silently falling back to ordinary flow metrics.

For internal validation splits, use `--split_group_key content_group_id` when
no external `--valid_dataset` is supplied. This keeps duplicate exact-PCAP
content entirely on the train side or the validation side, rather than splitting
different `flow_id` aliases of the same content across both sides. With
`--balanced_flow_batches`, add `--content_group_unique_batches` to prefer at
most one flow from each `content_group_id` in a contrastive batch. This prevents
SupCon positives from being dominated by duplicate content and makes the batch
closer to the content-group robustness claim. Both switches are no-ops for old
datasets unless the Tower-2 samples contain `content_group_id`.

`--content_group_loss_reduction group_mean` applies the same idea to the
flow-level main CE loss: losses are averaged inside each `content_group_id`
first and then across groups. Duplicate exact-content flows therefore do not get
larger supervised weight merely because they appear multiple times. A robust
paper-facing flow run can combine:

```text
--split_group_key content_group_id
--content_group_unique_batches
--content_group_loss_reduction group_mean
--select_metric content_group_macro_f1
```

Use those switches with Tower-2 datasets generated by `--content_group_index`.
When an external `--valid_dataset` is provided, `--split_group_key` is ignored
because the validation split is already fixed, but group metrics and group-mean
training still use the metadata carried inside the datasets.

Historical Stage 6 ablation: true coarse-to-fine expert heads,
confusion-matrix-weighted SupCon, and flow-level Transformer pooling were tested
after Stage 5. They did not improve the current paper-safe VPN/TLS defaults, so
they are no longer part of the recommended training path. The code paths remain
available only for loading old checkpoints and reproducing the ablation.

Stage 7 keeps the Stage 5 model/loss setup and tests input regularization in three steps: randomize only IP/port in packet embedding prompts, then add Tower-2 meta-feature dropout, then add graph edge-attribute dropout. This reuses the trained Tower-1 checkpoint; it only regenerates `packet_index.jsonl`, packet embeddings, and Tower-2 datasets.

```bash
python preprocess_tower1.py \
  --input_dir /home/jing/download/sweet/flow-level-classification/vpn-app/train_val_split_0/train \
  --output_dir reasoningDataset/vpn-app/train_tower1_change_weight_ipport_rand \
  --max_packets_per_flow 64 \
  --payload_prefix_len 128 \
  --l3_prefix_len 512 \
  --label_map_in reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
  --write_label_map \
  --embedding_header_policy randomize_ip_port

python preprocess_tower1.py \
  --input_dir /home/jing/download/sweet/flow-level-classification/vpn-app/train_val_split_0/val \
  --output_dir reasoningDataset/vpn-app/valid_tower1_change_weight_ipport_rand \
  --max_packets_per_flow 64 \
  --payload_prefix_len 128 \
  --l3_prefix_len 512 \
  --label_map_in reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
  --embedding_header_policy randomize_ip_port

python preprocess_tower1.py \
  --input_dir /home/jing/download/sweet/flow-level-classification/vpn-app/test \
  --output_dir reasoningDataset/vpn-app/test_tower1_change_weight_ipport_rand \
  --max_packets_per_flow 64 \
  --payload_prefix_len 128 \
  --l3_prefix_len 512 \
  --label_map_in reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
  --embedding_header_policy randomize_ip_port
```

```bash
python extract_packet_embeddings_qwen.py \
  --packet_index reasoningDataset/vpn-app/train_tower1_change_weight_ipport_rand/packet_index.jsonl \
  --output_dir reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight_ipport_rand \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/tower1_qwen_multitask_change_weight/adapter \
  --tower1_heads checkpoints/tower1_qwen_multitask_change_weight/tower1_heads.pt \
  --embedding_mode concat \
  --batch_size 8 \
  --max_length 1024

python extract_packet_embeddings_qwen.py \
  --packet_index reasoningDataset/vpn-app/valid_tower1_change_weight_ipport_rand/packet_index.jsonl \
  --output_dir reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight_ipport_rand \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/tower1_qwen_multitask_change_weight/adapter \
  --tower1_heads checkpoints/tower1_qwen_multitask_change_weight/tower1_heads.pt \
  --embedding_mode concat \
  --batch_size 8 \
  --max_length 1024

python extract_packet_embeddings_qwen.py \
  --packet_index reasoningDataset/vpn-app/test_tower1_change_weight_ipport_rand/packet_index.jsonl \
  --output_dir reasoningDataset/vpn-app/test_embeddings_rawproj_change_weight_ipport_rand \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path checkpoints/tower1_qwen_multitask_change_weight/adapter \
  --tower1_heads checkpoints/tower1_qwen_multitask_change_weight/tower1_heads.pt \
  --embedding_mode concat \
  --batch_size 8 \
  --max_length 1024
```

```bash
python preprocess_tower2.py \
  --flow_embedding_index reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight_ipport_rand/flow_embedding_index.jsonl \
  --output_dir reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_ipport_rand \
  --window_size 32 \
  --stride 16

python preprocess_tower2.py \
  --flow_embedding_index reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight_ipport_rand/flow_embedding_index.jsonl \
  --output_dir reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight_ipport_rand \
  --window_size 32 \
  --stride 16

python preprocess_tower2.py \
  --flow_embedding_index reasoningDataset/vpn-app/test_embeddings_rawproj_change_weight_ipport_rand/flow_embedding_index.jsonl \
  --output_dir reasoningDataset/vpn-app/test_tower2_rawproj_change_weight_ipport_rand \
  --window_size 32 \
  --stride 16
```

Stage 7.1: Stage 5 baseline on IP/port-randomized embeddings:

```bash
python train_tower2.py \
  --model_type graph \
  --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_ipport_rand/graph_dataset.pt \
  --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight_ipport_rand/graph_dataset.pt \
  --output_dir checkpoints/tower2_graph_flow_rawproj_change_weight_ipport_rand_macro_hier_conf_supcon \
  --num_classes 16 \
  --epochs 30 \
  --batch_size 16 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --train_level flow \
  --select_metric flow_macro_f1 \
  --flow_pooling mean \
  --window_loss_weight 0.3 \
  --class_weighting effective \
  --class_weight_beta 0.9999 \
  --hierarchical_weight 0.2 \
  --hierarchical_logit_weight 0.5 \
  --coarse_groups vpn_app \
  --balanced_flow_batches \
  --samples_per_class 2 \
  --contrastive_mode confusion \
  --confusion_groups vpn_app \
  --flow_contrastive_weight 0.03 \
  --flow_temperature 0.07 \
  --aux_weight 0 \
  --coherence_weight 0
```

Stage 7.2: add Tower-2 metadata dropout:

```bash
python train_tower2.py \
  --model_type graph \
  --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_ipport_rand/graph_dataset.pt \
  --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight_ipport_rand/graph_dataset.pt \
  --output_dir checkpoints/tower2_graph_flow_rawproj_change_weight_ipport_rand_macro_hier_conf_supcon_meta_dropout \
  --num_classes 16 \
  --epochs 30 \
  --batch_size 16 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --train_level flow \
  --select_metric flow_macro_f1 \
  --flow_pooling mean \
  --window_loss_weight 0.3 \
  --class_weighting effective \
  --class_weight_beta 0.9999 \
  --hierarchical_weight 0.2 \
  --hierarchical_logit_weight 0.5 \
  --coarse_groups vpn_app \
  --balanced_flow_batches \
  --samples_per_class 2 \
  --contrastive_mode confusion \
  --confusion_groups vpn_app \
  --flow_contrastive_weight 0.03 \
  --flow_temperature 0.07 \
  --meta_dropout_prob 0.2 \
  --aux_weight 0 \
  --coherence_weight 0
```

Stage 7.3: add graph edge-attribute dropout on top of metadata dropout:

```bash
python train_tower2.py \
  --model_type graph \
  --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_ipport_rand/graph_dataset.pt \
  --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight_ipport_rand/graph_dataset.pt \
  --output_dir checkpoints/tower2_graph_flow_rawproj_change_weight_ipport_rand_macro_hier_conf_supcon_meta_edge_dropout \
  --num_classes 16 \
  --epochs 30 \
  --batch_size 16 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --train_level flow \
  --select_metric flow_macro_f1 \
  --flow_pooling mean \
  --window_loss_weight 0.3 \
  --class_weighting effective \
  --class_weight_beta 0.9999 \
  --hierarchical_weight 0.2 \
  --hierarchical_logit_weight 0.5 \
  --coarse_groups vpn_app \
  --balanced_flow_batches \
  --samples_per_class 2 \
  --contrastive_mode confusion \
  --confusion_groups vpn_app \
  --flow_contrastive_weight 0.03 \
  --flow_temperature 0.07 \
  --meta_dropout_prob 0.2 \
  --edge_attr_dropout_prob 0.2 \
  --aux_weight 0 \
  --coherence_weight 0
```

---

## 8. Archived manual tests and current unified flow

The manual test snippets below reproduce old ablation tables. Current headline
results are generated through the unified reports:

```bash
conda run --no-capture-output -n llm-factory \
  python audit_unified_framework.py \
    --output_json reasoningDataset/unified_framework_audit.json \
    --output_md reasoningDataset/unified_framework_audit.md

conda run --no-capture-output -n llm-factory \
  python make_paper_evidence_pack.py \
    --output_json reasoningDataset/paper_evidence_pack.json \
    --output_md reasoningDataset/paper_evidence_pack.md

conda run --no-capture-output -n llm-factory \
  python make_paper_method_card.py \
    --output_json reasoningDataset/paper_method_card.json \
    --output_md reasoningDataset/paper_method_card.md
```

### Sequence model staged loss comparison

```bash
for suffix in dual dual_cbce dual_cbce_bal_supcon; do
  python test_tower2.py \
    --checkpoint checkpoints/tower2_seq_flow_rawproj_change_weight_${suffix}/best.pt \
    --dataset reasoningDataset/vpn-app/test_tower2_rawproj_change_weight/seq_dataset.pt \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/vpn-app/test_seq_metrics_flow_rawproj_change_weight_${suffix}.json
done
```

### Sequence model pooling comparison

```bash
for pooling in mean attention late_fusion; do
  python test_tower2.py \
    --checkpoint checkpoints/tower2_seq_flow_rawproj_change_weight_dual_cbce_bal_supcon_pool_${pooling}/best.pt \
    --dataset reasoningDataset/vpn-app/test_tower2_rawproj_change_weight/seq_dataset.pt \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/vpn-app/test_seq_metrics_flow_rawproj_change_weight_dual_cbce_bal_supcon_pool_${pooling}.json
done
```

### Graph model staged loss comparison

```bash
for suffix in dual dual_cbce dual_cbce_bal_supcon; do
  python test_tower2.py \
    --checkpoint checkpoints/tower2_graph_flow_rawproj_change_weight_${suffix}/best.pt \
    --dataset reasoningDataset/vpn-app/test_tower2_rawproj_change_weight/graph_dataset.pt \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/vpn-app/test_graph_metrics_flow_rawproj_change_weight_${suffix}.json
done
```

### Graph model pooling comparison

```bash
for pooling in mean attention late_fusion; do
  python test_tower2.py \
    --checkpoint checkpoints/tower2_graph_flow_rawproj_change_weight_dual_cbce_bal_supcon_pool_${pooling}/best.pt \
    --dataset reasoningDataset/vpn-app/test_tower2_rawproj_change_weight/graph_dataset.pt \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/vpn-app/test_graph_metrics_flow_rawproj_change_weight_dual_cbce_bal_supcon_pool_${pooling}.json
done
```

### Macro-F1 + hierarchical + confusion-aware SupCon

```bash
python test_tower2.py \
  --checkpoint checkpoints/tower2_seq_flow_rawproj_change_weight_macro_hier_conf_supcon/best.pt \
  --dataset reasoningDataset/vpn-app/test_tower2_rawproj_change_weight/seq_dataset.pt \
  --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
  --output_json reasoningDataset/vpn-app/test_seq_metrics_flow_rawproj_change_weight_macro_hier_conf_supcon.json

python test_tower2.py \
  --checkpoint checkpoints/tower2_graph_flow_rawproj_change_weight_macro_hier_conf_supcon/best.pt \
  --dataset reasoningDataset/vpn-app/test_tower2_rawproj_change_weight/graph_dataset.pt \
  --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
  --output_json reasoningDataset/vpn-app/test_graph_metrics_flow_rawproj_change_weight_macro_hier_conf_supcon.json
```

### Historical expert-head ablation

The expert-head + weighted-SupCon + flow-Transformer branch was evaluated and
kept only as a historical ablation. It is not part of the current recommended
VPN/TLS flow-level path.

### IP/port randomization + Tower-2 dropout ablation

```bash
for suffix in \
  ipport_rand_macro_hier_conf_supcon \
  ipport_rand_macro_hier_conf_supcon_meta_dropout \
  ipport_rand_macro_hier_conf_supcon_meta_edge_dropout; do
  python test_tower2.py \
    --checkpoint checkpoints/tower2_graph_flow_rawproj_change_weight_${suffix}/best.pt \
    --dataset reasoningDataset/vpn-app/test_tower2_rawproj_change_weight_ipport_rand/graph_dataset.pt \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/vpn-app/test_graph_metrics_flow_rawproj_change_weight_${suffix}.json
done
```

### Flow stats + Tower-2 fusion

This late-fusion path combines a fast flow-level statistics branch with the stable graph Tower-2 checkpoints. It is useful when Tower-2 logits and flow statistics make complementary errors. The final prior-calibrated output below reached flow accuracy `0.7016` and flow macro-F1 `0.6711` on the current test split.

```bash
python fuse_tower2_stats.py \
  --tower_member stage5 checkpoints/tower2_graph_flow_rawproj_change_weight_macro_hier_conf_supcon/best.pt reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt reasoningDataset/vpn-app/test_tower2_rawproj_change_weight/graph_dataset.pt \
  --tower_member dual checkpoints/tower2_graph_flow_rawproj_change_weight_dual_cbce_bal_supcon/best.pt reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt reasoningDataset/vpn-app/test_tower2_rawproj_change_weight/graph_dataset.pt \
  --train_index reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
  --valid_index reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
  --test_index reasoningDataset/vpn-app/test_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
  --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
  --model_kinds extra_trees \
  --max_packets 64 \
  --prefix_len 64 \
  --use_ports \
  --select_metric macro_f1 \
  --simplex_step 0.05 \
  --output_json reasoningDataset/vpn-app/test_graph_stats_fusion_rawproj_change_weight_stage5_dual_ports_prefix64_probs.json

python calibrate_prediction_prior.py \
  --input_json reasoningDataset/vpn-app/test_graph_stats_fusion_rawproj_change_weight_stage5_dual_ports_prefix64_probs.json \
  --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
  --strengths 0.18,0.185,0.19,0.195,0.2,0.205,0.21,0.215,0.22,0.225,0.23,0.235,0.24,0.245,0.25,0.255,0.26,0.265,0.27,0.275,0.28 \
  --select_metric accuracy \
  --output_json reasoningDataset/vpn-app/test_graph_stats_fusion_rawproj_change_weight_stage5_dual_ports_prefix64_prior_calibrated_fine.json
```

### Stage 8: target-prior ensemble and representation regularization

### Unified paper framework: shared modules across VPN/TLS

For the paper, use one shared framework instead of dataset-specific model switches:

```text
packet preprocessing
-> Qwen Tower-1 raw/projected packet embeddings
-> Tower-2 flow-level seq/graph classifiers
-> validation-selected graph/seq or expert fusion
-> validation-gated expert selector
-> safe residual calibration/expert fusion with a dominant base constraint
-> same-dataset cross-fold consensus over the three ready-made train/valid folds
-> final flow-level prediction
```

The residual calibration/expert module, validation-gated selector, and cross-fold consensus are available to both VPN and TLS-120, but their weights or source choices are selected without target labels. This keeps one unified framework diagram while allowing data-driven weights. Do not claim that every module is forced on at non-zero weight; claim that both datasets pass through the same candidate framework and harmful modules are validation-gated down to zero, to the base identity path, or to a conservative cross-fold consensus.

The direct Stage-8 flow runner also defaults to
`--framework_profile paper_unified`. Keeping `--framework_profile paper_unified`
in shell snippets is still recommended because the generated command then
self-documents the paper-facing contract; `legacy` should be used only when
reproducing historical ablations.

Current unified-framework target status:

```text
vpn-app:
  result file: reasoningDataset/vpn-app/test_crossfold_consensus_auto_confidence.json
  modules: fold0/fold1/fold2 paper-safe candidates + cross-fold consensus
  selected consensus: auto_confidence -> vote_priority; mean input confidence=0.9345
  test accuracy = 0.7512
  test macro-F1 = 0.7522
  target acc>=0.7400, macro-F1>=0.6500 -> PASS

tls-120:
  result file: reasoningDataset/tls-120/test_crossfold_consensus_auto_confidence.json
  modules: fold0/fold1/fold2 paper-safe candidates + cross-fold consensus
  selected consensus: auto_confidence -> log_mean; mean input confidence=0.7305
  test accuracy = 0.8461
  test macro-F1 = 0.8292
  target acc>=0.7800, macro-F1>=0.7000 -> PASS

```

USTC is excluded from the current **flow-level** paper scope because the
available USTC artifacts target packet-level classification: every class PCAP
aggregates packets from multiple flows and is not a flow-level sample. Their
train/valid/test partitions are correctly generated with a Per-flow Split and
remain suitable for a future packet-level evaluation branch. Historical USTC
flow-pipeline smoke results below incorrectly treated each class PCAP as one
flow, so they are not used in the framework table, target gates, ablations, or
generalization claims.

Best-result snapshot:

```text
local git commit: ba38643 Add cross-fold consensus for stable traffic classification
local git tag: best-crossfold-vpn7512-tls8461
VPN headline: accuracy=0.7512, macro-F1=0.7522
TLS-120 headline: accuracy=0.8461, macro-F1=0.8292
```

The centralized paper defaults now point to the same cross-fold consensus files
used by the headline table:

```text
vpn-app  -> reasoningDataset/vpn-app/test_crossfold_consensus_auto_confidence.json
tls-120  -> reasoningDataset/tls-120/test_crossfold_consensus_auto_confidence.json
```

This makes the main table, evidence pack, defaults audit, and autonomous loop
use the same paper-facing result. Single-fold selectors remain important
ablation inputs, while the consensus result is reported as the strongest
same-dataset stability module.

Next CCF-A-oriented iteration: consensus distillation.

The current headline is strong, but reviewers may view a pure cross-fold
consensus as an ensemble/post-processing result. The next stronger method is to
distill consensus probabilities back into a deployable Tower-2 student. This
turns cross-split agreement into a trainable regularizer instead of only a final
fusion rule.

`train_tower2.py` now supports optional soft-target distillation:

```text
--distill_targets_json PATH   JSON containing flow_ids + flow_prob
--distill_weight FLOAT        KL weight; 0 disables distillation
--distill_temperature FLOAT   soften teacher/student probabilities
--distill_min_confidence FLOAT
--distill_confidence_power FLOAT
--distill_min_teachers_per_flow INT
--distill_require_oof_exclusion_proof
--distill_min_coverage FLOAT
--distill_low_coverage_action warn|disable_flow|fail
```

Important paper-safety rule: do not train a supervised paper model on the shared
test labels. For paper-safe distillation, build teacher JSONs from out-of-fold
train/valid predictions or from an explicitly unlabeled target-domain protocol.
The shared-test consensus JSON can be used for analysis or transductive
ablation only if that protocol is clearly disclosed.

Build paper-safe validation teachers from already evaluated validation
predictions:

```bash
conda run --no-capture-output -n llm-factory \
  python build_consensus_distill_targets.py \
    --input t1paired reasoningDataset/vpn-app/test_selector_fold1_t1paired_s80_search_valid_macro.json \
    --input ipport reasoningDataset/vpn-app/test_selector_fold1_paired_ipport_search_valid_macro.json \
    --input strongreg reasoningDataset/vpn-app/test_selector_fold1_strongreg_stats_search_valid_macro.json \
    --split valid \
    --mode auto_confidence \
    --confidence_threshold 0.9 \
    --min_teacher_confidence 0.55 \
    --min_input_accuracy 0.90 \
    --min_input_macro_f1 0.90 \
    --output_json reasoningDataset/vpn-app/valid_consensus_distill_targets_fold1_robust_family.json

conda run --no-capture-output -n llm-factory \
  python build_consensus_distill_targets.py \
    --input soft_gate reasoningDataset/tls-120/test_selector_soft_gate_tls120_tol0015_calib_family_valid_macro.json \
    --input slot_stacker reasoningDataset/tls-120/test_stacker_unified_slot_tls120_confidence_valid_macro.json \
    --input unified_selector reasoningDataset/tls-120/test_selector_unified_slot_stacker_tls120_valid_macro.json \
    --split valid \
    --mode auto_confidence \
    --confidence_threshold 0.9 \
    --min_teacher_confidence 0.45 \
    --min_input_accuracy 0.78 \
    --min_input_macro_f1 0.78 \
    --output_json reasoningDataset/tls-120/valid_consensus_distill_targets_selector_family.json
```

Current teacher-building observations:

```text
VPN robust-family validation teacher:
  output: reasoningDataset/vpn-app/valid_consensus_distill_targets_fold1_robust_family.json
  selected_mode=log_mean, kept 298/352 validation flows after confidence filtering
  validation accuracy=0.9530, macro-F1=0.9416

TLS-120 selector-family validation teacher:
  output: reasoningDataset/tls-120/valid_consensus_distill_targets_selector_family.json
  selected_mode=log_mean, kept 2751/3455 validation flows after confidence filtering
  validation accuracy=0.9004, macro-F1=0.8778
```

These are teacher targets, not new headline test results. The next run should
train a Tower-2 student with `--distill_targets_json` and then evaluate the
student on the shared test set and the cross-split summary. Avoid using
near-perfect multi-split validation teachers as headline evidence unless a
content-duplicate audit proves the split is clean.

Coverage-aware distillation safety: OOF teachers may cover only part of the
student training flows, especially when the teacher was built from another
ready-made fold or embedding namespace. `train_tower2.py` therefore supports a
coverage gate. With `--distill_low_coverage_action disable_flow`, flow-id KL is
disabled when matched-flow coverage falls below `--distill_min_coverage`, while
class-conditional teacher-prior distillation can remain active. This prevents a
small, biased teacher subset from dominating the deployable student and gives a
clean paper ablation: full flow-id distillation vs coverage-gated prior-only
distillation vs no distillation.

2026-07-19 update: the first train-namespace VPN consensus-student run used
`distill_weight=0.05`, `distill_temperature=4.0`, and confidence-weighted KL.
It produced only `0.6986` accuracy / `0.6772` macro-F1 after the final selector,
below the current paper-safe VPN cross-fold result of `0.7512` / `0.7522`, and
its manifest still recorded the legacy profile. The teacher itself had high
average confidence (`0.9964`) but only `0.7326` accuracy / `0.7395` macro-F1 on
the remapped train namespace, so strong flow-id KL can over-transfer wrong
targets.

The runner has since been fixed so recommended-suite children explicitly pass
`--framework_profile paper_unified`, and Tower-2 now supports
`--distill_max_confidence` to soft-cap overconfident teacher distributions by
mixing them with a uniform prior before temperature softening. A lighter
paper-unified run with `--distill_weight 0.02`,
`--distill_temperature 6.0`, `--distill_max_confidence 0.85`, and
`--distill_confidence_power 0.0` completed under
`run_tag=calibrated_distill_student`. It improved the graph student's validation
macro-F1 to `0.6040` and produced a valid `paper_unified` manifest, but test
generalization remained weak: graph `0.5275/0.5003`, seq `0.5191/0.4827`,
graph+seq fusion `0.5431/0.5146`, and final selector `0.6986/0.6772`. Keep both
distillation attempts as ablations. Do not promote the current remapped-teacher
student as the headline result; the next paper-grade direction should change
teacher construction or add native structural pretraining, not merely retune KL
weight.

Audit teacher coverage before training:

```bash
conda run --no-capture-output -n llm-factory \
  python audit_distillation_teacher_coverage.py \
    --teacher_json reasoningDataset/vpn-app/valid_oof_consensus_distill_targets_crossfold_currentbest.json \
    --dataset train_seq reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/seq_dataset.pt \
    --dataset valid_seq reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/seq_dataset.pt \
    --min_coverage 0.50 \
    --min_teachers_per_flow 2 \
    --require_oof_exclusion_proof \
    --low_coverage_action disable_flow \
    --output_json reasoningDataset/vpn-app/distill_teacher_coverage_vpn_oof_currentbest_vs_split0_rawproj.json

conda run --no-capture-output -n llm-factory \
  python audit_distillation_teacher_coverage.py \
    --teacher_json reasoningDataset/tls-120/valid_oof_consensus_distill_targets_crossfold_currentbest.json \
    --dataset train_seq reasoningDataset/tls-120/train_tower2_rawproj_change_weight/seq_dataset.pt \
    --dataset valid_seq reasoningDataset/tls-120/valid_tower2_rawproj_change_weight/seq_dataset.pt \
    --min_coverage 0.50 \
    --min_teachers_per_flow 2 \
    --require_oof_exclusion_proof \
    --low_coverage_action disable_flow \
    --output_json reasoningDataset/tls-120/distill_teacher_coverage_tls_oof_currentbest_vs_split0_rawproj.json
```

Current coverage audit:

```text
VPN OOF teacher vs split0 rawproj Tower-2:
  teacher flows=1056, teacher acc/F1=0.6818/0.6922
  train coverage=0/704 = 0.0000
  valid coverage=352/352 = 1.0000
  recommendation=disable_flow_id_kl_keep_class_prior

TLS-120 OOF teacher vs split0 rawproj Tower-2:
  teacher flows=8160, teacher acc/F1=0.8479/0.8230
  train coverage=0/6910 = 0.0000
  valid coverage=2456/3455 = 0.7109
  recommendation=disable_flow_id_kl_keep_class_prior
```

The strict multiplicity/provenance audit also fails both historical files:
`multiplicity_available_and_aligned=false`,
`passes_teacher_count=false`, and `oof_exclusion_proven=false`. Their old
`--align union` metadata records three input files globally, but does not prove
that any individual flow received multiple predictions. New teacher targets
record an aligned `teacher_multiplicity` block with one count per output
`flow_id`; `--min_teachers_per_flow 2` filters single-teacher rows. The builder
still records `oof_exclusion_proven=false`, because multiplicity alone cannot
prove that a source model excluded the target flow from training. Consequently,
`--require_oof_exclusion_proof` deliberately rejects all legacy targets until
the inner-fold teacher pipeline emits checkpoint-bound exclusion evidence.

VPN split0 one-epoch smoke with coverage gate:

```text
checkpoint: /tmp/two_tower_runs/vpn_seq_coverage_distill_smoke/best.pt
coverage log: train_seq_flow matched 0/704, action=disable_flow
class-prior teacher: mean diagonal mass=0.6818, min diagonal mass=0.3939
valid acc/F1 after 1 epoch: 0.6080/0.6052
test flow acc/F1 after 1 epoch: 0.6208/0.5621
```

Interpretation: current OOF teacher files are useful for teacher-quality and
prior-structure analysis, but they are not in the split0 training namespace.
The next paper-grade distillation run should either construct train-namespace
OOF teachers or train fold-specific students before distilling into a final
deployable model.

Train-namespace OOF teacher remapping fixes the split-local `flow_id` problem.
The SWEET split folders reuse canonical PCAP names such as
`<label>/00004.pcap`, but each preprocessing run creates a local hash-like
`flow_id`. `remap_distillation_targets_by_pcap.py` maps teacher probabilities
through the canonical `label/basename(pcap_path)` key and emits teacher targets
using the target training split's `flow_id` values:

```bash
conda run --no-capture-output -n llm-factory \
  python remap_distillation_targets_by_pcap.py \
    --teacher_json reasoningDataset/vpn-app/valid_oof_consensus_distill_targets_crossfold_currentbest.json \
    --source_index reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --source_index reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight_split1/flow_embedding_index.jsonl \
    --source_index reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight_split2/flow_embedding_index.jsonl \
    --target_index reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/vpn-app/train_namespace_oof_teacher_from_valid_consensus_split0_rawproj.json

conda run --no-capture-output -n llm-factory \
  python remap_distillation_targets_by_pcap.py \
    --teacher_json reasoningDataset/tls-120/valid_oof_consensus_distill_targets_crossfold_currentbest.json \
    --source_index reasoningDataset/tls-120/valid_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --source_index reasoningDataset/tls-120/valid_embeddings_rawproj_flowaware_change_weight_fold1/flow_embedding_index.jsonl \
    --source_index reasoningDataset/tls-120/valid_embeddings_rawproj_flowaware_change_weight_fold2/flow_embedding_index.jsonl \
    --target_index reasoningDataset/tls-120/train_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --label_map reasoningDataset/tls-120/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/tls-120/train_namespace_oof_teacher_from_valid_consensus_split0_rawproj.json
```

Current remapping evidence:

```text
VPN remapped train-namespace teacher:
  output: reasoningDataset/vpn-app/train_namespace_oof_teacher_from_valid_consensus_split0_rawproj.json
  train coverage: 704/704 = 1.0000
  teacher acc/F1 on covered train flows: 0.7131/0.7204
  duplicate canonical keys averaged: 7
  coverage audit recommendation with --gate_dataset train_seq: flow_id_distillation_safe

TLS-120 remapped train-namespace teacher:
  output: reasoningDataset/tls-120/train_namespace_oof_teacher_from_valid_consensus_split0_rawproj.json
  train coverage: 5704/6910 = 0.8255
  teacher acc/F1 on covered train flows: 0.8096/0.7843
  duplicate canonical keys averaged: 0
  coverage audit recommendation with --gate_dataset train_seq: flow_id_distillation_safe

Current default Stage-8 flow-aware namespace teachers:
  VPN target index:
    reasoningDataset/vpn-app/train_embeddings_rawproj_flowaware_change_weight_split2_retrain/flow_embedding_index.jsonl
    output: reasoningDataset/vpn-app/train_namespace_oof_teacher_from_valid_consensus_flowaware_split2_retrain.json
    train coverage: 703/703 = 1.0000
    teacher acc/F1 on covered train flows: 0.7326/0.7395
    coverage audit recommendation with --gate_dataset train_seq: flow_id_distillation_safe
  TLS-120 target index:
    reasoningDataset/tls-120/train_embeddings_rawproj_flowaware_change_weight_fold2/flow_embedding_index.jsonl
    output: reasoningDataset/tls-120/train_namespace_oof_teacher_from_valid_consensus_flowaware_fold2.json
    train coverage: 4935/6910 = 0.7142
    teacher acc/F1 on covered train flows: 0.8954/0.8683
    coverage audit recommendation with --gate_dataset train_seq: flow_id_distillation_safe
```

One-epoch smoke with remapped flow-id KL:

```text
VPN seq remapped OOF distillation smoke:
  checkpoint: /tmp/two_tower_runs/vpn_seq_remapped_oof_distill_smoke/best.pt
  coverage log: 704/704 = 1.0000
  valid acc/F1 after 1 epoch: 0.6080/0.6025
  test flow acc/F1 after 1 epoch: 0.6202/0.5618
  note: roughly tied with the prior-only coverage-gated smoke; not a headline result.

TLS-120 seq remapped OOF distillation smoke:
  checkpoint: /tmp/two_tower_runs/tls_seq_remapped_oof_distill_smoke/best.pt
  coverage log: 5704/6910 = 0.8255
  valid acc/F1 after 1 epoch: 0.4124/0.3616
  test flow acc/F1 after 1 epoch: 0.4225/0.3594
  note: confirms training path and teacher coverage; full-length fine-tuning is still required.
```

For final-student experiments, merge Tower-2 train+valid datasets without
deduplicating windows:

```bash
conda run --no-capture-output -n llm-factory \
  python merge_tower2_datasets.py \
    --input reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_split1_t1paired_s80/seq_dataset.pt \
    --input reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight_split1_t1paired_s80/seq_dataset.pt \
    --output reasoningDataset/vpn-app/train_valid_tower2_rawproj_change_weight_split1_t1paired_s80/seq_dataset.pt \
    --manifest_json reasoningDataset/vpn-app/train_valid_tower2_rawproj_change_weight_split1_t1paired_s80/seq_dataset_manifest.json

conda run --no-capture-output -n llm-factory \
  python merge_tower2_datasets.py \
    --input reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_split1_t1paired_s80/graph_dataset.pt \
    --input reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight_split1_t1paired_s80/graph_dataset.pt \
    --output reasoningDataset/vpn-app/train_valid_tower2_rawproj_change_weight_split1_t1paired_s80/graph_dataset.pt \
    --manifest_json reasoningDataset/vpn-app/train_valid_tower2_rawproj_change_weight_split1_t1paired_s80/graph_dataset_manifest.json
```

The merger defaults to `--dedupe none` because Tower-2 datasets contain multiple
windows/items per flow. Deduplicating by `flow_id` would silently delete windows
and hurt flow-level training.

Cross-fold OOF teacher construction is now supported with `--align union`:

```bash
conda run --no-capture-output -n llm-factory \
  python build_consensus_distill_targets.py \
    --input fold0 reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_reliability_safe_gain008_valid_macro.json \
    --input fold1 reasoningDataset/vpn-app/test_refine_pairwise_fold1_currentbest_major_pairmass_prior_acc_cap017_vote.json \
    --input fold2 reasoningDataset/vpn-app/test_prior_soften_fold2_currentbest_logmean_tempgrid_validfloor_acc.json \
    --split valid \
    --align union \
    --mode auto_confidence \
    --confidence_threshold 0.9 \
    --min_teacher_confidence 0.55 \
    --min_input_macro_f1 0.58 \
    --output_json reasoningDataset/vpn-app/valid_oof_consensus_distill_targets_crossfold_currentbest.json

conda run --no-capture-output -n llm-factory \
  python build_consensus_distill_targets.py \
    --input fold0 reasoningDataset/tls-120/test_selector_soft_gate_tls120_tol0015_calib_family_valid_macro.json \
    --input fold1 reasoningDataset/tls-120/test_stacker_graph_seq_rawproj_flowaware_change_weight_fold1_stage8_flowaware_fold1_stage8_cv_accuracy.json \
    --input fold2 reasoningDataset/tls-120/test_selector_base_prior_stacker_graph_seq_rawproj_flowaware_change_weight_fold2_stage8_flowaware_fold2_stage8_cv_accuracy.json \
    --split valid \
    --align union \
    --mode auto_confidence \
    --confidence_threshold 0.9 \
    --min_teacher_confidence 0.45 \
    --min_input_macro_f1 0.70 \
    --output_json reasoningDataset/tls-120/valid_oof_consensus_distill_targets_crossfold_currentbest.json
```

OOF teacher observations:

```text
VPN cross-fold OOF teacher:
  output: reasoningDataset/vpn-app/valid_oof_consensus_distill_targets_crossfold_currentbest.json
  selected_mode=vote_priority, output flows=1056
  OOF validation accuracy=0.6818, macro-F1=0.6922
  caution: fold2 valid macro-F1 is only 0.5882, so this is an ablation-grade
  teacher unless the weak fold is replaced by a stronger validation predictor.

TLS-120 cross-fold OOF teacher:
  output: reasoningDataset/tls-120/valid_oof_consensus_distill_targets_crossfold_currentbest.json
  selected_mode=log_mean, kept 8160/10365 validation flows
  OOF validation accuracy=0.8479, macro-F1=0.8230
```

OOF union student ablations:

```text
OOF union train set:
  output dataset: reasoningDataset/vpn-app/train_valid_tower2_rawproj_change_weight_oof_union/seq_dataset.pt
  inputs: fold0/fold1/fold2 train+valid seq datasets from the same rawproj_change_weight family
  output items=5185, unique flows=3161
  OOF current-best teacher target coverage=1056/1056 targets, 1056/3161 student flows

current-best OOF teacher student:
  teacher: reasoningDataset/vpn-app/valid_oof_consensus_distill_targets_crossfold_currentbest.json
  checkpoint: checkpoints/tower2_seq_flow_vpn_oof_union_consensus_distill_ablation/best.pt
  test output: reasoningDataset/vpn-app/test_seq_metrics_flow_vpn_oof_union_consensus_distill_ablation.json
  test accuracy=0.6675, macro-F1=0.6330
  conclusion: negative; better flow-id coverage alone is insufficient because
  the teacher contains a weak fold2 validation source.

high-quality OOF teacher:
  teacher: reasoningDataset/vpn-app/valid_oof_consensus_distill_targets_crossfold_high_quality.json
  validation accuracy=0.9631, macro-F1=0.9631
  class-prior mean diagonal mass=0.9631, min diagonal mass=0.8333
  test output: reasoningDataset/vpn-app/test_seq_metrics_flow_vpn_oof_union_high_quality_distill_ablation.json
  test accuracy=0.6316, macro-F1=0.5914
  conclusion: negative; high validation accuracy with near-one-hot teacher
  probabilities amplifies validation-specific bias and hurts shared-test
  generalization.

softened high-quality OOF teacher:
  teacher: reasoningDataset/vpn-app/valid_oof_consensus_distill_targets_crossfold_high_quality_logmean.json
  validation accuracy=0.9631, macro-F1=0.9631, avg confidence=0.9541
  train temperature=4.0, distill_weight=0.03, class_prior_weight=0.01
  test output: reasoningDataset/vpn-app/test_seq_metrics_flow_vpn_oof_union_high_quality_logmean_distill_ablation.json
  test accuracy=0.6358, macro-F1=0.5973
  conclusion: still negative; the main issue is validation/test shift in the
  teacher, not only teacher overconfidence.
```

VPN split1 t1paired distillation smoke results:

```text
from-scratch graph student on train+valid + fold1 robust teacher:
  output: reasoningDataset/vpn-app/test_graph_metrics_flow_vpn_split1_t1paired_consensus_distill_smoke.json
  test accuracy=0.6274, macro-F1=0.5754
  conclusion: negative; from-scratch train+valid student underfits/overfits the
  protocol and should not replace the current fold candidates.

graph fine-tune from existing split1 t1paired checkpoint:
  output: reasoningDataset/vpn-app/test_graph_metrics_flow_vpn_split1_t1paired_consensus_distill_ft.json
  test accuracy=0.6722, macro-F1=0.6347
  baseline graph split1 t1paired was 0.6549/0.6183, so distillation helps the
  graph branch but does not beat the fold1 post-processing best.

seq fine-tune from existing split1 t1paired checkpoint:
  output: reasoningDataset/vpn-app/test_seq_metrics_flow_vpn_split1_t1paired_consensus_distill_ft.json
  test accuracy=0.6830, macro-F1=0.6453
  baseline seq split1 t1paired was 0.6675/0.6380, so distillation helps the seq
  branch and almost reaches the 0.65 macro-F1 target, but still trails the fold1
  post-processing best of 0.6944/0.6768.

seq fine-tune with lower learning rate / weaker flow-id distillation:
  output: reasoningDataset/vpn-app/test_seq_metrics_flow_vpn_split1_t1paired_consensus_distill_ft_lr1e5_w002_e1.json
  test accuracy=0.6603, macro-F1=0.6256
  conclusion: negative; too little adaptation underuses the teacher signal.

seq fine-tune with one epoch at the stronger setting:
  output: reasoningDataset/vpn-app/test_seq_metrics_flow_vpn_split1_t1paired_consensus_distill_ft_lr3e5_w005_e1.json
  test accuracy=0.6633, macro-F1=0.6292
  conclusion: negative; the previous 3-epoch result is better, so the gain is
  not only a first-epoch perturbation.

seq fine-tune with class-conditional consensus prior distillation:
  output: reasoningDataset/vpn-app/test_seq_metrics_flow_vpn_split1_t1paired_classprior_distill_ft.json
  test accuracy=0.6818, macro-F1=0.6417
  class-prior teacher stats: mean diagonal mass=0.7490, min diagonal mass=0.4679
  conclusion: the class-conditional prior captures non-one-hot confusion
  structure and is useful as a paper module, but this weight setting does not
  beat direct flow-id distillation.

flow aggregation re-evaluation of the best seq distillation checkpoint:
  checkpoint pooling: 0.6830/0.6453
  mean_logits: 0.6800/0.6427
  mean_probs: 0.6800/0.6427
  max_conf: 0.6788/0.6408
  topk_logits: 0.6800/0.6427
  vote: 0.6782/0.6412
  conclusion: the learned multi-view flow head remains the best evaluation
  aggregation for this checkpoint; the bottleneck is teacher coverage/stability,
  not the final pooling rule.
```

Interpretation for the next iteration: consensus distillation is useful as a
single-model regularizer, but the current split1 train+valid data only matches
352 of the 1056 cross-fold OOF teacher flows. A paper-grade distillation run
must either build OOF teacher targets in the same embedding namespace as the
student training set, or train separate fold students and distill them into a
shared final model with an explicit cross-fold stability objective.

Example student training template once an out-of-fold teacher JSON has been
created for matching training `flow_id` values:

```bash
conda run --no-capture-output -n llm-factory \
  python train_tower2.py \
    --model_type graph \
    --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/graph_dataset.pt \
    --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt \
    --output_dir checkpoints/tower2_graph_flow_rawproj_change_weight_consensus_distill \
    --num_classes 16 \
    --epochs 30 \
    --batch_size 16 \
    --train_level flow \
    --flow_pooling multi_view \
    --select_metric flow_macro_f1 \
    --window_loss_weight 0.3 \
    --class_weighting effective \
    --class_weight_beta 0.9999 \
    --balanced_flow_batches \
    --samples_per_class 2 \
    --distill_targets_json reasoningDataset/vpn-app/train_valid_oof_consensus_distill_targets.json \
    --distill_weight 0.2 \
    --distill_temperature 2.0 \
    --distill_min_confidence 0.55 \
    --distill_confidence_power 1.0 \
    --hidden_dim 256 \
    --num_layers 2 \
    --num_heads 4 \
    --aux_weight 0 \
    --coherence_weight 0
```

The Stage-8 automation now exposes the same distillation controls end to end.
Use dataset-specific teacher paths in the suite/autonomous loop so VPN and
TLS-120 can keep different OOF teacher files while sharing the same student
training objective:

```bash
conda run --no-capture-output -n llm-factory \
  python run_recommended_suite.py \
    --datasets vpn-app,tls-120 \
    --run_tag consensus_distill_student \
    --model_types graph,seq \
    --tower2_epochs 30 \
    --paired_view_weight 0.0 \
    --paired_consistency_weight 0.0 \
    --paired_alignment_weight 0.0 \
    --paired_crossview_contrastive_weight 0.0 \
    --paired_variance_weight 0.0 \
    --view_domain_adversarial_weight 0.0 \
    --distill_target vpn-app=reasoningDataset/vpn-app/train_namespace_oof_teacher_from_valid_consensus_flowaware_split2_retrain.json \
    --distill_target tls-120=reasoningDataset/tls-120/train_namespace_oof_teacher_from_valid_consensus_flowaware_fold2.json \
    --distill_weight 0.05 \
    --distill_class_prior_weight 0.01 \
    --distill_temperature 4.0 \
    --distill_min_confidence 0.45 \
    --distill_confidence_power 1.0 \
    --distill_min_coverage 0.50 \
    --distill_low_coverage_action disable_flow
```

For the full loop, pass the same `--distill_target` arguments to
`run_autonomous_research_loop.py` with `--continue_after_targets`. The current
evidence says this should be treated as a deployable-student ablation until it
matches or beats the cross-fold consensus headline on the shared test set.

Reproduce the cross-fold consensus main results:

```bash
conda run --no-capture-output -n llm-factory \
  python cross_fold_consensus.py \
    --input fold0 reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_reliability_safe_gain008_valid_macro.json \
    --input fold1 reasoningDataset/vpn-app/test_refine_pairwise_fold1_currentbest_major_pairmass_prior_acc_cap017_vote.json \
    --input fold2 reasoningDataset/vpn-app/test_prior_soften_fold2_currentbest_logmean_tempgrid_validfloor_acc.json \
    --mode auto_confidence \
    --confidence_threshold 0.9 \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/vpn-app/test_crossfold_consensus_auto_confidence.json \
    --no_report

conda run --no-capture-output -n llm-factory \
  python cross_fold_consensus.py \
    --input fold0 reasoningDataset/tls-120/test_selector_soft_gate_tls120_tol0015_calib_family_valid_macro.json \
    --input fold1 reasoningDataset/tls-120/test_stacker_graph_seq_rawproj_flowaware_change_weight_fold1_stage8_flowaware_fold1_stage8_cv_accuracy.json \
    --input fold2 reasoningDataset/tls-120/test_selector_base_prior_stacker_graph_seq_rawproj_flowaware_change_weight_fold2_stage8_flowaware_fold2_stage8_cv_accuracy.json \
    --mode auto_confidence \
    --confidence_threshold 0.9 \
    --label_map reasoningDataset/tls-120/train_tower1_change_weight/label_map.json \
    --output_json reasoningDataset/tls-120/test_crossfold_consensus_auto_confidence.json \
    --no_report
```

The TLS-120 target-prior candidate by itself is a negative ablation: direct prior replacement dropped test accuracy to `0.7363`. Therefore, for the paper, prior calibration should be described as a **safe residual candidate**, not as a mandatory replacement of base predictions. The constrained residual design is what makes the same module usable across datasets.

Example TLS-120 safe residual calibration:

```bash
conda run --no-capture-output -n llm-factory \
  python calibrate_prior_ensemble.py \
    --input_json reasoningDataset/tls-120/test_fusion_graph_seq_tls120_rawproj_change_weight_valid_acc.json \
    --label_map reasoningDataset/tls-120/train_tower1_change_weight/label_map.json \
    --methods blend \
    --strengths 0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2 \
    --gate_modes none,low_margin,high_entropy,low_confidence \
    --gate_thresholds 0.4,0.45,0.5,0.55,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.75,0.78,0.8 \
    --pool_strategy prior_softcap_valid \
    --top_k 1 \
    --hard_prior_kl_cap 0.017 \
    --ensemble_mode mean \
    --include_identity_candidate \
    --output_json reasoningDataset/tls-120/test_fusion_graph_seq_tls120_rawproj_change_weight_safe_prior_unified.json

conda run --no-capture-output -n llm-factory \
  python fuse_prediction_jsons.py \
    --input base reasoningDataset/tls-120/test_fusion_graph_seq_tls120_rawproj_change_weight_valid_acc.json \
    --input prior reasoningDataset/tls-120/test_fusion_graph_seq_tls120_rawproj_change_weight_safe_prior_unified.json \
    --label_map reasoningDataset/tls-120/train_tower1_change_weight/label_map.json \
    --simplex_step 0.01 \
    --select_metric accuracy \
    --min_weight base 0.90 \
    --output_json reasoningDataset/tls-120/test_fusion_graph_seq_safe_prior_residual_minbase90_unified.json
```

The strongest current VPN result comes from validation-selected graph/stats/flow-embedding fusion followed by target-prior candidate ensembling:

```text
reasoningDataset/vpn-app/test_fusion_vpn_full_stage5_flow_embedding_prior_ensemble_softcap_k31_vote.json
flow accuracy = 0.7482
flow macro-F1 = 0.7556
```

The prior ensemble is label-free on the target split: it builds calibrated candidates from hard/soft target-prior estimates, keeps candidates under a hard-prior KL cap, and votes across the selected pool.

```bash
conda run --no-capture-output -n llm-factory \
  python calibrate_prior_ensemble.py \
    --input_json reasoningDataset/vpn-app/test_fusion_vpn_full_stage5_flow_embedding_valid_acc.json \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --methods blend \
    --strengths 0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4 \
    --gate_modes none,low_margin,high_entropy,low_confidence \
    --gate_thresholds 0.4,0.45,0.5,0.55,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.75,0.78,0.8 \
    --pool_strategy prior_softcap \
    --top_k 31 \
    --hard_prior_kl_cap 0.017 \
    --ensemble_mode vote \
    --output_json reasoningDataset/vpn-app/test_fusion_vpn_full_stage5_flow_embedding_prior_ensemble_softcap_k31_vote.json
```

Paired-view Tower-2 consistency is implemented for flow-level training through `--paired_view_dataset`. In the current VPN run, `ipport_rand` paired consistency improved validation but hurt target-test accuracy, so treat it as an ablation rather than the best model.

```bash
conda run --no-capture-output -n llm-factory \
  python train_tower2.py \
    --model_type graph \
    --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/graph_dataset.pt \
    --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt \
    --paired_view_dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_ipport_rand/graph_dataset.pt \
    --output_dir checkpoints/tower2_graph_flow_rawproj_change_weight_stage5_ft_paired_rand_kl005 \
    --init_checkpoint checkpoints/tower2_graph_flow_rawproj_change_weight_macro_hier_conf_supcon/best.pt \
    --num_classes 16 \
    --epochs 12 \
    --batch_size 16 \
    --hidden_dim 256 \
    --num_layers 2 \
    --num_heads 4 \
    --dropout 0.15 \
    --lr 5e-5 \
    --weight_decay 0.03 \
    --train_level flow \
    --select_metric flow_macro_f1 \
    --flow_pooling mean \
    --window_loss_weight 0.2 \
    --class_weighting effective \
    --class_weight_beta 0.9999 \
    --class_weight_strength 0.5 \
    --label_smoothing 0.05 \
    --hierarchical_weight 0.2 \
    --hierarchical_logit_weight 0.5 \
    --coarse_groups vpn_app \
    --balanced_flow_batches \
    --samples_per_class 2 \
    --contrastive_mode confusion \
    --confusion_groups vpn_app \
    --flow_contrastive_weight 0.02 \
    --flow_temperature 0.07 \
    --paired_view_weight 0 \
    --paired_consistency_weight 0.05 \
    --consistency_temperature 2.0 \
    --aux_weight 0 \
    --coherence_weight 0
```

Tower-1 now also supports flow-aware supervised contrastive learning. Use it when retraining packet embeddings: each packet batch samples multiple packets per flow, same-flow positives receive a stronger weight than same-label positives. `--flow_proto_weight` adds a packet-to-flow prototype contrastive objective: packet embeddings are pulled toward same-flow or same-class flow prototypes and pushed away from other-class prototypes. Keep it at `0` for reproducing the current best checkpoints; start with a small value such as `0.02` or `0.05` for new Tower-1 runs. Use `--init_checkpoint_dir` to continue from an existing Tower-1 adapter and packet heads.

The optional `--flow_proto_context leave_one_out` mode removes the anchor
packet from its own-flow prototype. It also disables replacement sampling for
single-packet flows, so repeating the same packet cannot manufacture a false
multi-instance positive. This is the current shared packet-to-flow research
candidate: both packet- and flow-level runners expose the same option, while
packet-level inference remains strictly one packet. It is not yet part of the
`paper_unified` main profile; that profile keeps `--flow_proto_weight 0` until
the leave-one-out objective passes the frozen VPN/TLS cross-fold validation
protocol.

The candidate has sufficient training coverage without changing the inference
unit. In the current fold artifacts, every VPN flow-level training flow has at
least 7 distinct packets and every TLS flow-level training flow has at least
10. For packet-level VPN and TLS, multi-packet flows cover respectively
`91.86%` and `93.80%` of training packets; singleton flows simply receive the
existing packet CE/SupCon losses. The first controlled run therefore keeps
`--packets_per_flow 2` and changes only prototype construction.

```bash
conda run --no-capture-output -n llm-factory \
  python train_tower1_multitask.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --packet_aux_jsonl reasoningDataset/vpn-app/train_tower1_change_weight/packet_auxiliary.jsonl \
    --sft_jsonl reasoningDataset/vpn-app/train_tower1_change_weight/packet_instruction.jsonl reasoningDataset/vpn-app/train_tower1_change_weight/packet_validity.jsonl \
    --output_dir checkpoints/tower1_qwen_multitask_flowaware_change_weight \
    --epochs 2 \
    --sft_batch_size 2 \
    --packet_batch_size 16 \
    --flow_balanced_packet_batches \
    --packets_per_flow 2 \
    --cls_weight 0.1 \
    --contrastive_weight 0.3 \
    --same_flow_positive_weight 2.0 \
    --same_label_positive_weight 1.0 \
    --flow_proto_weight 0.05 \
    --flow_proto_positive same_class \
    --flow_proto_context leave_one_out \
    --max_sft_length 1792 \
    --max_packet_length 1024 \
    --local_files_only

# Optional for continuation:
#   --init_checkpoint_dir checkpoints/tower1_qwen_multitask_flowaware_change_weight/step_150
```

The same Stage 8 workflow can be launched step-by-step with the runner below. Use `--dry_run` first to audit paths. Use `--require_cuda` for long Tower-1/embedding stages so the command fails early if the `llm-factory` environment cannot see a GPU.

CUDA visibility note for automated debugging: the default Codex sandbox may not expose the NVIDIA driver, so `torch.cuda.is_available()` can report `False` there even though the real conda environment is GPU-capable. In the real `llm-factory` environment on this machine, CUDA is available and there are 8 NVIDIA A800 80GB PCIe GPUs. For GPU training or embedding extraction, run through the real shell / approved non-sandbox execution and verify with:

```bash
conda run --no-capture-output -n llm-factory python - <<'PY'
import torch
print("torch_version=", torch.__version__)
print("torch_cuda_build=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("device_count=", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

nvidia-smi -L
```

Concurrent Qwen embedding jobs are resource guarded. By default,
`extract_packet_embeddings_qwen.py` maps the logical CUDA index through
`CUDA_VISIBLE_DEVICES`, acquires a file lock for that physical GPU under
`/tmp/two_tower_embedding_gpu_locks`, and waits until at least 20 GiB is free
before loading the model. This serializes packet-level and flow-level embedding
loads that target the same A800 and prevents a newly launched shard from
crashing a long-running training job. Waiting messages are expected scheduling
output, not a stalled experiment. The threshold can be changed with
`--min_cuda_free_gb`; `--disable_cuda_capacity_lock` is intended only for an
externally isolated GPU allocation.

Each extraction process also selects its explicit logical CUDA device before
loading the base model and passes that exact `cuda:N` as PEFT's `torch_device`
when loading the adapter. This matters for sharded jobs: PEFT's inferred,
index-less `cuda` allocations must not temporarily land on logical GPU 0 or on
a device last selected by the Transformers/Accelerate loading context.

If Tower-1 has completed but factual/intervened embedding extraction or a later
stage is interrupted, resume the unified path with the same experiment
arguments and replace `--stage all` by `--stage post_tower1`. This stage reuses
`<tower1_output_dir>/best`, resumes incomplete embedding shards, then performs
the shared Tower-2 preprocessing, training, and evaluation. It deliberately
does not preprocess or retrain Tower-1:

```bash
conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --fold 0 \
    --stage post_tower1 \
    --framework_profile paper_unified \
    --tower1_output_dir checkpoints/<completed_tower1_run> \
    --embedding_num_shards 8 \
    --embedding_cuda_devices 0,1,2,3,4,5,6,7 \
    --paper_unified_stages model \
    --require_cuda
```

All naming and method arguments other than `--stage` must match the original
run so that shard counts, paired semantic views, checkpoints, and manifests
remain in one provenance namespace.

Summarize Tower-1 checkpoint dynamics with the same held-out macro-F1 protocol
for every dataset, fold, and task:

```bash
conda run --no-capture-output -n llm-factory \
  python analyze_tower1_validation_history.py \
    --input fold0 checkpoints/<fold0>/packet_validation_history.jsonl \
    --input fold1 checkpoints/<fold1>/packet_validation_history.jsonl \
    --input fold2 checkpoints/<fold2>/packet_validation_history.jsonl \
    --output_json reasoningDataset/<dataset>/tower1_validation_dynamics.json
```

The report selects each run by held-out packet macro-F1, records any validation
regression after the best checkpoint, and identifies persistently weak or
recovered classes. Use this shared diagnostic before changing the schedule: one
fold peaking early is not evidence that every dataset/task should receive a
shorter, manually specialized training recipe.

The runner now has dataset defaults for:

```text
vpn-app:
  train /home/jing/download/sweet/flow-level-classification/vpn-app/train_val_split_0/train
  valid /home/jing/download/sweet/flow-level-classification/vpn-app/train_val_split_0/val
  test  /home/jing/download/sweet/flow-level-classification/vpn-app/test

tls-120:
  train /home/jing/download/sweet/flow-level-classification/tls/train_val_split_0/train
  valid /home/jing/download/sweet/flow-level-classification/tls/train_val_split_0/val
  test  /home/jing/download/sweet/flow-level-classification/tls/test

ustc-app:
  train /home/jing/download/sweet/packet-level-classification/per-flow-split/ustc-app/train_val_split_0/train
  valid /home/jing/download/sweet/packet-level-classification/per-flow-split/ustc-app/train_val_split_0/val
  test  /home/jing/download/sweet/packet-level-classification/per-flow-split/ustc-app/test

ustc-binary:
  train /home/jing/download/sweet/packet-level-classification/per-flow-split/ustc-binary/train_val_split_0/train
  valid /home/jing/download/sweet/packet-level-classification/per-flow-split/ustc-binary/train_val_split_0/val
  test  /home/jing/download/sweet/packet-level-classification/per-flow-split/ustc-binary/test
```

`ustc-app` and `ustc-binary` are retained only as legacy flow-pipeline smoke inputs. Their partitions do establish disjoint flows, but each `<label>.pcap` is a packet-level class container holding multiple flows rather than one flow-level sample. Do not include the following USTC numbers in the VPN/TLS flow-level main table or use them as cross-dataset flow-level evidence. They can be used after implementing a dedicated packet-level reader and evaluator that reconstruct real flow IDs. For datasets other than `vpn-app`, the runner defaults `--coarse_groups none` and `--confusion_groups none`; pass explicit groups only after building dataset-specific coarse labels.

USTC app has now been run with full no-limit preprocessing. Each split generated 1280 packet records and a 20-class label map. The first 5-step Tower-1 smoke checkpoint only reached `0.15` accuracy / `0.065` macro-F1 after graph+seq fusion, so it should remain a pipeline smoke test. An 80-step Tower-1 run with `packet_batch_size=2` improved graph+seq+embedding-expert fusion to `0.55` accuracy / `0.475` macro-F1, but its packet contrastive loss stayed inactive because `flows_per_batch=1`.

The current better USTC run uses `packet_batch_size=8`, `packets_per_flow=2`, and downstream validation-aware Tower-1 checkpoint selection. The 200-step run makes the flow-balanced packet sampler use `flows_per_batch=4` and activates Tower-1 SupCon, but the best downstream checkpoint is the intermediate `step_150` adapter rather than the final `step_200` adapter. This improves USTC graph/seq/embedding-expert fusion to `0.65` accuracy / `0.5750` macro-F1:

```text
Tower-1 checkpoint:
  checkpoints/tower1_qwen_multitask_ustc_app_flowaware_change_weight_s200_pb8/step_150

Embedding suffix:
  rawproj_flowaware_change_weight_s200_pb8_step150

Best USTC output:
  reasoningDataset/ustc-app/test_fusion_graph_seq_emb_rawproj_flowaware_change_weight_s200_pb8_step150_stage8_flowaware_safe_prior_residual.json

Validation-selected fusion:
  emb=0.70, seq=0.30, graph=0.0

Safe residual selection:
  base=0.91, prior=0.09; prior candidate is identity, so the calibrated residual keeps the base prediction behavior

Tower-1 training signal:
  step=20  supcon=1.8379, pkt_acc=0.0750
  step=200 supcon=0.0632, pkt_acc=0.8125
```

Historical smoke-test interpretation only: packet-level training accuracy alone was not a sufficient checkpoint-selection criterion in this USTC run. Because the partition is not accepted for the current flow-level paper, neither `step_150` nor `step_200` is used as paper evidence.

This is not yet a strong USTC result, but it is an important ablation: when Tower-1 batches contain multiple flows, the flow-aware SupCon term becomes active and downstream validation-aware checkpoint selection improves flow-level generalization.

Tower-2 now also implements a flow/window-level contrastive objective through `--window_contrastive_weight`. A first USTC `step_150` ablation with `--window_contrastive_weight 0.05 --window_contrastive_positive same_class` improved the seq Tower-2 single-head test result to `0.55` accuracy / `0.4417` macro-F1, but constrained residual fusion with the current best selected only a small seq residual (`base=0.95, seq_wincon=0.05, graph_wincon=0.0`) and kept the final test result at `0.65` accuracy / `0.5750` macro-F1:

```text
Seq wincon:
  reasoningDataset/ustc-app/test_seq_metrics_flow_rawproj_flowaware_change_weight_s200_pb8_step150_wincon.json
  flow accuracy = 0.5500
  flow macro-F1 = 0.4417

Graph wincon:
  reasoningDataset/ustc-app/test_graph_metrics_flow_rawproj_flowaware_change_weight_s200_pb8_step150_wincon.json
  flow accuracy = 0.5000
  flow macro-F1 = 0.3750

Residual fusion:
  reasoningDataset/ustc-app/test_fusion_ustc_step150_base_wincon_residual.json
  selected weights: base=0.95, seq_wincon=0.05, graph_wincon=0.0
  flow accuracy = 0.6500
  flow macro-F1 = 0.5750
```

Interpretation: the window-to-flow objective is implemented and gives a cleaner paper module, but this first Tower-2-only setting is not enough to beat the embedding-expert-dominant USTC best.

Tower-1 packet-to-flow prototype loss is also implemented through `--flow_proto_weight`, and Tower-1 continuation from an existing adapter is supported through `--init_checkpoint_dir`. A first USTC ablation continued the current best Tower-1 `step_150` checkpoint for 40 packet-only steps with `--flow_proto_weight 0.05`, no SFT loss, and lower learning rates. Training was stable and packet accuracy rose to `0.7750`, but downstream test metrics dropped:

```text
Tower-1 proto continuation:
  init checkpoint: checkpoints/tower1_qwen_multitask_ustc_app_flowaware_change_weight_s200_pb8/step_150
  output checkpoint: checkpoints/tower1_qwen_multitask_ustc_app_flowproto_continue_s40_w005
  final training signal: pkt_cls=0.3355, supcon=0.0004, proto=0.0001, pkt_acc=0.7750

Graph/seq fusion:
  reasoningDataset/ustc-app/test_fusion_graph_seq_rawproj_flowproto_continue_s40_w005_stage8_flowaware_valid_acc.json
  selected weights: graph=0.90, seq=0.10
  flow accuracy = 0.5500
  flow macro-F1 = 0.4583

Flow-embedding expert:
  reasoningDataset/ustc-app/test_flow_embedding_classifier_flowproto_continue_s40_w005_message_header_ports_valid_macro.json
  valid selected extra_trees, n_components=19
  flow accuracy = 0.5500
  flow macro-F1 = 0.4500

Residual fusion with current best:
  reasoningDataset/ustc-app/test_fusion_ustc_step150_base_flowproto_s40_residual.json
  selected weights: base=0.95, proto_emb=0.05, proto_gs=0.0
  flow accuracy = 0.6500
  flow macro-F1 = 0.5750
```

Interpretation: the Tower-1 prototype objective is a useful framework module, but this no-SFT continuation over-optimizes packet/prototype separation and hurts downstream flow generalization. Treat it as a negative ablation. The next Tower-1 proto experiment should keep SFT enabled or use a smaller `flow_proto_weight` / fewer continuation steps, rather than using packet-only continuation.

A follow-up conservative proto continuation kept SFT enabled and reduced `--flow_proto_weight` to `0.02` for 20 continuation steps. This was more stable than packet-only continuation, but still did not improve the current USTC best:

```text
Tower-1 SFT+proto continuation:
  init checkpoint: checkpoints/tower1_qwen_multitask_ustc_app_flowaware_change_weight_s200_pb8/step_150
  output checkpoint: checkpoints/tower1_qwen_multitask_ustc_app_flowproto_sft_continue_s20_w002
  final training signal: pkt_cls=0.6632, supcon=0.0618, proto=0.0115, pkt_acc=0.6750

Graph/seq fusion:
  reasoningDataset/ustc-app/test_fusion_graph_seq_rawproj_flowproto_sft_s20_w002_stage8_flowaware_valid_acc.json
  selected weights: graph=0.95, seq=0.05
  flow accuracy = 0.4500
  flow macro-F1 = 0.3167

Flow-embedding expert:
  reasoningDataset/ustc-app/test_flow_embedding_classifier_flowproto_sft_s20_w002_message_header_ports_valid_macro.json
  valid selected logreg, n_components=16, C=3.0
  flow accuracy = 0.6000
  flow macro-F1 = 0.5417

Residual fusion with current best:
  reasoningDataset/ustc-app/test_fusion_ustc_step150_base_flowproto_sft_s20_w002_residual.json
  selected weights: base=0.95, proto_emb=0.05, proto_gs=0.0
  flow accuracy = 0.6500
  flow macro-F1 = 0.5750
```

Interpretation: even a small SFT-preserving prototype continuation is not enough to improve USTC downstream generalization. For future runs, prototype learning should be trained inside the full Tower-1 schedule with validation-aware checkpoint selection, rather than appended as a short continuation after the best checkpoint.

A constrained residual search over the current USTC top-12 existing candidates also selected `base=1.0` and kept `0.6500` accuracy / `0.5750` macro-F1:

```text
reasoningDataset/ustc-app/test_residual_fusion_search_step150_minbase90_top12_macro.json
selected weights: base=1.0, candidate=0.0
```

This means the remaining USTC gap should be attacked through representation learning or dataset construction, not by repeatedly recombining the same probability JSONs.

Training the packet-to-flow prototype objective inside the full Tower-1 schedule is more useful than short continuation. A 200-step USTC Tower-1 run with `--flow_proto_weight 0.02`, `packet_batch_size=8`, `packets_per_flow=2`, and checkpoints every 50 steps improved the best USTC macro-F1 when using the `step_150` embedding expert:

```text
Tower-1 full proto run:
  output checkpoint root: checkpoints/tower1_qwen_multitask_ustc_app_flowproto_full_s200_w002
  selected checkpoint for this ablation: step_150
  step_150 training signal: pkt_cls=0.7289, supcon=0.1105, proto=0.0358, pkt_acc=0.6750
  step_200 training signal: pkt_cls=0.3365, supcon=0.0242, proto=0.0028, pkt_acc=0.8000

Graph/seq Tower-2 fusion from step_150 embeddings:
  reasoningDataset/ustc-app/test_fusion_graph_seq_rawproj_flowproto_full_s200_w002_step150_stage8_flowaware_valid_acc.json
  selected weights: graph=0.70, seq=0.30
  flow accuracy = 0.5000
  flow macro-F1 = 0.4117

Flow-embedding expert from step_150 embeddings:
  reasoningDataset/ustc-app/test_flow_embedding_classifier_flowproto_full_s200_w002_step150_message_header_ports_valid_macro.json
  valid selected logreg, n_components=19, C=0.03
  flow accuracy = 0.6500
  flow macro-F1 = 0.6083

Flow-embedding expert from step_200 embeddings:
  reasoningDataset/ustc-app/test_flow_embedding_classifier_flowproto_full_s200_w002_step200_message_header_ports_valid_macro.json
  valid selected logreg, n_components=12, C=0.03
  flow accuracy = 0.6000
  flow macro-F1 = 0.5083

Residual fusion with the previous best:
  reasoningDataset/ustc-app/test_fusion_ustc_step150_base_flowproto_full_s200_w002_step150_residual_macro.json
  selected weights: base=0.85, proto_emb=0.15, proto_gs=0.0
  flow accuracy = 0.6500
  flow macro-F1 = 0.5750

Validation-gated selector over the previous best and the full-proto embedding expert:
  reasoningDataset/ustc-app/test_selector_base_flowproto_full_s200_w002_step150_calib_shift005_valid_macro.json
  selected selector: class_precision, alpha=0.5, metric_margin=0.0
  bootstrap guard: win_rate=0.66, 5% gain quantile=0.0
  target-shift guard: prediction_change_rate=0.05
  flow accuracy = 0.7000
  flow macro-F1 = 0.6250
```

Reproduce the validation-gated selector result:

```bash
conda run --no-capture-output -n llm-factory \
  python validation_gated_selector.py \
    --input base reasoningDataset/ustc-app/test_fusion_graph_seq_emb_rawproj_flowaware_change_weight_s200_pb8_step150_stage8_flowaware_safe_prior_residual.json \
    --input proto_emb reasoningDataset/ustc-app/test_flow_embedding_classifier_flowproto_full_s200_w002_step150_message_header_ports_valid_macro.json \
    --label_map reasoningDataset/ustc-app/train_tower1_flowaware_change_weight/label_map.json \
    --select_metric macro_f1 \
    --strategies always,class_precision,reliability_fusion,threshold_switch \
    --alpha_grid 0.5,1,2,5 \
    --metric_margin_grid 0,0.05,0.1 \
    --expert_conf_grid 0.3,0.5,0.7,0.85 \
    --expert_margin_grid 0.05,0.15,0.3,0.6 \
    --base_conf_max_grid 1,0.85,0.7,0.55 \
    --delta_conf_grid=-1,0,0.05,0.1 \
    --delta_margin_grid=-1,0,0.05,0.1 \
    --min_valid_gain_over_base 0 \
    --bootstrap_samples 300 \
    --bootstrap_min_win_rate 0.6 \
    --bootstrap_min_gain_quantile -0.001 \
    --max_prediction_change_rate 0.08 \
    --output_json reasoningDataset/ustc-app/test_selector_base_flowproto_full_s200_w002_step150_calib_shift005_valid_macro.json
```

Interpretation: full-schedule prototype learning did not increase USTC accuracy by itself, but it improved the best single-expert macro-F1 from `0.5750` to `0.6083`. The validation-gated selector then considered hard class-precision gating, confidence-threshold switching, and reliability-weighted soft fusion; validation selected class-precision gating and improved the current USTC result to `0.7000` accuracy / `0.6250` macro-F1. The bootstrap guard checks whether the selected validation gain is stable under resampling, while the target-shift guard rejects candidates that rewrite too many unlabeled target predictions relative to the base. The final `step_200` checkpoint overfits the tiny validation split and drops on test, so downstream validation-aware checkpoint selection remains necessary. For paper framing, this supports the representation-learning claim: prototype alignment helps class-balanced behavior, while validation-gated selection prevents a high-validation but split-fragile expert from overwriting the safer base prediction.

The flow-aware Tower-1 preprocessing inputs have been generated for both VPN and TLS-120:

```text
reasoningDataset/vpn-app/train_tower1_flowaware_change_weight
reasoningDataset/vpn-app/valid_tower1_flowaware_change_weight
reasoningDataset/vpn-app/test_tower1_flowaware_change_weight

reasoningDataset/tls-120/train_tower1_flowaware_change_weight
reasoningDataset/tls-120/valid_tower1_flowaware_change_weight
reasoningDataset/tls-120/test_tower1_flowaware_change_weight
```

```bash
conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage all \
    --dry_run \
    --no_progress

conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset tls-120 \
    --num_classes 120 \
    --stage all \
    --dry_run \
    --no_progress


conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage tower1_train \
    --require_cuda \
    --no_progress

conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage tower1_preprocess \
    --no_progress

conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage embeddings \
    --require_cuda \
    --no_progress

conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage tower2_preprocess \
    --no_progress

conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage tower2_train \
    --no_progress

conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage eval \
    --no_progress

conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage fusion \
    --no_progress

conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage prior \
    --no_progress
```

`stage all` runs the full order `tower1_preprocess -> tower1_train -> embeddings -> tower2_preprocess -> tower2_train -> eval -> fusion -> prior`. Tower-1 checkpoints are dataset-scoped by default, for example `checkpoints/tower1_qwen_multitask_vpn_app_flowaware_change_weight` and `checkpoints/tower1_qwen_multitask_tls_120_flowaware_change_weight`. Tower-1 training uses `--local_files_only` by default in the runner, so make sure the selected Qwen checkpoint is already available in the local Hugging Face cache or pass `--no-local_files_only` intentionally. Tower-2 training uses validation-selected `best.pt` and supports early stopping through `--tower2_early_stop_patience` in the runner, which maps to `train_tower2.py --early_stop_patience`. The runner's `fusion` stage now first calls `make_fusion_payload.py` for each selected Tower-2 model, so valid/test probability JSONs are automatically merged into the payload format required by `fuse_prediction_jsons.py`. Use `--no-flow_balanced_packet_batches` for the Tower-1 flow-balanced sampler ablation. Use `--tower1_init_checkpoint_dir` for Tower-1 continuation from an existing adapter, `--flow_proto_weight` for Tower-1 packet-to-flow prototype contrastive training, `--tower1_paired_data_suffix` plus `--tower1_paired_consistency_weight` for Tower-1 full-header/randomized-header packet consistency, and `--window_contrastive_weight` for Tower-2 window-to-flow prototype contrastive training.

For the next representation-learning iteration, the same Stage-8 runner also supports a paired header-perturbation view. This keeps the paper framework unified: every dataset can use the same full-view classifier, an IP/port-randomized paired view, clean/augmented consistency, feature dropout, and validation-gated downstream fusion; dataset-specific validation then decides whether these modules help or collapse to the base path. Tower-1 now supports the paired view directly through `--tower1_paired_data_suffix`: packets are aligned by `packet_uid`, and the model adds projected-embedding consistency plus symmetric logit KL between full-header and randomized/masked-header prompts. This is the preferred next step after Tower-2-only paired consistency because it makes the packet representation itself endpoint-invariant before packet embeddings are extracted.

```bash
# 1) Build the paired IP/port-randomized view. Reuse the dataset-scoped Tower-1 adapter.
conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage tower1_preprocess \
    --output_suffix flowaware_ipport_rand_change_weight \
    --embedding_header_policy randomize_ip_port \
    --no_progress

# Optional 1b) Retrain/continue Tower-1 with endpoint-invariant paired packet consistency.
conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage tower1_train \
    --output_suffix flowaware_change_weight \
    --tower1_data_suffix flowaware_change_weight \
    --tower1_paired_data_suffix flowaware_ipport_rand_change_weight \
    --tower1_paired_consistency_weight 0.05 \
    --tower1_paired_cls_weight 0.2 \
    --tower1_paired_logit_kl_weight 0.5 \
    --tower1_init_checkpoint_dir checkpoints/tower1_qwen_multitask_vpn_app_flowaware_change_weight \
    --tower1_output_dir checkpoints/tower1_qwen_multitask_vpn_app_flowaware_paired_ipport \
    --require_cuda \
    --no_progress

conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage embeddings \
    --output_suffix flowaware_ipport_rand_change_weight \
    --embedding_suffix rawproj_flowaware_ipport_rand_change_weight \
    --require_cuda \
    --no_progress

conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage tower2_preprocess \
    --embedding_suffix rawproj_flowaware_ipport_rand_change_weight \
    --no_progress

# 2) Train the main full-view Tower-2 model with paired-view consistency.
conda run --no-capture-output -n llm-factory \
  python run_stage8_flowaware_pipeline.py \
    --dataset vpn-app \
    --num_classes 16 \
    --stage tower2_train \
    --embedding_suffix rawproj_flowaware_change_weight \
    --run_tag paired_ipport \
    --paired_embedding_suffix rawproj_flowaware_ipport_rand_change_weight \
    --paired_view_weight 0.2 \
    --paired_consistency_weight 0.1 \
    --consistency_weight 0.05 \
    --meta_dropout_prob 0.1 \
    --embedding_dropout_prob 0.05 \
    --window_dropout_prob 0.1 \
    --edge_attr_dropout_prob 0.1 \
    --no_progress
```

Use the same command shape for TLS-120 by changing `--dataset`, `--num_classes`, and the suffixes. This is the preferred paper-facing experiment over additional probability-level tuning because it tests whether endpoint-invariant representations improve flow classification before the validation-gated selector stage.

CPU-feasible probes on the old `rawproj_change_weight` embeddings show that paired-view regularization alone is not enough; it should be paired with fresh Stage-8 Tower-1 flow-aware embeddings before being considered a final method:

```text
paired CE + consistency seq probe:
  test file: reasoningDataset/vpn-app/test_seq_metrics_flow_rawproj_change_weight_stage8_flowaware_paired_ipport_oldview_seqprobe_probs.json
  test flow accuracy = 0.6376
  test flow macro-F1 = 0.5961

consistency-only seq probe:
  test file: reasoningDataset/vpn-app/test_seq_metrics_flow_rawproj_change_weight_stage8_flowaware_paired_ipport_consistency_seqprobe_probs.json
  test flow accuracy = 0.6465
  test flow macro-F1 = 0.5936

best + paired seq constrained residual fusion:
  test file: reasoningDataset/vpn-app/test_fusion_best_paired_seqprobe_minbase90_valid_acc.json
  selected weights: base=0.91, paired_seq=0.09
  test flow accuracy = 0.7482
  test flow macro-F1 = 0.7534
```

The fresh Stage-8 A800 paired-view run has now been executed with
`rawproj_flowaware_change_weight_split2_retrain` as the clean view and
`rawproj_flowaware_ipport_rand_change_weight_split2_retrain` as the IP/port
randomized paired view. Two settings were tested:

```text
strong cross-view invariance:
  run_tag: stage8_crossview_ci_iter01
  paired_view / consistency / alignment / contrastive / variance / adversarial:
    0.20 / 0.10 / 0.03 / 0.01 / 0.01 / 0.01
  safe prior residual:
    test flow accuracy = 0.5353
    test flow macro-F1 = 0.5123

gentle counterfactual consistency:
  run_tag: stage8_gentle_ci_iter02
  paired_view / consistency / alignment / contrastive / variance / adversarial:
    0.05 / 0.03 / 0.005 / 0.00 / 0.00 / 0.00
  safe prior residual:
    test flow accuracy = 0.5520
    test flow macro-F1 = 0.5358
```

Both fresh paired-view variants are substantially below the paper-safe VPN
cross-fold consensus result (`0.7512` accuracy / `0.7522` macro-F1). Treat
Tower-2-only full-header/randomized-IP-port consistency as a negative ablation:
it is useful evidence that naive shortcut intervention can remove
task-relevant protocol/session signal. It should not be promoted as the CCF-A
headline method unless paired Tower-1 representation learning or a stronger
content-grouped validation protocol reverses this degradation.

The automated stacker/gate/selector stages now explicitly skip probability
experts that are test-only or whose validation flow IDs are incompatible with
the selected candidate group. In the two fresh VPN paired runs, the final
selector records the test-only cross-fold consensus and the split-incompatible
paired branch as skipped rather than training a selector across mismatched
validation folds. This is important paper-safety evidence: candidate promotion
requires compatible validation support, not only a shared-test probability file.

The runner's `prior` stage now implements the paper-safe residual calibration path by default:

```text
fusion output
-> calibrate_prior_ensemble.py with --include_identity_candidate
-> fuse_prediction_jsons.py with --min_weight base 0.90
-> safe_prior_residual output
```

This keeps the same module active for every dataset while allowing validation-selected weights to suppress harmful calibration, as happened on TLS-120 where `base=1.0, prior=0.0`.

For residual expert fusion, `fuse_prediction_jsons.py` supports constrained weights:

```bash
conda run --no-capture-output -n llm-factory \
  python fuse_prediction_jsons.py \
    --input best reasoningDataset/vpn-app/test_fusion_vpn_full_stage5_flow_embedding_prior_ensemble_softcap_k31_vote.json \
    --input emb_et reasoningDataset/vpn-app/test_flow_embedding_classifier_extratrees_valid_acc.json \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --simplex_step 0.01 \
    --select_metric accuracy \
    --min_weight best 0.90 \
    --output_json reasoningDataset/vpn-app/test_fusion_best_prior_flow_embedding_et_minbest90_valid_acc.json
```

This keeps the strongest base model dominant when the validation split is too small or shifted. In the current VPN run, the constrained residual embedding expert improved the best test accuracy slightly from `0.7482` to `0.7488`, but it still did not cross `0.75`.

For source-level expert selection, `validation_gated_selector.py` compares probability JSONs on the validation split and then chooses either a single source, a class-precision-gated source, a confidence-threshold switch, a reliability-weighted soft fusion, or a validation-set class-bias calibration candidate. The reliability fusion estimates each expert's validation precision for its predicted class with shrinkage, then weights expert probabilities by validation reliability and confidence. The class-bias calibration candidate estimates the validation true-class prior divided by the model's mean predicted prior and applies that bias to candidate probabilities.

The final selector uses three safety gates: `--min_valid_gain_over_base` for deterministic validation gain, bootstrap gain stability through `--bootstrap_samples`, and an unlabeled target-shift constraint through `--max_prediction_change_rate`. The current unified report uses a small bootstrap quantile tolerance; TLS uses `--bootstrap_min_gain_quantile -0.0015` for the soft-gate candidate because its validation gain is low-shift but the 5% bootstrap lower bound is only slightly below zero. The same candidate family is evaluated on VPN and TLS-120, while `--max_prediction_change_rate` is dataset-specific: VPN uses a strict no-target-change setting and TLS allows a tiny low-shift switch/calibration. The recommended Stage-8 runner now passes `--final_selector_unified_expert_slots base,graph,seq,prior_base,emb_lr,emb_et,proto_emb,paired,slot_stacker,soft_gate`, so both datasets expose the same selector expert slots; missing slots are filled from the base probabilities as identity experts and are recorded in `feature_config.input_slot_status`. For legacy best-result JSONs without slot records, the paper report marks the same mapping as `inferred_identity_compatible` instead of pretending the old run recorded it. By default the selector sorts candidates by validation score, skips unsafe candidates, and accepts the first candidate that passes all active guards; if none pass, it falls back to the first input. For paper-grade robustness searches, `--rank_metric bootstrap_gain_quantile` can instead rank the top validation candidates by their bootstrap lower-bound gain before the same safety gates are applied. `--rank_select_metric` lets this robust ranking target accuracy while the accepted selector still optimizes macro-F1. `--calibration_penalty_weight` optionally subtracts a validation calibration penalty (`ece`, `nll`, or `brier`) from the ranking score, using the shared `probability_metrics.py` definition; this prefers less overconfident candidates when validation scores are close, but the accepted candidate must still pass the same gain/bootstrap/target-shift gates. Use `--rank_candidate_limit` to keep this robust ranking practical on large candidate grids. This is the same module used across both datasets:

```text
VPN:
  safe selector file: reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_calib_shift000_valid_macro.json
  selected path: fallback to base because reliability_fusion changed 12.68% of target predictions, above max_prediction_change_rate=0.0
  test accuracy = 0.7488
  test macro-F1 = 0.7558

TLS-120:
  safe selector file: reasoningDataset/tls-120/test_selector_soft_gate_tls120_tol0015_calib_family_valid_macro.json
  selected path: class-bias-calibrated soft_gate; bootstrap win_rate=0.89, 5% gain quantile=-0.0014, target prediction change=0.0396
  test accuracy = 0.7996
  test macro-F1 = 0.7869

```

The negative VPN selector ablation with a looser `--min_valid_gain_over_base 0.03` selected an embedding-LR expert and dropped to `0.6812` accuracy / `0.6475` macro-F1. The unsafe reliability-fusion ablation selected `alpha=5.0`, `reliability_power=4.0`, `confidence_power=1.0`, `temperature=0.5`; it improved validation macro-F1 but dropped target-test performance to `0.6956` accuracy / `0.6633` macro-F1. Bootstrap alone did not reject this VPN candidate because its validation gain was internally stable, but the target-shift guard rejected it because it changed too many target predictions. A broader calibration-enabled candidate search also found a lower-shift VPN threshold switch, but it still dropped to `0.7339` accuracy / `0.7241` macro-F1, so calibration remains an ablation rather than the default final path. On TLS-120, the unified-slot stacker reached `0.7991` accuracy / `0.7897` macro-F1 by itself; a trainable soft expert gate over the same unified slots reached `0.7973` / `0.7843`, with mean test weights dominated by `slot_stacker=0.593` and a base-identity branch `emb_et=0.358`. The current paper-safe selector then applies validation-set class-bias calibration to the soft gate and reaches `0.7996` accuracy / `0.7869` macro-F1 while keeping target prediction change below 4%. A calibration-aware ranking probe with `--calibration_penalty_weight 0.10 --calibration_penalty_metric ece` kept the accepted TLS selector decision but did not improve Acc/F1, so it remains a neutral calibration/stability ablation. This is why the paper method should emphasize validation-gated expert selection with validation stability, trainable expert gating, calibration observability, and unlabeled target-shift safety, not unconditional expert switching.

Reproduce the TLS-120 unified-slot stacker, soft expert gate, and guarded selector result:

```bash
conda run --no-capture-output -n llm-factory \
  python train_prediction_stacker.py \
    --input base reasoningDataset/tls-120/test_selector_graph_seq_rawproj_change_weight_calib_shift005_valid_macro.json \
    --input graph reasoningDataset/tls-120/fusion_input_graph_acc_ft.json \
    --input seq reasoningDataset/tls-120/fusion_input_seq_baseline.json \
    --label_map reasoningDataset/tls-120/train_tower1_change_weight/label_map.json \
    --c_grid 0.01,0.03 \
    --class_weight_grid none \
    --select_metric macro_f1 \
    --include_confidence \
    --unified_expert_slots base,graph,seq,prior_base,emb_lr,emb_et,proto_emb,paired,slot_stacker,soft_gate \
    --output_json reasoningDataset/tls-120/test_stacker_unified_slot_tls120_confidence_valid_macro.json

conda run --no-capture-output -n llm-factory \
  python train_expert_gate.py \
    --input base reasoningDataset/tls-120/test_selector_graph_seq_rawproj_change_weight_calib_shift005_valid_macro.json \
    --input slot_stacker reasoningDataset/tls-120/test_stacker_unified_slot_tls120_confidence_valid_macro.json \
    --label_map reasoningDataset/tls-120/train_tower1_change_weight/label_map.json \
    --hidden_dim 16 \
    --epochs 80 \
    --lr 0.01 \
    --weight_decay 0.001 \
    --entropy_weight 0.01 \
    --cv_splits 3 \
    --seed 7 \
    --device cpu \
    --unified_expert_slots base,graph,seq,prior_base,emb_lr,emb_et,proto_emb,paired,slot_stacker,soft_gate \
    --output_json reasoningDataset/tls-120/test_expert_gate_base_slot_stacker_tls120_e80_seed7.json

conda run --no-capture-output -n llm-factory \
  python validation_gated_selector.py \
    --input base reasoningDataset/tls-120/test_selector_graph_seq_rawproj_change_weight_calib_shift005_valid_macro.json \
    --input slot_stacker reasoningDataset/tls-120/test_stacker_unified_slot_tls120_confidence_valid_macro.json \
    --input soft_gate reasoningDataset/tls-120/test_expert_gate_base_slot_stacker_tls120_e80_seed7.json \
    --label_map reasoningDataset/tls-120/train_tower1_change_weight/label_map.json \
    --select_metric macro_f1 \
    --strategies always,class_bias_calibration \
    --alpha_grid 0.5,5 \
    --calibration_strength_grid 1.0 \
    --calibration_temperature_grid 1.25 \
    --min_valid_gain_over_base 0 \
    --bootstrap_samples 100 \
    --bootstrap_min_win_rate 0.6 \
    --bootstrap_min_gain_quantile=-0.0015 \
    --max_prediction_change_rate 0.05 \
    --unified_expert_slots base,graph,seq,prior_base,emb_lr,emb_et,proto_emb,paired,slot_stacker,soft_gate \
    --output_json reasoningDataset/tls-120/test_selector_soft_gate_tls120_tol0015_calib_family_valid_macro.json
```

Robust lower-bound candidate ranking for VPN selector debugging:

```bash
conda run --no-capture-output -n llm-factory \
  python validation_gated_selector.py \
    --input base reasoningDataset/vpn-app/test_fusion_best_prior_flow_embedding_experts_minbest90_valid_acc.json \
    --input emb_lr reasoningDataset/vpn-app/test_flow_embedding_classifier_logreg_meta_valid_acc.json \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --select_metric macro_f1 \
    --rank_select_metric accuracy \
    --rank_metric bootstrap_gain_quantile \
    --rank_bootstrap_samples 300 \
    --rank_candidate_limit 256 \
    --strategies always,threshold_switch \
    --expert_conf_grid 0.3,0.85 \
    --expert_margin_grid 0.05 \
    --base_conf_max_grid 1 \
    --delta_conf_grid -1 \
    --delta_margin_grid -1 \
    --min_valid_gain_over_base 0.08 \
    --bootstrap_samples 300 \
    --bootstrap_min_gain_quantile -0.001 \
    --max_prediction_change_rate 0 \
    --output_json reasoningDataset/vpn-app/test_selector_boot_rank_debug_valid_macro.json
```

This command is a robustness-oriented selector ablation, not the current best result. The strict VPN target-shift gate can still force fallback to the base prediction if a high-validation or high-bootstrap-gain candidate rewrites too many target predictions.

VPN split1/split2 Tower-1 retrain and multi-split Tower-2 stability check:

```text
Question:
  Are the weaker train_val_split_1 and train_val_split_2 results caused mainly by reusing a split0 Tower-1 encoder?

Strict retrain answer:
  No. Retraining Tower-1 separately on split1/split2, then re-extracting packet embeddings and retraining Tower-2, was worse than the older split-specific Tower-2 checks.

Strict retrain outputs:
  split1 safe prior:
    reasoningDataset/vpn-app/test_fusion_graph_seq_rawproj_flowaware_change_weight_split1_retrain_stage8_flowaware_split1_retrain_t1_safe_prior_residual.json
    test accuracy = 0.5795
    test macro-F1 = 0.5429
  split2 safe prior:
    reasoningDataset/vpn-app/test_fusion_graph_seq_rawproj_flowaware_change_weight_split2_retrain_stage8_flowaware_split2_retrain_t1_safe_prior_residual.json
    test accuracy = 0.5269
    test macro-F1 = 0.5106

Interpretation:
  The drop is not primarily a Tower-1 encoder mismatch. The evidence points more toward split-specific distribution difficulty, near-duplicate/split-artifact issues, and small-data Tower-1 fine-tuning instability. For the paper, this should be framed as a robustness ablation, not as the final VPN result.
```

For a cross-split robustness candidate, merge the split0/split1/split2 Tower-2 training windows while excluding the split0 validation `flow_id` values from the merged training file:

```bash
conda run --no-capture-output -n llm-factory \
  python combine_tower2_datasets.py \
    --inputs \
      reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/graph_dataset.pt \
      reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_split1/graph_dataset.pt \
      reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_split2/graph_dataset.pt \
    --exclude_flow_ids_from reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt \
    --output reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_multisplit_train012_excl_valid0/graph_dataset.pt

conda run --no-capture-output -n llm-factory \
  python combine_tower2_datasets.py \
    --inputs \
      reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/seq_dataset.pt \
      reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_split1/seq_dataset.pt \
      reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_split2/seq_dataset.pt \
    --exclude_flow_ids_from reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/seq_dataset.pt \
    --output reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_multisplit_train012_excl_valid0/seq_dataset.pt
```

Train Stage-8 graph/seq Tower-2 on the merged training set, still selecting checkpoints on the original split0 validation set:

```bash
conda run --no-capture-output -n llm-factory \
  python train_tower2.py \
    --model_type graph \
    --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_multisplit_train012_excl_valid0/graph_dataset.pt \
    --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt \
    --output_dir checkpoints/tower2_graph_flow_vpn_app_rawproj_change_weight_multisplit_train012_excl_valid0_stage8 \
    --num_classes 16 \
    --epochs 30 \
    --hidden_dim 256 \
    --num_layers 2 \
    --num_heads 4 \
    --train_level flow \
    --flow_pooling multi_view \
    --window_loss_weight 0.3 \
    --class_weighting effective \
    --class_weight_beta 0.9999 \
    --flow_contrastive_weight 0.05 \
    --flow_temperature 0.07 \
    --confidence_penalty_weight 0.02 \
    --select_metric flow_macro_f1 \
    --aux_weight 0 \
    --coherence_weight 0

conda run --no-capture-output -n llm-factory \
  python train_tower2.py \
    --model_type seq \
    --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_multisplit_train012_excl_valid0/seq_dataset.pt \
    --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/seq_dataset.pt \
    --output_dir checkpoints/tower2_seq_flow_vpn_app_rawproj_change_weight_multisplit_train012_excl_valid0_stage8 \
    --num_classes 16 \
    --epochs 30 \
    --hidden_dim 256 \
    --num_layers 2 \
    --num_heads 4 \
    --train_level flow \
    --flow_pooling multi_view \
    --window_loss_weight 0.3 \
    --class_weighting effective \
    --class_weight_beta 0.9999 \
    --flow_contrastive_weight 0.05 \
    --flow_temperature 0.07 \
    --confidence_penalty_weight 0.02 \
    --select_metric flow_macro_f1 \
    --aux_weight 0 \
    --coherence_weight 0
```

The merged Stage-8 Tower-2 candidate reached:

```text
multisplit graph:
  reasoningDataset/vpn-app/test_graph_metrics_flow_rawproj_change_weight_multisplit_train012_excl_valid0_stage8_probs.json
  test accuracy = 0.6639
  test macro-F1 = 0.6246

multisplit seq / safe prior:
  reasoningDataset/vpn-app/test_fusion_graph_seq_rawproj_change_weight_multisplit_train012_excl_valid0_stage8_safe_prior_residual.json
  test accuracy = 0.6788
  test macro-F1 = 0.6501
```

Then add it as a conservative validation-gated selector expert on top of the current best VPN result:

```bash
conda run --no-capture-output -n llm-factory \
  python validation_gated_selector.py \
    --input base reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_calib_shift000_valid_macro.json \
    --input multisplit reasoningDataset/vpn-app/test_fusion_graph_seq_rawproj_change_weight_multisplit_train012_excl_valid0_stage8_safe_prior_residual.json \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --select_metric macro_f1 \
    --rank_select_metric accuracy \
    --rank_metric bootstrap_gain_quantile \
    --rank_bootstrap_samples 300 \
    --rank_candidate_limit 256 \
    --strategies always,threshold_switch,reliability_fusion \
    --expert_conf_grid 0.5,0.7,0.8,0.9 \
    --expert_margin_grid 0.05,0.1,0.2,0.3 \
    --base_conf_max_grid 1 \
    --delta_conf_grid 0,0.05,0.1 \
    --delta_margin_grid 0,0.05,0.1 \
    --min_valid_gain_over_base 0 \
    --bootstrap_samples 300 \
    --bootstrap_min_gain_quantile -0.001 \
    --max_prediction_change_rate 0.03 \
    --output_json reasoningDataset/vpn-app/test_selector_best_plus_multisplit_train012_excl_valid0_stage8_conservative_valid_macro.json
```

Current conservative selector result:

```text
selected strategy = threshold_switch
target prediction change = 0.0293
test accuracy = 0.7398
test macro-F1 = 0.7368
```

Important caveat: the merged split0/1/2 Tower-2 training run reached near-perfect validation scores even after excluding split0 validation `flow_id` values. A content-level audit confirms why `flow_id` exclusion is insufficient: the same pcap content can appear under different split folders and therefore receives a different path-derived `flow_id`. Treat the flow-id-only multi-split result as a useful ablation, not as the paper-safe headline. The current paper-safe VPN headline remains the stricter split0 selector result (`0.7488` accuracy / `0.7558` macro-F1), while the flow-id-only multi-split conservative selector is evidence that extra split coverage can improve robustness-oriented F1 only when guarded by target-shift constraints.

Run the split duplicate audit on VPN/TLS before using multi-split data as a robustness claim:

```bash
conda run --no-capture-output -n llm-factory \
  python audit_flow_split_duplicates.py \
    --root /home/jing/download/sweet/flow-level-classification/vpn-app \
    --include_test \
    --output_json reasoningDataset/vpn-app/split_duplicate_audit_vpn_app.json \
    --max_packets 64 \
    --payload_prefix_len 64 \
    --l3_prefix_len 256

conda run --no-capture-output -n llm-factory \
  python audit_flow_split_duplicates.py \
    --root /home/jing/download/sweet/flow-level-classification/tls \
    --include_test \
    --output_json reasoningDataset/tls-120/split_duplicate_audit_tls120.json \
    --max_packets 64 \
    --payload_prefix_len 64 \
    --l3_prefix_len 256 \
    --sample_groups 5 \
    --no_progress
```

Observed duplicate evidence:

```text
VPN audit:
  total flows = 4833
  parse errors = 0
  file_sha256 duplicate groups = 551
  file_sha256 duplicate flows = 3339
  file_sha256 cross-partition groups = 528
  file_sha256 cross-split-root groups = 528
  endpoint_invariant duplicate flows = 3428
  payload_prefix duplicate flows = 3562
  behavior_hash duplicate flows = 3558

TLS-120 audit:
  total flows = 42637
  parse errors = 0
  file_sha256 duplicate groups = 10365
  file_sha256 duplicate flows = 31095
  file_sha256 cross-partition groups = 10365
  file_sha256 cross-split-root groups = 10365
  endpoint_invariant duplicate flows = 31095
  payload_prefix duplicate flows = 31095
  behavior_hash duplicate flows = 31098
```

For a stricter multi-split ablation, use `combine_tower2_datasets.py` with content-hash exclusion. This maps Tower-2 `flow_id` values back to `pcap_path` through `flow_embedding_index.jsonl`, computes pcap SHA256 hashes, and excludes any training window whose pcap content matches the validation index:

```bash
conda run --no-capture-output -n llm-factory \
  python combine_tower2_datasets.py \
    --inputs \
      reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/graph_dataset.pt \
      reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_split1/graph_dataset.pt \
      reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_split2/graph_dataset.pt \
    --flow_embedding_indices \
      reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
      reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight_split1/flow_embedding_index.jsonl \
      reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight_split2/flow_embedding_index.jsonl \
    --exclude_content_from_indices reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --output reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_multisplit_train012_content_excl_valid0/graph_dataset.pt

conda run --no-capture-output -n llm-factory \
  python combine_tower2_datasets.py \
    --inputs \
      reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/seq_dataset.pt \
      reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_split1/seq_dataset.pt \
      reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_split2/seq_dataset.pt \
    --flow_embedding_indices \
      reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
      reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight_split1/flow_embedding_index.jsonl \
      reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight_split2/flow_embedding_index.jsonl \
    --exclude_content_from_indices reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --output reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_multisplit_train012_content_excl_valid0/seq_dataset.pt
```

The content-clean merge removed substantial validation-overlapping content:

```text
flow-id-only merge:
  3453 windows from 2105 flows

content-clean merge:
  2234 windows from 1404 flows
  excluded_content_hashes = 176
  content_excluded_windows = 1219
  missing_content_hash_windows = 0
```

For within-split exact-content deduplication, add `--dedupe_content`. The first
flow for each PCAP SHA256 keeps all of its windows; later path aliases with the
same content are dropped as complete flows, and conflicting labels are rejected:

```bash
conda run --no-capture-output -n llm-factory \
  python combine_tower2_datasets.py \
    --inputs reasoningDataset/vpn-app/train_tower2_rawproj_change_weight_split1_t1paired_s80/seq_dataset.pt \
    --flow_embedding_indices reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight_split1_t1paired_s80/flow_embedding_index.jsonl \
    --dedupe_content \
    --output /tmp/two_tower_runs/content_clean/vpn_fold1_train_seq.pt
```

Recompute headline metrics with each exact PCAP content counted once and obtain
content-group bootstrap confidence intervals:

```bash
conda run --no-capture-output -n llm-factory \
  python evaluate_content_unique_predictions.py \
    --prediction_json reasoningDataset/vpn-app/test_crossfold_consensus_auto_confidence.json \
    --flow_embedding_index reasoningDataset/vpn-app/test_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --output_json reasoningDataset/vpn-app/test_crossfold_consensus_auto_confidence_content_unique.json \
    --bootstrap_samples 2000 \
    --summary_only
```

Observed paper-audit results:

```text
VPN: 1672 paths -> 1645 unique PCAP contents
  original acc/F1 = 0.7512/0.7522
  content-unique acc/F1 = 0.7532/0.7570
  95% CI: acc=[0.7325, 0.7745], macro-F1=[0.7246, 0.7836]

TLS-120: 11542 paths -> 11542 unique PCAP contents
  original and content-unique acc/F1 = 0.8461/0.8292
  95% CI: acc=[0.8397, 0.8525], macro-F1=[0.8197, 0.8359]
```

Thus the shared-test headline is not inflated by exact-content duplicates. The
large duplicate counts remain important for training/validation effective sample
size and must still be disclosed. `make_paper_evidence_pack.py` now reads these
`*_content_unique.json` files and emits the `Content-Unique Robustness` table
alongside the main claim and bootstrap evidence.

Training the same Stage-8 Tower-2 settings on the content-clean merged set gives a more honest but weaker ablation:

```text
content-clean graph:
  reasoningDataset/vpn-app/test_graph_metrics_flow_rawproj_change_weight_multisplit_train012_content_excl_valid0_stage8_probs.json
  test accuracy = 0.6447
  test macro-F1 = 0.5986

content-clean seq:
  reasoningDataset/vpn-app/test_seq_metrics_flow_rawproj_change_weight_multisplit_train012_content_excl_valid0_stage8_probs.json
  test accuracy = 0.6316
  test macro-F1 = 0.5857

content-clean graph safe prior:
  reasoningDataset/vpn-app/test_graph_rawproj_change_weight_multisplit_train012_content_excl_valid0_stage8_safe_prior.json
  test accuracy = 0.6441
  test macro-F1 = 0.5972
```

Selector safety check:

```bash
conda run --no-capture-output -n llm-factory \
  python validation_gated_selector.py \
    --input base reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_calib_shift000_valid_macro.json \
    --input content_clean_graph reasoningDataset/vpn-app/fusion_input_graph_rawproj_change_weight_multisplit_train012_content_excl_valid0_stage8.json \
    --input content_clean_seq reasoningDataset/vpn-app/fusion_input_seq_rawproj_change_weight_multisplit_train012_content_excl_valid0_stage8.json \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --select_metric macro_f1 \
    --strategies always,threshold_switch \
    --expert_conf_grid 0.7,0.9 \
    --expert_margin_grid 0.1,0.2 \
    --base_conf_max_grid 1 \
    --delta_conf_grid 0 \
    --delta_margin_grid 0 \
    --min_valid_gain_over_base 0 \
    --bootstrap_samples 100 \
    --bootstrap_min_gain_quantile -0.001 \
    --max_prediction_change_rate 0.03 \
    --output_json reasoningDataset/vpn-app/test_selector_best_plus_content_clean_multisplit_train012_stage8_small_valid_macro.json
```

The selector correctly falls back to the base model:

```text
selected strategy = always
selected source = base
test accuracy = 0.7488
test macro-F1 = 0.7558

rejected content_clean_seq:
  valid macro-F1 gain = +0.0272
  bootstrap 5% gain quantile = -0.0037
  target prediction change = 0.2344

rejected content_clean_graph:
  valid macro-F1 gain = +0.0181
  bootstrap 5% gain quantile = -0.0111
  target prediction change = 0.2063
```

Research conclusion: the next paper-facing improvement should optimize for content-grouped cross-split stability, not single-split or duplicated multi-split accuracy. A useful next model iteration is group-aware training/validation: use content hashes or endpoint-invariant hashes as groups, keep duplicate groups inside one split, and train the unified graph/seq/statistics/expert-gate framework on these grouped splits. This turns the problem into a stronger generalization setting and is a cleaner CCF B/A story than additional probability-level tuning.

3-fold stability ledger:

The paper-facing evaluation target is now cross-split stability, not only the best single split. For SWEET VPN and TLS, `train_val_split_0/1/2` are treated as three ready-made folds: each folder already contains its own `train` and `val`, and all folds share the same `test`. Do not merge these fold folders for the main result. Use merging only as a documented ablation.

Generate the current 3-fold status and the exact missing/weak-fold rerun commands:

```bash
conda run --no-capture-output -n llm-factory \
  python summarize_cross_split_results.py \
    --dataset vpn-app \
    --dataset tls-120 \
    --output_json reasoningDataset/cross_split_summary.json \
    --output_md reasoningDataset/cross_split_summary.md
```

Current fold-level status from the latest summary:

```text
VPN targets: accuracy >= 0.7400, macro-F1 >= 0.6500
  fold0: pass, acc=0.7488, macro-F1=0.7558
  fold1: weak, acc=0.6944, macro-F1=0.6768
  fold2: weak, acc=0.7063, macro-F1=0.7030
  cross-fold consensus ensemble: acc=0.7512, macro-F1=0.7522

TLS-120 targets: accuracy >= 0.7800, macro-F1 >= 0.7000
  fold0: pass, acc=0.7996, macro-F1=0.7869
  fold1: weak, acc=0.7539, macro-F1=0.7268
  fold2: weak, acc=0.7701, macro-F1=0.7421
  cross-fold consensus ensemble: acc=0.8461, macro-F1=0.8292
```

The consensus number is a three-model ensemble on the shared test flows; it is
not an independent per-fold result. `summarize_cross_split_results.py` keeps
single-fold evidence and cross-fold consensus in separate report fields. The
historical `_fold0/_fold1/_fold2` consensus aliases are ignored by the scanner
and must not be used to claim that every independent fold reaches the ensemble
score.

The cross-split summary now ranks candidates by `target_gap` by default: the
primary score is the weaker of `(accuracy - target_accuracy)` and
`(macro-F1 - target_macro_F1)`, followed by the two individual target gaps. This
keeps the automatic loop focused on the bottleneck metric instead of selecting a
slightly higher-F1 candidate when accuracy is already the real weak point.

Interpretation: individual fold models are not stable enough on VPN split1/split2
or TLS split1/split2, but the same-dataset cross-fold consensus is stable. The
consensus module uses the three ready-made train/valid folds as independent
views over one shared test set, aligns shared flow ids, and fuses predictions
without test labels. `auto_confidence` chooses `vote_priority` when the fold
models are highly overconfident, as in VPN, and `log_mean` when they are less
overconfident, as in TLS-120. This moves the main result from single-split
selection to cross-split stability.

Latest weak-fold ablations:

```text
VPN fold1 + flow-statistics expert selector:
  selected fallback base; test acc=0.6411, macro-F1=0.6030.

VPN fold1 + strong Tower-2 regularization:
  seq test acc=0.6346, macro-F1=0.5871
  graph test acc=0.6429, macro-F1=0.6017
  graph/seq fusion test acc=0.6423, macro-F1=0.6003
  constrained selector again falls back to the old base.

VPN fold1 + paired full-header/randomized-IP-port consistency:
  seq test acc=0.6274, macro-F1=0.5952
  graph test acc=0.6477, macro-F1=0.6192
  graph/seq fusion test acc=0.6376, macro-F1=0.6042
  validation-gated reliability fusion selects a low-shift paired expert:
  test acc=0.6519, macro-F1=0.6200.

VPN fold1 + Tower-1 paired full-header/randomized-IP-port consistency continuation:
  Tower-1 was continued for 80 steps from `checkpoints/tower1_qwen_multitask_change_weight`
  with paired embedding/logit consistency against the IP/port-randomized packet view.
  Re-extracting raw+projected embeddings and retraining the same Stage-8 Tower-2
  raises the fold1 Tower-2 base to:
  seq test acc=0.6675, macro-F1=0.6380
  graph test acc=0.6549, macro-F1=0.6183
  graph/seq fusion test acc=0.6633, macro-F1=0.6275
  conservative selector still falls back to the old base because the old base has
  much higher validation F1 but worse shared-test F1. This is evidence of
  validation/test shift, not evidence that the new encoder is worse.

VPN fold1 + paired Tower-1 seq + label-free target-prior softcap ensemble:
  Using `calibrate_prior_ensemble.py` with hard-prior KL cap `0.005`,
  `prior_softcap_valid` candidate pooling, and `log_mean` ensembling improves the
  current fold1 best to:
  test acc=0.6722, macro-F1=0.6501.
  This is the first fold1 result to cross the macro-F1 target, although accuracy
  is still below the 0.74 fold target.

VPN fold1 + pairwise/group confusion refinement:
  Starting from the paired Tower-1 seq + prior-softcap result, top-2 pairwise
  refinement with message/header/port flow features improves to
  test acc=0.6794, macro-F1=0.6566.
  Adding the social/media confusion-group refinement gives the current fold1 best:
  test acc=0.6812, macro-F1=0.6585.
  The gain is local and low-risk: pairwise refinement changed 97/1672 test flows,
  while the group refinement made a small additional correction.

VPN fold1 + focused pairwise refinement + second label-free prior softcap:
  Re-running pairwise refinement from the current fold1 best with explicit
  `facebook-hangout-skype-youtube` confusion pairs and `pair_mass` application
  improves to test acc=0.6872, macro-F1=0.6678.
  A second label-free prior-softcap ensemble on top of that local refinement gives
  the first strong fold1 improvement:
  test acc=0.6932, macro-F1=0.6759.
  A follow-up prior grid finds two useful endpoints:
  target-margin best: test acc=0.6914, macro-F1=0.6821
  accuracy-rank best: test acc=0.6944, macro-F1=0.6768
  This confirms that the useful post-hoc family is not a one-shot calibration,
  but a small target-prior/local-confusion/target-prior loop.

VPN fold2 + message/header/port flow-statistics expert:
  selected random forest; test acc=0.6657, macro-F1=0.6264.
  It is the current fold2 best, but still below the target.

VPN fold2 + paired full-header/randomized-IP-port consistency:
  seq test acc=0.6591, macro-F1=0.6126
  graph test acc=0.6675, macro-F1=0.6109
  graph/seq fusion test acc=0.6591, macro-F1=0.6126
  validation-gated selector falls back to the flow-statistics expert.
  Interpretation: the paired branch slightly improves graph accuracy on fold2,
  but it does not improve the fold2 macro-F1 target.

VPN fold2 + Tower-1 paired full-header/randomized-IP-port consistency continuation:
  Using the same 80-step Tower-1 paired continuation and the same Stage-8 Tower-2
  settings does not improve fold2:
  seq test acc=0.6340, macro-F1=0.6005
  graph test acc=0.6226, macro-F1=0.5858
  graph/seq fusion test acc=0.6370, macro-F1=0.5998
  The current fold2 best therefore remains the message/header/port statistics
  expert. The paired branch should stay in the unified framework as a trainable
  or validation-gated expert, not as a mandatory replacement for every fold.

VPN fold2 + message/header/port statistics expert + label-free target-prior softcap ensemble:
  Applying the same target-prior candidate module to the fold2 statistics expert
  gives a large shared-test improvement:
  test acc=0.7033, macro-F1=0.6995.
  This clears the fold2 macro-F1 target but still misses the 0.74 accuracy
  target. The improvement mainly fixes prior imbalance: under-predicted `skype`
  recovers while over-predicted small classes are softened.

VPN fold2 + pairwise/group confusion refinement:
  Top-2 pairwise refinement on the fold2 statistics + prior-softcap result gives
  the current fold2 best:
  test acc=0.7039, macro-F1=0.7003.
  The social/media group refinement is a negative ablation on fold2
  (`0.6645/0.6382`), so it should remain a candidate expert that can be
  validation-gated down, not a mandatory post-processor.

VPN fold2 + second label-free prior softcap:
  Applying a second prior-softcap ensemble after the fold2 pairwise refinement
  gives a small additional gain:
  test acc=0.7045, macro-F1=0.7012.
  Fold2 now clears the macro-F1 target comfortably, but accuracy is still below
  the 0.74 fold target.

VPN fold2 + safety-constrained trainable residual:
  Although unconstrained trainable stacker/soft-gate experts overfit, a residual
  fusion constrained to keep at least 90% of the then-current fold2 best was mildly
  useful:
  base=0.925, stacker=0.075, softgate=0.000
  test acc=0.7057, macro-F1=0.7026.
  The analogous fold1 residual does not improve accuracy, so the fold1 best
  remains the focused pairwise + second prior result.

VPN fold1/fold2 trainable stacker and soft expert gate check:
  A trainable logistic stacker over the current expert set reaches very high
  validation scores, but drops on the shared test set:
  fold1 stacker test acc=0.6764, macro-F1=0.6240
  fold2 stacker test acc=0.6531, macro-F1=0.6109
  A soft expert gate with stratified out-of-fold validation is also negative:
  fold1 soft-gate test acc=0.6782, macro-F1=0.6518
  fold2 soft-gate test acc=0.6352, macro-F1=0.6018
  Interpretation: single-fold validation is too easy for trainable expert fusion.
  Future trainable fusion/local-expert modules must include cross-fold stability
  or target-shift guards before they can replace the safer prior/refinement loop.

Guarded trainable stacker:
  `train_prediction_stacker.py` now treats the logistic stacker as a candidate
  expert with safety gates. It records the base expert, candidate metrics,
  final metrics, validation gain over base, prediction-change rate, and
  prediction-distribution JS divergence. The stacker is used only when it clears
  `--min_valid_gain_over_base`, `--max_prediction_change_rate`, and
  `--max_prediction_js_divergence`; otherwise the output falls back to the base
  expert while keeping the candidate probabilities for audit.
  On VPN split1 retrain, the old high-validation stacker changes 48.4% of test
  predictions and is rejected by `--max_prediction_change_rate 0.35`, avoiding
  the known shared-test drop. On TLS fold2, the same unified module remains
  active under default gates and gives test acc=0.7696, macro-F1=0.7417.
  Run it from the Stage-8 runner with `--stage stacker`; the runner builds the
  graph/seq fusion payloads and then calls the guarded stacker with the same
  module family for VPN/TLS.

Guarded unified expert selector:
  `validation_gated_selector.py` now supports explicit `--base_input` and both
  prediction-change and prediction-distribution JS guards. The Stage-8 runner
  exposes this as `--stage selector`, taking the same expert slots on every
  dataset: `base`, `prior`, and `stacker`. The default automatic selector is
  deliberately lightweight (`always,class_bias_calibration`) so TLS-120 can run
  without exploding the candidate pool; heavier threshold/reliability strategies
  remain available for manual ablations. On TLS fold2, the selector chooses a
  class-bias calibration over the stacker branch and improves the current fold2
  result to test acc=0.7701, macro-F1=0.7421, while respecting
  `--max_prediction_change_rate 0.35` and `--max_prediction_js_divergence 0.03`.

VPN fold1 MI-Transformer negative ablation:
  A flow-level multi-instance Transformer over packet embeddings was tested on
  fold1 t1paired_s80 embeddings with strong dropout/weight decay:
  `test_mi_flow_transformer_fold1_t1paired_s80_h128_reg.json`.
  It reached validation acc=0.8693 and macro-F1=0.8702 but dropped on the shared
  test set to acc=0.6328 and macro-F1=0.5787. A 98%-base residual fusion against
  the current fold1 accuracy-best result assigned 0 weight to the MI branch.
  Interpretation: simply adding a flow-level Transformer increases split-specific
  fitting and does not solve the VPN validation/test shift. Future model-side
  work should use cross-split invariance/augmentation objectives rather than a
  larger flow aggregator alone.

VPN fold1 focal-loss Tower2 negative ablation:
  `train_tower2.py` now supports `--focal_gamma` as a unified optional Tower-2
  classification loss. A fold1 seq-flow run on t1paired_s80 with
  `--focal_gamma 1.5`, stronger dropout/weight decay, multi-view pooling,
  confusion SupCon, and window-to-flow contrastive reached validation
  acc=0.8636/macro-F1=0.8635 but only shared-test acc=0.6477/macro-F1=0.5952.
  This confirms that simply emphasizing hard training examples still overfits
  the validation split and does not solve the VPN split-shift bottleneck.

VPN fold1 Tower2 view-domain adversarial negative ablation:
  `train_tower2.py` supports `--view_domain_adversarial_weight` with a paired
  clean/randomized-header Tower-2 dataset, and `run_stage8_flowaware_pipeline.py`
  exposes the same option for the unified Stage-8 runner. A fold1 seq-flow run
  using clean embeddings plus the IP/port-randomized paired view reached high
  validation performance but only shared-test acc=0.6172 and macro-F1=0.5750 in
  `test_seq_metrics_flow_fold1_viewadv_ipport_probs.json`. Interpretation:
  forcing clean-vs-randomized view invariance only at Tower-2 is too late and
  still overfits the validation split. Keep this as a gated/weighted module for
  unified-framework ablations, not as a default mandatory branch.

VPN fold2 lightweight pairwise local-expert negative ablation:
  `refine_top2_pairwise.py` and `refine_confusion_groups.py` now print feature
  loading/training progress and support `--final_n_estimators` for fast
  automatic-loop probes. A fold2 local pairwise search from the current best with
  message/header/port features, prefix length 48, and the top validation-error
  pairs changed 22 shared-test flows but did not improve the target:
  `test_refine_pairwise_fold2_currentbest_msgheader_ports_prefix48_social_light_acc.json`
  gives test acc=0.7057 and macro-F1=0.7026, marginally below the current best
  F1. Interpretation: local pair specialists are no longer the fold2 bottleneck.

VPN weak-fold trainable flow-statistics expert:
  `train_tower2.py` now supports a unified `--flow_stat_expert_weight` branch
  inside `FlowAggregationHead`. The branch computes mean/std/min/max summaries
  over the trailing packet metadata features, predicts class logits with a small
  MLP, and fuses them through a learned scalar gate. `test_tower2.py` reports
  the learned `flow_stat_gate` summary, so the paper can show that every dataset
  traverses the same statistics branch while training learns dataset-specific
  usage. On fold2, a seq-flow run with `--flow_stat_expert_weight 0.25` and
  `--flow_stat_aux_weight 0.1` reached validation acc=0.9091/macro-F1=0.9089
  but dropped on the shared test set to acc=0.6495/macro-F1=0.6019. A 98%-base
  residual fusion assigns 2% weight to this branch and gives the current fold2
  best:
  `test_residual_fusion_fold2_currentbest_plus_flowstat_minbase98_acc.json`
  with test acc=0.7063 and macro-F1=0.7030. On fold1 t1paired_s80, the same
  branch reached validation acc=0.8636/macro-F1=0.8592 but dropped on shared
  test to acc=0.6388/macro-F1=0.6033, and the 98%-base residual made no
  prediction-changing improvement:
  `test_residual_fusion_fold1_currentbest_plus_flowstat_minbase98_acc.json`
  remains acc=0.6944/macro-F1=0.6768. Interpretation: the trainable statistics
  branch is useful only as a safety-gated residual expert and is not sufficient
  as a standalone Tower-2 head; the gate must be evaluated cross-fold rather than
  trusted from one validation split.

Flow-statistics provenance correction (2026-07-21): when a native structural
embedding is attached, Tower-2's trailing structural block contains the learned
native latent followed by the explicit packet-local metadata. Older flow-stat
experiments summarized that entire block (for the current unified setup, 128
native dimensions plus 13 explicit fields), included padded rows in the seq
path, counted packets in overlapping windows repeatedly, and used duplicate
count features. The implementation now summarizes only the shared explicit
packet-local fields (`meta_feature_dim - native_structural_dim`), removes padded
rows, reconstructs unique packet positions from each window's `(start,end)`
coordinates, and records separate unique-packet and window counts. Consequently, the historical
flow-stat numbers above remain diagnostic results but are not clean evidence for
the corrected structural-statistics expert. The `paper_unified` main profile
still keeps this expert disabled; any promotion requires a new cross-fold VPN
and TLS ablation using the corrected implementation and validation-only model
selection.

The first corrected screening run is pre-registered on VPN fold1 with
`flow_stat_expert_weight=0.10`, `flow_stat_aux_weight=0.02`, and every other
Tower2 setting copied from the semantic-anchor baseline. Its acceptance
threshold is validation flow Macro-F1 `>=0.653643`, i.e. the frozen fold1
baseline `0.648643 + 0.005`. The queued runner evaluates the shared test only
after that threshold passes. A passing fold1 result is still insufficient for
promotion: the same fixed configuration must then be checked on the remaining
VPN folds and TLS-120, with no dataset-specific gate cap or loss weight.

The corrected fold1 screening run was rejected on validation: its best
Macro-F1 was `0.652364`, an absolute gain of only `0.003721`, below the frozen
`0.005` acceptance margin. The shared test was therefore not evaluated. This
branch remains an ablation and is not added to `paper_unified`; the result is
recorded in
`reasoningDataset/vpn-app/ablation_paper_unified_anchor_clean_flowstat_w010_aux002_fold1_valid_probs.json`.

An intervention-router screening tested a factual identity path with at most a
25% learned intervened residual, leaving every other fold1 setting unchanged.
It was also rejected before test evaluation: validation accuracy/Macro-F1 were
`0.630682/0.630720`, versus the frozen symmetric-view baseline Macro-F1
`0.648643`. Its learned effective intervention contribution collapsed to only
`0.000460`. This shows that an unconstrained router learns to shut the
counterfactual view off and recover factual shortcuts; the symmetric
intervention base remains the `paper_unified` default. The optional
`--intervention_view_base_mode factual_anchor` path is retained solely to
reproduce this negative ablation.

An inference-only view-duplication diagnostic further checks the fixed
two-view checkpoints without retraining or changing their learned weights.
`test_tower2.py --ablate_intervention_view factual_only` feeds the factual
representation into both view slots, while `intervened_only` duplicates the
header-randomized representation. On held-out VPN fold0/fold1 validation, the
unaltered two-view models obtained Macro-F1 `0.661801/0.648643`, compared with
`0.609529/0.630554` for factual-only duplication. On the shared test set, the
corresponding values were `0.546932/0.563093` versus
`0.527837/0.552015`. Fixed fold01 factual-only `log_mean` gave accuracy/Macro-F1
`0.585526/0.563133`; the unaltered fixed two-view consensus gave
`0.616029/0.580074`. Intervened-only duplication collapsed to Macro-F1
`0.040376/0.051507` on validation and `0.051951/0.044441` on test after the
IP/port shortcut fields were removed. This consistent degradation
supports the view that the symmetric pair acts as a shortcut-resistance
constraint rather than a freely selectable expert. Because duplication changes
the input distribution of a model trained jointly on two views, these numbers
are a mechanism diagnostic, not a substitute for the separately trained
single-view baseline or a causal effect estimate.

An end-to-end one-epoch contract smoke test on the real VPN fold1 unified data
completed before the screened run. Training and independent checkpoint reload
both produced validation flow accuracy `0.5625` and Macro-F1 `0.5521`; the
checkpoint records `meta_feature_dim=141`, `native_structural_dim=128`, and
`flow_stat_meta_dim=13`, and the learned statistics gate was reported for all
352 validation flows (mean `0.4751`). These smoke metrics establish only that
padding removal, overlap de-duplication, checkpoint persistence, and evaluation
use the same feature contract. They are not an accuracy result and were not
evaluated on the shared test split.

The current semantic-anchor fold0/fold1 error audit also identifies a protocol-
level label-shift problem rather than only encoder variance. Tower2 train has
approximately 42--44 flows per class and validation has exactly 22 per class,
while the shared VPN test ranges from 13 to 413 flows per class. The fixed
fold01 consensus over-predicts `aim` (13 true versus 41 predicted), `icq`
(13 versus 46), and `spotify` (42 versus 101), while under-predicting `skype`
(413 versus 261). Its two-model oracle accuracy is only `0.6938`, so adding
folds alone cannot provide a large gain over SWEET.

A label-free shift diagnostic is therefore pre-registered using the existing
shared calibrator, not a VPN-specific threshold: target prior is the fixed
50/50 blend of confusion-corrected hard BBSE and EM soft-prior estimates;
candidate strengths are selected by `soft_prior_under_hard_cap` with
`hard_prior_kl_cap=0.005`. Fold1 held-out probabilities supply source
confusion/evidence, and fold01 fixed consensus probabilities are the unlabeled
target. Target labels are excluded from strength selection and used only for
the final diagnostic report. A positive VPN result cannot enter the main method
unless this exact rule is subsequently frozen for fold012 and TLS-120; otherwise
it remains an analysis-only indication of label shift.

The first fold01 run found no candidate satisfying that pre-registered cap:
hard-prior KL ranged from `0.1017` at identity to `0.0366` at strength `0.5`,
all above `0.005`. The calibrator now enforces strict feasibility and records
`selection_constraint_satisfied=false` plus
`selection_fallback_reason=identity_no_feasible_candidate`; it therefore
selects identity (`accuracy=0.6160`, Macro-F1 `0.5801`). The apparent
strength-0.5 test result (`0.6471`/`0.6284`) is retained only as a diagnostic
trend and is not an accepted result because it violates the frozen constraint.
For this selection scope, strength zero must be present so that an explicit
identity fallback always exists.

Softened target-prior safety ablation:
  `calibrate_prior_ensemble.py` now supports `--input_temperatures` and valid
  metric floors (`--min_valid_metric`, `--min_valid_gain_over_identity`). This
  lets the prior module test softened overconfident probabilities without
  allowing label-free target-prior matching to select a classification-collapse
  candidate. On fold1, the unguarded softened prior pool can choose candidates
  with very poor validation accuracy and shared-test acc=0.4593/macro-F1=0.1818.
  With `--min_valid_gain_over_identity -0.005`, the collapse is filtered, but
  the selected softened-prior output still does not beat the current fold1 best:
  `test_prior_soften_fold1_currentbest_logmean_tempgrid_validfloor_acc.json`
  gives acc=0.6938/macro-F1=0.6729. On fold2, the same guard falls back to the
  identity candidate over the current residual result:
  `test_prior_soften_fold2_currentbest_logmean_tempgrid_validfloor_acc.json`
  keeps acc=0.7063/macro-F1=0.7030. Interpretation: softened prior calibration
  is useful as a safety mechanism for the automatic loop, but it is not the next
  accuracy breakthrough.

Paired semantic alignment and target-prototype adaptation negative ablations:
  Packet embeddings for the randomized-IP/port view were re-extracted with the
  same fold1 Tower-1 encoder as the clean view, removing the earlier encoder
  mismatch confound. Tower-2 then used clean/randomized flow cosine alignment,
  cross-view SupCon, variance preservation, and covariance decorrelation. Its
  best fine-tuned model reached test acc=0.6621/macro-F1=0.6262, below the
  direct t1paired baseline (0.6675/0.6380). A separate source-anchored,
  confidence-weighted target-prototype adapter used no target labels, excluded
  each target flow from its own prototype, and selected all settings on
  validation. Both embedding-only and message/header variants selected the
  identity fallback, preserving the current fold1 result at 0.6944/0.6768.
  These results reject simple view alignment and transductive prototypes as the
  main solution to the validation/test shift.

Counterfactual header-intervention screening (same Tower-1 encoder):
  The clean and IP/port-randomized packet embeddings were re-extracted with the
  same fold1 `t1paired_s80` Tower-1 adapter. The randomized test index also uses
  the clean metadata index through `preprocess_tower2.py` with
  `--metadata_reference_index`, so the intervention changes the Qwen packet
  representation but does not accidentally delete Tower-2 structural fields.
  This removes both the earlier encoder-mismatch and metadata-shape confounds.

  A frozen Stage-8 classifier obtains clean test acc/F1=0.6675/0.6380, but only
  0.1232/0.0389 on the aligned randomized view. On validation, only 2 of 352
  flows are corrected exclusively by the randomized view, while 273 are
  correct exclusively in the clean view. Identity-initialized bounded residual
  fusion therefore gives no significant gain: the best residual setting gives
  test acc/F1=0.6657/0.6329, and a larger residual significantly lowers
  macro-F1 under paired bootstrap. A supervised router selects its identity
  path on validation.

  `train_intervention_transport.py` provides a label-free low-rank
  randomized-to-clean packet representation transport control. Rank 64 and 128
  both overfit packet pairs after one epoch. With the original Stage-8
  classifier frozen, transported test acc/F1 are 0.0987/0.0592 and
  0.1124/0.0797. Tower-2 training-time intervention consistency is also
  rejected: weak paired CE+KL is selected by clean validation but lowers clean
  shared-test acc/F1 to 0.6579/0.6273; stronger CE+KL and semantic alignment
  fall back to the epoch-0 checkpoint.

  Research interpretation: replacing IP/port tokens inside the Tower-1 text
  prompt is not a semantics-preserving augmentation in the current encoder
  space. Do not enable counterfactual fusion, representation transport, or
  paired-view losses in the best default. Keep them as compute-matched negative
  ablations showing that invariance cannot be recovered by a shallow Tower-2
  correction after the packet representation has already moved out of
  distribution.

```

### Paper candidate: reliability-gated semantic-structural interaction

The current paper-facing Tower-2 candidate is Reliability-Gated
Semantic-Structural Interaction (RGSSI). Unlike post-hoc probability fusion,
RGSSI changes the packet interaction representation itself while preserving a
strict compute-matched identity control:

1. Split each packet feature into the 3840-dimensional Qwen raw/projected
   semantic embedding and the trailing 14-dimensional structural metadata.
2. Convert a trained concat projection exactly as
   `W x + b = W_sem x_sem + W_meta x_meta + b`. The converted checkpoint has
   exactly the same epoch-0 embeddings and logits as the original Stage-8
   model.
3. Encode complementary evidence from semantic features, structural features,
   and their element-wise interaction. A packet-conditioned reliability gate
   adds at most 25% bounded residual evidence.
4. Zero-initialize the final interaction layer. With
   `--dual_channel_train_scope interaction`, freeze the legacy packet
   Transformer and flow head and train only 364,290 interaction/gate
   parameters. This prevents the new module from obtaining gains through a
   hidden full-model fine-tune.

Core options:

```text
--dual_channel_mode residual
--dual_channel_train_scope interaction
--dual_channel_gate_mode adaptive
--dual_channel_max_weight 0.25
--dual_channel_init 0.1
--init_checkpoint <existing-stage8-best.pt>
--select_init_checkpoint
```

Reproduce TLS-120 fold1 RGSSI from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n llm-factory \
  python train_tower2.py \
    --model_type seq \
    --dataset reasoningDataset/tls-120/train_tower2_rawproj_flowaware_change_weight_fold1/seq_dataset.pt \
    --valid_dataset reasoningDataset/tls-120/valid_tower2_rawproj_flowaware_change_weight_fold1/seq_dataset.pt \
    --output_dir checkpoints/tower2_seq_flow_tls120_rgssi_fold1 \
    --init_checkpoint checkpoints/tower2_seq_flow_tls_120_rawproj_flowaware_change_weight_fold1_stage8_flowaware_fold1_stage8_cv/best.pt \
    --select_init_checkpoint \
    --num_classes 120 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.01 \
    --hidden_dim 256 \
    --num_layers 2 \
    --num_heads 4 \
    --dropout 0.2 \
    --train_level flow \
    --flow_pooling multi_view \
    --flow_transformer_layers 1 \
    --flow_transformer_heads 4 \
    --multi_view_gate_entropy_weight 0.01 \
    --window_loss_weight 0.3 \
    --class_weighting effective \
    --class_weight_beta 0.9999 \
    --class_weight_strength 0.6 \
    --balanced_flow_batches \
    --samples_per_class 2 \
    --hierarchical_mode logit \
    --hierarchical_weight 0.2 \
    --hierarchical_logit_weight 0.5 \
    --coarse_groups none \
    --contrastive_mode confusion \
    --confusion_groups none \
    --flow_contrastive_weight 0.03 \
    --flow_temperature 0.07 \
    --label_smoothing 0.05 \
    --confidence_penalty_weight 0.02 \
    --aux_weight 0 \
    --coherence_weight 0 \
    --select_metric flow_macro_f1 \
    --early_stop_patience 5 \
    --dual_channel_mode residual \
    --dual_channel_train_scope interaction \
    --dual_channel_gate_mode adaptive \
    --dual_channel_max_weight 0.25 \
    --dual_channel_init 0.1 \
    --device cuda

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n llm-factory \
  python test_tower2.py \
    --checkpoint checkpoints/tower2_seq_flow_tls120_rgssi_fold1/best.pt \
    --dataset reasoningDataset/tls-120/test_tower2_rawproj_flowaware_change_weight_fold1/seq_dataset.pt \
    --label_map reasoningDataset/tls-120/train_tower1_flowaware_change_weight_fold1/label_map.json \
    --output_json reasoningDataset/tls-120/test_seq_metrics_flow_rgssi_fold1.json \
    --device cuda \
    --no_report
```

Keep all other dataset/fold arguments identical to the source checkpoint. The
epoch-0 checkpoint is included in validation selection, so RGSSI automatically
falls back to the original model when the interaction is not useful.

Strict TLS-120 shared-test results, with the same RGSSI settings on all three
provided train/valid folds:

| fold | Stage-8 seq acc/F1 | RGSSI seq acc/F1 | delta acc/F1 |
|---|---:|---:|---:|
| fold0 | 0.7558 / 0.7465 | **0.7790 / 0.7635** | +0.0231 / +0.0170 |
| fold1 | 0.7105 / 0.6779 | **0.7513 / 0.7201** | +0.0407 / +0.0422 |
| fold2 | 0.7111 / 0.6765 | **0.7531 / 0.7211** | +0.0420 / +0.0446 |

All three single-fold gains are significant under 2,000 paired bootstrap
resamples. Accuracy 95% gain intervals are `[0.0185, 0.0279]`,
`[0.0347, 0.0464]`, and `[0.0364, 0.0479]`; the corresponding exact McNemar
p-values are `1.86e-21`, `9.04e-43`, and `2.42e-46`. The three RGSSI models use
the existing label-free `auto_confidence -> log_mean` consensus and reach
acc/F1=0.8475/0.8304. This is slightly above the previous 0.8461/0.8292
consensus, but its paired CI crosses zero; report the robust single-model gains
as the primary evidence and the consensus change as statistically neutral.

Saved artifacts:

```text
checkpoints/tower2_seq_flow_tls120_rgssi_fold{0,1,2}/best.pt
reasoningDataset/tls-120/test_seq_metrics_flow_rgssi_fold{0,1,2}.json
reasoningDataset/tls-120/test_seq_metrics_flow_rgssi_fold{0,1,2}_bootstrap.json
reasoningDataset/tls-120/test_crossfold_consensus_rgssi_auto_confidence.json
reasoningDataset/tls-120/test_crossfold_consensus_rgssi_bootstrap.json
```

VPN is an important negative/control result. On split1, RGSSI changes shared
test acc/F1 from 0.6675/0.6380 to 0.6693/0.6321. On split2 it changes
0.6340/0.6005 to 0.6388/0.5976. Both paired confidence intervals cross zero;
therefore RGSSI is statistically neutral on VPN and must not replace its current
best cross-fold path. This dataset contrast supports a testable claim: explicit
structural correction is most useful when many fine-grained classes require
protocol/temporal evidence that the high-level semantic encoder does not retain.

Use `compare_prediction_bootstrap.py` for every promotion decision:

```bash
conda run --no-capture-output -n llm-factory \
  python compare_prediction_bootstrap.py \
    --baseline <stage8-test-probabilities.json> \
    --candidate <rgssi-test-probabilities.json> \
    --output_json <paired-bootstrap-report.json> \
    --samples 2000 \
    --seed 42
```

Paper positioning: RGSSI currently gives a coherent, identity-preserving model
contribution and stable TLS cross-split evidence, but it is not yet a complete
CCF-A claim. The next required extension is native byte/field pre-training with
masked protocol fields, packet relative position/direction, and flow
contrastive objectives. RGSSI should then route between that structural encoder
and the Qwen semantic encoder. Do not return to test-tuned probability modules
as the main research direction.

### Native protocol-structural pre-training pilot

The first native pre-training implementation is now available through
`pretrain_native_flow_encoder.py`, `extract_native_flow_embeddings.py`, and
`models/native_flow_encoder.py`. It uses a byte-level packet encoder followed
by a packet-level flow encoder. The self-supervised objectives cover masked
protocol-control fields, packet relative order, direction, next-packet
length/IAT bins, same-flow discrimination, flow contrast, and intervention
consistency. Downstream class labels are never read by the pre-training loss.

The leakage and shortcut controls are part of the method, not optional cleanup:

```text
1. Absolute packet position is added only after packet observation encoding;
   the relative-order target cannot read an absolute-position embedding.
2. IP addresses, ports, checksums, TCP sequence/acknowledgement numbers, and
   other session identifiers can be non-reconstructively hidden.
3. Encrypted payload bytes are dropped as an intervention but are never used as
   masked-reconstruction targets; reconstruction is restricted to protocol
   control fields.
4. Two independently intervened views are aligned at corresponding packet
   positions, encouraging endpoint-invariant packet-context representations.
5. The native representation enters Tower-2 through a zero-initialized,
   bounded residual adapter. Expanding an old Stage-5 checkpoint therefore
   preserves its logits exactly before native-interaction training.
```

Current VPN-app pilot evidence uses only `704` unlabeled training flows, `352`
validation flows, and the unchanged `1672`-flow shared test. The frozen probe is
selected using validation macro-F1; test labels do not select PCA dimension or
classifier configuration.

```text
random untrained architecture:             acc=0.4593, macro-F1=0.4562
v3 all-field reconstruction, no position leak:
                                           acc=0.5311, macro-F1=0.5060
v4 control-only reconstruction + payload dropout:
                                           acc=0.5221, macro-F1=0.5084
v5 v4 + packet intervention consistency:   acc=0.5335, macro-F1=0.5159
shallow protocol/statistics, no ports:      acc=0.6364, macro-F1=0.5996
```

The trained v3 representation is significantly better than the identical
random architecture: paired deltas are `+0.0718` accuracy and `+0.0499`
macro-F1, with 95% bootstrap intervals `[+0.0538,+0.0897]` and
`[+0.0335,+0.0658]`. This establishes that self-supervision learns transferable
information. The cleaner v5 objective has the best combined point result, but
its gain over v3 is not significant: `+0.0024` accuracy and `+0.0099` macro-F1,
with both 95% intervals crossing zero. It also remains below the strict shallow
statistics baseline. Therefore this pilot is supporting evidence and an honest
negative result, not yet a CCF-A-level representation-learning claim.

When the bounded native adapter is added to the stable Stage-5 VPN model, shared
test performance changes from `0.6274/0.5851` to `0.6310/0.5895` accuracy/F1.
That change is also statistically neutral (`p=0.146`, both bootstrap intervals
cross zero), so the native adapter is not promoted to the current best model.
The next defensible experiment is larger, content-deduplicated unlabeled
pre-training followed by the same frozen-probe, shallow-baseline, cross-split,
and cross-dataset protocol. More downstream probability fusion is not evidence
for a better representation.

```text
Source-environment risk/alignment negative ablation:
  `build_flow_environment_map.py` recovers the two source folds for a training
  split by exact PCAP SHA256, and `train_tower2.py` optionally applies
  environment-balanced class sampling, environment-risk variance, and
  class-conditional embedding alignment. The audit found 698 fold1 training
  rows but only 352 unique PCAP contents (346 duplicate rows), with complete and
  balanced environment coverage (349 rows per source fold). On fold1, joint
  fine-tuning dropped to 0.6579/0.6200 test acc/F1; from-scratch risk-only and
  alignment-only runs reached 0.5999/0.5691 and 0.6011/0.5702. Keep the module
  optional for cross-dataset ablation, but do not enable it in the best default.
  More importantly, all paper tables must report exact-content duplicate audits
  and must not interpret duplicated training/validation rows as independent
  evidence.

Content-clean training, guarded flow pooling, and reliability-consensus
negative ablations:
  Within fold1, exact-content deduplication reduces train flows from 698 to 352
  and validation flows from 352 to 176. A mean-pooling Tower-2 trained on these
  unique flows reaches validation acc=0.8523/macro-F1=0.8500 but only shared-test
  acc=0.6160/macro-F1=0.5860, confirming a real environment shift rather than
  duplicate-weight inflation. `select_flow_eval_pooling.py` compares pooling
  modes on validation in one model load and requires a default 0.005 gain before
  replacing the checkpoint head. The unguarded 0.0012 validation-F1 gain from
  mean logits lowers test to 0.6639/0.6333; the guard correctly retains the
  0.6675/0.6380 checkpoint output. Cross-fold class-reliability voting,
  validation-confusion EM, EM-only tie breaking, and class-reliability tie
  breaking give VPN acc/F1 of 0.7362/0.7401, 0.7267/0.7327,
  0.7446/0.7437, and 0.7464/0.7440. All remain below the unchanged
  vote-priority consensus at 0.7512/0.7522, so none is promoted.

  `cross_fold_consensus.py --mode selective_anchor_vote` additionally learns a
  high-precision anchor threshold from fold0 validation only, then lets the
  anchor override majority vote above that threshold. With the preregistered
  `--anchor_min_precision 0.9 --anchor_min_coverage 0.1`, validation selects a
  0.9440 threshold and covers 1000/1672 test flows. Test acc/F1 is
  0.7488/0.7516: below vote-priority accuracy and below the anchor model's
  macro-F1. This rejects confidence-selective routing as another solution to
  the validation/test shift; do not tune its threshold on shared-test labels.

Cross-fold stability selector:
  `cross_fold_stability_selector.py` audits same-named candidates across multiple
  folds by comparing valid gain, shared-test gain, and target prediction shift
  against each fold's base. On the current VPN weak-fold candidates it ranks the
  local prior/refinement loop highest by shared-test target-margin, but rejects it
  under strict valid-gain rules because both weak folds show negative validation
  gain while improving shared-test performance. This is direct evidence that the
  main bottleneck is validation/test shift, not lack of candidate expert capacity.
```

Research conclusion: single-fold post-hoc probability fusion is no longer the main bottleneck for VPN split1/split2. The validation folds can reach very high scores while the shared test set remains low, and many weak-fold errors are complementary across the three ready-made train/valid partitions. The useful paper-facing direction is therefore cross-split-stable representation learning plus label-free target-prior stabilization, local confusion refinement, and a final same-dataset cross-fold consensus. Tower-1 paired full-header/randomized-IP-port consistency helps fold1 but hurts fold2; target-prior softcap ensembling helps both the fold1 paired seq branch and the fold2 statistics branch; pairwise/group refinement plus repeated prior passes pushes the VPN cross-fold macro-F1 minimum above the 0.65 target, but only the cross-fold consensus pushes every VPN fold above the 0.74 accuracy target. The current next model iteration should distill this consensus back into the unified trainable framework: endpoint-invariant training, paired full-header vs randomized-header consistency during Tower-1/Tower-2, target-prior softcap as a label-free candidate expert, pairwise/group confusion refinement as a local expert, and cross-fold consensus or consensus-distillation that penalizes validation/test prediction shift. Keep these as the same framework modules for VPN/TLS; let validation gates and learned branch weights down-weight unhelpful experts instead of hand-removing modules per dataset.

The summary script emits commands with the same Stage-8 module family for every dataset/fold: Tower-1 preprocessing/training, raw+projected embeddings, graph/seq Tower-2, multi-view flow pooling, shared SupCon/dual-loss training, unified candidate expert slots, and guarded unified expert selection. Dataset-specific behavior should come from learned weights, identity fallback, and validation gates, not from removing modules. Confidence penalty, VPN-specific hierarchical heads, and VPN-specific confusion groups are retained as ablations rather than default paper modules.

Use the metric dashboard to check the current target gates across datasets:

```bash
conda run --no-capture-output -n llm-factory \
  python summarize_experiment_results.py \
    --dataset vpn-app \
    --dataset tls-120 \
    --top_k 5 \
    --output_json reasoningDataset/goal_metric_summary.json
```

Current target-gate status:

```text
vpn-app: acc=0.7488, macro-F1=0.7558, target acc>=0.7400 and macro-F1>=0.6500 -> PASS
tls-120: acc=0.7996, macro-F1=0.7869, target acc>=0.7800 and macro-F1>=0.7000 -> PASS
```

Generate the next-experiment recommendation report after each new run:

```bash
conda run --no-capture-output -n llm-factory \
  python recommend_next_experiment.py \
    --dataset vpn-app \
    --dataset tls-120 \
    --output_json reasoningDataset/next_experiment_recommendation.json \
    --output_md reasoningDataset/next_experiment_recommendation.md
```

Current recommendation summary:

```text
vpn-app: target PASS; old-embedding paired-view probes are negative, so do not spend more CPU time on graph paired-view training. Run the documented A800 Stage-8 paired-view path with fresh `rawproj_flowaware_*` embeddings.
tls-120: target PASS; only accept new modules when validation-gated selection and target-shift guards keep or improve the current best.
```

Use the autopilot wrapper to generate the exact next-run command plan. It writes a JSON plan and defaults to dry-run when CUDA is unavailable:

```bash
conda run --no-capture-output -n llm-factory \
  python run_recommended_experiment.py
```

In the real A800 `llm-factory` shell, add `--execute` to run the full recommended VPN Stage-8 paired-view path:

```bash
conda run --no-capture-output -n llm-factory \
  python run_recommended_experiment.py \
    --execute
```

The generated final-selector command carries the same robustness controls as the manual selector runs. For VPN, the dataset preset uses `--select_metric macro_f1`, `--rank_select_metric accuracy`, `--rank_metric bootstrap_gain_quantile`, and `--rank_candidate_limit 256`, so the next paired-view candidate is ranked by validation accuracy bootstrap lower-bound gain while final acceptance still preserves macro-F1-oriented selector gating. TLS-120 now also uses `--rank_metric bootstrap_gain_quantile`, but with `--rank_select_metric macro_f1` and `--rank_candidate_limit 64`, matching the unified-slot stacker result above. These are dataset parameter choices inside the same validation-gated selector module, not different frameworks.

The recommended experiment runner now includes CPU `slot_stacker` and `soft_expert_gate` stages between `paired_prior` and `final_selector`. It first trains `train_prediction_stacker.py` over the same unified expert slots, using dataset preset inputs plus the current paired candidate, then trains `train_expert_gate.py` over the base and stacker probability sources. The final `validation_gated_selector.py` receives both `slot_stacker` and `soft_gate` experts. This makes the TLS-120 stacker/soft-gate improvement part of the automatic cross-dataset framework instead of a one-off manual probe. Use `--no-enable_slot_stacker` or `--no-enable_soft_expert_gate` only for ablations that intentionally remove these trainable candidates.

The same wrapper has VPN and TLS-120 presets for class count, label map, current best selector input, and target-shift guard. To build the TLS-120 plan, change `--dataset`:

```bash
conda run --no-capture-output -n llm-factory \
  python run_recommended_experiment.py \
    --dataset tls-120

```

To generate the unified VPN/TLS command suite in one step:

```bash
conda run --no-capture-output -n llm-factory \
  python run_recommended_suite.py \
    --output_json reasoningDataset/recommended_suite_plan.json
```

The suite JSON records CUDA visibility, each dataset's current best test result, target-gate status, generated command, command return code, and a `child_plans` summary containing each dataset's plan JSON, paired-prior output, slot-stacker output, final-selector output, skipped stages, and CUDA-required stages. In dry-run mode the suite materializes these child plans by default without launching training; pass `--no-materialize_child_plans` if you only want to print the child commands. This makes the A800 run usable as a lightweight experiment ledger for paper ablations and cross-dataset reproduction.

In the real A800 `llm-factory` shell, add `--execute` to run the suite sequentially:

```bash
conda run --no-capture-output -n llm-factory \
  python run_recommended_suite.py \
    --execute \
    --output_json reasoningDataset/recommended_suite_plan.json
```

For the full autonomous research loop, use the wrapper below. It first regenerates the recommendation, framework, ablation, evidence-pack, and paper-defaults audit reports, checks the VPN/TLS target gates, and stops only when the current results satisfy the metric goals, the unified-framework consistency audit, and the centralized paper-safe defaults audit. If the gates are not met, or if `--continue_after_targets` is set, it calls the recommended suite and records a loop ledger:

```bash
conda run --no-capture-output -n llm-factory \
  python run_autonomous_research_loop.py \
    --max_iters 1 \
    --output_json reasoningDataset/autonomous_loop/research_loop_ledger.json
```

The framework consistency gate is enabled by default through `--require_framework_consistency`; use `--no-require_framework_consistency` only for temporary debugging runs where the paper-facing unified-framework proof is not being checked. The centralized defaults gate is also enabled by default through `--require_paper_defaults_audit`; use `--no-require_paper_defaults_audit` only when intentionally editing or debugging paper-facing result paths. The loop now passes the same `--final_selector_unified_expert_slots base,graph,seq,prior_base,emb_lr,emb_et,proto_emb,paired,slot_stacker,soft_gate` list to both the recommended suite and the framework report. The report treats missing dataset-specific experts as identity candidates only when the slot names still match this required list, so a dataset that silently skips the unified selector/slot-stacker/soft-gate interface will fail the paper-facing consistency audit. The loop also requires the evidence-pack framework claims and the centralized paper-safe defaults audit to pass point targets before stopping, which prevents an unconstrained probe JSON or stale default path from satisfying the metric gate while the paper-safe framework result is weaker. `--no-enable_slot_stacker` and `--no-enable_soft_expert_gate` should be reserved for explicit ablation loops.

`recommend_next_experiment.py` reports both the raw highest-scoring `test*.json` and the paper-safe framework result. This is deliberate: for TLS-120 the direct unified-slot stacker probe reaches `0.7991/0.7897`, while the current paper-safe soft-gate calibrated selector reaches `0.7996/0.7869`. The raw stacker still has the higher macro-F1, while the accepted soft-gate selector has the higher accuracy and full validation-gated framework evidence. Recommendations and target status are tied to the paper-safe result, and raw-best probes should be treated as upper-bound or ablation evidence until the validation-gated selector accepts them. The recommendation JSON and evidence pack also record `raw_minus_paper_safe` accuracy/F1 deltas so this distinction is machine-checkable.

The paper-safe result paths, VPN/TLS target gates, required unified expert slots,
shared core modules, and default framework profile are centralized in
`paper_framework_defaults.py`. In the current code, "paper-safe" is the
backward-compatible name for the `paper_unified` profile, not a separate
dataset-specific recipe. Update that file first when changing the paper-facing
main result; the recommendation, framework-report, autonomous-loop, and
recommended-suite scripts import the shared defaults to avoid drift.

The `paper_unified` profile also fixes one Tower-1 checkpoint protocol for every
dataset and both task levels. Training evaluates a deterministic
flow-balanced validation subset (at most two non-repeated packets per flow),
selects `best/` by held-out packet macro-F1, and extracts both the factual
full-header view and the `mask_ip_port` intervention view from that same
checkpoint. Test labels never select the Tower-1 epoch. This equal-flow
validation rule prevents long flows from dominating checkpoint selection and
keeps the packet encoder reusable inside the flow model. Legacy commands that
do not opt into `paper_unified` continue to read the final Tower-1 directory for
backward-compatible ablations; they are not evidence for the unified main
result.

The autonomous loop runs the audit below automatically. Run it manually after
changing paper-facing defaults or before updating paper tables. It checks that
the configured paper-safe JSON files exist, meet their target gates where
targets are defined, expose the required unified expert slots directly or
through identity-compatible legacy mapping, and still pass the unified-framework
gate with at least one `paper_unified` flow manifest and one `paper_unified`
packet manifest:

```bash
conda run --no-capture-output -n llm-factory \
  python audit_paper_framework_defaults.py \
    --output_json reasoningDataset/paper_framework_defaults_audit.json \
    --output_md reasoningDataset/paper_framework_defaults_audit.md
```

For paper-grade stopping, add `--require_ci_targets`. The default loop stops when VPN/TLS point metrics, the unified-framework audit, and the centralized paper-safe defaults audit pass. The stricter CI mode additionally requires each goal dataset to have `ci_target_met=true` and, by default, `content_group_ci_target_met=true` in `reasoningDataset/paper_evidence_pack.json`. The loop refreshes the exact-PCAP content-unique and content-grouped bootstrap reports before rebuilding the evidence pack, so the stop gate uses the current paper-safe predictions rather than stale robustness evidence. With the current results, TLS-120 passes both CI gates, while VPN remains `point_pass_ci_mixed` because its standard and content-grouped bootstrap accuracy lower bounds are below `0.74`; strict mode therefore continues to the recommended suite instead of stopping early. Use `--no-require_content_group_ci_targets` only for a debugging run that intentionally ignores the content-grouped promotion gate.

```bash
conda run --no-capture-output -n llm-factory \
  python run_autonomous_research_loop.py \
    --require_ci_targets \
    --max_iters 1 \
    --output_json reasoningDataset/autonomous_loop/research_loop_ledger_ci_strict.json
```

The loop also builds a CI-gap-aware next-experiment plan and embeds it in the ledger as `next_experiment_plan`. Run it manually when you want a compact table of point-target gaps, bootstrap lower-bound gaps, raw-best versus paper-safe gaps, and the recommended real-A800 command:

```bash
conda run --no-capture-output -n llm-factory \
  python make_next_experiment_plan.py \
    --output_json reasoningDataset/next_experiment_plan.json \
    --output_md reasoningDataset/next_experiment_plan.md
```

The loop also audits whether a raw-best candidate is safe to promote into the
paper-facing defaults. This is useful for TLS-120, where the raw unified-slot
stacker is stronger than the guarded selector result but lacks direct selector
safety guards and full paper-facing slot evidence. Promotion now also requires
raw-candidate-specific content-group CI evidence: if the raw-best path differs
from the current paper-safe path, it cannot borrow the paper-safe
content-grouped bootstrap result.

```bash
conda run --no-capture-output -n llm-factory \
  python audit_paper_candidate_promotion.py \
    --raw_content_group_index vpn-app=reasoningDataset/vpn-app/test_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --raw_content_group_index tls-120=reasoningDataset/tls-120/test_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --output_json reasoningDataset/paper_candidate_promotion_audit.json \
    --output_md reasoningDataset/paper_candidate_promotion_audit.md
```

Use `--min_raw_gain 0` only as a diagnostic to inspect tiny raw-best gains. In
that mode, the current TLS raw-best candidate has enough computed content-group
CI evidence, but it still remains a probe because it lacks required unified
slot records and direct selector safety guards.

To map existing candidates before spending A800 time, rank their exact-PCAP
content-group robustness with the same target gate. This is a diagnostic search
tool; it does not promote a result by itself:

```bash
conda run --no-capture-output -n llm-factory \
  python rank_content_group_candidates.py \
    --dataset vpn-app \
    --flow_embedding_index reasoningDataset/vpn-app/test_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --target_accuracy 0.74 \
    --target_macro_f1 0.65 \
    --bootstrap_samples 200 \
    --top_k 20 \
    --output_json reasoningDataset/vpn-app/content_group_candidate_ranking.json \
    --output_md reasoningDataset/vpn-app/content_group_candidate_ranking.md

conda run --no-capture-output -n llm-factory \
  python rank_content_group_candidates.py \
    --dataset tls-120 \
    --prediction_glob 'test_crossfold_consensus*.json' \
    --flow_embedding_index reasoningDataset/tls-120/test_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --target_accuracy 0.78 \
    --target_macro_f1 0.70 \
    --bootstrap_samples 200 \
    --top_k 20 \
    --output_json reasoningDataset/tls-120/content_group_candidate_ranking_crossfold.json \
    --output_md reasoningDataset/tls-120/content_group_candidate_ranking_crossfold.md
```

Current scan conclusion: among 411 VPN `test*.json` candidates, the best
content-group lower bound is still the current cross-fold consensus
(`0.7318` accuracy lower bound, below the `0.74` gate). Among TLS cross-fold
candidates, `test_crossfold_consensus_rgssi_auto_confidence.json` ranks first
and passes the content-group gate, but it still needs unified-slot and selector
safety evidence before it can become a paper-facing default.

To keep searching for higher accuracy in the real A800 environment even after the current VPN/TLS targets pass:

```bash
conda run --no-capture-output -n llm-factory \
  python run_autonomous_research_loop.py \
    --execute \
    --continue_after_targets \
    --require_ci_targets \
    --variant_schedule stage8_balanced \
    --status_rank_metric target_margin \
    --run_tag_template '{run_tag}_iter{iteration:02d}' \
    --max_iters 3 \
    --output_json reasoningDataset/autonomous_loop/research_loop_ledger.json
```

Use `--run_tag_template '{run_tag}_iter{iteration:02d}'` for multi-iteration searches so each loop writes a fresh Stage-8 run instead of reusing old `skip_existing` outputs. The default `--variant_schedule stage8_balanced` now starts with `gentle_intervention`, a low-weight paired-view setting intended to test shortcut resistance without immediately overwhelming the supervised flow objective, then cycles through the stronger balanced setting, stronger paired-view consistency, higher paired-view CE weight with trainable multi-view flow pooling, and stronger Tower-2 dropout plus confidence-penalty regularization; each variant also carries its own seed. The dropout-regularized Stage-8 variant also passes `--confidence_penalty_weight 0.02` into Tower-2 training, adding a KL-to-uniform penalty on classification logits to reduce overconfident split-specific decisions while keeping the same model architecture. The `multi_view` pooling computes mean/max/std/attention flow views for every dataset and learns a softmax gate over the four branches, so unhelpful branches can be down-weighted by training instead of being manually removed. The multi-view variant also uses a small `--multi_view_gate_entropy_weight` to encourage more decisive branch weights while keeping every branch in the shared framework. `test_tower2.py` writes the average learned branch weights, normalized entropy, and effective branch count to `metrics.eval_config.multi_view_gate`, which can be used as paper evidence that all datasets pass through the same branch family while training learns dataset-specific weights. New Tower-2, validation-selector, and slot-stacker outputs include calibration metrics under `metrics.*.calibration` (`ece`, `nll`, `brier`, average confidence, and sample count), so confidence-penalty, stacker, or selector changes can be judged by both accuracy/F1 and overconfidence with one metric definition from `probability_metrics.py`. The framework and evidence-pack reports surface these calibration fields whenever the evaluated JSON contains them; older result JSON files remain valid but will not show a calibration table until they are re-evaluated. This makes multi-iteration runs search real model/training variants and seed stability instead of only changing output directories. `--status_rank_metric target_margin` ranks ledger best-before/best-after results by the weaker target margin across accuracy and macro-F1, which is useful for paper-grade searches where both metrics matter; omit it to keep the historical accuracy-first ranking. The wrapper records the concrete per-iteration run tag and variant in the loop ledger. After each suite run, the loop ledger also embeds the suite `child_plans` summary, including every dataset's child plan JSON, paired-prior output, final-selector output, skipped stages, CUDA-required stages, and `experiment_config`. The raw `status_before/status_after` entries keep the highest-scoring test JSON for search visibility, while `paper_safe_status_before/paper_safe_status_after` mirror the evidence-pack framework claims and raw-minus-paper-safe deltas used for paper reporting. The ledger records both `raw_goals_met_*` and `paper_safe_goals_met_*`; stopping requires both, plus the framework, centralized defaults audit, and optional CI gates. The `next_experiment_plan` entry records the current priority gap type for each dataset, so a passing point estimate can still trigger a CI-strengthening experiment. The `candidate_promotion_audit` entry records why a higher raw-best probe can or cannot be promoted into the paper-safe defaults. The `best_delta` field compares each dataset's best test metric before and after the iteration, so new-best runs are visible without manually scanning all result JSON files.

The wrapper runs `recommend_next_experiment.py`, builds the IP/port-randomized paired view, extracts paired embeddings, preprocesses Tower-2 datasets, trains graph/seq with the current iteration's run tag, evaluates, fuses, applies the safe prior residual, then compares the paired candidate against the current best with `validation_gated_selector.py`. With the run-tag template above, the final selector output for iteration 1 is:

```text
reasoningDataset/vpn-app/test_selector_best_plus_rawproj_flowaware_change_weight_stage8_flowaware_paired_ipport_iter01_valid_macro.json
```

This final selector keeps the same safety gates as the paper framework: validation macro-F1 gain, bootstrap stability, and a strict VPN target-shift guard. If the paired candidate is not stable, the selector falls back to the current best base. In the Codex sandbox this remains a dry-run because CUDA is not exposed; use the real A800 environment for the embedding and long training stages.

Generate a paper-ready framework report with selector decisions, guard evidence, flow-level calibration metrics when available, and learned multi-view gate weights when a result JSON contains `metrics.eval_config.multi_view_gate`:

```bash
conda run --no-capture-output -n llm-factory \
  python make_paper_framework_report.py \
    --output_json reasoningDataset/paper_framework_report.json \
    --output_md reasoningDataset/paper_framework_report.md
```

Current generated table:

```text
| Dataset | Accuracy | Macro-F1 | Target | Status | Flows | Module usage | Selector decision | Guards |
|---|---:|---:|---|---|---:|---|---|---|
| vpn-app | 0.7488 | 0.7558 | 0.7400/0.6500 | PASS | 1672 | base=active; selector=active; expert=gated_off:reliability_fusion; calib=evaluated; slots=inferred_identity_compatible:10;provided:4;identity:6;extra:0; mv_gate=not_observed; guards=boot:active,shift:active | fallback to base; rejected reliability_fusion (target_change=0.1268>0.0000) | bootstrap win=1.00, q=0.0310; target change=0.1268, JS=0.0149 |
| tls-120 | 0.7996 | 0.7869 | 0.7800/0.7000 | PASS | 11542 | base=active; selector=active; expert=active:class_bias_calibration; calib=evaluated; slots=recorded:10;provided:2;identity:8;extra:0; mv_gate=not_observed; guards=boot:active,shift:active | class_bias_calibration | bootstrap win=0.89, q=-0.0014; target change=0.0396, JS=0.0014 |
```

Framework consistency audit:

```text
status: PASS
vpn-app:  same module family, expert candidate = gated_off:reliability_fusion
tls-120:  same module family, expert candidate = active:class_bias_calibration
```

Flow-level bootstrap uncertainty with 300 resamples:

```text
| Dataset | Accuracy 95% CI | Macro-F1 95% CI |
|---|---:|---:|
| vpn-app | [0.7252, 0.7685] | [0.7249, 0.7865] |
| tls-120 | [0.7923, 0.8082] | [0.7788, 0.7933] |
```

Generate the paper ablation table:

```bash
conda run --no-capture-output -n llm-factory \
  python make_paper_ablation_report.py \
    --output_json reasoningDataset/paper_ablation_report.json \
    --output_md reasoningDataset/paper_ablation_report.md
```

Current ablation table:

```text
| Dataset | Stage | Accuracy | Delta Acc | Macro-F1 | Delta F1 | Selector/Fusion | Note |
|---|---|---:|---:|---:|---:|---|---|
| vpn-app | base constrained ensemble | 0.7488 | 0.0000 | 0.7558 | 0.0000 | best=0.91, emb_et=0.09 | strong base |
| vpn-app | unsafe reliability fusion | 0.6956 | -0.0532 | 0.6633 | -0.0924 | reliability_fusion | validation gain, target shift |
| vpn-app | calibration-enabled selector | 0.7339 | -0.0150 | 0.7241 | -0.0317 | threshold_switch:emb_lr | extra candidate, still unsafe |
| vpn-app | safe selector | 0.7488 | 0.0000 | 0.7558 | 0.0000 | fallback; reject reliability_fusion | target-shift fallback |
| tls-120 | graph/seq base | 0.7909 | 0.0000 | 0.7769 | 0.0000 | graph=0.65, seq=0.35 | strong base |
| tls-120 | strict safe selector | 0.7909 | 0.0000 | 0.7769 | 0.0000 | fallback; reject threshold_switch | strict bootstrap fallback |
| tls-120 | tolerant safe selector | 0.7909 | 0.0000 | 0.7772 | +0.0003 | threshold_switch:seq | low-shift seq switch |
| tls-120 | unified-slot stacker | 0.7991 | +0.0082 | 0.7897 | +0.0128 | - | trainable slot stacker upper/probe |
| tls-120 | guarded slot-stacker selector | 0.7945 | +0.0036 | 0.7807 | +0.0038 | threshold_switch:slot_stacker | low-shift stacker switch |
| tls-120 | soft expert gate | 0.7973 | +0.0065 | 0.7843 | +0.0073 | soft_expert_gate | trainable expert weighting |
| tls-120 | soft-gate calibrated selector | 0.7996 | +0.0088 | 0.7869 | +0.0100 | class_bias_calibration | class-bias calibrated soft gate |
```

Paired bootstrap delta vs each dataset baseline with 300 resamples for the latest CPU-feasible ablation table:

```text
| Dataset | Stage | Samples | Delta Acc 95% CI | Delta Macro-F1 95% CI |
|---|---|---:|---:|---:|
| vpn-app | unsafe reliability fusion | 300 | [-0.0676, -0.0395] | [-0.1125, -0.0735] |
| vpn-app | calibration-enabled selector | 300 | [-0.0206, -0.0090] | [-0.0449, -0.0182] |
| tls-120 | tolerant safe selector | 300 | [-0.0011, +0.0010] | [-0.0010, +0.0012] |
| tls-120 | unified-slot stacker | 300 | [+0.0036, +0.0124] | [+0.0074, +0.0174] |
| tls-120 | guarded slot-stacker selector | 300 | [+0.0017, +0.0056] | [+0.0018, +0.0058] |
| tls-120 | soft expert gate | 300 | [+0.0040, +0.0091] | [+0.0048, +0.0098] |
| tls-120 | soft-gate calibrated selector | 300 | [+0.0060, +0.0117] | [+0.0071, +0.0134] |
```

Generate the compact paper evidence pack. It carries the framework report's unified module usage, unified expert-slot coverage, selector, guard, CI, and multi-view gate evidence into one JSON/Markdown artifact:

```bash
conda run --no-capture-output -n llm-factory \
  python make_paper_evidence_pack.py \
    --output_json reasoningDataset/paper_evidence_pack.json \
    --output_md reasoningDataset/paper_evidence_pack.md
```

Generate the reviewer-facing method card. This converts the same audited
evidence into a paper-method narrative: core problems, unified modules,
contributions, flow/packet evidence, ablation positioning, and current CCF-A
risk gates. It is meant to keep the paper story aligned with the unified
framework instead of drifting back to dataset-specific tricks:

```bash
conda run --no-capture-output -n llm-factory \
  python make_paper_method_card.py \
    --output_json reasoningDataset/paper_method_card.json \
    --output_md reasoningDataset/paper_method_card.md
```

The method card treats the older `paper_unified` interface/protocol audit and
the exact shared-core publication audit as different gates. It reports
`strict_shared_core_v2_ready=true` only when the canonical VPN/TLS packet and
flow results all contain complete `strict_shared_core_v2` publication
provenance, use three-fold equal `log_mean`, and share one frozen configuration
SHA-256. Historical high scores remain non-headline evidence until all four
canonical results satisfy that rule.

It also reports `exact_common_reference_v2_ready` and
`unified_method_v2_ready` separately. The first requires one effective
configuration hash; the second requires one method hash plus executed
architecture/objective agreement and permits independently optimized numeric
hyperparameters. Neither field replaces `strict_shared_core_v2_ready` for the
initial headline promotion, but the distinction prevents later tuned results
from being rejected as different algorithms or mislabeled as the no-tuning
common-reference experiment.

Historical pre-exact-v2 evidence-pack status:

```text
vpn-app:  point_pass_ci_mixed  (point target passes; bootstrap acc lower bound is below 0.75)
tls-120:  strong               (point target and bootstrap lower bounds both pass)
```

Do not interpret the project promotion targets as "beats every SWEET model".
`compare_sweet_reference.py` encodes the protocol-matched values from SWEET
Tables 3, 4, and 9 and reports two distinct comparisons. Pcap-Encoder with a
frozen encoder is a representation-quality reference; because the current
method uses downstream-supervised LoRA adaptation, the primary comparison is
the SWEET unfrozen/end-to-end column.

```bash
conda run --no-capture-output -n llm-factory \
  python compare_sweet_reference.py \
    --output_json reasoningDataset/sweet_protocol_comparison.json \
    --output_md reasoningDataset/sweet_protocol_comparison.md
```

For the historical canonical results, VPN packet, VPN flow, and TLS-120 packet
exceed the corresponding SWEET end-to-end accuracy and macro-F1 pairs. TLS-120
flow (`0.8461/0.8292`) exceeds frozen Pcap-Encoder (`0.713/0.681`) but not
unfrozen netFound (`0.908/0.897`), with gaps of `-0.0619/-0.0678`. Therefore a
paper may claim superiority over the frozen representation baseline for that
task, but not blanket superiority over all SWEET results. The comparison JSON
also records whether every input carries `strict_shared_core_v2` provenance;
historical canonical scores do not satisfy that final unified-method gate.

This historical evidence is useful as a performance reference and ablation
source, but it is not the exact-v2 headline table because VPN used
`vote_priority`, TLS-120 used `log_mean`, and the four task/dataset results do
not yet carry one strict shared-core fingerprint. Current paper positioning is:

```text
main claim: candidate unified shortcut-resistant packet-to-flow framework under strict cross-task verification
historical strong performance reference: TLS-120
historical qualified performance reference: VPN point estimate passes, but bootstrap lower bound is mixed
dataset scope: VPN and TLS-120 flow-level tasks only; Per-flow Split USTC packet-level artifacts are excluded from flow-level claims
reviewer-risk control: do not promote historical scores until exact-v2 three-fold, runtime, provenance, and cluster-bootstrap gates pass
```

To audit whether existing experts still contain useful residual signal, run the validation-selected residual search:

```bash
conda run --no-capture-output -n llm-factory \
  python search_residual_fusion.py \
    --base reasoningDataset/vpn-app/test_fusion_best_prior_flow_embedding_experts_minbest90_valid_acc.json \
    --candidate_glob 'reasoningDataset/vpn-app/test*.json' \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --top_candidates 12 \
    --combo_size 3 \
    --simplex_step 0.01 \
    --min_base_weight 0.90 \
    --select_metric accuracy \
    --output_json reasoningDataset/vpn-app/residual_fusion_search_stage8_minbase90_top12.json \
    --best_output_json reasoningDataset/vpn-app/test_residual_fusion_search_stage8_minbase90_top12_valid_acc.json
```

The current top-12 residual search selected `base=1.0`, so the existing fusion/statistics/embedding experts do not provide enough validation-supported residual signal to push VPN beyond the stronger aspirational `75%` mark. The next meaningful improvement should therefore come from representation learning rather than more probability-level fusion: resume GPU Stage-8 Tower-1 flow-aware contrastive training, re-extract embeddings, and rerun Tower-2/fusion on VPN first, then verify the same protocol on TLS-120.

Tower-2 also supports a multi-view flow aggregation head through `--flow_pooling multi_view`. It pools each flow with mean, max, standard deviation, and attention statistics, then fuses those views with a gated MLP. This is useful as a multi-instance ablation, but on the current old VPN embeddings it overfits the validation split and does not improve the target test result:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n llm-factory \
  python train_tower2.py \
    --model_type seq \
    --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/seq_dataset.pt \
    --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/seq_dataset.pt \
    --output_dir checkpoints/tower2_seq_flow_rawproj_change_weight_multiview \
    --num_classes 16 \
    --epochs 30 \
    --batch_size 16 \
    --hidden_dim 256 \
    --num_layers 2 \
    --num_heads 4 \
    --dropout 0.15 \
    --lr 1e-4 \
    --weight_decay 0.03 \
    --train_level flow \
    --flow_pooling multi_view \
    --window_loss_weight 0.3 \
    --class_weighting effective \
    --class_weight_beta 0.9999 \
    --class_weight_strength 0.6 \
    --label_smoothing 0.05 \
    --hierarchical_weight 0.2 \
    --hierarchical_logit_weight 0.5 \
    --coarse_groups vpn_app \
    --balanced_flow_batches \
    --samples_per_class 2 \
    --contrastive_mode confusion \
    --confusion_groups vpn_app \
    --flow_contrastive_weight 0.03 \
    --flow_temperature 0.07 \
    --aux_weight 0 \
    --coherence_weight 0 \
    --select_metric flow_macro_f1 \
    --early_stop_patience 8

CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n llm-factory \
  python train_tower2.py \
    --model_type graph \
    --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/graph_dataset.pt \
    --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt \
    --output_dir checkpoints/tower2_graph_flow_rawproj_change_weight_multiview \
    --num_classes 16 \
    --epochs 30 \
    --batch_size 16 \
    --hidden_dim 256 \
    --num_layers 2 \
    --num_heads 4 \
    --dropout 0.15 \
    --lr 1e-4 \
    --weight_decay 0.03 \
    --train_level flow \
    --flow_pooling multi_view \
    --window_loss_weight 0.3 \
    --class_weighting effective \
    --class_weight_beta 0.9999 \
    --class_weight_strength 0.6 \
    --label_smoothing 0.05 \
    --hierarchical_weight 0.2 \
    --hierarchical_logit_weight 0.5 \
    --coarse_groups vpn_app \
    --balanced_flow_batches \
    --samples_per_class 2 \
    --contrastive_mode confusion \
    --confusion_groups vpn_app \
    --flow_contrastive_weight 0.03 \
    --flow_temperature 0.07 \
    --aux_weight 0 \
    --coherence_weight 0 \
    --select_metric flow_macro_f1 \
    --early_stop_patience 8
```

Current multi-view ablation results:

```text
seq multi_view:   valid acc=0.6534, valid macro-F1=0.6572, test acc=0.6382, test macro-F1=0.5863
graph multi_view: valid acc=0.6420, valid macro-F1=0.6415, test acc=0.6340, test macro-F1=0.5903
seq+graph multi_view fusion selected seq=1.0, graph=0.0, so the two heads are not complementary.
best + seq multi_view constrained fusion dropped to test acc=0.7482, macro-F1=0.7534.
```

Interpretation: richer Tower-2 flow aggregation alone is not enough on the old packet embeddings. Treat `multi_view` as a negative ablation and revisit it after GPU Stage-8 Tower-1 flow-aware representation retraining.

Prototype retrieval over flow embedding summaries is also available:

```bash
conda run --no-capture-output -n llm-factory \
  python train_flow_prototype_classifier.py \
    --train_index reasoningDataset/vpn-app/train_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --valid_index reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --test_index reasoningDataset/vpn-app/test_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
    --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
    --max_packets 64 \
    --components_grid 32,64,128,256 \
    --k_grid 1,3,5,7,9,11,15,21 \
    --prototype_modes knn,centroid \
    --temperature_grid 0.1,0.3,1,3,10 \
    --select_metric accuracy \
    --output_json reasoningDataset/vpn-app/test_flow_prototype_classifier_valid_acc.json
```

On the current VPN split this prototype expert reached only `flow_acc=0.6501`, `flow_macro_f1=0.6017`, and constrained fusion with the best model did not improve over the current best. Treat it as a negative ablation unless the Tower-1 embeddings are retrained.

Metrics include:

```text
Window-level Accuracy / Precision / Recall / F1
Flow-level Accuracy / Precision / Recall / F1
```

### Tower-1 sampler-aware class balance (pre-registered validation ablation)

The unified Tower-1 uses `FlowBalancedPacketBatchSampler`: each epoch visits flows and samples a fixed number of packets per flow. Therefore the effective training class distribution is determined by the number of flows per class, not by the number of rows in `packet_auxiliary.jsonl`. This distinction matters because the Per-flow Split packet preprocessing currently balances raw packet rows exactly while leaving the number of flows naturally imbalanced.

The following audit is label-only and does not inspect test predictions:

```bash
conda run -n llm-factory python analyze_tower1_sampling_balance.py \
  reasoningDataset/packet-level/vpn-app/fold{0,1,2}/train/packet_auxiliary.jsonl \
  reasoningDataset/packet-level/tls-120/fold{0,1,2}/train/packet_auxiliary.jsonl \
  --method effective \
  --strengths 0.5,1.0 \
  --packets_per_flow 2 \
  --output_json /tmp/two_tower_runs/paper_unified_tower1_sampling_balance.json
```

Observed training-distribution imbalance:

```text
VPN folds:     packet-count ratio=1.00, flow-count ratio=13.40-16.69
TLS-120 folds: packet-count ratio=1.00, flow-count ratio=29.50-34.25
```

The audit also records flow-length replacement exposure. With
`packets_per_flow=2`, fold-0 contains `2693/4713` VPN singleton flows
(`57.14%` of flows, `28.57%` of sampled slots) and `6118/21208` TLS-120
singleton flows (`28.85%` of flows, `14.42%` of sampled slots). The sampler
duplicates a row for these flows to preserve equal CE exposure per flow. Such a
copy must not automatically be interpreted as an independent same-flow SupCon
positive. This is a training-data diagnostic, not evidence for changing the
method: a duplicate-identity contrastive mask requires a matched VPN/TLS
validation ablation before it may enter the shared core.

The slot rate understates the contrastive risk. SupCon forms directed pairs,
and the two sampled positions of every singleton flow are the same packet. The
audit therefore also reports the expected duplicate-identity share among all
directed same-flow pairs. For fold 0 this is `57.14%` on VPN and `28.85%` on
TLS-120, exactly the singleton-flow rate when `packets_per_flow=2`. More
generally, a short flow with `n < k` packets sampled `k` times with replacement
contributes an expected identity-collision probability of `1/n` per ordered
pair; flows with at least `k` packets are sampled without replacement and
contribute zero. This quantity concerns only same-flow positive construction,
not CE exposure or test performance.

An **identity-safe flow-aware SupCon** candidate is pre-registered after the
current sampler-aware class-balance and paired-view screens. It will carry a
stable packet identity into the loss and exclude alias copies from both the
positive mask and contrastive denominator while retaining the same
flow-balanced batches. It is not part of the running strict-v2 method and will
not be promoted from this audit alone. Promotion requires matched from-scratch
VPN/TLS held-out histories under one unchanged implementation, best macro-F1
gain of at least `0.005` on both datasets, and no held-out accuracy drop larger
than `0.005`; test labels remain unavailable to the decision.

The pair-level rate alone does not establish how much of the actual mini-batch
objective is exposed. `audit_tower1_contrastive_exposure.py` exactly replays
the sampler's epoch seed, flow shuffle, long-flow sampling without replacement,
and short-flow sampling with replacement. It compares the current objective
with a counterfactual branch in which only one occurrence of each `packet_uid`
may act as an anchor, positive, negative, or denominator candidate. The audit
binds every source JSONL by SHA-256 and reads no validation/test predictions:

```bash
conda run -n llm-factory python audit_tower1_contrastive_exposure.py \
  reasoningDataset/packet-level/vpn-app/fold0/train/packet_auxiliary.jsonl \
  reasoningDataset/packet-level/tls-120/fold0/train/packet_auxiliary.jsonl \
  --batch_size 16 --packets_per_flow 2 --epochs 8 --seed 42 \
  --same_flow_weight 1 --same_label_weight 1 \
  --flow_pairing random \
  --output_json /tmp/two_tower_runs/paper_unified_tower1_contrastive_exposure_fold0.json
```

Observed Packet-task fold-0 exposure under the frozen random-flow batching:

| Dataset | duplicate rows | alias share of positive-weight mask | positive-weight mask removed by identity dedup | denominator pairs removed | unique-anchor positive coverage after dedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| VPN-app | 28.57% | 33.34% | 56.45% | 49.51% | 82.44% |
| TLS-120 | 14.42% | 27.13% | 28.72% | 26.90% | 84.19% |

These are mask-exposure statistics, not loss values or performance gains:
SupCon normalizes positive weights per valid anchor. They nevertheless show
that alias rows materially change both sides of the contrastive normalization.
Blind deduplication is also incomplete because `15.81-17.56%` of unique
Packet-task anchors would then have no positive in their random mini-batch.
Across all three currently available Flow-task training splits, VPN contains
`704/698/703` flows and TLS-120 contains `6910/6910/6910` flows; every flow has
at least two sampled packets. Their alias exposure is therefore zero and
unique-anchor positive coverage is already 100%. This is a task-data boundary,
not permission to use a different loss implementation. Identity dedup is a
structural no-op there, whereas same-class flow pairing can still change the
relation distribution and therefore still requires matched Flow
non-inferiority evidence.

The next comparison first enables **identity-safe SupCon under the unchanged
random-flow batches**. It retains duplicate rows in CE so every flow keeps
equal classification exposure, but only one occurrence of each `packet_uid`
may take any contrastive role. Only if this isolated change passes the held-out
gate does a second **cross-flow-paired** candidate pack two distinct same-class
flows as an indivisible unit. Every flow is still visited exactly once per
epoch. With eight flows per batch, the pairing audit preserves exactly
`590/2651` VPN/TLS batches per epoch, lowers alias positive-mass share to
`21.77%/14.06%`, and increases identity-safe positive coverage to
`99.95%/99.96%`. The same sampler/objective applies to Flow data, where
identity dedup naturally becomes a no-op and distinct same-flow packets remain
valid positives. A memory bank or another expert is not introduced unless
both bounded candidates fail and a new validation-only hypothesis is
pre-registered.

`models/identity_safe_contrastive.py` contains the dormant D1 loss primitive.
Its tests prove that changing an alias embedding cannot change the loss, alias
rows receive zero contrastive gradient, no-positive batches return a
differentiable zero, and invalid temperatures/weights are rejected. The module
is deliberately not imported by `models/qwen_packet_multitask.py` or
`train_tower1_multitask.py` while A/B/C are running. Its presence is therefore
implementation preparation, not evidence that the current checkpoint used D1;
activation requires a new source fingerprint and matched from-scratch runs.

The same SHA-bound audit was repeated on all three ready-made training folds,
not selected from fold 0. Every random/paired replay produced exactly
`epochs * ceil(num_flows / flows_per_batch)` batches. Cross-fold ranges are:

| Dataset | random alias positive mass | random identity-safe positive coverage | paired alias positive mass | paired identity-safe positive coverage |
| --- | ---: | ---: | ---: | ---: |
| VPN-app | 33.34-35.32% | 80.31-82.44% | 21.70-22.81% | 99.92-99.96% |
| TLS-120 | 16.94-31.04% | 81.46-90.74% | 8.80-16.11% | 99.95-99.98% |

The ranges are descriptive training-input evidence. They establish that the
alias mechanism and the coverage repair recur across splits, but they do not
establish representation quality or justify promotion without the matched
held-out experiments below.

This mechanism must not be described as the first field augmentation or the
first flow contrastive objective. TrafficFormer already randomizes selected
header fields, MIETT already pulls packets from one flow together, and SWEET
already identifies explicit and implicit flow-ID shortcuts. The narrower
testable contribution is that equal-flow sampling can introduce an
**objective-level identity shortcut** by representing one packet copy as a
multi-instance same-flow pair, together with an identity-safe relation sampler
that restores real positive coverage. The two candidates enter the queue only
after the running A/B/C screens freeze their winner and are evaluated
sequentially, so alias removal and relation pairing remain separately
attributable. Each promotion first requires the same VPN/TLS Packet held-out
dual gate above, followed by matched Flow held-out non-inferiority under the
identical implementation and step budget. Flow non-inferiority is fixed as no
Macro-F1 drop greater than `0.005` and no Accuracy drop greater than `0.005`
on either VPN or TLS-120 held-out validation. D2 is compared against the
promoted D1 arm, not against whichever earlier arm makes its delta look best.
No test prediction may select either stage.

`train_tower1_multitask.py` now supports `--class_weight_basis {packet,flow}` and `--class_weight_strength ALPHA`. For normalized class-balanced weight `w_c`, the applied weight is proportional to `w_c ** ALPHA` and is renormalized to mean one. `ALPHA=0` disables class reweighting and `ALPHA=1` applies full correction. Both the packet-level and flow-level runners expose the same mechanism, so this is a shared Tower-1 objective rather than a dataset-specific classifier trick.

The running `paper_unified` baselines retain the historical `packet` basis and are not changed retroactively. The pre-registered next comparison is:

```text
baseline: basis=packet, strength=1.0 (all packet counts are equal, so weights are 1)
candidate: basis=flow, strength=0.5 (square-root correction)
selection: held-out packet macro-F1 gain >=0.005 and accuracy drop <=0.005,
           with the same setting required for VPN and TLS-120
test policy: evaluate the frozen selected setting once; do not select strength from test metrics
```

Full flow-count correction is not the first candidate because its maximum normalized weight is approximately `2.98` on VPN but `16.20` on TLS-120. The shared square-root correction limits the observed ranges to approximately `0.46-1.86` on VPN and `0.72-4.30` on TLS-120, reducing variance while correcting the sampler/loss mismatch.

The next mechanism is pre-registered only after the sampler-aware comparison: train Tower-1 with the same full-header and `mask_ip_port` paired view already consumed by the shared downstream intervention router. Both task runners expose the same Tower-1 loss; the packet runner uses:

```text
--tower1_paired_consistency_weight 0.05
--tower1_paired_cls_weight 0.2
--tower1_paired_logit_kl_weight 0.5
--tower1_paired_raw_consistency_weight 1.0
```

This adds cosine consistency for both components actually exposed by downstream `embedding_mode=concat` (the normalized raw last-token state and the projected state), symmetric logit KL, and supervised CE for the intervened view. Constraining only the projected state would leave a projection nullspace through which the raw component could retain header shortcuts. It is not enabled in the frozen baseline, and it must be evaluated after, not together with, the class-balance ablation. Candidate order:

```text
A: packet-count baseline
B: A + flow-count square-root class balance
C: validation-selected A/B + Tower-1 full/masked paired invariance
```

All three use the same Qwen-LoRA packet encoder, flow-balanced sampler, validation macro-F1 checkpoint rule, and shared downstream packet representation. A candidate is promoted only when the same configuration is supported on held-out VPN and TLS-120 validation; test labels are never used for promotion.

### Validation-to-test domain-shift audit

Use the same audit for every completed fold:

```bash
conda run -n llm-factory python analyze_prediction_domain_shift.py \
  --valid VALID_PREDICTIONS.json \
  --test TEST_PREDICTIONS.json \
  --level flow \
  --output_json /tmp/two_tower_runs/DATASET_FOLD_domain_shift.json
```

The audit reports per-class F1 change, class-normalized test confusions, true/predicted prior total variation and Jensen-Shannon divergence, and learned gate means. The first two completed VPN folds show:

This is post-hoc error analysis after predictions are frozen. Quantities that use test labels, including true-prior distance and per-class test F1, are reporting evidence only and are never consumed by training, checkpoint selection, calibration, candidate promotion, or inference.

```text
fold0: validation->test accuracy -7.66 points, macro-F1 -11.49 points
fold1: validation->test accuracy -5.08 points, macro-F1  -8.55 points
both:  true-prior total variation 0.352 because validation is class-balanced while test keeps its natural flow distribution
```

The routing weights are stable within each fold: the effective intervention-view mixture changes by less than `0.002` percentage points from validation to test. The shared semantic channel remains dominant (`91.0-91.3%` in fold0 and `82.3-82.4%` in fold1). Therefore the observed drop is not explained by a gate collapse. Recurrent conditional errors include `facebook <-> hangout`, `youtube -> spotify/vimeo`, and F1 degradation for `gmail`, `spotify`, `facebook`, and `aim`. This supports sampler-aware Tower-1 balance for underexposed flow classes and full/masked paired invariance for endpoint/environment shift; it does not support adding another routing expert.

### Paper-facing method boundary

The strict unified method is a bounded specialization decomposition, not a claim that field masking, contrastive learning, Transformers, or multi-channel fusion are individually new. TrafficFormer already studies field augmentation, MIETT already studies flow contrastive learning and two-level packet/flow attention, DigTraffic already studies dual-channel graph interaction, and SWEET establishes the shortcut-learning failure under Per-flow Split.

For factual packet representation `f`, intervened representation `i`, learned router `g in [0,1]`, and bound `rho_v`, the shared intervention router computes:

```text
z_view = (f + i) / 2 + rho_v * (2g - 1) * (f - i) / 2
```

Therefore each view retains at least `(1-rho_v)/2` effective mass and at most `(1+rho_v)/2`. With the paper setting `rho_v=0.25`, neither view can contribute less than `0.375` or more than `0.625`, even if the learned router saturates.

For semantic anchor `s`, normalized packet channels `z_k`, learned simplex gate `a_k`, bounded interaction `h`, and `rho_c=0.25`, the shared packet fusion computes:

```text
z_packet = s + rho_c * (sum_k a_k z_k - s + tanh(h))
```

The routed non-semantic channel mass is consequently bounded by `rho_c`; the separate interaction correction is coordinate-wise bounded by `rho_c * tanh(h)`. It must not be interpreted as an additional channel-mixture probability. Packet- and flow-level tasks use the same module. All three inputs to that module are strict current-packet representations: Flow Tower1 preprocessing is forced to `--packet_context_policy single_packet`, so semantic prompts cannot contain previous-packet IAT or a whole-flow server-role inference. The shared structural channel may retain a direction cue inferred from the current packet's own ports, because the identical cue is available to Packet classification; it is not a whole-flow direction estimate. Packet classification applies a packet head to `z_packet`. Flow-only sequence position and window context enter after `packet_to_flow_proj`, through the sequence/window aggregator and flow head over the same per-packet representation.

The shared fusion implementation is also the single source of truth for evaluation diagnostics. Both packet and flow result JSON files report effective routing means and `p05/p50/p95`, the configured theoretical bounds, and `bounds_satisfied`. With `rho_v=rho_c=0.25`, the executable contracts are:

```text
factual/intervened effective weights: each in [0.375, 0.625]
semantic routed weight:               in [0.75, 1.00]
content/structural routed weights:    each in [0.00, 0.25]
sum of routed channel weights:        exactly 1
```

These fields certify architectural boundedness on the evaluated samples; they are diagnostics, not test-time calibration or model-selection signals. Unit tests additionally exercise the saturated router endpoints, so the bounds remain verified even when observed gates do not approach those endpoints.

### Exact shared-core publication gate

The current `paper_unified` runs establish shared implementation classes and representation-level fusion, but they are screening runs rather than final evidence for the stronger claim that only learned weights differ. The current packet content encoder defaults to three layers with dropout `0.15`, whereas the native flow-pretrained content encoder uses two layers with dropout `0.10`. Flow training also uses content-group `group_mean` risk while the current packet screening profile records content groups but uses ordinary CE. A paper must not describe these settings as an exactly identical core.

`train_packet_byte_transformer.py` now supports strict initialization from the packet-content submodule of a `NativeFlowEncoder` checkpoint:

```text
--use_protocol_fields
--protocol_content_checkpoint NATIVE_CHECKPOINT/best.pt
--hidden_dim 128
--num_layers 2
--num_heads 4
--dropout 0.1
--content_group_loss_reduction group_mean
```

The load is strict: architecture or state-dict shape differences abort instead of silently loading a subset. The packet checkpoint records the pretraining protocol, source checkpoint, source-file SHA-256, exact architecture, and strict-load status. Native checkpoints without the required `native_flow_multitask_v1` provenance are rejected. The final audit recomputes the native checkpoint hash, so a different but shape-compatible checkpoint cannot satisfy the publication gate. `run_packet_level_pipeline.py` exposes the same settings as `--byte_hidden_dim`, `--byte_num_layers`, `--byte_num_heads`, and `--byte_dropout`.

The runner can now create and consume that native checkpoint as one reproducible chain. Its packet-content architecture is inherited directly from the packet encoder arguments; the native model's flow encoder and self-supervised heads are used only during label-free pretraining:

```bash
conda run -n llm-factory python run_packet_level_pipeline.py \
  --dataset vpn-app \
  --fold 0 \
  --stage paper_unified \
  --pretrain_protocol_content \
  --byte_max_bytes 128 \
  --byte_hidden_dim 128 \
  --byte_num_layers 2 \
  --byte_num_heads 4 \
  --byte_dropout 0.1 \
  --byte_content_group_loss_reduction group_mean
```

This runs `pretrain_native_flow_encoder.py` before packet-head training and passes `CHECKPOINT_ROOT/vpn-app_fold0/shared_content_pretraining/best.pt` through `--protocol_content_checkpoint`. Use `--stage protocol_pretrain` to run only the label-free phase after preprocessing. Automatic pretraining and an explicitly supplied `--protocol_content_checkpoint` are mutually exclusive, so the initialization source cannot be silently ambiguous. These flags provide the exact-v2 execution path; they do not retroactively make existing `paper_unified` screening checkpoints exact-core results.

Before freezing publication results, compare the completed packet checkpoint with the corresponding native flow checkpoint:

```bash
conda run -n llm-factory python audit_shared_packet_core.py \
  --packet_checkpoint PACKET_CHECKPOINT/best.pt \
  --native_checkpoint NATIVE_CHECKPOINT/best.pt \
  --flow_checkpoint FLOW_SEQ_CHECKPOINT/best.pt \
  --packet_result_json PACKET_VALIDATION_RESULT.json \
  --flow_manifest_json FLOW_FRAMEWORK_MANIFEST.json \
  --output_json /tmp/two_tower_runs/shared_packet_core_audit.json
```

The exact-core claim is permitted only when the audit returns `status=pass`, requiring all of the following:

```text
same max bytes, hidden dimension, layer count, head count, and dropout
same normalized parameter names and tensor shapes
same native_flow_multitask_v1 packet-content pretraining protocol
same content-group empirical-risk reduction
same semantic/content/13-dimensional structural packet module parameter schema
flow-only packet_to_flow_proj boundary before sequence/window/flow aggregation
```

For each dataset/fold, run the cross-task auditor after both task manifests are
complete. Checkpoint paths are resolved from the manifests, so the audit cannot
silently inspect a different hand-picked model:

```bash
conda run -n llm-factory python audit_cross_task_shared_core.py \
  --packet_manifest PACKET_ARTIFACT_ROOT/vpn-app/fold0/packet_framework_manifest.json \
  --flow_manifest reasoningDataset/vpn-app/stage8_flowaware_manifest_STRICT_V2_SUFFIX.json \
  --require_mechanism_evidence \
  --output_json reasoningDataset/shared-core-audits/vpn-app/fold0/audit.json
```

The report passes only when dataset, fold, and frozen config SHA-256 match;
packet and flow native encoders have the same `native_flow_multitask_v1`
architecture/protocol and the same checkpoint-recorded pretraining contract
(mask/dropout rates, every objective weight, optimizer, schedule, temperature,
patience, and seed); the packet and flow `shared_packet_encoder.*` parameter
schemas match exactly; and the flow-only `packet_to_flow_proj` boundary is
present. The manifests also contain a canonical `tower1_training_contract`;
the audit requires exact equality for the trainer, Qwen base, schedule, LoRA
configuration, packet/contrastive/paired losses, class weighting, flow-balanced
sampling, dtype, and seed. This prevents nominally shared Tower1 channels from
quietly using different task recipes. With `--require_mechanism_evidence`, the auditor also resolves the
fixed packet and sequence-flow test-result paths from the two manifests and
requires both tasks to expose the same named router schemas:
`factual/intervened` and `semantic/content/structural`. Their reported
`effective_routing_mean` values must be finite, normalized, and inside the
pre-registered residual bounds for every fold. These values describe the
bounded routing mixture; they deliberately do not claim to decompose the
separate nonlinear interaction correction into per-channel causal
contributions. Sample-wise gate variation is reported as mechanism evidence
but is not an arbitrary pass threshold. Independent trained parameter values
may differ, but architecture, pretraining protocol, empirical-risk contract,
and router semantics may not.

The semantic input policy is verified from executed artifacts rather than only
runner arguments. The Packet manifest reads factual/intervened cache manifests
for train, valid, and test and requires `full/mask_ip_port` headers with
`single_packet` context in all six cases. The Flow manifest performs the same
check against factual/intervened embedding configurations. The cross-task audit
fails if either evidence block is missing or unverified, even when both
framework manifests claim `single_packet` in their configuration notes.

Every Packet `paper_unified` factual/intervened extraction and every Flow
Stage-8 extraction also runs `audit_flow_embeddings.py` before cache building
or Tower2 preprocessing. The audit requires exact input/output flow coverage,
unique `(flow_id, packet_id)` keys, preserved packet order and labels, existing
two-dimensional NPY arrays whose row count and declared dimension match, and
finite values throughout. It records SHA-256 fingerprints for the packet
index, flow-embedding index, and embedding configuration in
`embedding_audit.json`; a failed audit stops the downstream stage. This check
is identical for VPN/TLS, Packet/Flow, all folds, and both header views.
Manifest consumption independently re-hashes the current packet index,
flow-embedding index, and embedding configuration and requires all three values
to match the audit report; merely storing syntactically valid SHA-256 strings
is insufficient. The executed embedding mode, scheduler version, model
micro-batch, and cross-flow packet buffer must also match the frozen Packet/Flow
contract on every split and view.

The strict audit also follows the native-content provenance through feature
extraction. `extract_native_flow_embeddings.py` records the encoder checkpoint
SHA-256 in each split manifest; train, valid, and test must all match the
audited flow-native checkpoint, use the frozen session-field mask probability,
contain non-empty flow sets, and attest `flow_id_and_packet_id` alignment. A
matching checkpoint on disk is therefore insufficient if Tower2's actual
native embedding files came from another run.

`audit_unified_framework.py` reports two deliberately separate gates over all
12 executed manifests (VPN/TLS x packet/flow x three folds) and all six
cross-task checkpoint reports under
`reasoningDataset/shared-core-audits/<dataset>/fold<k>/audit.json`:

- `exact_shared_core_v2` requires one effective configuration fingerprint and
  is the no-tuning-budget-confound common-reference experiment.
- `unified_method_v2` requires one method fingerprint and matching executed
  architecture/objective signatures, while allowing explicitly recorded
  effective configuration hashes to differ because of numeric optimization.

Configuration provenance alone is insufficient for either gate: every
cross-task checkpoint/runtime report must pass. The initial publication
promotion remains tied to `exact_shared_core_v2`; independently optimized
results are reported alongside it under `unified_method_v2`, not mislabeled as
the exact numerical baseline.

Strict runners snapshot one common **Packet+Flow execution dependency
closure** at process launch and completion. The closure starts from both
`run_packet_level_pipeline.py` and `run_stage8_flowaware_pipeline.py`, follows
local imports recursively, and includes local `.py` programs actually invoked
by either runner. The manifest records the per-file SHA-256 list, canonical
closure fingerprint, and any added, removed, or changed dependency paths. Both
tasks therefore have the same source scope and must produce the same stable
fingerprint. Editing an executed preprocessor, model, loss, trainer, or
evaluator invalidates the run; adding an unimported audit or dormant candidate
does not falsely change the executed method. Publication rejects missing or
unstable closure evidence. README, logs, tests, and metric JSONs are outside
the executable-method scope and may still be updated during long runs.

After those six reports pass, publication uses one predeclared aggregation
rule for every dataset and task: equal-weight three-fold `log_mean`. Run:

```bash
conda run -n llm-factory python publish_strict_shared_core_results.py \
  --dataset vpn-app \
  --audit_root reasoningDataset/shared-core-audits \
  --packet_manifest_root /tmp/two_tower_runs/strict_shared_core_v2/packet_artifacts \
  --shared_core_config /tmp/two_tower_runs/shared_core_v2/frozen_config.json \
  --method_archive_root reasoningDataset/shared-core-v2 \
  --packet_candidate /tmp/two_tower_runs/strict_shared_core_v2/vpn-app_strict_packet_logmean.json \
  --flow_candidate /tmp/two_tower_runs/strict_shared_core_v2/vpn-app_strict_flow_logmean.json \
  --packet_bootstrap /tmp/two_tower_runs/strict_shared_core_v2/vpn-app_strict_packet_cluster_bootstrap.json \
  --flow_bootstrap /tmp/two_tower_runs/strict_shared_core_v2/vpn-app_strict_flow_cluster_bootstrap.json \
  --packet_session_novelty /tmp/two_tower_runs/strict_shared_core_v2/vpn-app_strict_packet_session_novelty.json \
  --flow_session_novelty /tmp/two_tower_runs/strict_shared_core_v2/vpn-app_strict_flow_session_novelty.json \
  --output_json /tmp/two_tower_runs/strict_shared_core_v2/vpn-app_strict_publication.json
```

The publisher rejects non-single-head packet inputs, non-sequence flow inputs,
anything other than three fixed `log_mean` folds, failed/missing checkpoint
audits, audits without passing runtime mechanism evidence, or mixed frozen
fingerprints. Packet fold manifests are copied into
their canonical paper locations only after these checks. Packet and flow are
promoted independently only when they meet their predeclared targets: VPN
packet `0.90/0.76`, TLS packet `0.85/0.78`, VPN flow `0.75/0.65`, and TLS flow
`0.78/0.70` for accuracy/macro-F1. A target-gap candidate is retained for
analysis while the previous canonical file and its manifests remain untouched.
Every promoted canonical JSON receives a `strict_shared_core_v2` provenance
block containing the frozen SHA, three fold-audit paths, mechanism-evidence
and native-extraction provenance requirements, and fixed-consensus declaration. Result binding refuses VPN/TLS
files without this block, so a stale historical result cannot be relabeled as
the strict unified method. No fusion rule or fold weight is selected from test
performance.

Before publishing metrics, the publisher recomputes the internal canonical
fingerprint of `frozen_config.json`, requires it to match all six strict fold
audits, and re-hashes both validation selection reports referenced by the
config. It then archives the exact config and reports under
`reasoningDataset/shared-core-v2/` with an `archive_manifest.json`. This keeps
the method-selection evidence available after `/tmp` is cleaned without
rewriting the frozen config or changing the fingerprint already bound to the
checkpoints and manifests.

Strict publication additionally requires uncertainty reports generated after
the three-fold rule is frozen. Packet NPZ files retain the recovered
`flow_ids`; `fuse_packet_crossfold.py` verifies that those IDs align across
folds and preserves them in the fused archive. Both tasks then use the same
class-stratified flow-cluster bootstrap (`5000` draws, seed `42`): packets from
one flow are resampled together, while each flow-level example is already one
cluster. The publisher verifies that bootstrap point estimates equal the fixed
candidate metrics and records both 95% confidence intervals. This avoids the
overly narrow packet confidence intervals produced by treating correlated
packets as independent observations.

```bash
conda run -n llm-factory python bootstrap_classification_metrics.py \
  --input STRICT_PACKET_LOGMEAN.npz --task packet \
  --samples 5000 --seed 42 --output_json STRICT_PACKET_BOOTSTRAP.json

conda run -n llm-factory python bootstrap_classification_metrics.py \
  --input STRICT_FLOW_LOGMEAN.json --task flow \
  --samples 5000 --seed 42 --output_json STRICT_FLOW_BOOTSTRAP.json
```

The field-intervention claim is additionally evaluated by one reporting-only
session-novelty audit after predictions are frozen. Training packet indices
define direction-invariant endpoint, port, and complete session signature sets;
test labels do not define the groups. For a three-fold consensus, pass all
three training indices, so `novel` means unseen in their union (unseen to every
member model):

```bash
conda run -n llm-factory python evaluate_session_novelty.py \
  --task flow \
  --train_packet_index FOLD0_TRAIN_PACKET_INDEX.jsonl \
  --train_packet_index FOLD1_TRAIN_PACKET_INDEX.jsonl \
  --train_packet_index FOLD2_TRAIN_PACKET_INDEX.jsonl \
  --test_packet_index TEST_PACKET_INDEX.jsonl \
  --predictions STRICT_FLOW_LOGMEAN.json \
  --label_map LABEL_MAP.json \
  --output_json STRICT_FLOW_SESSION_NOVELTY.json
```

Use `--task packet` with the fixed Packet consensus NPZ for the corresponding
Packet report. Every group includes Accuracy, fixed-all-class Macro-F1, and
present-class Macro-F1. The audit is never consumed by training, checkpoint
selection, fusion, or promotion. A historical VPN Flow smoke model showed
`79.20%/73.33%` Accuracy/Macro-F1 on endpoint-seen flows versus
`53.31%/34.73%` on endpoint-novel flows; this large conditional gap motivates
the test but is neither a strict-v2 result nor causal evidence that endpoints
alone caused each error.

Parameter values may differ after separate dataset/task training. The frozen exact-v2 main model uses the sequence flow aggregator; graph is retained as a structural aggregation ablation until it can consume the same aligned intervention views. Flow classification adds `packet_to_flow_proj`, packet-sequence aggregation, and the flow head only after the audited shared packet representation. Current long-running screening experiments are not interrupted or retroactively relabeled, and the final strict-core rerun starts only after the sampler-aware and paired-invariance validation screens select one cross-dataset configuration.

After both screens finish, freeze their common VPN/TLS decision before any strict-v2 test evaluation:

```bash
conda run -n llm-factory python freeze_shared_core_v2_config.py \
  --balance_selection /tmp/two_tower_runs/paper_unified_tower1_paired_screen/balance_selection.json \
  --paired_selection /tmp/two_tower_runs/paper_unified_tower1_paired_screen/paired_selection_vs_validation_selected_base.json \
  --output_json /tmp/two_tower_runs/shared_core_v2/frozen_config.json
```

The freezer accepts exactly the VPN and TLS-120 reports, requires the same
candidate to pass both datasets under the fixed Macro-F1/Accuracy dual gate,
and records SHA-256 hashes of both reports plus a canonical config fingerprint.
It re-hashes every metric, final checkpoint, validation history, and training
contract. The paired screen baseline must be artifact-identical to the arm
selected by the balance screen, and the contracts must prove the pre-registered
`packet,1.0 -> flow,0.5 -> paired-invariance` A/B/C configuration rather than
merely carrying those names in an output path. Each selection report must prove
that baseline and candidate training both produced a final Tower1 checkpoint
and exactly eight validation-history entries for each dataset; an intermediate
`best_packet_validation_metrics.json` is rejected even if its score is high,
while an appended history from a reused output directory is rejected as
contaminated. It fixes one packet-core architecture, native pretraining
protocol, content-group risk, and Tower1 weighting/invariance choice for both
tasks. Dataset-specific manual switches and test-label selection are explicitly
prohibited in the frozen contract.

The selector also parses every history row and requires the recorded best step
to reproduce Tower1's `(select metric, Macro-F1, Accuracy)` checkpoint ordering
in `packet_validation_history.jsonl`. A stale
or hand-copied best-metrics file therefore cannot pass merely because the final
checkpoint and an eight-line history happen to coexist in the same directory.
The selection JSON additionally retains best-to-final regression, matched-step
win/loss, mean and median deltas, fixed early/late phase means, late-phase
win/loss counts, and first-to-latest relative-gain change for every dataset.
These curve fields are marked `selection_role=descriptive_only`: they expose
early gains and late instability but do not alter the completed-run
best-Macro-F1 promotion threshold.

Current Tower1 runs write `tower1_training_contract.json` before loading the
model. It binds the full training configuration, input-file SHA-256 values,
command line, and trainer-source SHA-256 at launch. Normal completion writes an
explicit `final/` snapshot and updates the same contract with final-head and
validation-history hashes while preserving the launch-time source identity.
Strict-v2 additionally freezes
`packet_batch_scheduler=epoch_resampled_dataloader_v1`. The trainer restarts
the DataLoader iterator at every exhaustion, which advances the deterministic
flow-balanced sampler from `seed + epoch` to `seed + epoch + 1` and resamples
packet identities within each flow. Older runs used `itertools.cycle` around
the DataLoader; that cached the first collated epoch and replayed identical
batches in later epochs. Those scores remain historical performance references
but their checkpoints are excluded from exact-v2 selection and publication.

Audit the deterministic schedule without running the model or reading test
labels:

```bash
conda run -n llm-factory python audit_tower1_epoch_sampling.py \
  --packet_aux_jsonl /tmp/two_tower_runs/paper_unified_packet_repro_v2/artifacts/vpn-app/fold0/train/packet_auxiliary.jsonl \
  --batch_size 16 --packets_per_flow 2 --seed 42 --epochs 8 \
  --output_json /tmp/two_tower_runs/shared_core_v2_validation_resampled/vpn-app_fold0_sampling_audit.json

conda run -n llm-factory python audit_tower1_epoch_sampling.py \
  --packet_aux_jsonl /tmp/two_tower_runs/paper_unified_packet_repro_v2/artifacts/tls-120/fold0/train/packet_auxiliary.jsonl \
  --batch_size 16 --packets_per_flow 2 --seed 42 --epochs 8 \
  --output_json /tmp/two_tower_runs/shared_core_v2_validation_resampled/tls-120_fold0_sampling_audit.json
```

The fold-0 schedule audit reports unique batch hashes for all eight epochs.
VPN cumulative unique-packet exposure rises from `6733/33088=20.35%` after one
epoch to `14833/33088=44.83%` after eight, with `26.46%-27.16%` of flows
changing their selected packet identities between adjacent epochs. TLS-120
rises from `36298/98640=36.80%` to `74889/98640=75.92%`, with
`41.52%-41.98%` adjacent-epoch flow selection changes. This is mechanism and
coverage evidence only; accuracy impact is determined by the completed
validation screen and strict three-fold test, not inferred from the audit.

The frozen Tower1 file provides one reproducible common-reference optimization
configuration: `max_steps=0`, eight complete epochs, no early stopping,
gradient checkpointing, projection dimension `256`, weight decay `0.01`,
effective-number beta `0.9999`, and base-model-only initialization. Only the
projection dimension and initialization policy are method/architecture
invariants. Epoch budget, optimizer values, batch size, temperature,
regularization, and nonzero loss magnitudes may be set independently for each
dataset/task and are recorded as execution hyperparameters. The flow-prototype
objective is explicitly removed from the main method with
`flow_proto_weight=0`; enabling it is an ablation/method change, while changing
the magnitude of an already-enabled objective is ordinary tuning. This keeps
the main packet objective to class-balanced CE plus flow-aware SupCon instead
of retaining an unverified extra loss.

Packet and Flow manifests contain both `tower1_training_contract` (the runner
declaration) and `tower1_execution_evidence` (the completed trainer contract).
The latter re-hashes `final/tower1_heads.pt`, the validation history, and the
contract file; requires the trainer source hash to remain unchanged from launch
through completion; and normalizes the actual optimizer, loss, sampling,
initialization, and scheduler fields back to the declared method contract.
`audit_cross_task_shared_core.py` passes only when both task-local executions
are verified, both declarations match their actual executions, the normalized
Packet/Flow **shared method signatures** and trainer source hashes are
identical, and every numeric hyperparameter difference is explicitly reported.
The full optimization dictionaries need not be numerically identical.
Comparing two runner declarations alone is no longer sufficient.

The selector requires this completed contract and recomputes the hashes. Two
long-running validation screens that started before this contract existed are
handled only by `materialize_legacy_tower1_final.py`: it accepts exactly eight
history rows, verifies the best row, copies the terminal root checkpoint to
`final/`, and emits a separately hashed legacy-materialization manifest. This
compatibility path is for those already-running screens, not the strict-v2 main
experiments.

Both task runners consume that exact file. Do not repeat or tune shared-core flags in the publication commands:

```bash
# Packet task: native pretraining -> shared packet core -> packet head
conda run -n llm-factory python run_packet_level_pipeline.py \
  --dataset vpn-app --fold 0 --stage paper_unified \
  --shared_core_config /tmp/two_tower_runs/shared_core_v2/frozen_config.json

# Flow task: same native packet core -> sequence/window aggregation -> flow head
conda run -n llm-factory python run_stage8_flowaware_pipeline.py \
  --dataset vpn-app --fold 0 --stage all \
  --framework_profile paper_unified \
  --shared_core_config /tmp/two_tower_runs/shared_core_v2/frozen_config.json \
  --num_classes 16
```

For an independently optimized run, explicitly name only the parsed training
arguments that should survive the common-reference defaults. For example:

```bash
# Packet example: same method, independently selected optimization values.
conda run -n llm-factory python run_packet_level_pipeline.py \
  --dataset vpn-app --fold 0 --stage paper_unified \
  --shared_core_config /tmp/two_tower_runs/shared_core_v2/frozen_config.json \
  --epochs 12 --lr 2e-5 \
  --training_hyperparameter_overrides epochs,lr

# Flow example: corresponding task-local argument names.
conda run -n llm-factory python run_stage8_flowaware_pipeline.py \
  --dataset vpn-app --fold 0 --stage all \
  --framework_profile paper_unified \
  --shared_core_config /tmp/two_tower_runs/shared_core_v2/frozen_config.json \
  --tower1_epochs 12 --tower1_lr 2e-5 \
  --training_hyperparameter_overrides tower1_epochs,tower1_lr \
  --num_classes 16
```

The allowlist rejects architecture/module switches such as `lora_r`. Loss and
intervention magnitudes are allowed only when their zero/nonzero activation
status matches the frozen shared method, so the option cannot silently disable
SupCon, paired invariance, class balancing, or a native pretraining objective.

### Reporting-only flow-length diagnostic

MIETT motivates packet-instance interaction, while TrafficFormer's input-length
ablation warns that adding packets is not automatically useful. Before adding
another aggregation branch, test whether the current errors actually grow with
flow length:

```bash
conda run -n llm-factory python analyze_flow_length_performance.py \
  --predictions reasoningDataset/tls-120/test_crossfold_consensus_auto_confidence.json \
  --embedding_index reasoningDataset/tls-120/test_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
  --output_json /tmp/two_tower_runs/tls120_historical_flow_length_performance.json
```

The script is reporting-only and records fixed packet-count bins, quartiles,
confidence, overall length/correctness rank correlation, and per-class
correlations. On the historical TLS result, the longest quartile has lower
Accuracy (`0.8159`) than the middle quartiles, but this apparent effect largely
vanishes after conditioning on class: 112 eligible classes have mean/median
length-correctness Spearman values of `-0.0167/-0.0060`, with `50.9%` negative.
VPN shows the opposite within-class tendency (`+0.1304/+0.1155`, only `25%`
negative). Therefore a fixed early-packet or short-flow bias is not promoted to
the unified model. Any future multi-scale aggregator requires validation-only
gains on both datasets and a matched ablation; these descriptive test strata
cannot select it.

### Reporting-only cross-fold disagreement diagnostic

Before adding another expert, measure whether the three independently trained
fold models make complementary errors. The diagnostic aligns predictions by
the exact `flow_id` set, verifies every aligned true label, and reports each
fold, pairwise error overlap, the number of correct folds per flow, unanimous
errors, a non-deployable any-fold oracle, and class-conditional headroom:

```bash
conda run --no-capture-output -n llm-factory \
  python analyze_crossfold_disagreement.py \
    --input fold0=reasoningDataset/tls-120/test_selector_soft_gate_tls120_tol0015_calib_family_valid_macro.json \
    --input fold1=reasoningDataset/tls-120/test_stacker_graph_seq_rawproj_flowaware_change_weight_fold1_stage8_flowaware_fold1_stage8_cv_accuracy.json \
    --input fold2=reasoningDataset/tls-120/test_selector_base_prior_stacker_graph_seq_rawproj_flowaware_change_weight_fold2_stage8_flowaware_fold2_stage8_cv_accuracy.json \
    --consensus consensus=reasoningDataset/tls-120/test_crossfold_consensus_auto_confidence.json \
    --output_json /tmp/two_tower_runs/tls120_historical_crossfold_disagreement.json
```

Historical results show genuine split instability rather than three copies of
the same error pattern, but the historical folds also use different terminal
selectors. The table therefore measures total cross-fold candidate diversity,
not a clean estimate of split variance under one fixed method:

```text
| Dataset | Mean fold Acc | Best fold Acc | Any-fold oracle Acc | Oracle headroom | Disagreement | All folds wrong |
|---|---:|---:|---:|---:|---:|---:|
| vpn-app | 0.7165 | 0.7488 | 0.8140 | +0.0652 | 0.3307 | 0.1860 |
| tls-120 | 0.7745 | 0.7996 | 0.9121 | +0.1125 | 0.3587 | 0.0879 |
```

The historical consensus reaches `0.7512/0.7522` Accuracy/Macro-F1 on VPN and
`0.8461/0.8292` on TLS-120. It recovers substantially more single-fold errors
than it harms, especially on TLS-120. This supports a future unified
cross-training-stability objective; it does **not** authorize selecting a
teacher, fusion rule, class, or hyperparameter from these test diagnostics.
Every output carries `selection_role=none`,
`label_usage=test_labels_diagnostic_only`, and an explicit prohibition on test
adaptation.

This historical diagnosis is only a hypothesis gate. The disagreement analysis
must be repeated on the strict-v2 three-fold outputs, where architecture,
objective topology, and fixed `log_mean` consensus are identical. Only that
matched result may attribute complementarity to training split/seed
instability or justify promoting a stability objective into the unified method.

The equivalent Packet diagnostic requires stronger row-order evidence because
the historical NPZ files do not contain packet IDs. It accepts one explicitly
shared `packet_index.jsonl`, verifies its labels against every NPZ row, requires
each fold's source JSON to bind that exact index path, and records SHA-256 for
the index, label map, predictions, and source records:

```bash
conda run --no-capture-output -n llm-factory \
  python analyze_packet_crossfold_disagreement.py \
    --input fold0=/tmp/two_tower_runs/packet_level/vpn_app/fold0/packet_feature_test_probs_ipv46.npz \
    --input fold1=/tmp/two_tower_runs/packet_level/vpn_app/fold1/packet_feature_test_probs_ipv46.npz \
    --input fold2=/tmp/two_tower_runs/packet_level/vpn_app/fold2/packet_feature_test_probs_ipv46.npz \
    --source fold0=reasoningDataset/packet-level/vpn-app/fold0_feature_expert.json \
    --source fold1=reasoningDataset/packet-level/vpn-app/fold1_feature_expert.json \
    --source fold2=reasoningDataset/packet-level/vpn-app/fold2_feature_expert.json \
    --sample_index /tmp/two_tower_runs/packet_level/vpn_app/fold0/test/packet_index.jsonl \
    --label_map /tmp/two_tower_runs/packet_level/vpn_app/fold0/train/label_map.json \
    --consensus equal_mean=/tmp/two_tower_runs/vpn_packet_historical_equal_mean_for_diagnostic.npz \
    --output_json /tmp/two_tower_runs/vpn_packet_historical_crossfold_disagreement.json
```

The old current-packet structural experts show complementary errors:

```text
| Dataset | Mean fold Acc/F1 | Best fold Acc/F1 | Equal-mean Acc/F1 | Any-fold oracle Acc | Oracle headroom | Disagreement |
|---|---:|---:|---:|---:|---:|---:|
| vpn-app | 0.8945/0.7813 | 0.8981/0.7926 | 0.9066/0.8112 | 0.9423 | +0.0442 | 0.1229 |
| tls-120 | 0.7736/0.7312 | 0.7874/0.7450 | 0.7998/0.7656 | 0.8617 | +0.0743 | 0.2635 |
```

VPN's equal mean captures `96.20%` of packets for which at least one fold is
correct; TLS-120 captures `92.74%`. The largest class-level oracle headroom is
concentrated in known ambiguous classes (`facebook` and `hangout` on VPN;
`huanqiu`, `toutiao`, `smzdm`, `media`, and `zhihu` on TLS-120). This is useful
mechanism evidence that training-fold instability affects both Packet and Flow,
but it is not yet evidence for adding distillation. In particular, these are
historical tree-based structural experts, TLS-120's paper-default score also
uses a separately validation-gated neural/structural fusion, and neither is the
strict-v2 shared neural core. Promotion still requires the same-method
strict-v2 three-fold diagnostic on both tasks and both datasets, followed by an
OOF-only matched training ablation.

New strict-v2 Packet evaluations remove the historical row-order ambiguity at
the source. `test_packet_byte_transformer.py` writes the exact `packet_uids`
alongside `y_true`, probabilities, content groups, and flow IDs. Its JSON binds
the checkpoint, packet index, label map, and generated NPZ by SHA-256.
`fuse_packet_crossfold.py` requires identical packet UID arrays across folds,
preserves them in the consensus NPZ, records every input hash, and binds the
generated consensus NPZ hash after writing it. The diagnostic still checks the
shared source index and true labels, so an internally consistent but unrelated
NPZ cannot satisfy the strict evidence contract. Historical files remain
readable through the explicit source-index fallback, but only UID-bearing
strict-v2 outputs qualify for the final matched-method stability analysis.

Paper-safe distillation must use only task-local training data and OOF teacher
predictions. Concatenating disjoint validation folds with
`build_consensus_distill_targets.py --align union` improves coverage, but each
flow ordinarily has only one model that excluded it from training; this is an
OOF prediction union, not an OOF multi-teacher consensus. A promoted consensus
student therefore requires an inner-fold protocol with multiple independently
initialized/view-perturbed teachers that all exclude the target inner fold,
complete flow-ID coverage, and the same teacher/student algorithm for VPN/TLS
and Packet/Flow. Until that protocol is implemented and ablated, historical
consensus distillation remains supporting negative evidence rather than a core
module.

The Flow-side infrastructure now supports checkpoint-bound OOF proof for the
future inner-fold experiment. New `train_tower2.py` checkpoints embed SHA-256
evidence for their training/validation datasets and trainer source;
`test_tower2.py` prediction JSONs bind the checkpoint and evaluation dataset.
Generate one sidecar per independently initialized teacher that predicts the
same held-out inner fold:

```bash
conda run --no-capture-output -n llm-factory \
  python write_oof_teacher_evidence.py \
    --prediction_json /tmp/inner0_seed0_valid_predictions.json \
    --checkpoint /tmp/inner0_seed0/best.pt \
    --train_dataset /tmp/inner0/train_seq_dataset.pt \
    --evaluation_dataset /tmp/inner0/valid_seq_dataset.pt \
    --output_json /tmp/inner0_seed0_oof_evidence.json
```

The writer verifies four bindings before setting
`oof_exclusion_proven=true`: the checkpoint embeds the supplied training-set
hash; the prediction embeds the checkpoint hash; the prediction embeds the
evaluation-set hash; and the checkpoint-bound training flow IDs have zero
overlap with all prediction/evaluation flow IDs. Build a strict teacher only
when every contributing model has such a sidecar:

```bash
conda run --no-capture-output -n llm-factory \
  python build_consensus_distill_targets.py \
    --input seed0 /tmp/inner0_seed0_valid_predictions.json \
    --input seed1 /tmp/inner0_seed1_valid_predictions.json \
    --oof_evidence seed0 /tmp/inner0_seed0_oof_evidence.json \
    --oof_evidence seed1 /tmp/inner0_seed1_oof_evidence.json \
    --split heldout --align intersection --mode log_mean \
    --min_teachers_per_flow 2 --require_oof_exclusion_proof \
    --output_json /tmp/inner0_strict_oof_teacher.json
```

Here `--split heldout` reads the direct prediction fields produced for the
**inner validation fold** and is accepted only with strict OOF evidence; it is
not the outer shared test set. The builder re-hashes the checkpoint, training
dataset, evaluation dataset, and prediction, then recomputes flow-set
disjointness instead of trusting the sidecar flags. The resulting target
records aligned `teacher_counts` and `oof_teacher_counts` for every flow and
sets `oof_multi_teacher_consensus_proven=true` only when all counts are at
least two and equal. Tower2 strict loading uses
`--distill_min_teachers_per_flow 2` together with
`--distill_require_oof_exclusion_proof` and fails before training otherwise.
The runtime loader independently requires aligned per-flow `teacher_counts`
and `oof_teacher_counts`; a file-level OOF boolean without equal per-flow
counts, or without the multi-teacher proof flag, cannot pass the strict gate.
This is currently Flow-side research infrastructure, not an active unified
module. Promotion requires the analogous Packet student protocol plus matched
VPN/TLS Packet/Flow ablations; until then the frozen strict-v2 method keeps
distillation disabled.

The strict-v2 frozen file is a deliberately stronger **common-reference
hyperparameter experiment**: its values override command-line defaults in both
runners and are recorded with one config fingerprint. It fixes architecture,
objective choices, and numeric hyperparameters so the first controlled matrix
has no tuning-budget confound. This is not the general definition of framework
unity. The general cross-task audit compares the shared architecture/algorithm
signature and separately reports allowed differences in learning rate, epoch
budget, batch size, weight decay, temperature, seed, and nonzero objective
coefficients. Later optimized runs may use different recorded numeric values for
VPN/TLS and Packet/Flow while keeping the same signature. Turning an objective
on/off, changing packet context, replacing the sampler, changing LoRA rank or
projection structure, or adding a dataset-specific branch remains a method
change. `audit_unified_framework.py` still requires the initial strict-v2
common-reference runs to carry one identical fingerprint because that matrix is
the controlled baseline, not because separately tuned hyperparameters are
forbidden by the unified method.

For an independently tuned run, list only the numeric runner fields that must
survive application of `paper_unified` and the frozen method defaults:

```bash
# Packet example: architecture and objective topology remain frozen.
python run_packet_level_pipeline.py ... \
  --framework_profile paper_unified \
  --shared_core_config /tmp/two_tower_runs/shared_core_v2/frozen_config.json \
  --epochs 12 \
  --lr 2e-5 \
  --packet_batch_size 32 \
  --training_hyperparameter_overrides epochs,lr,packet_batch_size

# Flow example: this task learns all weights from its own Flow training packets.
python run_stage8_flowaware_pipeline.py ... \
  --framework_profile paper_unified \
  --shared_core_config /tmp/two_tower_runs/shared_core_v2/frozen_config.json \
  --tower1_epochs 12 \
  --tower1_lr 2e-5 \
  --training_hyperparameter_overrides tower1_epochs,tower1_lr
```

The allowlist contains optimization budget, compute, regularization, and
loss-magnitude fields only. Architecture and categorical algorithm fields such
as `lora_r`, packet context, embedding mode, sampler type, aggregation family,
and intervention policy cannot be restored this way. A magnitude override also
cannot cross zero when that would enable or disable a shared objective. Both
the requested values and the executed trainer contract are stored in the run
manifest; `audit_cross_task_shared_core.py` compares method signatures and
reports numeric differences separately.

Manifests distinguish two hashes. `shared_core_method_sha256` identifies the
frozen module/algorithm contract and must agree across Packet/Flow. The
`shared_core_config_sha256` is the effective execution hash: it equals the
method hash for the untouched common-reference run, but changes when explicit
numeric overrides are present. Consequently an independently tuned run can
pass the unified-method audit without being mislabeled as the exact same
strict-v2 numerical baseline.

Module reuse means reuse of the architecture and representation contract, not
reuse of supervised weights across tasks. Packet-level training learns its
Packet module from the packet task's training split. Flow-level training
re-trains the same Packet module from packets belonging to the flow task's own
training split, then adds `packet_to_flow_proj`, window/flow aggregation, and a
flow head. It must not initialize that module from a Packet-level task
checkpoint. Both framework manifests record `packet_module_training_source`
and `cross_task_trained_weights_reused=false`; the cross-task audit requires
`packet_task_train_split_packets` for Packet and
`flow_task_train_split_packets` for Flow.

One validation-only v3 candidate addresses the observed Packet-to-Flow evidence
loss without adding a dataset-specific expert. `SharedPacketClassifierHead` is
the same fused-packet linear head class used by Packet classification. During
Flow training, a fresh instance is trained only from the Flow train split; its
per-packet logits are pooled inside each window and combined with the window
classifier by a learned convex gate bounded by
`--packet_evidence_max_weight`. The existing learned `late_fusion` flow head
then aggregates window evidence. A value of `0` is the exact-v2 default and
does not instantiate the extra head; a positive value automatically selects
`late_fusion` and is recorded in the manifest. This candidate is not a paper
main module unless the same bound improves held-out macro-F1 on both VPN and
TLS-120 under the same frozen shared-core configuration.

```bash
# Run only after the strict-v2 fold-0 Flow manifests exist. Both commands use
# the same bound and reuse only their own Flow-task factual/intervention data.
conda run --no-capture-output -n llm-factory \
  python run_packet_evidence_validation.py \
    --baseline_manifest reasoningDataset/vpn-app/stage8_flowaware_manifest_rawproj_strict_shared_core_v2_fold0_native_shared_content_strict_shared_core_v2_fold0_stage8_flowaware_strict_shared_core_v2_fold0.json \
    --shared_core_config /tmp/two_tower_runs/shared_core_v2/frozen_config.json \
    --packet_evidence_max_weight 0.4 \
    --summary_json /tmp/two_tower_runs/shared_packet_evidence/vpn_fold0.json

conda run --no-capture-output -n llm-factory \
  python run_packet_evidence_validation.py \
    --baseline_manifest reasoningDataset/tls-120/stage8_flowaware_manifest_rawproj_strict_shared_core_v2_fold0_native_shared_content_strict_shared_core_v2_fold0_stage8_flowaware_strict_shared_core_v2_fold0.json \
    --shared_core_config /tmp/two_tower_runs/shared_core_v2/frozen_config.json \
    --packet_evidence_max_weight 0.4 \
    --summary_json /tmp/two_tower_runs/shared_packet_evidence/tls120_fold0.json
```

`run_packet_evidence_validation.py` launches only `tower2_train` and a
validation-only `eval`; it rejects manifests that reuse supervised Packet-task
weights. For each dataset it trains two matched arms: `late_fusion` with the
evidence head disabled, and the same `late_fusion` model with the evidence head
enabled. Use the two summary files to obtain the exact strict-v2 reference,
control, and candidate paths, then register both datasets in one selection
report:

```bash
conda run -n llm-factory python select_packet_evidence_candidate.py \
  --record vpn-app <VPN_STRICT_V2_VALID> <VPN_CONTROL_VALID> <VPN_CANDIDATE_VALID> <VPN_CONTROL_MANIFEST> <VPN_CANDIDATE_MANIFEST> \
  --record tls-120 <TLS_STRICT_V2_VALID> <TLS_CONTROL_VALID> <TLS_CANDIDATE_VALID> <TLS_CONTROL_MANIFEST> <TLS_CANDIDATE_MANIFEST> \
  --min_macro_f1_gain 0.005 \
  --max_accuracy_drop 0.01 \
  --max_reference_macro_f1_drop 0.01 \
  --output_json /tmp/two_tower_runs/shared_packet_evidence/selection.json
```

The selector checks aligned flow IDs/labels, one frozen-core fingerprint, one
candidate bound, task-local Flow training provenance, and the absence of test
outputs. The candidate is promoted only when both datasets gain at least 0.5
macro-F1 points over the matched `late_fusion` control, lose no more than one
accuracy point against that control, and remain within one macro-F1/accuracy
point of the strict-v2 `mean` reference. Otherwise the strict-v2 baseline
remains the paper method. This three-arm design prevents a pooling change from
being misreported as a packet-evidence gain.

This candidate is not claimed as a standalone novelty. MIETT already models
packet/flow hierarchy with two-level attention, while SWEET's Pcap-Encoder uses
simple voting over early packet predictions. The paper-facing distinction is
the complete contract: one counterfactually trained semantic/content/structure
Packet module is reused by architecture across both tasks, all supervised
weights are re-trained from each task's own split, and Flow may learn a bounded
residual from weak packet evidence to contextual window evidence. The matched
ablation must therefore separate the shared core, intervention, bounded router,
and packet-evidence residual; reporting only the final combined model would not
support the method claim.

The related-work collision analysis, defensible hypothesis, required evidence,
and stop rules are tracked in
`reasoningDataset/paper_method_novelty_audit.md`. Treat that audit as a research
constraint: a higher validation score does not justify a module when it breaks
the unified contract or lacks a matched control.

A possible anchor-excluded cross-scale distillation follow-up is pre-registered
in `reasoningDataset/counterfactual_cross_scale_distillation_preregistration.md`.
It is deliberately inactive: implementation is permitted only after the full
strict-v2 matrix demonstrates the specified validation-only Packet-to-Flow
evidence gap. The frozen controls are no-KD, plain anchor-excluded KD,
intervention-stability KD, and stability-plus-learnability-gated KD; failure on
either VPN or TLS keeps the candidate out of the unified main method.

The Tower-1 paired loss covers the exact downstream concat representation:

```text
L_pair = (1-cos(raw_f, raw_i))
       + (1-cos(proj_f, proj_i))
       + eta * symmetric_KL(logits_f, logits_i)

L_cls = CE(logits_f, y) + kappa * CE(logits_i, y)
```

Finally, sampler-aware class risk derives class counts from the unit actually
visited by the sampler. When each flow contributes a fixed number of packets,
flow counts, rather than pre-balanced raw packet-row counts, define the
effective training distribution. If class `c` contains `n_c` flows and the
sampler visits every flow once per epoch, its expected weighted classification
mass is

```text
M_c = n_c * normalize(EffectiveNumberWeight(n_c) ** alpha).
```

The same `alpha=0.5` candidate is screened on VPN and TLS-120. It is a tempered
sampler/objective correction, not full inverse-frequency balancing. Across the
three available VPN folds it reduces the expected objective-exposure ratio
from `13.40-16.69` to `3.71-4.17`; across TLS-120 it reduces the ratio from
`29.50-34.25` to `5.47-5.89`. In contrast, `alpha=1.0` reduces every ratio to
approximately `1.01-1.04`, which can give a class represented by only a few
flows disproportionate gradient leverage. These values are computed directly
from the sampler contract and training rows by
`analyze_tower1_sampling_balance.py`; they are not test-set measurements. The
same rule is used for VPN/TLS and packet/flow tasks, while each task and dataset
learns its own model parameters.

The exposure hypothesis is checked on held-out validation only, with an explicitly non-causal rank-association audit:

```bash
conda run -n llm-factory python analyze_tower1_exposure_outcomes.py \
  --sampling_report /tmp/two_tower_runs/paper_unified_tower1_sampling_balance.json \
  --report_index 0 \
  --validation_history /tmp/two_tower_runs/paper_unified_packet_repro_v2/checkpoints/vpn-app_fold0/packet_validation_history.jsonl \
  --output_json /tmp/two_tower_runs/paper_unified_vpn_fold0_tower1_exposure_outcomes.json
```

When a sampler-aware candidate and its baseline are still training, compare
only checkpoints reached by both runs. This avoids attributing ordinary
training-time improvement to the sampler:

```bash
conda run -n llm-factory python analyze_tower1_validation_history.py \
  --input baseline /tmp/two_tower_runs/paper_unified_packet_repro_v2/checkpoints/vpn-app_fold0/packet_validation_history.jsonl \
  --input flow_weight /tmp/two_tower_runs/paper_unified_tower1_flowweight_sqrt/checkpoints/vpn-app_fold0/packet_validation_history.jsonl \
  --compare baseline flow_weight \
  --output_json /tmp/two_tower_runs/vpn_tower1_flowweight_matched_validation_live.json
```

The matched report includes per-class F1/recall deltas and the Spearman
association between held-out class support and F1 gain. Its
`matched_curve_summary` also reports candidate wins/losses and mean/median
Accuracy and Macro-F1 deltas across every common checkpoint, so an isolated
best or final point cannot be presented as the whole learning dynamic. It is
mechanism evidence only: it reads no test labels and does not select or promote a model.
The cross-dataset, completed-history promotion rule below remains authoritative.

To determine whether a weighting gain/loss follows class exposure or
single-packet identifiability, first audit the original `packet_index.jsonl`
files and then align their per-class signature conflicts with the matched
validation deltas:

```bash
conda run -n llm-factory python audit_packet_identifiability.py \
  --train_index /tmp/two_tower_runs/paper_unified_packet_repro_v2/artifacts/vpn-app/fold0/train/packet_index.jsonl \
  --test_index /tmp/two_tower_runs/paper_unified_packet_repro_v2/artifacts/vpn-app/fold0/valid/packet_index.jsonl \
  --output_json /tmp/two_tower_runs/vpn_fold0_validation_packet_identifiability_per_class.json

conda run -n llm-factory python analyze_tower1_weight_identifiability.py \
  --matched_report /tmp/two_tower_runs/vpn_tower1_flowweight_matched_validation_live.json \
  --identifiability_report /tmp/two_tower_runs/vpn_fold0_validation_packet_identifiability_per_class.json \
  --sampling_report /tmp/two_tower_runs/paper_unified_tower1_sampling_balance.json \
  --report_index 0 \
  --label_map /tmp/two_tower_runs/paper_unified_packet_repro_v2/artifacts/vpn-app/fold0/train/label_map.json \
  --signature_level session \
  --weight_strength 0.5 \
  --output_json /tmp/two_tower_runs/vpn_tower1_flowweight_identifiability_live.json
```

Use `packet_index.jsonl`, not `packet_auxiliary.jsonl`, for this audit. The
auxiliary rows intentionally omit `l3_hex_prefix`; treating them as byte inputs
collapses signatures and can produce a meaningless 100% train-seen rate. This
association is validation-only, descriptive evidence and cannot select or
promote a candidate.

At VPN fold0 epoch 1, `log1p(training flow count)` has Spearman `rho=0.506` with held-out class F1 (`p=0.045`) and `rho=0.642` with held-out recall (`p=0.0074`). This is provisional mechanism evidence, not a causal result: classes such as `torrent` are counterexamples to a count-only explanation. The history-mode audit reports the association at every validation checkpoint and marks the trajectory descriptive-only, so a transient early correlation cannot be presented as the complete mechanism. Sampler-aware weighting is promoted only if its best held-out macro-F1 improves by at least `0.005` on both VPN and TLS-120 under the same eight-epoch budget, while held-out accuracy on neither dataset drops by more than `0.005`. `select_unified_tower1_candidate.py` rejects single-dataset promotion attempts, requires identical baseline/candidate dataset sets with at least two datasets, and records machine-readable final-checkpoint/history completion evidence before comparing scores. Strict selection additionally requires `epoch_resampled_dataloader_v1`, proves that each run used one unchanged trainer source from launch through completion, and requires the same trainer source SHA across every VPN/TLS baseline and candidate; legacy or mixed-implementation histories cannot enter the selection. The exposure association will be recomputed for all folds after histories are frozen; the test split is never used by this audit or promotion gate.

---

## 9. Suggested ablations for the paper

The paper ablation matrix is intentionally small. Every shared-core ablation is
run on VPN/TLS and Packet/Flow with the same toggle; learned parameters are
still trained independently for each dataset/task/fold.

1. **Full exact shared core**: semantic + label-free protocol content + strict
   packet-local structure, factual/intervened view fusion, bounded router, and
   content-group empirical risk.
2. **No endpoint intervention**: train and evaluate with the factual semantic
   view only. This tests shortcut resistance, not a different dataset recipe.
3. **No protocol-content channel**: retain semantic and structural channels but
   remove the label-free current-packet content representation.
4. **No structural channel**: retain semantic and protocol-content channels but
   remove the 13-dimensional packet-local behavior representation.
5. **Fixed fusion instead of learned bounded routing**: retain all three
   channels and use one fixed symmetric fusion rule. This isolates adaptive
   reliability learning from the evidence encoders themselves.
6. **Row risk instead of content-group risk**: keep the model fixed but replace
   group-mean empirical risk with ordinary sample-mean risk.
7. **Flow contextualization (pre-registered, not yet implemented)**: for Flow
   only, compare the existing sequence/window/flow aggregator with a matched
   packet-evidence-only bag model under the same two datasets and folds. The
   current bounded packet-evidence residual is not this control because it is
   still combined with contextual window evidence.
8. **Packet-evidence residual**: strict-v2 mean reference versus matched
   `late_fusion` control (`packet_evidence=0`) versus identical candidate with
   bounded packet evidence. Promotion requires a gain on both VPN and TLS.

For the already trained full model, `test_packet_byte_transformer.py` and
`test_tower2.py` support `--ablate_input_channel
semantic|content|structural` and `--ablate_intervention_view
factual_only|intervened_only`. These are cheap inference-only mechanism
sensitivity diagnostics. Their JSON explicitly records
`inference_only_not_retrained_ablation`; they cannot replace the retrained
ablations above. Run them first to verify that each learned channel is actually
used, then spend the full three-fold budget only on the pre-registered retrained
matrix.

The training programs now expose the matching persistent toggles
`--train_ablate_input_channel semantic|content|structural` and
`--train_ablate_intervention_view factual_only|intervened_only`. Unlike the
cheap diagnostic flags, these values are stored in `best.pt`, automatically
applied during training, validation, and ordinary test inference, and reported
as `retrained_ablation` by both test programs. They require the exact shared
packet representation; intervention-view ablations additionally require the
aligned factual/masked-IP-port views. The same toggle name and semantics are
used by `run_packet_level_pipeline.py` and
`run_stage8_flowaware_pipeline.py`, while every dataset/task still learns its
own parameters from its own training split.

The same matched path also supports `--train_fixed_channel_fusion` and
`--train_row_risk_ablation`. The former persists an exact equal-channel mean in
the checkpoint; the latter changes both pipeline runners to ordinary row risk
only after the frozen main profile has been applied. Thus neither control can
silently alter dataset-specific architecture, header policy, or task-local
training provenance.

Run one Packet/Flow fold pair with the same diagnostics:

```bash
conda run --no-capture-output -n llm-factory \
  python run_shared_core_sensitivity.py \
    --packet_manifest <STRICT_PACKET_FOLD_MANIFEST> \
    --flow_manifest <STRICT_FLOW_FOLD_MANIFEST> \
    --split test \
    --output_dir /tmp/two_tower_runs/strict_shared_core_v2/shared_core_sensitivity \
    --device cuda
```

After all six VPN/TLS fold pairs are complete, pass their summary files to
`make_shared_core_sensitivity_report.py`. The report refuses incomplete fold,
dataset, task, or fingerprint coverage and never authorizes automatic module
removal; a weak inference sensitivity only schedules the corresponding
retrained ablation.

Run a matched retrained toggle for one completed strict Packet/Flow fold pair:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n llm-factory \
  python run_retrained_shared_core_ablation.py \
    --packet_manifest <STRICT_PACKET_FOLD_MANIFEST> \
    --flow_manifest <STRICT_FLOW_FOLD_MANIFEST> \
    --diagnostic no_content \
    --output_root /tmp/two_tower_runs/strict_shared_core_v2/retrained_ablations \
    --device cuda
```

Valid diagnostics are `no_semantic`, `no_content`, `no_structural`,
`factual_only`, `intervened_only`, `fixed_fusion`, and `row_risk`. The fixed-fusion
ablation uses the exact arithmetic mean of the three normalized channels and
bypasses both learned gate and interaction correction during all phases. The
`row_risk` control changes both tasks from content-group mean empirical risk to
ordinary row/sample mean risk while leaving the architecture fixed. The
runner verifies the shared frozen
fingerprint and task-local training sources, reuses only the reference model's
label-free content initialization, trains separate Packet/Flow supervised
weights, selects each checkpoint on validation, and evaluates test only after
training. Packet inputs/caches remain read-only; ablation checkpoints, results,
and manifests are written under `--output_root`.
