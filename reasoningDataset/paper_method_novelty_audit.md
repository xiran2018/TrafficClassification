# Unified Packet-to-Flow Novelty Audit

Status: working research constraint, not a claim of publication readiness.

## Protocol Scope

- Tasks: per-flow-split packet classification and per-flow-split flow classification.
- Datasets: VPN-app and TLS-120, each trained independently under one architecture and protocol.
- USTC and other packet-only datasets may be reported as supplementary evidence,
  but cannot support the unified Packet-to-Flow claim without a protocol-matched
  per-flow-split Flow task.
- Shared Packet module reuse means the same classes, strict current-packet inputs, channel schema, and bounded routing rule. Flow semantic prompts cannot contain previous-packet IAT or whole-flow server-role inference. A direction cue inferred from the current packet alone is allowed in both tasks; Flow-only sequence position and window context enter only after the shared Packet representation.
- Supervised weights are task-local. Flow re-trains the Packet module from packets in the Flow training split; it never imports a Packet-task checkpoint.
- Test labels cannot select modules, pooling, thresholds, priors, or checkpoints.

## Related-Work Collisions

The uploaded papers impose the following claim boundary. This table is a
design constraint: a checked implementation component is not automatically a
paper contribution.

| Prior work | Capability already established | Claim that is not allowed | Evidence needed for a narrower claim here |
| --- | --- | --- | --- |
| SWEET | Per-packet leakage diagnosis, Per-flow Split, downstream header filtering, strong simple packet encoder and packet-vote flow inference | Per-flow evaluation, field masking, or packet voting is novel | One fixed Packet-to-Flow protocol, independently retrained on each task, with protocol-matched Packet and Flow gains over the SWEET baseline |
| TrafficFormer | Masked traffic pretraining, same-flow/order supervision, and random initialization field augmentation during fine-tuning | Masked bytes, IP/port randomization, or same-flow pretraining is novel | Aligned factual/intervened views must feed a learned bounded reliability mechanism, and that mechanism must beat factual-only and ordinary random augmentation on VPN and TLS |
| MIETT | Multi-instance packet-to-flow modeling, intra/inter-packet attention, relative packet position, and flow contrastive learning | Hierarchical attention or flow SupCon is novel | The shared current-packet representation contract must improve both strict Packet inference and Flow aggregation, not only a bag classifier |
| DigTraffic | Length/timing interaction channels, typed message edges, and edge-aware graph attention | Dual structural channels, edge attributes, or a graph Transformer is novel | A single task-independent structural schema must provide reproducible cross-dataset gains; otherwise graph processing remains an ablation |
| TrafficLLM | Traffic-aware LLM representation, multi-task instruction tuning, masked meta-information, and parameter-efficient adaptation | Qwen/LoRA, traffic prompts, or generic multi-task applicability is novel | Demonstrate auditable single-packet input scope, task-local retraining, and intervention-conditioned reliability under domain/content-group evaluation |

Consequently, the main contribution cannot be described as a collection of
field randomization, LLM embeddings, contrastive learning, attention pooling,
and graph edges. The algorithmic claim must be the **learned relationship**
between aligned shortcut interventions, packet-local semantic/content/
structural evidence, and cross-scale reuse. Protocol rigor and provenance are
required evidence, but are not substitutes for an algorithmic contribution.

### MIETT

MIETT already treats packets as instances in a flow bag, applies intra-packet and inter-packet attention, and introduces packet-relative-position and flow-contrastive pretraining. Therefore the following are not sufficient novelty claims here:

- hierarchical packet-to-flow modeling;
- two-level packet/window attention by itself;
- flow contrastive learning by itself;
- preserving packet order by itself.

Our differentiating question is whether one shortcut-resistant Packet representation contract can execute in both packet and flow tasks while task-local supervision learns different parameters, and whether counterfactual channel reliability is identifiable under per-flow split.

### DigTraffic

DigTraffic already combines length and timing channels, constructs heterogeneous message edges, and injects structural/edge encodings into graph attention. Therefore graph structure, edge attributes, or dual-channel fusion alone are not defensible contributions. The graph implementation remains an ablation unless the same graph definition and settings improve VPN and TLS under the strict protocol.

### TrafficFormer And Other Pretrained Encoders

Masked-byte or masked-flow pretraining, relative-order objectives, and generic protocol encoders are established ideas. The native pretraining module is supporting machinery. Its paper role is to supply a label-free current-packet content channel whose session fields are masked under a declared protocol, not to claim a new masked-modeling objective in isolation.

TrafficFormer also randomizes shortcut-prone header fields, including endpoint,
port, and sequence-related values, while retaining other header content. Header
randomization by itself is therefore not a contribution. The intervention claim
requires aligned factual/intervened views of the same packet, a bounded learned
reliability rule, and matched factual-only versus paired-view evidence under the
same Per-flow Split protocol.

### TrafficLLM

TrafficLLM already combines traffic-domain representations with an LLM and
studies heterogeneous tasks, masked meta-information, and adaptation to unseen
environments. Using Qwen, LoRA, multimodal traffic features, or feature masking
is not sufficient novelty. Our narrower differentiating hypothesis is an
auditable current-packet representation shared by Packet and Flow classifiers,
with independently trained task-local weights and cross-packet context confined
to the Flow aggregator. Claims of environment generalization still require an
explicit capture/domain-shift evaluation; ordinary random train/test accuracy
does not establish them.

### Flow-To-Packet Distillation And Bag Teachers

Flow-Packet Hybrid Traffic Classification (FPHTC, arXiv:2105.00074) already
transfers predictions from a flow-statistics teacher to a packet-level routing
policy, and ICMIL (arXiv:2312.01099) already studies a bag classifier as an
instance-level teacher. Ordinary flow-to-packet knowledge distillation,
training-time privileged context, or a bag-teacher/instance-student schedule is
therefore not a defensible contribution by itself.

If strict-v2 results expose a genuine Packet-to-Flow evidence gap, one
pre-registered follow-up may study **counterfactual learnability-gated
cross-scale distillation**. Its teacher must exclude the anchor packet, use only
same-training-flow context, and distill only the class relations that remain
stable across aligned factual/header-intervened views. The hypothesis is that
this gate suppresses privileged flow information that a current-packet student
cannot represent, while transferring environment-stable class geometry. It
must be implemented by the same optional training loss for Packet and Flow
tasks, keep Packet inference strictly single-packet, and beat plain KD plus no-KD
controls on held-out VPN and TLS validation before entering any main result.
This is a contingent candidate, not part of the current method.

Mechanism-oriented and causal distillation also have direct prior art. Wu et
al. (UniReps 2024) connect distillation to agreement on invariant outputs under
counterfactual latent changes and study Jacobian/contrastive mechanisms;
Dissanayake et al. (AISTATS 2025) use partial information decomposition to
separate task-relevant redundant teacher information from nuisance information.
Counterfactual invariance must additionally match the assumed causal structure
(Wang and Veitch, SCIS 2022), and suppressing all domain information can be
harmful when domain and class are dependent (Akuzawa et al., ICLR 2019). Thus
counterfactual KD, Jacobian matching, invariant-feature transfer, information
decomposition, or domain-information removal are not standalone novelty claims.

The network-specific candidate is deliberately narrower: an anchor-excluded
same-flow teacher supplies cross-scale evidence; aligned factual and
header-intervened predictions define which class relations are stable; and a
learnability/conflict gate prevents transfer when a strict current-packet
student cannot identify the teacher relation. The gate must be label-free at
test time and estimated from training/validation data only. Promotion requires
matched no-KD, plain-KD, stability-only KD, and learnability-gated KD controls,
plus evidence that any gain survives endpoint/session-novel strata on both VPN
and TLS. Otherwise this candidate remains a negative-result ablation.
The trigger, objective, cross-fitting protocol, promotion thresholds, and
falsification outcomes are frozen in
`counterfactual_cross_scale_distillation_preregistration.md`; that document
does not authorize implementation before strict-v2 satisfies its trigger.

Primary references for this boundary:

- Wu et al., "What Mechanisms Does Knowledge Distillation Distill?", UniReps
  2024: https://proceedings.mlr.press/v243/wu24a.html
- Dissanayake et al., "Quantifying Knowledge Distillation using Partial
  Information Decomposition", AISTATS 2025:
  https://proceedings.mlr.press/v258/dissanayake25a.html
- Wang and Veitch, "A Unified Causal View of Domain Invariant Representation
  Learning", SCIS 2022: https://openreview.net/forum?id=-l9cpeEYwJJ
- Akuzawa et al., "Infinite-dimensional Feature Learning in the Presence of
  Domain and Category Shift", ICLR 2019:
  https://openreview.net/forum?id=HJx38iC5KX

### SWEET

SWEET establishes that per-packet split leaks explicit and implicit flow identifiers, that downstream header removal is a model-design choice, and that a simple Pcap-Encoder can be strong. SWEET also uses simple aggregation of early packet predictions for flow classification. Therefore neither IP/port masking nor packet voting alone is novel.

The required comparison is protocol matched: per-flow split, three train/validation folds, one shared test set, accuracy and macro-F1, frozen versus end-to-end distinctions preserved, and no test-selected candidate.

### Class And Sample Reweighting

Effective-number class-balanced loss (Cui et al., CVPR 2019), class-wise
difficulty-balanced loss (Sinha et al., ACCV 2020), influence-balanced loss
(Park et al., ICCV 2021), and validation-gradient example reweighting (Ren et
al., ICML 2018) already establish that class counts alone need not determine
training weight and that difficult, noisy, or high-influence examples may need
different treatment. Therefore flow-count weighting, class-difficulty
weighting, sample uncertainty weighting, gradient-based reweighting, or
identifiability-normalized CE are not standalone novelty claims.

The VPN fold-0 diagnostic currently shows that session-signature conflict is
associated with the *change* caused by flow-count weighting, while weight
magnitude itself is not. This is descriptive validation evidence, not a causal
result and not a promotion rule. No identifiability-aware weighting candidate
may enter the shared core unless the same matched weighted/unweighted
association is evaluated on TLS and a pre-registered, class-mass-preserving
form improves both datasets. Even then, its role is supporting risk design;
the paper cannot claim generic sample reweighting as novel.

Primary references for this boundary:

- Cui et al., "Class-Balanced Loss Based on Effective Number of Samples",
  CVPR 2019:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html
- Sinha et al., "Class-Wise Difficulty-Balanced Loss for Solving
  Class-Imbalance", ACCV 2020:
  https://openaccess.thecvf.com/content/ACCV2020/html/Sinha_Class-Wise_Difficulty-Balanced_Loss_for_Solving_Class-Imbalance_ACCV_2020_paper.html
- Park et al., "Influence-Balanced Loss for Imbalanced Visual
  Classification", ICCV 2021:
  https://openaccess.thecvf.com/content/ICCV2021/html/Park_Influence-Balanced_Loss_for_Imbalanced_Visual_Classification_ICCV_2021_paper.html
- Ren et al., "Learning to Reweight Examples for Robust Deep Learning", ICML
  2018: https://proceedings.mlr.press/v80/ren18a.html

### Identity Aliases In Flow-Aware Contrastive Sampling

The current Packet-task sampler visits flows uniformly and requests two packet
rows per selected flow. A flow containing only one packet must therefore copy
that row when replacement is enabled. Standard supervised contrastive loss
excludes the anchor only by batch index and treats every remaining same-label
index as a positive. It does not know that two distinct indices may denote the
same original packet. The copied row can consequently become both a trivial
same-flow positive and a denominator entry. This is an **objective-level
identity alias** induced by the interaction of fixed-cardinality flow sampling
and the contrastive relation definition; it is not evidence that SupCon itself
was implemented incorrectly.

The exact eight-epoch replay audit quantifies why this matters before any model
change is attempted:

| Dataset | Singleton flows | Copied batch rows | Positive mass removed by identity deduplication | Identity-safe anchor positive coverage |
| --- | ---: | ---: | ---: | ---: |
| VPN Packet fold 0 | 57.14% | 28.57% | 56.45% | 82.44% |
| TLS-120 Packet fold 0 | 28.85% | 14.42% | 28.72% | 84.19% |

This observation has a strict prior-work boundary. Khosla et al. define
multi-positive SupCon over all same-label batch indices and show that easy
positives contribute smaller gradients. Robinson et al. establish more broadly
that pair construction and instance-discrimination difficulty can create
contrastive shortcuts and feature suppression. MIETT already defines packets
from one flow as positives and packets from different flows as negatives.
Therefore identity deduplication, harder positive mining, same-class pairing,
or flow-aware contrastive learning is not a standalone novelty claim.

The only admissible follow-up is a pre-registered **relation-complete
identity-safe flow sampler**: it must prevent one physical packet from
occupying multiple semantic roles, preserve useful same-flow or same-class
positive coverage without replacement aliases, and use one rule on VPN and
TLS. The ordered validation screen is:

1. random flow pairing with the existing loss versus the same batches with
   physical-packet identity masking;
2. only if identity masking passes, random pairing versus identity-safe
   same-class flow pairing;
3. promotion only for a validation gain on both VPN and TLS Packet, followed by
   Flow non-inferiority under the same relation rule. Packet promotion requires
   at least `+0.005` Macro-F1 on each dataset with no Accuracy drop over
   `0.005`; Flow non-inferiority permits at most `0.005` loss in either metric
   on either dataset. D2 is compared only with promoted D1.

The audit-only same-class pairing replay raises identity-safe positive coverage
to 99.95% on VPN and 99.96% on TLS while reducing alias-positive mass to 21.77%
and 14.06%, respectively. These are sampling diagnostics, not accuracy results,
and do not authorize promotion before the currently frozen balance/paired-view
screen finishes. If the matched model experiment fails, the mechanism remains
a documented negative result rather than part of the unified method.

Primary references for this boundary:

- Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020:
  https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html
- Robinson et al., "Can Contrastive Learning Avoid Shortcut Solutions?",
  NeurIPS 2021:
  https://proceedings.neurips.cc/paper/2021/hash/27934a1f19d678a1377c257b9a780e80-Abstract.html
- Wang et al., "MIETT: Multi-Instance Encrypted Traffic Transformer", AAAI
  2025: the uploaded paper, Section 3.3 Flow Contrastive Learning.

## Defensible Method Hypothesis

The current hypothesis is a unified counterfactual Packet-to-Flow representation contract:

1. A task-local semantic channel is learned with Qwen LoRA from the task's own training packets under a strict single-current-packet prompt policy.
2. A label-free protocol-content channel masks endpoint, checksum, sequence, and session fields.
3. A strict packet-local structural channel exposes declared behavior features without cross-packet context.
4. Factual and endpoint-intervened semantic views are fused before a semantic-anchored, bounded tri-channel router.
5. Packet classification adds only a shared linear classifier-head class.
6. Flow classification applies the same Packet module to every packet, then adds window/flow aggregation and a flow head.
7. A validation-only candidate may add a bounded residual from a fresh Flow-trained packet-evidence head; it is not part of the method unless it beats a matched late-fusion control on both datasets.

The potentially novel object is the complete contract and its testable
counterfactual-reliability hypothesis, not any single encoder, loss, or pooling
operation. The current implementation does not by itself establish the
stronger reliability claim defined below.

### Current Reliability-Claim Limit

The implemented `SharedInterventionViewFusion` observes the factual view, the
intervened view, their signed difference, and their absolute difference. It
starts at the symmetric mean and permits only a bounded sample-dependent
residual. This proves that the intervention can condition the mixture and that
neither view can be assigned an unbounded shortcut path.

It does **not** yet prove that the router weight estimates view reliability.
The router currently receives only the downstream task gradient; no loss
directly identifies which leave-one-view prediction has lower conditional
risk. Moreover, Tower-1 paired consistency can make the two views similar and
therefore reduce the very disagreement used by the router. Until additional
evidence exists, the allowed description is **bounded intervention-conditioned
fusion**, not identifiable counterfactual reliability learning.

A stronger reliability claim requires, on held-out validation and without
changing the test decision rule:

- per-view factual-only and intervened-only losses from the same checkpoint;
- calibration between the learned gate and signed per-view excess loss;
- non-degenerate gate variation after paired Tower-1 consistency;
- matched fixed-mean, ordinary random-augmentation, and learned-router
  controls on both VPN and TLS for both Packet and Flow;
- a pre-registered training objective if explicit reliability supervision is
  introduced later.

These diagnostics may falsify the reliability interpretation. They must not be
used to tune a test-set gate or to rename ordinary mixture-of-experts routing
as a causal estimator.

## Required Evidence

- Exact class/schema audit showing both tasks execute `SharedPacketRepresentationEncoder`, `SharedInterventionViewFusion`, `SharedPacketChannelFusion`, and the same classifier-head class where applicable.
- Input-scope audit proving factual/intervened semantic prompts and exported native content are current-packet only in both tasks; neighboring packets may enter only after `packet_to_flow_proj`.
- Manifest audit proving Packet-task and Flow-task supervised weights were trained from their own splits and were not transferred across tasks.
- Fixed three-fold equal log-mean results for both tasks and both datasets.
- Flow-cluster/content-group bootstrap confidence intervals, not only point estimates.
- Reporting-only train-seen versus train-novel endpoint/port/session strata for
  every VPN/TLS Packet/Flow result. Three-fold consensus uses the union of all
  member training signatures; test labels never define strata or select models.
- Matched ablations for protocol content, endpoint intervention, bounded router, and aggregation.
- The pre-registered core matrix uses persistent training toggles for no-semantic,
  no-content, no-structural, factual-only, fixed equal fusion, and row risk.
  Fold0 decisions use validation only; module removal requires three-fold
  non-inferiority on both datasets and both tasks.
- Three-arm packet-evidence ablation: strict-v2 mean reference, late-fusion control with evidence disabled, and identical late-fusion candidate with evidence enabled.
- Runtime gate diagnostics showing finite bounded weights and non-degenerate sample variation.
- Cost table for parameters, training time, inference throughput, and memory. This is evidence of practical trade-offs, not the main novelty.

## Stop Rules

- Do not promote a module that helps only VPN or only TLS.
- Do not call a pooling change a packet-evidence gain.
- Do not call field randomization, Qwen/LoRA use, or multimodal fusion a novel
  method without the aligned intervention, bounded-routing, and cross-task
  contract ablations.
- Do not report historical best-of-many test results as the unified framework result.
- Do not add distillation until the teacher method and matched ablations are frozen.
- Do not describe a flow/bag teacher as novel; any distillation follow-up must
  isolate the counterfactual learnability gate against plain KD and no-KD.
- Do not describe class/sample reweighting as novel; any identifiability-aware
  risk follow-up must preserve per-class objective mass, use one VPN/TLS rule,
  and remain a supporting component even if it passes validation.
- Do not claim CCF-A readiness until strict provenance, protocol-matched SWEET comparison, grouped confidence intervals, and all required ablations pass.
