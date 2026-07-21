# Counterfactual Cross-Scale Distillation Preregistration

Status: contingent follow-up; not part of the current unified method.

## Trigger

Implementation is allowed only after the frozen `strict_shared_core_v2` matrix
has complete three-fold VPN/TLS Packet and Flow results, passing cross-task
provenance audits, and shows a reproducible Packet-to-Flow evidence gap. Test
labels cannot trigger, configure, select, or stop this experiment.

A reproducible evidence gap means both of the following hold on held-out
validation data in at least one dataset:

1. an anchor-excluded same-flow teacher improves macro-F1 over the strict
   current-packet predictor by at least 0.02; and
2. at least 10% of validation packets have a correct, confident teacher and an
   incorrect current-packet prediction.

If neither VPN nor TLS passes this trigger, do not implement the loss.

## Shared Training Contract

The same optional loss and hyperparameter-selection rule must be available to
VPN/TLS Packet and Flow training. Each task trains its own parameters from its
own training split. Flow-task supervision cannot initialize a Packet-task
model, and Packet-task supervision cannot initialize a Flow-task model.

For anchor packet `i` from training flow `f`, the student receives only the
strict current-packet channels already defined by shared-core v2. The teacher
receives all packets in `f` except `i` from the same training flow. No
validation/test packet, test-derived prior, endpoint lookup table, or cross-flow
neighbor is allowed. Packet inference remains one current packet. Flow
inference remains the shared Packet module followed by the standard Flow
aggregator.

The context teacher is an exponential-moving-average copy of the same shared
Packet core followed by the standard Flow aggregator and classifier-head
class. It is trained only from the current task's training flows and removed at
inference. For each anchor, context packets preserve their observed order but
exclude the anchor before aggregation. A flow with fewer than two eligible
packets receives `w_i = 0`. The maximum context size and deterministic sampling
rule are fixed once for VPN/TLS and cannot be selected separately by dataset.
Teacher logits and gates are stop-gradient targets; student gradients cannot
update the teacher outside the declared EMA update.

## Candidate Objective

Let `p_i^F` and `p_i^I` be student class distributions for factual and
header-intervened views. Let `q_-i^F` and `q_-i^I` be anchor-excluded teacher
distributions. Define teacher stability

```text
s_i = exp(-JS(q_-i^F, q_-i^I) / tau_s)
```

and a training-only learnability score

```text
l_i = stopgrad(g(phi_i))
```

where `phi_i` contains only label-free agreement, entropy, margin, and
factual/intervened representation-distance features available during training.
The gate is bounded by `w_i = min(w_max, s_i * l_i)`. It cannot consume the
ground-truth label, sample identity, endpoint identity, or test statistics.

Only teacher class relations stable across the intervention are transferred.
With centered log-probabilities `r(p) = log(p) - mean(log(p))`, use

```text
R_-i = 0.5 * (r(q_-i^F) + r(q_-i^I))
L_ccsd = mean_i w_i * [
    Huber(r(p_i^F), R_-i) +
    Huber(r(p_i^I), R_-i)
]
L_total = L_shared_core_v2 + lambda_ccsd * L_ccsd
```

Relation distillation is used instead of copying the teacher argmax so that
the objective transfers stable class geometry without requiring the current
packet to reproduce all privileged flow evidence.

## Learnability Gate Identification

The gate must be fitted without test labels. A permissible implementation uses
cross-fitting inside the training split:

1. partition training flows into inner folds;
2. produce out-of-inner-fold student and anchor-excluded teacher predictions;
3. define a binary target indicating whether teacher guidance reduces student
   cross-entropy on that held-out inner fold;
4. fit the small gate `g` from label-free `phi_i` to that target;
5. freeze or jointly fine-tune the gate on the outer training fold without
   exposing outer validation labels to gradient updates.

The gate is a shared algorithm, not a dataset-specific threshold. Dataset/task
specific gate weights may be learned. The gate architecture, feature list,
EMA coefficient, context-size candidate set, `w_max` candidate set, and
selection metric are fixed across datasets/tasks.

## Required Controls

All arms use identical data, initialization policy, optimizer budget, shared
core, validation metric, and random seeds:

1. `no_kd`: strict shared-core v2;
2. `plain_kd`: anchor-excluded teacher without intervention stability or
   learnability gating;
3. `stability_kd`: intervention-stability weighting without learnability gate;
4. `ccsd`: stability plus cross-fitted learnability gate.

An additional `inclusive_teacher` diagnostic may quantify anchor leakage, but
cannot be a promotion candidate.

## Promotion Rule

Freeze one configuration using fold-0 validation from both VPN and TLS. Promote
only if `ccsd`, relative to `no_kd`, satisfies all conditions on both datasets:

- macro-F1 gain at least 0.005;
- accuracy drop no greater than 0.005;
- no degradation greater than 0.01 on endpoint-novel or session-novel
  validation strata;
- improvement over `plain_kd` and `stability_kd` in mean macro-F1;
- finite gates, non-degenerate gate variance, and mean effective weight below
  `0.75 * w_max`;
- the same selected configuration survives all three folds.

After selection is frozen, evaluate the shared test set once per fold and use
the predeclared equal log-mean consensus. Report flow/content-group clustered
bootstrap confidence intervals. Failure on either dataset keeps this method as
a negative-result ablation.

## Falsification Outcomes

- If plain KD equals CCSD, the learnability mechanism is unsupported.
- If stability-only KD equals CCSD, the learnability gate is unsupported.
- If gains occur only on train-seen sessions, the claimed shortcut resistance
  is unsupported.
- If Flow improves but Packet degrades materially, the shared cross-task loss
  claim is unsupported.
- If only TLS or only VPN improves, the candidate is not a unified main module.
- If an inclusive teacher is substantially stronger than the anchor-excluded
  teacher, the apparent gain is likely anchor leakage rather than transferable
  cross-scale evidence.

## Novelty Boundary

Privileged-information distillation, flow-to-packet teachers, bag-to-instance
distillation, counterfactual invariance, Jacobian matching, and information
decomposition all have prior art. Any defensible contribution must be the
network-specific combination of anchor exclusion, aligned field intervention,
relation-level stability, strict current-packet inference, and an empirically
identified learnability gate, supported by the controls above. No individual
component should be claimed as novel.
