# Conditional-Risk Router Preregistration

Status: pending the frozen base-milestone Packet/Flow development Test and
held-out gate diagnostics. This registration does not promote or enable the
candidate.

## Hypothesis

The factual and endpoint-intervened semantic observations have sample-dependent
reliability. The current bounded router is trained only through classification,
so a varying gate is not necessarily an identifiable reliability estimate. A
task-local leave-one-view conditional-risk target may identify that gate while
preserving the same Packet/Flow inference graph.

## Frozen Candidate

For prediction unit `u`, obtain final factual-duplicated and
intervened-duplicated logits from the same task-local model. Let their ordinary,
unweighted per-unit cross-entropies be `l_f(u)` and `l_i(u)`. The detached
factual-reliability target and only candidate-specific loss are

```text
q_f(u) = sigmoid(stop_gradient((l_i(u) - l_f(u)) / 1.0))
L_reliability = 0.1 * BCE(r_f(u), q_f(u))
```

`r_f` is the raw factual router probability before the bounded effective-weight
map. Temperature `1.0` and coefficient `0.1` are fixed, not selected from the
new Test. The ordinary classification loss, bounded mixture, optimizer family,
heads, and inference graph remain unchanged.

Packet uses one current Packet and one Packet label per prediction unit. Flow
uses the final one-Flow logits after the unchanged Packet encoder, sequence or
window aggregator, and Flow head; a window label is not a valid substitute.
Each dataset/task trains all supervised weights independently from its own
training split.

## Teacher And Alignment Semantics

Training-only teacher paths share the student's parameters, run under
`no_grad` with dropout disabled, and restore training mode before student
backpropagation. They consume unaugmented aligned observations. Applying class
weights to `l_f/l_i` is forbidden because that would introduce a class-specific
reliability temperature.

The factual teacher duplicates the factual semantic observation into both view
inputs. The intervened teacher duplicates the intervened semantic observation.
Native content and strict current-Packet structural inputs remain identical
between teachers. For Flow, only the semantic slice may change; content and
structural slices are copied from the factual input and asserted equal before
both teachers traverse the same Flow aggregator and head.

For the differentiable Flow student gate, valid gate rows are scattered to
their source Packet positions through recorded `window_ranges`. Repeated
observations are summed and divided by occurrence counts, yielding one raw gate
per unique source Packet; unique-Packet gates are then mean-pooled to one Flow
gate. Padding is excluded. Missing or inconsistent ranges are hard errors.
Tests must prove that duplicating an overlapping interior window does not alter
the resulting Flow gate.

## Promotion Gate

Selection and checkpoint choice use Valid only. A matched base/candidate run
must keep data, seed policy, architecture width, optimizer family, and training
budget fixed.

- Accuracy and Macro-F1 deltas must be non-negative on VPN and TLS for Packet
  and Flow.
- At least two of the four primary cells must gain at least `0.005` Macro-F1.
- No cell may lose more than `0.003` Accuracy.
- Gate association with signed leave-one-view excess loss must have Pearson and
  Spearman correlations of at least `0.10`.
- Factual effective weight must differ by at least `0.02` between top and
  bottom advantage quintiles.
- Flow-cluster bootstrap Pearson 95% lower confidence bound must be positive.
- Gains restricted to one dataset, one task, or a larger inference graph reject
  the candidate.

After Valid promotion, one tagged development Test is allowed. If it informs a
later method change, final claims require a fresh outer holdout or nested cross-
validation.

## Required Controls

1. Symmetric mean without a learned residual.
2. Current bounded router with classification gradients only.
3. Ordinary factual/intervened random augmentation without risk distillation.
4. Conditional-risk-distilled bounded router.

## Stop Rule

Do not search a loss-weight grid. Use the registered default and permit at most
one scale-normalized correction only when a pre-update gradient audit proves
the default numerically inactive. Failure of the matched Valid gate records a
negative ablation and ends this direction.
