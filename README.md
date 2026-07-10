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

---

## 0. Dataset format

Each pcap file is treated as one flow. The class label is the subfolder name.

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
  --output_dir reasoningDataset/vpn-app/train_tower2_rawproj_change_weight \
  --window_size 32 \
  --stride 16
```

### Valid/test data

```bash
python preprocess_tower2.py \
  --flow_embedding_index reasoningDataset/vpn-app/valid_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
  --output_dir reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight \
  --window_size 32 \
  --stride 16

python preprocess_tower2.py \
  --flow_embedding_index reasoningDataset/vpn-app/test_embeddings_rawproj_change_weight/flow_embedding_index.jsonl \
  --output_dir reasoningDataset/vpn-app/test_tower2_rawproj_change_weight \
  --window_size 32 \
  --stride 16
```

Outputs:

```text
seq_dataset.pt
 graph_dataset.pt
```

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

## 6. Train Tower 2: staged sequence Transformer experiments

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

## 7. Train Tower 2: staged graph Transformer experiments

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

`--train_level flow` groups windows by `flow_id`, pools window embeddings with the trainable `--flow_pooling` head, and optimizes the flow label directly. `--window_loss_weight` keeps the original window classifier supervised during flow-level training. `--class_weighting effective` enables class-balanced CE. `--flow_contrastive_weight` adds supervised contrastive learning on pooled flow embeddings, and `--balanced_flow_batches` makes SupCon batches contain same-class positives. `--flow_pooling late_fusion` combines the trainable flow head with mean window logits.

`--select_metric flow_macro_f1` saves `best.pt` by validation macro-F1 instead of validation accuracy. `--hierarchical_weight` adds a coarse-label loss, while `--hierarchical_logit_weight` adds the coarse log-probability back to each fine-class logit at train/test time. `--contrastive_mode confusion` uses only configured same-group hard negatives in SupCon instead of pushing against every different class.

Stage 6 replaces the flat fine classifier with true coarse-to-fine expert heads, weights SupCon negatives by a validation confusion matrix, and uses a flow-level Transformer over window embeddings. Generate the validation confusion file from Stage 5 first; do not use the test metrics JSON for training.

```bash
python test_tower2.py \
  --checkpoint checkpoints/tower2_graph_flow_rawproj_change_weight_macro_hier_conf_supcon/best.pt \
  --dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt \
  --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
  --output_json reasoningDataset/vpn-app/valid_graph_metrics_flow_rawproj_change_weight_macro_hier_conf_supcon.json

python train_tower2.py \
  --model_type graph \
  --dataset reasoningDataset/vpn-app/train_tower2_rawproj_change_weight/graph_dataset.pt \
  --valid_dataset reasoningDataset/vpn-app/valid_tower2_rawproj_change_weight/graph_dataset.pt \
  --output_dir checkpoints/tower2_graph_flow_rawproj_change_weight_expert_weighted_supcon_flowtrans \
  --num_classes 16 \
  --epochs 30 \
  --batch_size 16 \
  --hidden_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --train_level flow \
  --select_metric flow_macro_f1 \
  --flow_pooling transformer \
  --flow_transformer_layers 1 \
  --flow_transformer_heads 4 \
  --window_loss_weight 0.3 \
  --class_weighting effective \
  --class_weight_beta 0.9999 \
  --hierarchical_mode expert \
  --hierarchical_weight 0.2 \
  --coarse_groups vpn_app \
  --balanced_flow_batches \
  --samples_per_class 2 \
  --contrastive_mode confusion_weighted \
  --confusion_groups vpn_app \
  --confusion_matrix_json reasoningDataset/vpn-app/valid_graph_metrics_flow_rawproj_change_weight_macro_hier_conf_supcon.json \
  --confusion_matrix_level flow \
  --confusion_weight_power 1.0 \
  --flow_contrastive_weight 0.03 \
  --flow_temperature 0.07 \
  --aux_weight 0 \
  --coherence_weight 0
```

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

## 8. Test

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

### Expert heads + weighted SupCon + flow Transformer

```bash
python test_tower2.py \
  --checkpoint checkpoints/tower2_graph_flow_rawproj_change_weight_expert_weighted_supcon_flowtrans/best.pt \
  --dataset reasoningDataset/vpn-app/test_tower2_rawproj_change_weight/graph_dataset.pt \
  --label_map reasoningDataset/vpn-app/train_tower1_change_weight/label_map.json \
  --output_json reasoningDataset/vpn-app/test_graph_metrics_flow_rawproj_change_weight_expert_weighted_supcon_flowtrans.json
```

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

Metrics include:

```text
Window-level Accuracy / Precision / Recall / F1
Flow-level Accuracy / Precision / Recall / F1
```

---

## 9. Suggested ablations for the paper

1. Tower1-QA only: `L_QA`
2. Tower1-QA + packet classification: `L_QA + alpha L_packet_cls`
3. Tower1-QA + SupCon: `L_QA + beta L_supcon`
4. Full Tower1: `L_QA + alpha L_packet_cls + beta L_supcon`
5. Tower2 sequence Transformer vs Graph Transformer
6. Raw last-token embedding vs projection-head embedding vs raw+projected concatenation
7. Window-level Tower-2 training vs flow/window dual-loss Tower-2 training
8. Class-balanced CE off vs on
9. Balanced SupCon off vs on
10. Flow pooling: mean vs attention vs late_fusion
11. Best checkpoint by flow accuracy vs flow macro-F1
12. Flat 16-class classifier vs hierarchical coarse-to-fine classifier
13. Standard SupCon vs confusion-aware SupCon

These ablations directly support the claim that Tower 1 learns protocol-aware packet semantics while Tower 2 learns flow-level packet interaction patterns.
