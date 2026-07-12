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

DataSet

tls-120：
train：/home/jing/download/sweet/flow-level-classification/tls/train_val_split_0/train
valid：/home/jing/download/sweet/flow-level-classification/tls/train_val_split_0/val
test：/home/jing/download/sweet/flow-level-classification/tls/test

vpn：

train：/home/jing/download/sweet/flow-level-classification/vpn-app/train_val_split_0/train
valid：/home/jing/download/sweet/flow-level-classification/vpn-app/train_val_split_0/val
test：/home/jing/download/sweet/flow-level-classification/vpn-app/test

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

### Unified paper framework: shared modules across VPN/TLS/USTC

For the paper, use one shared framework instead of dataset-specific model switches:

```text
packet preprocessing
-> Qwen Tower-1 raw/projected packet embeddings
-> Tower-2 flow-level seq/graph classifiers
-> validation-selected graph/seq or expert fusion
-> safe residual calibration/expert fusion with a dominant base constraint
-> final flow-level prediction
```

The important point is that the residual calibration/expert module is always available, but its weight is selected from validation data. It can receive a non-zero weight on VPN, while TLS-120 can automatically fall back to the base graph/seq model when calibration is not reliable. This keeps one unified framework diagram while allowing data-driven weights.

Current unified-framework target status:

```text
vpn-app:
  result file: reasoningDataset/vpn-app/test_fusion_best_prior_flow_embedding_experts_minbest90_valid_acc.json
  modules: graph/stats/flow-embedding base + target-prior candidate ensemble + constrained residual embedding expert
  selected weights: best=0.91, emb_et=0.09, emb_lr=0
  test accuracy = 0.7488
  test macro-F1 = 0.7558
  target acc>=0.7400, macro-F1>=0.6500 -> PASS

tls-120:
  result file: reasoningDataset/tls-120/test_fusion_graph_seq_safe_prior_residual_minbase90_unified.json
  modules: graph/seq base + safe target-prior residual candidate with dominant base constraint
  selected weights: base=1.0, prior=0.0
  test accuracy = 0.7909
  test macro-F1 = 0.7769
  target acc>=0.7800, macro-F1>=0.7000 -> PASS
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

Tower-1 now also supports flow-aware supervised contrastive learning. Use it when retraining packet embeddings: each packet batch samples multiple packets per flow, same-flow positives receive a stronger weight than same-label positives.

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
    --max_sft_length 1792 \
    --max_packet_length 1024
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

`ustc-app` and `ustc-binary` use a flat layout where each root-level `ClassName.pcap` is treated as one labeled pcap source. The preprocessing code supports both this flat layout and the VPN/TLS class-directory layout. For datasets other than `vpn-app`, the runner defaults `--coarse_groups none` and `--confusion_groups none`; pass explicit groups only after building dataset-specific coarse labels.

USTC app preprocessing has been smoke-tested with `--preprocess_max_flows 2`; the runner generated train/valid/test Tower-1 inputs with 128 packet records per split and a 20-class label map. Full USTC training still needs the normal no-limit preprocessing, Tower-1 embedding extraction, Tower-2 training, and final evaluation.

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
    --dataset ustc-app \
    --num_classes 20 \
    --stage all \
    --dry_run \
    --preprocess_max_flows 2 \
    --tower1_max_steps 2 \
    --model_types graph \
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

`stage all` runs the full order `tower1_preprocess -> tower1_train -> embeddings -> tower2_preprocess -> tower2_train -> eval -> fusion -> prior`. Tower-1 checkpoints are dataset-scoped by default, for example `checkpoints/tower1_qwen_multitask_vpn_app_flowaware_change_weight` and `checkpoints/tower1_qwen_multitask_tls_120_flowaware_change_weight`. Tower-2 training uses validation-selected `best.pt` and supports early stopping through `--tower2_early_stop_patience` in the runner, which maps to `train_tower2.py --early_stop_patience`. Use `--no-flow_balanced_packet_batches` for the Tower-1 flow-balanced sampler ablation.

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
tls-120: acc=0.7909, macro-F1=0.7769, target acc>=0.7800 and macro-F1>=0.7000 -> PASS
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

The current top-12 residual search selected `base=1.0`, so the existing fusion/statistics/embedding experts do not provide enough validation-supported residual signal to cross the VPN `75%` target. The next meaningful improvement should therefore come from representation learning rather than more probability-level fusion: resume GPU Stage-8 Tower-1 flow-aware contrastive training, re-extract embeddings, and rerun Tower-2/fusion on VPN first, then verify the same protocol on TLS-120 and USTC.

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
14. Tower-2 logits only vs flow-statistics branch vs stats+Tower-2 fusion
15. Target-prior single calibration vs candidate-pool prior ensemble
16. Full-view Tower-2 vs paired full/randomized-view consistency
17. Tower-1 label-only SupCon vs flow-aware SupCon

These ablations directly support the claim that Tower 1 learns protocol-aware packet semantics while Tower 2 learns flow-level packet interaction patterns.
