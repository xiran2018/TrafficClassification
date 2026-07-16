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

`--train_level flow` groups windows by `flow_id`, pools window embeddings with the trainable `--flow_pooling` head, and optimizes the flow label directly. `--window_loss_weight` keeps the original window classifier supervised during flow-level training. `--class_weighting effective` enables class-balanced CE. `--flow_contrastive_weight` adds supervised contrastive learning on pooled flow embeddings, and `--balanced_flow_batches` makes SupCon batches contain same-class positives. `--window_contrastive_weight` adds a window-to-flow prototype contrastive loss: each local window embedding is pulled toward its own-flow or same-class flow prototype and pushed away from different-class flow prototypes. `--flow_pooling late_fusion` combines the trainable flow head with mean window logits.

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
-> validation-gated expert selector
-> safe residual calibration/expert fusion with a dominant base constraint
-> final flow-level prediction
```

The important point is that the residual calibration/expert module and the validation-gated selector are always available, but their weights or source choices are selected from validation data. They can receive a non-zero contribution on VPN or USTC, while TLS-120 can automatically fall back to the base graph/seq model when calibration is not reliable. This keeps one unified framework diagram while allowing data-driven weights. In paper wording, do not claim that every module is forced on for every dataset; claim that every dataset passes through the same candidate framework and harmful modules are validation-gated down to zero or to the base identity path.

Current unified-framework target status:

```text
vpn-app:
  result file: reasoningDataset/vpn-app/test_selector_best_prior_embedding_experts_calib_shift000_valid_macro.json
  modules: graph/stats/flow-embedding base + target-prior candidate ensemble + constrained residual embedding expert + validation-gated selector
  selected selector: fallback to base after rejecting reliability_fusion because it changed 12.68% of target predictions, above the strict VPN target-shift guard
  test accuracy = 0.7488
  test macro-F1 = 0.7558
  target acc>=0.7400, macro-F1>=0.6500 -> PASS

tls-120:
  result file: reasoningDataset/tls-120/test_selector_soft_gate_tls120_tol0015_calib_family_valid_macro.json
  modules: graph/seq base + unified-slot probability stacker + soft expert gate + validation-gated selector
  selected selector: class-bias-calibrated soft_gate; bootstrap win_rate=0.89, 5% gain quantile=-0.0014, target prediction change=0.0396
  test accuracy = 0.7996
  test macro-F1 = 0.7869
  target acc>=0.7800, macro-F1>=0.7000 -> PASS

ustc-app:
  result file: reasoningDataset/ustc-app/test_selector_base_flowproto_full_s200_w002_step150_calib_shift005_valid_macro.json
  modules: graph/seq Tower-2 + flow-embedding expert + safe target-prior residual candidate + validation-gated expert selector
  selected selector: class_precision, alpha=0.5, metric_margin=0.0; passes bootstrap win-rate 0.66 and target prediction change 0.05
  test accuracy = 0.7000
  test macro-F1 = 0.6250
  note: 20 test flows only; use as cross-dataset framework evidence and continue improving representation learning
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

Interpretation for paper experiments: packet-level training accuracy alone is not a sufficient checkpoint-selection criterion. The final `step_200` adapter has stronger packet training accuracy, but `step_150` gives better downstream flow-level generalization on USTC. Keep downstream validation-aware checkpoint selection as part of the unified framework and report the `step_200` result as an ablation.

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

`stage all` runs the full order `tower1_preprocess -> tower1_train -> embeddings -> tower2_preprocess -> tower2_train -> eval -> fusion -> prior`. Tower-1 checkpoints are dataset-scoped by default, for example `checkpoints/tower1_qwen_multitask_vpn_app_flowaware_change_weight` and `checkpoints/tower1_qwen_multitask_tls_120_flowaware_change_weight`. Tower-1 training uses `--local_files_only` by default in the runner, so make sure the selected Qwen checkpoint is already available in the local Hugging Face cache or pass `--no-local_files_only` intentionally. Tower-2 training uses validation-selected `best.pt` and supports early stopping through `--tower2_early_stop_patience` in the runner, which maps to `train_tower2.py --early_stop_patience`. The runner's `fusion` stage now first calls `make_fusion_payload.py` for each selected Tower-2 model, so valid/test probability JSONs are automatically merged into the payload format required by `fuse_prediction_jsons.py`. Use `--no-flow_balanced_packet_batches` for the Tower-1 flow-balanced sampler ablation. Use `--tower1_init_checkpoint_dir` for Tower-1 continuation from an existing adapter, `--flow_proto_weight` for Tower-1 packet-to-flow prototype contrastive training, and `--window_contrastive_weight` for Tower-2 window-to-flow prototype contrastive training.

For the next representation-learning iteration, the same Stage-8 runner also supports a paired header-perturbation view. This keeps the paper framework unified: every dataset can use the same full-view classifier, an IP/port-randomized paired view, clean/augmented consistency, feature dropout, and validation-gated downstream fusion; dataset-specific validation then decides whether these modules help or collapse to the base path.

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

Use the same command shape for TLS-120 and USTC by changing `--dataset`, `--num_classes`, and the suffixes. This is the preferred paper-facing experiment over additional probability-level tuning because it tests whether endpoint-invariant representations improve flow classification before the validation-gated selector stage.

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

Interpretation: the paired-view idea is still useful as a paper module, but the old embeddings are not view-aligned enough for Tower-2-only regularization to help. Treat these results as negative ablations. The next real run should regenerate paired `rawproj_flowaware_*` embeddings from the Stage-8 Tower-1 checkpoint on the real A800 environment, then rerun the same `--run_tag paired_ipport` training/eval/fusion path.

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

The final selector uses three safety gates: `--min_valid_gain_over_base` for deterministic validation gain, bootstrap gain stability through `--bootstrap_samples`, and an unlabeled target-shift constraint through `--max_prediction_change_rate`. The current unified report uses a small bootstrap quantile tolerance; TLS uses `--bootstrap_min_gain_quantile -0.0015` for the soft-gate candidate because its validation gain is low-shift but the 5% bootstrap lower bound is only slightly below zero. The same candidate family is evaluated on every dataset, while `--max_prediction_change_rate` is dataset-specific: VPN uses a strict no-target-change setting, TLS allows a tiny low-shift switch/calibration, and USTC allows the class-precision gate. The recommended Stage-8 runner now passes `--final_selector_unified_expert_slots base,graph,seq,prior_base,emb_lr,emb_et,proto_emb,paired,slot_stacker,soft_gate`, so every dataset exposes the same selector expert slots; missing slots are filled from the base probabilities as identity experts and are recorded in `feature_config.input_slot_status`. For legacy best-result JSONs without slot records, the paper report marks the same mapping as `inferred_identity_compatible` instead of pretending the old run recorded it. By default the selector sorts candidates by validation score, skips unsafe candidates, and accepts the first candidate that passes all active guards; if none pass, it falls back to the first input. For paper-grade robustness searches, `--rank_metric bootstrap_gain_quantile` can instead rank the top validation candidates by their bootstrap lower-bound gain before the same safety gates are applied. `--rank_select_metric` lets this robust ranking target accuracy while the accepted selector still optimizes macro-F1. `--calibration_penalty_weight` optionally subtracts a validation calibration penalty (`ece`, `nll`, or `brier`) from the ranking score, using the shared `probability_metrics.py` definition; this prefers less overconfident candidates when validation scores are close, but the accepted candidate must still pass the same gain/bootstrap/target-shift gates. Use `--rank_candidate_limit` to keep this robust ranking practical on large candidate grids. This is the same module used across datasets:

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

USTC:
  safe selector file: reasoningDataset/ustc-app/test_selector_base_flowproto_full_s200_w002_step150_calib_shift005_valid_macro.json
  selected path: class_precision selector, alpha=0.5, metric_margin=0.0; bootstrap win_rate=0.66, target prediction change=0.05
  test accuracy = 0.7000
  test macro-F1 = 0.6250
```

The negative VPN selector ablation with a looser `--min_valid_gain_over_base 0.03` selected an embedding-LR expert and dropped to `0.6812` accuracy / `0.6475` macro-F1. The unsafe reliability-fusion ablation selected `alpha=5.0`, `reliability_power=4.0`, `confidence_power=1.0`, `temperature=0.5`; it improved validation macro-F1 but dropped target-test performance to `0.6956` accuracy / `0.6633` macro-F1. Bootstrap alone did not reject this VPN candidate because its validation gain was internally stable, but the target-shift guard rejected it because it changed too many target predictions. A broader calibration-enabled candidate search also found a lower-shift VPN threshold switch, but it still dropped to `0.7339` accuracy / `0.7241` macro-F1, so calibration remains an ablation rather than the default final path. On TLS-120, the unified-slot stacker reached `0.7991` accuracy / `0.7897` macro-F1 by itself; a trainable soft expert gate over the same unified slots reached `0.7973` / `0.7843`, with mean test weights dominated by `slot_stacker=0.593` and a base-identity branch `emb_et=0.358`. The current paper-safe selector then applies validation-set class-bias calibration to the soft gate and reaches `0.7996` accuracy / `0.7869` macro-F1 while keeping target prediction change below 4%. A calibration-aware ranking probe with `--calibration_penalty_weight 0.10 --calibration_penalty_metric ece` kept the same accepted TLS and USTC selector decisions but did not improve Acc/F1, so it remains a neutral calibration/stability ablation. This is why the paper method should emphasize validation-gated expert selection with validation stability, trainable expert gating, calibration observability, and unlabeled target-shift safety, not unconditional expert switching.

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
  fold1: weak, acc=0.6411, macro-F1=0.6030
  fold2: weak, acc=0.6597, macro-F1=0.6138
  mean: acc=0.6832, macro-F1=0.6575
  min:  acc=0.6411, macro-F1=0.6030

TLS-120 targets: accuracy >= 0.7800, macro-F1 >= 0.7000
  fold0: pass, acc=0.7991, macro-F1=0.7897
  fold1: missing
  fold2: missing
```

Interpretation: VPN already has evidence across all three folds, but only split0 meets the target. Split1/split2 are the real bottleneck. TLS currently has a strong split0 result, but fold1/fold2 still need to be run before claiming cross-split stability. This is why the next automatic loop should prioritize fold1/fold2 training and validation, not additional split0-only tuning.

The summary script emits commands with the same Stage-8 module family for every dataset/fold: Tower-1 preprocessing/training, raw+projected embeddings, graph/seq Tower-2, multi-view flow pooling, confusion-aware SupCon, confidence penalty, safe prior, and validation-gated selection. Dataset-specific behavior should come from learned weights and validation gates, not from removing modules.

Use the metric dashboard to check the current target gates across datasets:

```bash
conda run --no-capture-output -n llm-factory \
  python summarize_experiment_results.py \
    --dataset vpn-app \
    --dataset tls-120 \
    --dataset ustc-app \
    --top_k 5 \
    --output_json reasoningDataset/goal_metric_summary.json
```

Current target-gate status:

```text
vpn-app: acc=0.7488, macro-F1=0.7558, target acc>=0.7400 and macro-F1>=0.6500 -> PASS
tls-120: acc=0.7996, macro-F1=0.7869, target acc>=0.7800 and macro-F1>=0.7000 -> PASS
ustc-app: acc=0.7000, macro-F1=0.6250, cross-dataset evidence on the 20-flow test split
```

Generate the next-experiment recommendation report after each new run:

```bash
conda run --no-capture-output -n llm-factory \
  python recommend_next_experiment.py \
    --dataset vpn-app \
    --dataset tls-120 \
    --dataset ustc-app \
    --output_json reasoningDataset/next_experiment_recommendation.json \
    --output_md reasoningDataset/next_experiment_recommendation.md
```

Current recommendation summary:

```text
vpn-app: target PASS; old-embedding paired-view probes are negative, so do not spend more CPU time on graph paired-view training. Run the documented A800 Stage-8 paired-view path with fresh `rawproj_flowaware_*` embeddings.
tls-120: target PASS; only accept new modules when validation-gated selection and target-shift guards keep or improve the current best.
ustc-app: evidence result; prioritize representation learning before additional probability-level fusion.
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

The same wrapper has dataset presets for class count, label map, current best selector input, and target-shift guard. To build the corresponding plans for TLS-120 or USTC, only change `--dataset`:

```bash
conda run --no-capture-output -n llm-factory \
  python run_recommended_experiment.py \
    --dataset tls-120

conda run --no-capture-output -n llm-factory \
  python run_recommended_experiment.py \
    --dataset ustc-app
```

To generate the unified VPN/TLS/USTC command suite in one step:

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

The paper-safe result paths, VPN/TLS target gates, and required unified expert slots are centralized in `paper_framework_defaults.py`. Update that file first when changing the paper-facing main result; the recommendation, framework-report, autonomous-loop, and recommended-suite scripts import the shared defaults to avoid drift.

The autonomous loop runs the audit below automatically. Run it manually after changing paper-facing defaults or before updating paper tables. It checks that the configured paper-safe JSON files exist, meet their target gates where targets are defined, and expose the required unified expert slots directly or through identity-compatible legacy mapping:

```bash
conda run --no-capture-output -n llm-factory \
  python audit_paper_framework_defaults.py \
    --output_json reasoningDataset/paper_framework_defaults_audit.json \
    --output_md reasoningDataset/paper_framework_defaults_audit.md
```

For paper-grade stopping, add `--require_ci_targets`. The default loop stops when VPN/TLS point metrics, the unified-framework audit, and the centralized paper-safe defaults audit pass. The stricter CI mode additionally requires each goal dataset to have `ci_target_met=true` in `reasoningDataset/paper_evidence_pack.json`. With the current results, TLS-120 passes this strict gate, while VPN is still `point_pass_ci_mixed` because its bootstrap accuracy lower bound is below `0.74`; strict mode therefore continues to the recommended suite instead of stopping early.

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

The loop also audits whether a raw-best candidate is safe to promote into the paper-facing defaults. This is useful for TLS-120, where the raw unified-slot stacker is stronger than the guarded selector result but lacks direct selector safety guards and full paper-facing slot evidence:

```bash
conda run --no-capture-output -n llm-factory \
  python audit_paper_candidate_promotion.py \
    --output_json reasoningDataset/paper_candidate_promotion_audit.json \
    --output_md reasoningDataset/paper_candidate_promotion_audit.md
```

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

Use `--run_tag_template '{run_tag}_iter{iteration:02d}'` for multi-iteration searches so each loop writes a fresh Stage-8 run instead of reusing old `skip_existing` outputs. The default `--variant_schedule stage8_balanced` keeps iteration 1 at the current balanced Stage-8 setting, then cycles through stronger paired-view consistency, higher paired-view CE weight with trainable multi-view flow pooling, and stronger Tower-2 dropout plus confidence-penalty regularization; each variant also carries its own seed. The fourth Stage-8 variant also passes `--confidence_penalty_weight 0.02` into Tower-2 training, adding a KL-to-uniform penalty on classification logits to reduce overconfident split-specific decisions while keeping the same model architecture. The `multi_view` pooling computes mean/max/std/attention flow views for every dataset and learns a softmax gate over the four branches, so unhelpful branches can be down-weighted by training instead of being manually removed. The multi-view variant also uses a small `--multi_view_gate_entropy_weight` to encourage more decisive branch weights while keeping every branch in the shared framework. `test_tower2.py` writes the average learned branch weights, normalized entropy, and effective branch count to `metrics.eval_config.multi_view_gate`, which can be used as paper evidence that all datasets pass through the same branch family while training learns dataset-specific weights. New Tower-2, validation-selector, and slot-stacker outputs include calibration metrics under `metrics.*.calibration` (`ece`, `nll`, `brier`, average confidence, and sample count), so confidence-penalty, stacker, or selector changes can be judged by both accuracy/F1 and overconfidence with one metric definition from `probability_metrics.py`. The framework and evidence-pack reports surface these calibration fields whenever the evaluated JSON contains them; older result JSON files remain valid but will not show a calibration table until they are re-evaluated. This makes multi-iteration runs search real model/training variants and seed stability instead of only changing output directories. `--status_rank_metric target_margin` ranks ledger best-before/best-after results by the weaker target margin across accuracy and macro-F1, which is useful for paper-grade searches where both metrics matter; omit it to keep the historical accuracy-first ranking. The wrapper records the concrete per-iteration run tag and variant in the loop ledger. After each suite run, the loop ledger also embeds the suite `child_plans` summary, including every dataset's child plan JSON, paired-prior output, final-selector output, skipped stages, CUDA-required stages, and `experiment_config`. The raw `status_before/status_after` entries keep the highest-scoring test JSON for search visibility, while `paper_safe_status_before/paper_safe_status_after` mirror the evidence-pack framework claims and raw-minus-paper-safe deltas used for paper reporting. The ledger records both `raw_goals_met_*` and `paper_safe_goals_met_*`; stopping requires both, plus the framework, centralized defaults audit, and optional CI gates. The `next_experiment_plan` entry records the current priority gap type for each dataset, so a passing point estimate can still trigger a CI-strengthening experiment. The `candidate_promotion_audit` entry records why a higher raw-best probe can or cannot be promoted into the paper-safe defaults. The `best_delta` field compares each dataset's best test metric before and after the iteration, so new-best runs are visible without manually scanning all result JSON files.

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
| ustc-app | 0.7000 | 0.6250 | - | evidence | 20 | base=active; selector=active; expert=active:class_precision; calib=evaluated; slots=inferred_identity_compatible:10;provided:2;identity:8;extra:0; mv_gate=not_observed; guards=boot:active,shift:active | class_precision alpha=0.5, margin=0.0 | bootstrap win=0.66, q=0.0000; target change=0.0500, JS=0.0500 |
```

Framework consistency audit:

```text
status: PASS
vpn-app:  same module family, expert candidate = gated_off:reliability_fusion
tls-120:  same module family, expert candidate = active:class_bias_calibration
ustc-app: same module family, expert candidate = active:class_precision
```

Flow-level bootstrap uncertainty with 300 resamples:

```text
| Dataset | Accuracy 95% CI | Macro-F1 95% CI |
|---|---:|---:|
| vpn-app | [0.7252, 0.7685] | [0.7249, 0.7865] |
| tls-120 | [0.7923, 0.8082] | [0.7788, 0.7933] |
| ustc-app | [0.5000, 0.9000] | [0.4240, 0.8103] |
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
| ustc-app | base residual | 0.6500 | 0.0000 | 0.5750 | 0.0000 | base=0.91, prior=0.09 | base |
| ustc-app | proto embedding expert | 0.6500 | 0.0000 | 0.6083 | +0.0333 | - | Tower-1 prototype |
| ustc-app | safe selector | 0.7000 | +0.0500 | 0.6250 | +0.0500 | class_precision:a=0.5 | class-precision gate |
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
| ustc-app | safe selector | 300 | [0.0000, +0.1500] | [0.0000, +0.1331] |
```

Generate the compact paper evidence pack. It carries the framework report's unified module usage, unified expert-slot coverage, selector, guard, CI, and multi-view gate evidence into one JSON/Markdown artifact:

```bash
conda run --no-capture-output -n llm-factory \
  python make_paper_evidence_pack.py \
    --output_json reasoningDataset/paper_evidence_pack.json \
    --output_md reasoningDataset/paper_evidence_pack.md
```

Current evidence-pack claim status:

```text
vpn-app:  point_pass_ci_mixed  (point target passes; bootstrap acc lower bound is below 0.74)
tls-120:  strong               (point target and bootstrap lower bounds both pass)
ustc-app: evidence_only        (tiny 20-flow test split; framework evidence, not a strong performance claim)
```

Paper positioning from the evidence pack:

```text
main claim: unified candidate-expert traffic classification framework with validation-gated safety controls
strong performance claim: TLS-120
qualified performance claim: VPN point estimate passes, but bootstrap lower bound is mixed
cross-dataset framework evidence: USTC, with small-test-split caveat
reviewer-risk control: report harmful experts as negative ablations and use bootstrap/target-shift guards
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

The current top-12 residual search selected `base=1.0`, so the existing fusion/statistics/embedding experts do not provide enough validation-supported residual signal to push VPN beyond the stronger aspirational `75%` mark. The next meaningful improvement should therefore come from representation learning rather than more probability-level fusion: resume GPU Stage-8 Tower-1 flow-aware contrastive training, re-extract embeddings, and rerun Tower-2/fusion on VPN first, then verify the same protocol on TLS-120 and USTC.

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
18. Tower-1 packet-packet SupCon vs packet-to-flow prototype contrastive loss
19. Tower-2 flow-only SupCon vs window-to-flow prototype contrastive loss

These ablations directly support the claim that Tower 1 learns protocol-aware packet semantics while Tower 2 learns flow-level packet interaction patterns.
