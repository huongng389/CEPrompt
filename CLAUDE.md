# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Official implementation of **CEPrompt: Cross-Modal Emotion-Aware Prompting for Facial Expression Recognition** (IEEE TCSVT 2024). This is a research codebase (PyTorch) built on top of OpenAI CLIP (ViT-B/16) for facial expression recognition (FER) on RAF-DB and AffectNet.

There is no test suite, linter, or CI in this repo — it's an academic training/eval pipeline driven by shell scripts and argparse.

## Setup

```
pip install -r requirements.txt
```

Requires a locally downloaded CLIP checkpoint (e.g. `ViT-B-16.pt` from [OpenAI CLIP](https://github.com/openai/CLIP)), pointed to via `--clip-path`.

### Data layout

Datasets are NOT included in the repo and must be prepared externally, then referenced via `--data-path`:

```
data/RAF-DB/basic/EmoLabel/{images.txt,image_class_labels.txt,train_test_split.txt}
data/RAF-DB/basic/Image/aligned_224/          # MTCNN re-aligned images (see APVIT)
data/AffectNet/affectnet_info/{images.txt,image_class_labels.txt,train_test_split.txt}
data/AffectNet/Manually_Annotated_Images/{1,2,...}/
```

The three `.txt` files are `idx | value` pairs (image filename, integer class label, train(1)/test(0) split flag). Sample copies of these files live at the repo root (`dataset/`) for reference, but real training data is expected under `--data-path`.

## Commands

### Train Stage 1 (EVA — visual adapter pretraining)
```
python3 train_fer_first_stage.py --dataset ${DATASET} --data-path ${DATAPATH}
# or
bash stage1.sh
```

### Train Stage 2 (CAT — conception-appearance tuning), from a Stage 1 checkpoint
```
python3 train_fer_second_stage.py --dataset ${DATASET} --data-path ${DATAPATH} --ckpt-path ${CKPTPATH}
# or
bash stage2.sh
```

### Evaluation
```
python3 train_fer_second_stage.py --eval \
  --dataset ${DATASET} --data-path ${DATAPATH} \
  --ckpt-path ${CKPTPATH} --eval-ckpt ${EVACKPTPATH}
```

`stage1.sh` and `stage2.sh` are the canonical examples of argument combinations (dataset paths, `--gpu`, `--alpha`, `--topk`, `--n_ctx`, `--prompts_depth`, etc.) — check/edit these when adjusting default runs rather than guessing flags.

`--dataset` must be one of `rafdb`, `affectnet`, `affectnet_8`; `--data-path` must match the corresponding local dataset directory.

Outputs (logs, checkpoints, config dumps) are written under `outputs/first_stage/<run_name>/` and `outputs/second_stage/<run_name>/`, named from a timestamp + hyperparameters.

## Architecture: two-stage training pipeline

The method trains in two sequential stages, each with its own entrypoint script, engine, and optimizer strategy. Both stages share the same model class, `CLIPVIT` (`models/clip_vit.py`), which wraps a loaded CLIP ViT-B/16's visual tower (conv1, positional/class embeddings, transformer blocks, ln_post) plus CLIP's text encoder/tokenizer.

### Stage 1 — EVA (Emotion Conception-guided Visual Adapter)
- Entry: `train_fer_first_stage.py` → `engine_fer_first_stage.py` (`train`/`test`)
- Uses `CLIPVIT.forward()` / `forward_features()` — plain CLIP visual forward pass, no learned text prompts. Text side uses fixed hand-written prompts (`"a photo of a {label} facial expression..."`) tokenized via `clip.tokenize`.
- Produces two prediction heads that are blended by `args.alpha`:
  - **Local/patch score** (`score1`): patch tokens (`x[:, 1:]`) projected to CLIP embedding space, cosine-scored against label text embeddings, top-k (`args.topk`) pooled.
  - **Global score** (`score2`): CLS token (`x[:, 0]`) projected and cosine-scored against label embeddings.
- Trains against a frozen "teacher" CLIP visual encoder (`clip_model.encode_image`) via an L1 **knowledge-distillation loss** (`dist_loss`, weighted by `args.lamda`) on the global feature, added to the classification loss (`args.loss_function`: ce/focal/balanced/cosine/gce, see `utils/LossFunctions.py`) — this is what prevents catastrophic forgetting of CLIP's pretrained knowledge.
- Optimizer: `build_optimizer(args, model)` with no `stage` arg → AdamW with **layer-wise LR decay** (`utils/lr_decay.py`, `args.layer_decay`, `args.fix_layer` freezes early layers).
- Saves a full model state_dict checkpoint every epoch (`model_epoch_{N}.pth`) — this is the `--ckpt-path` consumed by Stage 2.

### Stage 2 — CAT (Conception-Appearance Tuner)
- Entry: `train_fer_second_stage.py` → `engine_fer_second_stage.py` (`train`/`test`/`eval`)
- Loads the Stage 1 checkpoint (`--ckpt-path`) into `CLIPVIT` (`strict=False`, since prompt-learner params are new).
- Only active when `args.stage2_name == "cat"` (there's a vestigial `"coop"` branch in `build_optimizer` for a CoOp-style baseline). Adds two submodules from `models/cat.py`:
  - `CATPromptLearner` — learns shared/compound text-prompt context vectors (`n_ctx`, `prompts_depth`, optional `ctxinit`, optional class-invariant context tokens gated by `--use_class_invariant`), plus per-layer projection layers (512→768) that turn the *text*-side compound prompts into *visual*-side deep prompt tokens for cross-modal coupling.
  - `TextEncoder` (aliased `TextEncoder_cat`) — a CLIP text transformer forward pass modified to splice in the compound prompt context at each layer up to `depth`.
- `CLIPVIT.forward_cat()` is the Stage 2 forward: prompt learner generates text prompts + projected visual deep-prompts; `get_features()`/`forward_vitblocks()` run the visual transformer, injecting the projected compound prompts into the visual token sequence at each layer < `depth` (VPT-style prefix/suffix splicing) — this is the actual conception↔appearance coupling mechanism. Same dual local/global scoring + `alpha` blend as Stage 1.
- Optimizer: `build_optimizer(args, model, stage='stage2')` → SGD, and **only params containing `"prompt_learner"` in their name are trainable** — the CLIP backbone (including the Stage-1-tuned visual weights) is frozen entirely in Stage 2.
- `--eval` mode in `train_fer_second_stage.py` skips training and loads a second checkpoint (`--eval-ckpt`, the Stage 2 output) for pure evaluation via `engine_fer_second_stage.eval`.

### Supporting modules
- `clip/` — a vendored copy of OpenAI's CLIP (model, tokenizer, BPE vocab); `clip.load()`/`clip.tokenize()`/`clip.build_model()` are used throughout.
- `dataloader/data_utils.py` — dispatches to `dataloader/rafdb/rafdb.py` or `dataloader/affectnet/{affectnet,affectnet_8}.py` based on `args.dataset`; also hardcodes each dataset's label name ordering (`get_labelname`) which is used to build the text prompts — **the order matters and must match the integer labels in `image_class_labels.txt`**. AffectNet variants use `dataloader/sampler.py`'s `ImbalancedDatasetSampler` for class-balanced sampling; RAF-DB does not.
- `utils/LossFunctions.py` — pluggable loss functions selected via `--loss_function`.
- `utils/lr_decay.py` / `utils/lr_sched.py` — layer-wise LR decay param grouping (Stage 1) and per-iteration LR schedule/warmup adjustment (`lrs.adjust_learning_rate`, called every training step in both stages).
- `utils/misc.py` — seeding, logging setup (`init_log`, writes to `<record_path>/recording.log`), fp32/fp16 conversion helpers (CLIP loads in fp16 by default; `convert_models_to_fp32` is applied after building `CLIPVIT`), distributed-training helpers (largely unused in the single-GPU scripts here — `--gpu` selects a single device via `torch.cuda.set_device`).

## Key gotchas when editing this code

- `CLIPVIT` behaves differently depending on `args.stage2_name`: it only constructs `CATPromptLearner`/`TextEncoder_cat` when `stage2_name == "cat"`, so Stage 1 training (`stage2_name=None`) does not have prompt-learner submodules at all — don't assume `forward_cat()` is always available.
- The `global_only` / `local_only` flags on `CLIPVIT` are ablation switches hardcoded in `__init__` (not exposed as CLI args) — flip them directly in `models/clip_vit.py` to reproduce ablation results from the paper.
- Text prompts differ between stages: Stage 1 uses fixed template strings; Stage 2 replaces them with learned context vectors via `CATPromptLearner`, but the class name list (`args.label_nms`) and its ordering must stay consistent between stages since it's baked into both the fixed prompts and the tokenized prompt buffers.
