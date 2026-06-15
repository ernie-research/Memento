# Memento - Reconstruct to Remember for Consistent Long Video Generation

[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white)](https://ernie-research.github.io/Memento)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxxx.xxxxx)
[![Models](https://img.shields.io/badge/HuggingFace-Models-ffd21e?logo=huggingface&logoColor=black)](https://huggingface.co/ernie-research/Memento)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?logo=apache&logoColor=white)](LICENSE)

[![Teaser](./docs/teaser_preview.png)](./teaser.pdf)

Memento generates long-form, multi-shot narrative videos with consistent subject identities across shots, scenes, and viewpoints. Given a global story caption and per-shot descriptions, Memento produces coherent minute-level videos through shot-by-shot autoregressive generation. See our [Project Page](https://ernie-research.github.io/Memento/) for more details and video results.

## TODO

- [x] Weights Release
- [x] Inference Code Release
- [x] Training Release

## Setup

### Environment

```bash
conda create -n memento python=3.10 -y
conda activate memento
pip install -r requirements.txt
```

### Model Weights

| Model | Description | Link |
|-------|-------------|------|
| Wan2.2-T2V-A14B | Base text-to-video model | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |
| Wan2.2-I2V-A14B | Base image-to-video model | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) |
| Memento LoRA | Trained LoRA + KeyframeQuery weights | [HuggingFace](https://huggingface.co/ernie-research/Memento) |

Download and place under `models/`:

```
models/
├── Wan2.2-T2V-A14B/
├── Wan2.2-I2V-A14B/
└── memento_lora/
    ├── backbone_low_noise.pth
    └── backbone_high_noise.pth
```

Or use the HuggingFace CLI:

```bash
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir models/Wan2.2-T2V-A14B
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir models/Wan2.2-I2V-A14B
huggingface-cli download ernie-research/Memento --local-dir models/memento_lora
```

## Inference

### Prerequisites

- 8x NVIDIA GPUs (A100 80GB recommended)

### Story Script Format

Each story is a JSON file describing a multi-shot video narrative:

```json
{
  "reconstruct_target": "[Person A] appearance description\n[Person B] appearance description",
  "scenes": [
    {
      "scene_num": 1,
      "cut": [true, false, false],
      "video_prompts": [
        "global caption: [Person A]: appearance, [Person B]: appearance; shot caption: [Person A] does something...",
        "global caption: [Person A]: appearance; shot caption: [Person A] continues doing...",
        "global caption: [Person A]: appearance; shot caption: [Person A] finishes..."
      ]
    }
  ]
}
```

Field descriptions:

| Field | Description |
|-------|-------------|
| `reconstruct_target` | Identity description for all main characters. One `[Person X] appearance` per line. Used for identity reconstruction conditioning. |
| `scenes[].scene_num` | Scene index (1-based) |
| `scenes[].cut` | Boolean array, one per shot. `true` = camera cut (no visual continuity from previous shot). `false` = continuous (uses last frame of previous shot as I2V condition). First shot of a scene is typically `true`. |
| `scenes[].video_prompts` | Array of prompts, one per shot. Format: `global caption: [Person X]: appearance, ...; shot caption: [Person X] action description` |

### Single Story Inference

```bash
bash run_inference.sh <story_json> <output_dir> <lora_weight_path>
```

Example:

```bash
bash run_inference.sh \
  ./infer_stories/3_astronaut.json \
  ./results/astronaut \
  ./models/memento_lora
```

### Batch Inference

```bash
bash run_inference_batch.sh <story_dir> <output_dir> <lora_weight_path>
```

Example:

```bash
bash run_inference_batch.sh \
  ./infer_stories \
  ./results/my_run \
  ./models/memento_lora
```

Optional environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NGPUS` | 8 | Number of GPUs |
| `T2V_MODEL_PATH` | `./models/Wan2.2-T2V-A14B` | T2V base model |
| `I2V_MODEL_PATH` | `./models/Wan2.2-I2V-A14B` | I2V base model |
| `VIDEO_SIZE` | `832*480` | Output resolution |
| `LORA_RANK` | 128 | LoRA rank (must match training) |
| `PORT` | 8200 | torchrun communication port (single) |
| `BASE_PORT` | 8200 | torchrun base port (batch) |

### Output

For each story, the script produces:
- `<output_dir>/<story_name>/XX_YY.mp4` — individual shots
- `<output_dir>/<story_name>/<story_name>.mp4` — concatenated final video
- `<output_dir>/<story_name>.log` — generation log (batch mode)

The batch script supports resume: completed stories (with final `.mp4`) are automatically skipped on re-run.

## Training

### Prerequisites

- 8x NVIDIA GPUs (A100 80GB recommended)
- PyTorch 2.0+
- Additional dependencies: `accelerate`, `bitsandbytes`, `tensorboard`, `imageio`
- Optional (for BOS data): `bce-python-sdk`, `pyyaml`

### Training Data Format

Training data is a JSON file containing sequences of video clips:

```json
[
  {
    "sequence_id": "story_001",
    "sequence_clips": [
      {"video_path": "/path/to/clip1.mp4", "caption": "global caption: ...; shot caption: ...", "sequence_order": 0},
      {"video_path": "/path/to/clip2.mp4", "caption": "global caption: ...; shot caption: ...", "sequence_order": 1}
    ],
    "reconstruct_targets": [
      {"clip_index": 0, "frame_in_clip_16fps": 10, "caption": "identity description"}
    ],
    "subjects": "[Person A] appearance description"
  }
]
```

Each clip can use either `video_path` (local file) or `bos_url` (Baidu BOS, requires `bos_config.yaml`).

### Running Training

```bash
DATA_PATH=/path/to/training_data.json bash run_train.sh
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | (required) | Path to training data JSON |
| `NGPUS` | 8 | Number of GPUs |
| `CHECKPOINT_DIR` | `./models/Wan2.2-I2V-A14B` | Base I2V model path |
| `OUTPUT_DIR` | `./outputs/memento_train` | Output for checkpoints and logs |
| `LORA_RANK` | 128 | LoRA rank (must match inference) |
| `MAX_STEPS` | 6000 | Maximum training steps |

### Training Low-Noise vs High-Noise Model

Memento uses a dual-DiT architecture (boundary at t=0.9). The two models must be trained separately:

```bash
# Train low-noise model (default):
DATA_PATH=/path/to/data.json bash run_train.sh

# Train high-noise model:
DATA_PATH=/path/to/data.json bash run_train.sh --train_high_noise_only
```

Note: `--train_both_models` exists but is not recommended — it will OOM on 8×A100 80GB.

### Training Output

- `<output_dir>/trainable_step{N}.pth` — LoRA + KeyframeQuery weights (for inference)
- `<output_dir>/checkpoint_step{N}.pth` — Full checkpoint with optimizer state (for resume)
- `<output_dir>/logs/` — TensorBoard logs

### Resume Training

```bash
# Resume from full checkpoint:
torchrun ... train.py --resume_from ./outputs/memento_train/checkpoint_step3000.pth ...

# Hot-start from inference weights (e.g., fine-tune further from released weights):
torchrun ... train.py \
  --resume_from_low ./models/memento_lora/backbone_low_noise.safetensors \
  --resume_from_high ./models/memento_lora/backbone_high_noise.safetensors ...
```

### Using Training Checkpoints for Inference

Training saves `trainable_step{N}.pth` containing both low-noise and high-noise LoRA weights (keyed by `low_noise_model.*` / `high_noise_model.*`). The inference script expects a **directory** with separate files, so you need to split the checkpoint:

```python
import torch

ckpt = torch.load("outputs/memento_train/trainable_step6000.pth", map_location="cpu")
low, high = {}, {}
for k, v in ckpt.items():
    if k.startswith("low_noise_model."):
        low[k] = v
    elif k.startswith("high_noise_model."):
        high[k] = v

torch.save(low, "models/memento_lora/backbone_low_noise.pth")
torch.save(high, "models/memento_lora/backbone_high_noise.pth")
```

If you trained low-noise and high-noise separately, each `trainable_step{N}.pth` only contains one side — rename directly:

```bash
cp outputs/low_noise_run/trainable_step6000.pth  models/memento_lora/backbone_low_noise.pth
cp outputs/high_noise_run/trainable_step6000.pth models/memento_lora/backbone_high_noise.pth
```

Then run inference as usual:

```bash
bash run_inference.sh ./story.json ./results/test ./models/memento_lora
```

### Demo Training

A small sample dataset (`train_data/`) is included for quick verification:

```bash
bash run_train_demo.sh
```

This runs 100 steps on the astronaut example data (11 clips, 36MB). Useful for validating the training pipeline before running on full data.

## Writing Story Scripts

### Story Script Conventions

Based on the showcase examples, story scripts follow these conventions:

1. **Characters are labeled** `[Person A]`, `[Person B]`, etc. Groups use `[Group A]`.

2. **`reconstruct_target`** lists only physical appearance (hair, clothing, accessories). No actions, no emotions. One character per line:
   ```
   [Person A] young man with shoulder-length dark hair wearing thin wire-frame glasses, black turtleneck, blue jeans
   [Person B] young man with curly brown hair and trimmed beard wearing patterned collared shirt
   ```

3. **`global caption`** in each shot prompt lists appearances of characters **present in that shot** (not all characters in the story). Uses `[Person X]: appearance` format with comma separation. Can be empty for establishing shots with no main characters.

4. **`shot caption`** describes the action in present tense, 1-3 sentences. Starts with the subject `[Person X]` or a scene description. Focuses on motion and visual change. Includes environment/lighting details.

5. **`cut` array logic**:
   - `true` for the first shot of each scene
   - `true` when camera angle/location changes drastically within a scene
   - `false` when the shot is a continuous continuation of the previous shot (camera follows the same action)

6. **Structure**: typically 3-4 scenes, 2-4 shots per scene, totaling 8-12 shots.

### LLM Prompt Template for Story Generation

Use the following prompt with Gemini / Qwen / GPT to generate story scripts:

````
You are a story-to-video script writer for a multi-shot video generation system. Given a story concept, produce a JSON script following the exact format below.

## Rules

1. Output valid JSON (no comments, no trailing commas).

2. Characters:
   - Label main characters as [Person A], [Person B], etc. Groups as [Group A].
   - `reconstruct_target`: one line per main character, format `[Person X] physical appearance only` (hair, clothing, accessories — no actions/emotions). Newline-separated.

3. Structure:
   - 3-4 scenes, 2-4 shots per scene (8-12 shots total).
   - Each scene has `scene_num` (1-based), `cut` (boolean array), and `video_prompts` (string array).

4. `cut` array:
   - First shot of each scene: `true`
   - Within a scene: `false` if camera follows same action continuously, `true` if camera angle or location jumps.

5. `video_prompts` format for each shot:
   ```
   global caption: [Person A]: appearance, [Person B]: appearance; shot caption: [Person X] action description
   ```
   - `global caption`: list ONLY characters visible in THIS shot. Use their full appearance from reconstruct_target. If no main character is visible (e.g., establishing shot), leave empty: `global caption: ; shot caption: ...`
   - `shot caption`: 1-3 sentences in present tense describing the visual action. Start with the subject. Include environment, lighting, and camera motion cues. Focus on motion and visual change.

6. Writing style for shot captions:
   - Present tense, descriptive, cinematic
   - Describe what the camera SEES, not internal thoughts
   - Include physical details: lighting direction, environment textures, character posture
   - Each shot should have clear visual motion/change (not static descriptions)

## Output format

```json
{
  "reconstruct_target": "[Person A] appearance\n[Person B] appearance",
  "scenes": [
    {
      "scene_num": 1,
      "cut": [true, false],
      "video_prompts": [
        "global caption: [Person A]: appearance; shot caption: ...",
        "global caption: [Person A]: appearance; shot caption: ..."
      ]
    }
  ]
}
```

## Story concept

{INSERT YOUR STORY CONCEPT HERE}
````
