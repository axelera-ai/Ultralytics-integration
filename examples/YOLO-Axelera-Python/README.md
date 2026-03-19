# YOLO on Axelera Voyager SDK

You've trained and exported your YOLO model -- these examples show how to build
complete inference pipelines around it using `axelera-rt`.

The typical workflow has two phases:

1. **Model preparation.** You use Ultralytics, PyTorch, and the Voyager SDK
   compiler to train, export, and compile your model to `.axm` format.
2. **Application development.** You build your video pipeline and business
   logic using only `axelera-rt` -- a lightweight, self-contained
   package with minimal dependencies.

**Why a pipeline SDK?** A real application does more than run a model -- it
reads camera or video frames, preprocesses them, runs AIPU inference, and
postprocesses the results. The runtime fuses all of these stages into a
single optimized pipeline, giving you high end-to-end FPS with low CPU
overhead.

```python
pipeline = op.seq(
    op.colorconvert('RGB', src='BGR'),  # OpenCV reads BGR; models expect RGB
    op.letterbox(640, 640),
    op.totensor(),
    op.load("yolo26n-pose.axm"),
    ConfidenceFilter(threshold=0.25),   # custom operator — see below
    op.to_image_space(keypoint_cols=range(6, 57, 3)),
).optimized()                           # runtime fuses ops for maximum throughput

poses = pipeline(frame)                 # frame in, results out
```

## Examples

We have written three examples, ordered from simple to advanced. It should be straightforward for you to repurpose them for other models and tasks.

| Script | Task | Model |
|--------|------|-------|
| `yolo26-pose.py` | Pose estimation — 17 COCO keypoints | YOLO26n-pose (NMS-free) |
| `yolo26-pose-tracker.py` | Pose estimation + multi-object tracking | YOLO26n-pose (NMS-free) |
| `yolo11-seg.py` | Instance segmentation | YOLO11n-seg |

**Suggested reading order:**
1. **`yolo26-pose.py`** -- Start here. Linear `op.seq()` pipeline with a custom operator.
2. **`yolo11-seg.py`** -- Branching with `op.par()` for multi-head models (detections + masks).
3. **`yolo26-pose-tracker.py`** -- Adds stateful `op.tracker()` for multi-object tracking.

## Quick Start

### Install

If you followed the [Axelera integration guide](../../docs/en/integrations/axelera.md), `axelera-rt` is already installed — running `yolo predict` or `yolo val` with an Axelera model auto-installs the runtime dependencies. If you need to install manually:

```bash
pip install axelera-rt==1.6.0rc2 --no-cache-dir --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple
```

You will also need `opencv-python` and `numpy` (likely already present).

### Compile

Export and compile your trained model directly to Axelera format using the Ultralytics CLI:

```bash
yolo export model=your-model.pt format=axelera
```

To reproduce these examples, export the pretrained Ultralytics models:

```bash
yolo export model=yolo26n-pose.pt format=axelera
yolo export model=yolo11n-seg.pt  format=axelera
```

The compiled models are written to `yolo26n-pose_axelera_model/` and
`yolo11n-seg_axelera_model/` respectively. Pass the `.axm` file inside to `--model`.

### Run

#### Pose Estimation (YOLO26)

```bash
python yolo26-pose.py --model yolo26n-pose.axm --source 0           # webcam
python yolo26-pose.py --model yolo26n-pose.axm --source video.mp4   # video
python yolo26-pose.py --model yolo26n-pose.axm --source image.jpg   # image
```

#### Pose Tracking (YOLO26 + TrackTrack)

```bash
python yolo26-pose-tracker.py --model yolo26n-pose.axm --source video.mp4
python yolo26-pose-tracker.py --model yolo26n-pose.axm --source 0 --tracker bytetrack
```

#### Instance Segmentation (YOLO11)

```bash
python yolo11-seg.py --model yolo11n-seg.axm --source 0
python yolo11-seg.py --model yolo11n-seg.axm --source video.mp4 --conf 0.3 --iou 0.5
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *required* | Path to compiled `.axm` model |
| `--source` | `0` | Image path, video path, or webcam index |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold *(segmentation only)* |
| `--tracker` | `tracktrack` | Tracking algorithm: `bytetrack`, `oc-sort`, `sort`, `tracktrack` *(pose-tracker only)* |

## What You'll See in the Code

### Custom Operators

The pipeline is fully composable — drop in your own operators anywhere in the sequence.
Just subclass `op.Operator` and implement `__call__`:

```python
class ConfidenceFilter(op.Operator):
    threshold: float = 0.25

    def __call__(self, x):
        # Each operator receives the output of the previous one in op.seq().
        # Here x is the output of the pose detection model — an array of
        # shape (batch, num_detections, num_values) where each row holds
        # [bbox, confidence, class_id, keypoints…]. Column 4 is the
        # confidence score. Squeeze the batch dim and filter by score.
        if x.ndim == 3:
            x = x[0]
        return x[x[:, 4] >= self.threshold]
```

The runtime treats custom operators as first-class citizens alongside the built-in ones.

### Multi-Object Tracking

Adding tracking to any detection or pose pipeline is a single line via `op.tracker()`:

```python
pipeline = op.seq(
    # ... your detection or pose pipeline ...
    op.tracker(algo='tracktrack'),   # one line adds tracking
)
```

| Algorithm | Key Strength | Reference |
|-----------|-------------|-----------|
| TrackTrack | Iterative matching with track-aware NMS (SOTA) | CVPR 2025 |
| ByteTrack | Handles low-confidence detections via dual threshold | Zhang et al., ECCV 2022 |
| OC-SORT | Observation-centric re-update + virtual trajectory | Cao et al., CVPR 2023 |
| SORT | Simple, fast IoU-only baseline | Bewley et al., ICIP 2016 |

The tracker operates on bounding boxes, but the original detection (with all its
metadata) is preserved — use `tracked.tracked` on each result to access the
original keypoints, masks, or other model outputs. You'll see this pattern in
`yolo26-pose-tracker.py`.

## Learn More

The [Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk) has everything
beyond these introductory examples: advanced pipeline patterns, ReID and
appearance-based tracking, the high-performance display system, production-ready
application templates, and the full API reference.

Questions or feedback? Join the [Axelera community](https://community.axelera.ai/).
