# YOLO on Axelera Voyager SDK

You've trained your model and validated it. Now you're thinking about production — and that
means picking the right hardware *and* the right software stack. A great accelerator means
nothing if productizing your model takes months of engineering effort.

Axelera partners with Ultralytics to offer both: purpose-built AIPU hardware *and* a
high-performance pipeline SDK that handles embedding, preprocessing, and postprocessing
internally. Write the pipeline in Python, call `.optimized()`, and let the runtime handle
the rest.

```python
pipeline = op.seq(
    op.letterbox(640, 640),
    op.totensor(),
    op.load("yolo26n-pose.axm"),
    ConfidenceFilter(threshold=0.25),  # custom operator — see below
    op.to_image_space(keypoint_cols=range(6, 57, 3)),
).optimized()                          # runtime fuses ops for maximum throughput

poses = pipeline(frame)                # that's it
```

## Examples

| Script | Task | Model |
|--------|------|-------|
| `yolo26-pose.py` | Pose estimation — 17 COCO keypoints | YOLO26n-pose (NMS-free) |
| `yolo11-seg.py` | Instance segmentation | YOLO11n-seg |

## Installation

Install `axelera-runtime2`, which is exposed for import as `axelera.runtime`:

```bash
pip install axelera-runtime2 opencv-python numpy
```

## Compile Your Model

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

## Usage

### Pose Estimation (YOLO26)

```bash
python yolo26-pose.py --model yolo26n-pose.axm --source 0           # webcam
python yolo26-pose.py --model yolo26n-pose.axm --source video.mp4   # video
python yolo26-pose.py --model yolo26n-pose.axm --source image.jpg   # image
```

### Instance Segmentation (YOLO11)

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

## Custom Operators

The pipeline is fully composable — you can drop in your own operators anywhere in the
sequence for custom filtering, domain-specific logic, or any pre/postprocessing you need.
Just subclass `op.Operator`:

```python
class ConfidenceFilter(op.Operator):
    threshold: float = 0.25

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            x = x[0]
        return x[x[:, 4] >= self.threshold]
```

The runtime treats custom operators as first-class citizens alongside the built-in ones.

## Preview Release

> `axelera-runtime2` is currently in **preview / experimental** status. APIs may change
> between releases, and we are actively improving inference performance and expanding
> supported model coverage.

Coming soon: **batching and streaming mode** to fully utilize all AIPU cores for even
higher throughput.

For the latest runtime, release notes, and roadmap visit the
**[Voyager SDK repository](https://github.com/axelera-ai-hub/voyager-sdk)**.

Questions, feedback, or want to connect with the team?
Join the **[Axelera community](https://community.axelera.ai/)**.
