# MULTISPECTOR RF-DETR Segmentation POC

Jupyter-first proof of concept for fine-tuning **RF-DETR Segmentation Large** to detect, segment, and directly classify wheat instances as:

- `bad seed`
- `healthy seed`
- `impurity`

## Environment

The project uses Python 3.11 and a CUDA-enabled PyTorch build pinned in `pyproject.toml` and `uv.lock`.

From PowerShell in the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
python -m jupyter lab
```

Open `notebooks/01_rfdetr_segmentation_large_poc_data_v2.ipynb` and run the cells in order.

To recreate the environment with `uv`:

```powershell
$projectRoot = (Resolve-Path '.').Path
$env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.uv-python'
uv sync
```

## Important dataset findings

- The original Dataset V2 split contained augmented versions of the same source image in different splits. The notebook rebuilds deterministic source-safe splits before training.
- Dense images contain up to 616 instances. The notebook creates an adaptive tiled training dataset whose tiles contain at most 180 instances, below Large's 200-query capacity.
- The final tiled dataset contains 3,113 train, 169 validation, and 156 test images.
- `impurity` is only about 0.2% of instances, so per-class and real-world testing remain essential.

Run the notebook one cell at a time and review each safety audit before training.

## Tiled RF-DETR Segmentation Large V2 web application

The web interface loads the best Dataset V2 Large checkpoint. It converts OpenCV images from BGR to the RGB input expected by RF-DETR, applies adaptive overlapping tiles to dense full-frame images, and merges duplicate masks across tile overlaps. It reports counts, percentages, processing statistics, and per-instance CSV/JSON data.

Ground Truth evaluation is optional. Upload the image's matching YOLO Segmentation `.txt` label (`class_id x1 y1 x2 y2 ...`, normalized coordinates) to calculate class-aware mask metrics at the selected IoU threshold: TP/FP/FN, precision, recall, F1, mean mask IoU, accuracy among spatial matches, per-class metrics, and a confusion matrix. Ground Truth is used for metrics and count tables only; the page does not generate a combined prediction/GT image. Without a label, the page remains inference-only and does not claim an accuracy value.

The deployed checkpoint is Approach 1, whose bad/healthy output semantics are reversed. All user-facing outputs therefore apply the fixed mapping `model bad -> displayed healthy`, `model healthy -> displayed bad`, and `impurity -> impurity`. This includes visualization colors and labels, aggregate counts, detection tables, CSV/JSON display classes, and Ground Truth evaluation. Ground Truth IDs and the global class order remain unchanged. Each exported detection also includes `raw_class_id` and `raw_class_name` for auditability.

Dense-image inference uses a layered anti-overcounting and seam-recovery pipeline. Overlapping tiles have non-overlapping midpoint ownership regions, so a mask centroid is accepted by only one first-pass tile. Predictions clipped by an internal tile edge are retained as seam candidates rather than counted. The analyzer groups those candidates into bounded crops centered across the relevant seams, reads each crop directly from the original image, and runs a targeted second inference pass. Complete masks that cross and overlap the original fragments replace those partial detections. Class-agnostic mask/box duplicate merging removes residual cross-tile or cross-pass duplicates. After merging, instances crossing the configurable inner image-border margin (10 px by default) are excluded from visualization, counts, exports, and evaluation because they represent partial grains. When a Ground Truth label is supplied, the same border rule is applied to its instances so ignored edge grains cannot become false negatives or distort the count comparison.


For local use, place the trained Large V2 checkpoint at:

```text
notebooks/model/large_v2/checkpoint_best_regular.pth
```

Then start the application from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m webapp.app
```

Open `http://127.0.0.1:7860`. The first analysis loads the checkpoint; subsequent requests reuse the loaded model. Use only a trusted checkpoint produced by this RF-DETR training project.

On the company server, the app automatically uses:

```text
/home/multispector2/testdino/outputRFDetrV2/training/checkpoint_best_regular.pth
```

Start it on the server:

```bash
cd /home/multispector2/testdino
source .venv/bin/activate
python -m webapp.app --host 127.0.0.1 --port 7860
```

Keep this SSH tunnel open on the local computer, then browse to `http://127.0.0.1:7860`:

```powershell
ssh -N -L 7860:127.0.0.1:7860 multispector2@144.76.63.2
```

## Dataset V2 / RF-DETR Segmentation Large

`notebooks/01_rfdetr_segmentation_large_poc_data_v2.ipynb` is the server notebook for Dataset V2 at `/home/multispector2/wheatv2/dataset_v2`. It runs strict label, leakage, class-order, and query-capacity audits; source-safe splitting; adaptive tiling; a 50-epoch Large run; best-checkpoint Test evaluation; and a matched-instance classification confusion matrix. All server artifacts are written below `/home/multispector2/testdino/outputRFDetrV2`.
