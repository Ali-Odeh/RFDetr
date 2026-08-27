# MULTISPECTOR RF-DETR — company server setup

Target project directory:

```text
/home/multispector2/testdino
```

The dataset already exists at:

```text
/home/multispector2/wheat_quality_project/dataset_segmented/wheatdecv1_seg/wheatdecv1_seg
```

## 1. Inspect the server before installing packages

Run these commands in the VS Code remote terminal and record their output:

```bash
cd /home/multispector2/testdino
nvidia-smi
python3 --version
command -v uv || true
```

Do not copy the Windows `.venv`; Linux needs its own environment. The PyTorch CUDA build must be compatible with the server's NVIDIA driver.

## 2. Link the existing dataset

From `/home/multispector2/testdino`:

```bash
ln -s /home/multispector2/wheat_quality_project/dataset_segmented/wheatdecv1_seg/wheatdecv1_seg wheatdecv1_seg
test -f wheatdecv1_seg/data.yaml && echo "dataset link: OK"
```

The link lets the notebook keep using `PROJECT_ROOT / "wheatdecv1_seg"` without duplicating the dataset.

## 3. Create the Linux environment

After confirming that the server driver supports the CUDA build pinned by the project:

```bash
cd /home/multispector2/testdino
uv sync --python 3.11
source .venv/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The last command must print `True` and the server GPU name before training.

## 4. Open the notebook remotely

Open `/home/multispector2/testdino` as the VS Code Remote-SSH folder, then open:

```text
notebooks/01_rfdetr_segmentation_nano_poc.ipynb
```

Select this kernel/interpreter:

```text
/home/multispector2/testdino/.venv/bin/python
```

Run cells from the beginning. The outputs and checkpoints will be written under:

```text
/home/multispector2/testdino/outputs/rfdetr_seg_nano
```

