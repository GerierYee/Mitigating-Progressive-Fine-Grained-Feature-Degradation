# Mitigating Progressive Fine-Grained Feature Degradation for UAV Small Object Detection

Official code release candidate for **CMD-YOLO26**, a compact detector that
instantiates Progressive Fine-Grained Response Recovery (PFGR) for UAV small
object detection.

> **Release status.** The architecture and runnable training/evaluation code
> are included. The manuscript numbers listed below are reported results; a
> verified public checkpoint and an exact artifact-reproduction table will be
> added after the release checkpoint audit. We intentionally do not claim that
> a fresh run will reproduce every decimal before that audit is complete.

## Overview

Compact feature pyramids can progressively weaken fine-grained object cues
relative to cluttered backgrounds. The paper describes this behavior as
**Progressive Fine-Grained Feature Degradation (PFGD)** and uses
**Object-Background Activation Contrast (OBAC)** as a feature-level diagnostic.
PFGR follows the same propagation path with four stages:

1. **Preserve** local detail during early downsampling with FLP-SPD.
2. **Inject** direct high-resolution supervision through a P2 prediction path.
3. **Transfer** P2/P3/P4 responses with lightweight P2-ASFF.
4. **Refine** prediction features using L-HMSP and FG-BSCA.

## Paper-to-code mapping

| Paper term | Released class | Historical checkpoint/YAML name |
|---|---|---|
| FLP-SPD | `FLPSPD` | `SPD_Downsample` |
| P2-ASFF | `P2ASFF` | `LightASFF` |
| L-HMSP | `LHMSP` | `MHP_Block` |
| FG-BSCA | `FGBSCA` | `CAE_Module` |

Historical aliases are retained so that earlier experiment YAML files and
checkpoints remain loadable. The canonical configuration is
`configs/models/cmd_yolo26n_pfgr.yaml`.

## Repository layout

```text
configs/
  data/                   Dataset path templates
  models/                 Canonical CMD-YOLO26/PFGR model definition
tools/
  train.py                Training entry point
  val.py                  Validation and per-class metric export
  obac.py                 Revised OBAC diagnostic
tests/
  test_pfgr_modules.py    Shape-level module checks
ultralytics/              Ultralytics 8.4.24 plus the PFGR integration
```

The repository is based on **Ultralytics 8.4.24** (upstream commit
`e95ce53a7f7ffbc695d012bd755345adc7d84926`). Unrelated private
diagnostic, temporal-detection, dataset, checkpoint, and server-log files are
not part of this release.

## Installation

```bash
git clone https://github.com/GerierYee/Mitigating-Progressive-Fine-Grained-Feature-Degradation.git
cd Mitigating-Progressive-Fine-Grained-Feature-Degradation

conda create -n pfgr python=3.11 -y
conda activate pfgr
pip install -e .
```

Confirm that the model graph can be constructed:

```bash
python -c "from ultralytics import YOLO; YOLO('configs/models/cmd_yolo26n_pfgr.yaml').info()"
```

## Dataset preparation

Download VisDrone2019-DET and/or UAVDT from their official sources and convert
the annotations to the standard YOLO detection format. Update the `path` field
in `configs/data/visdrone.yaml` or `configs/data/uavdt.yaml`.

The official VisDrone folder layout can be converted with:

```bash
python tools/convert_visdrone.py \
  --source /path/to/official/VisDrone2019-DET \
  --output /path/to/converted/VisDrone
```

Expected layout:

```text
DATASET_ROOT/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

The released evaluator reports the Ultralytics metric on converted YOLO labels.
It is not presented as the official VisDrone challenge-server metric; keep the
evaluation protocol fixed when comparing methods.

## Training

The command below exposes the manuscript-level 800-pixel, 300-epoch protocol.
Adjust `--batch` only when GPU memory requires it and report any change.

```bash
python tools/train.py \
  --model configs/models/cmd_yolo26n_pfgr.yaml \
  --data configs/data/visdrone.yaml \
  --epochs 300 \
  --imgsz 800 \
  --batch 16 \
  --device 0 \
  --seed 0
```

An optional compatible initialization checkpoint may be supplied with
`--pretrained /path/to/checkpoint.pt`. Runs without this flag start from random
initialization; these protocols must not be mixed in the same comparison.

## Validation

```bash
python tools/val.py \
  --weights runs/train/cmd_yolo26_pfgr_seed0/weights/best.pt \
  --data configs/data/visdrone.yaml \
  --imgsz 800 \
  --device 0 \
  --output runs/val/cmd_yolo26_pfgr_seed0.json
```

The JSON output records overall AP, per-class AP, thresholds, image size, and
the evaluation-protocol note.

## Inference

```bash
yolo predict \
  model=/path/to/best.pt \
  source=/path/to/images \
  imgsz=800 \
  device=0
```

## OBAC diagnostic

`tools/obac.py` hooks the feature maps directly entering the detection head,
selects resized small objects, excludes every annotated/ignored region and
letterbox padding from the background pool, and performs equal-size seeded
foreground/background sampling.

```bash
python tools/obac.py \
  --weights /path/to/best.pt \
  --images /path/to/VisDrone2019-DET-val/images \
  --annotations /path/to/VisDrone2019-DET-val/annotations \
  --imgsz 800 \
  --device 0 \
  --output runs/obac/cmd_yolo26_pfgr
```

This is the **revised public diagnostic protocol**. It deliberately fixes
background contamination and sampling ambiguities found during the artifact
audit. Its absolute values must not be mixed with legacy private OBAC tables;
use it to compare checkpoints under the same released protocol.

## Manuscript-reported results

| Dataset | Model | Input | Params | GFLOPs | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|---:|---:|
| VisDrone2019 val | YOLO26 baseline | 800 | 2.38M | 5.2 | 37.5 | 21.8 |
| VisDrone2019 val | CMD-YOLO26 | 800 | 3.23M | 11.2 | 42.3 | 25.4 |
| UAVDT | CMD-YOLO26 | 800 | 3.23M | 12.7 | 34.3 | - |

These values are transcribed from the current manuscript. The checkpoint hash,
runtime environment, and independently regenerated metric JSON will be linked
here when the artifact audit is complete.

## Reproducibility checklist

- [x] Model graph and paper-to-code name mapping
- [x] Training, validation, and inference entry points
- [x] Dataset path templates
- [x] Revised OBAC implementation with explicit sampling protocol
- [x] Dependency and license baseline
- [ ] Audited public checkpoint and SHA-256
- [ ] Multi-seed reproduction summary
- [ ] Device-specific latency reproduction instructions

## Citation

The bibliographic entry will be updated after publication. For the current
manuscript, please use:

```bibtex
@article{ye2026pfgd,
  title   = {Mitigating Progressive Fine-Grained Feature Degradation for UAV Small Object Detection},
  author  = {Ye, Ziqi and Chai, Xiaojie and Tang, Zihuan and Zhai, Yirui and Qin, Yanfei and Sang, Shengbo},
  journal = {Under review},
  year    = {2026}
}
```

## License and acknowledgements

This release is built on Ultralytics 8.4.24 and follows the included
AGPL-3.0 license. Please review the upstream Ultralytics licensing terms before
commercial deployment. We thank the Ultralytics project and the maintainers of
the VisDrone and UAVDT benchmarks.
