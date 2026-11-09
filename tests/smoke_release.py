"""CPU-only release smoke test; optionally checks a historical checkpoint."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    canonical = YOLO("configs/models/cmd_yolo26n_pfgr.yaml")
    legacy = YOLO("configs/legacy/yolo26-cmd-P2.yaml")
    assert list(canonical.model.state_dict()) == list(legacy.model.state_dict())
    print("graph_params", sum(parameter.numel() for parameter in canonical.model.parameters()))
    print("legacy_state_keys_compatible", True)

    if args.checkpoint:
        checkpoint = YOLO(args.checkpoint)
        canonical.model.load_state_dict(checkpoint.model.float().state_dict(), strict=True)
        print("checkpoint_params", sum(parameter.numel() for parameter in checkpoint.model.parameters()))
        print("checkpoint_state_strict_load", True)
        print("checkpoint_loaded", True)


if __name__ == "__main__":
    main()
