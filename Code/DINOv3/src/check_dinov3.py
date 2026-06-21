"""Smoke test: verify that the DINOv3 backbone loads and runs a forward pass."""

from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPO_DIR = PROJECT_ROOT / "external" / "dinov3"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def main():
    print("Project root:", PROJECT_ROOT)
    print("DINOv3 repo:", REPO_DIR)
    print("Weights:", WEIGHTS_PATH)

    if not REPO_DIR.exists():
        raise FileNotFoundError(f"DINOv3 repo not found: {REPO_DIR}")

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = torch.hub.load(
        str(REPO_DIR),
        "dinov3_vits16",
        source="local",
        weights=str(WEIGHTS_PATH),
    )

    model = model.to(device)
    model.eval()

    x = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(x)

    print("Output type:", type(output))

    if isinstance(output, torch.Tensor):
        print("Output shape:", output.shape)
    elif isinstance(output, dict):
        print("Output keys:", output.keys())
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(key, value.shape)
    else:
        print(output)

    print("DINOv3 loaded successfully.")


if __name__ == "__main__":
    main()
