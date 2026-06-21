"""Smoke test: inspect the forward_features output structure of DINOv3 ViT-S/16."""

from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPO_DIR = PROJECT_ROOT / "external" / "dinov3"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def print_tensor(name, value):
    if isinstance(value, torch.Tensor):
        print(f"{name}: {tuple(value.shape)}")
    else:
        print(f"{name}: {type(value)}")


def main():
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

    print("\nNormal forward output:")
    with torch.no_grad():
        y = model(x)

    print_tensor("model(x)", y)

    print("\nChecking available model methods:")
    useful_methods = [
        name for name in dir(model)
        if "feature" in name.lower() or "intermediate" in name.lower()
    ]

    for name in useful_methods:
        print(name)

    print("\nTrying forward_features:")
    if hasattr(model, "forward_features"):
        with torch.no_grad():
            features = model.forward_features(x)

        print("forward_features type:", type(features))

        if isinstance(features, dict):
            print("Keys:")
            for key, value in features.items():
                print_tensor(key, value)

            if "x_norm_patchtokens" in features:
                patch_tokens = features["x_norm_patchtokens"]
                print("\nPatch tokens found.")
                print_tensor("patch_tokens", patch_tokens)

                batch_size, num_patches, hidden_dim = patch_tokens.shape
                height_patches = width_patches = int(num_patches ** 0.5)

                dense_features = patch_tokens.transpose(1, 2)
                dense_features = dense_features.reshape(
                    batch_size,
                    hidden_dim,
                    height_patches,
                    width_patches,
                )

                print_tensor("dense_features", dense_features)
        else:
            print_tensor("features", features)
    else:
        print("This model does not expose forward_features.")


if __name__ == "__main__":
    main()