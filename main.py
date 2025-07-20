from custom.models.saint_transformer.train import train_model
from pathlib import Path
import modal
import torch

modal_app = modal.App("neural-learning")
modal_img = (
    modal.Image.debian_slim(python_version="3.12.2")
    .apt_install("git")
    .pip_install(
        [
            "torch",
            "lightning",
            "black",
            "modal",
            "neptune",
            "polars",
            "prodigy-plus-schedule-free",
            "prodigyopt",
            "pyarrow",
            "requests",
            "rich",
        ]
    )
    .pip_install("git+https://github.com/manujosephv/pytorch_tabular.git@main")
    .pip_install("git+https://github.com/PriorLabs/TabPFN.git@main")
    .add_local_dir(Path.cwd() / "custom", remote_path="/root/custom")
    .add_local_dir(Path.cwd() / "datasets", remote_path="/root/datasets")
)
modal_gpu = "H100"

GROUNDED_RAW_DATASET_PATH = Path.cwd() / "datasets" / "sample_horses_v2.csv"

torch.set_float32_matmul_precision("medium")


@modal_app.function(gpu=modal_gpu, image=modal_img, timeout=10800, cpu=2)
def run_with_modal() -> None:
    has_cuda = torch.cuda.is_available()
    print(f"CUDA status: {has_cuda}")

    modal.interact()
    train_model(path_to_csv=GROUNDED_RAW_DATASET_PATH, perform_eval=True)

    return


@modal_app.local_entrypoint()
def main() -> None:
    run_with_modal.remote()
    return


if __name__ == "__main__":
    train_model(path_to_csv=GROUNDED_RAW_DATASET_PATH, perform_eval=True)
