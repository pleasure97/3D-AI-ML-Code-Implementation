from pathlib import Path
import wandb

def version_to_int(artifact) -> int:
    return int(artifact.version[1:])

def download_checkpoint(run_id: str, download_dir: Path, version: str | None) -> Path:
    api = wandb.Api()
    run = api.run(run_id)

    chosen = None
    for logged_artifact in run.logged_artifacts():
        if logged_artifact.type != "model" or logged_artifact.state != "COMMITED":
            continue

        if version is None:
            if chosen is None or version_to_int(logged_artifact) > version_to_int(chosen):
                chosen = logged_artifact
        elif version == logged_artifact.version:
            chosen = logged_artifact
            break

    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_id
    chosen.download(root=root)
    return root / "model.ckpt"


def update_checkpoint_path(path: str | None, wandb_config: dict) -> Path | None:
    if path is None:
        return None

    if not str(path).startswith("wandb://"):
        return Path(path)

    # path is named as follows - "wandb://run_id:(version)"
    run_id, *version = path[len("wandb://"):].split(":")

    if len(version) == 0:
        version = None
    elif len(version) == 1:
        version = version[0]
    else:
        raise ValueError("Invalid Version!")

    project = wandb_config["project"]
    return download_checkpoint(f"{project}/{run_id}", Path("checkpoints"), version)
