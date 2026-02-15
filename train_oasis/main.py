import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_path)

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from train_oasis.utils import cyan, is_rank_zero, download_latest_checkpoint, is_run_id
from train_oasis.experiment.experiment import VideoPredictionExperiment
from typing import Optional, Union
from lightning.pytorch.loggers.wandb import WandbLogger

exp_registry = dict(
    exp=VideoPredictionExperiment,
)

# import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load()

def build_experiment(
    cfg: DictConfig,
    logger: Optional[WandbLogger] = None,
    ckpt_path: Optional[Union[str, Path]] = None,
) -> VideoPredictionExperiment:
    """
    Build an experiment instance based on registry
    :param cfg: configuration file
    :param logger: optional logger for the experiment
    :param ckpt_path: optional checkpoint path for saving and loading
    :return:
    """
    if cfg.experiment._name not in exp_registry:
        raise ValueError(
            f"Experiment {cfg.experiment._name} not found in registry {list(exp_registry.keys())}. "
            "Make sure you register it correctly in 'experiments/__init__.py' under the same name as yaml file."
        )

    return exp_registry[cfg.experiment._name](cfg, logger, ckpt_path)

def run_local(cfg: DictConfig):
    # delay some imports in case they are not needed in non-local envs for submission
    from utils import OfflineWandbLogger, SpaceEfficientWandbLogger
    
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    # os.environ["NCCL_DEBUG"] = "INFO"

    # Get yaml names
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["dataset"] is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        if cfg_choice["algorithm"] is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]
        if cfg_choice["model"] is not None:
            cfg.model._name = cfg_choice["model"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    if is_rank_zero:
        print(cyan(f"Outputs will be saved to:"), output_dir)
        (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
        (output_dir.parents[1] / "latest-run").symlink_to(output_dir, target_is_directory=True)

    # Set up logging with wandb.
    if cfg.wandb.mode != "disabled":
        # If resuming, merge into the existing run on wandb.
        resume = cfg.get("resume", None)
        name = f"{cfg.name} ({output_dir.parent.name}/{output_dir.name})" if resume is None else None

        if "_on_compute_node" in cfg and cfg.cluster.is_compute_node_offline:
            logger_cls = OfflineWandbLogger
        else:
            logger_cls = SpaceEfficientWandbLogger

        offline = cfg.wandb.mode != "online"
        logger = logger_cls(
            name=name,
            save_dir=str(output_dir),
            offline=offline,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            log_model="all" if not offline else False,
            config=OmegaConf.to_container(cfg),
            id=resume,
        )
    else:
        logger = None

    # Load ckpt
    resume = cfg.get("resume", None)
    load = cfg.get("load", None)
    checkpoint_path = None
    load_id = None
    if load and not is_run_id(load):
        checkpoint_path = load
    if resume:
        load_id = resume
    elif load and is_run_id(load):
        load_id = load
    else:
        load_id = None

    if load_id:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        checkpoint_path = Path("outputs/downloaded") / run_path / "model.ckpt"

    if checkpoint_path and is_rank_zero:
        print(f"Will load checkpoint from {checkpoint_path}")

    # launch experiment
    experiment = build_experiment(cfg, logger, checkpoint_path)
    for task in cfg.experiment.tasks:
        experiment.exec_task(task)

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="latent_diffusion",
)
def run(cfg: DictConfig):
    if "name" not in cfg:
        raise ValueError("must specify a name for the run with command line argument '+name=[name]'")

    if not cfg.wandb.get("entity", None):
        raise ValueError(
            "must specify wandb entity in 'config/config.yaml' or with command line"
            " argument 'wandb.entity=[entity]' \n An entity is your wandb user name or group"
            " name. This is used for logging. If you don't have an wandb account, please signup at https://wandb.ai/"
        )

    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.name)

    # If resuming or loading a wandb ckpt and not on a compute node, download the checkpoint.
    resume = cfg.get("resume", None)
    load = cfg.get("load", None)

    if resume and load:
        raise ValueError(
            "When resuming a wandb run with `resume=[wandb id]`, checkpoint will be loaded from the cloud"
            "and `load` should not be specified."
        )

    if resume:
        load_id = resume
    elif load and is_run_id(load):
        load_id = load
    else:
        load_id = None

    if load_id and "_on_compute_node" not in cfg:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        download_latest_checkpoint(run_path, Path("outputs/downloaded"))

    run_local(cfg)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
