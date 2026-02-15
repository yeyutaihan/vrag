from train_oasis.dataset.minecraft_pos import MinecraftPosDataset
from train_oasis.trainer.DF_Trainer import DiffusionForcingVideo
from train_oasis.trainer.Hist_Buffer_Trainer import HistBufferTrainer
from train_oasis.trainer.Attn_Mem_Trainer import AttentionMemoryTrainer
from train_oasis.trainer.DF_Rag_Trainer import DiffusionForcingRagVideo


from typing import Optional, Union
import pathlib
import os

import hydra
import torch
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from omegaconf import DictConfig

from train_oasis.utils import cyan, is_rank_zero

class VideoPredictionExperiment:
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        df_video=DiffusionForcingVideo,
        hist_buffer=HistBufferTrainer,
        attn_mem_video=AttentionMemoryTrainer,
        df_rag=DiffusionForcingRagVideo,
    )

    compatible_datasets = dict(
        # video datasets
        minecraft_pos=MinecraftPosDataset,
    )

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug
        self.logger = logger
        self.ckpt_path = ckpt_path
        self.algo = None

        # set random seed
        if self.cfg.seed is not None:
            pl.seed_everything(self.cfg.seed, workers=True)

    def _build_algo(self, model_ckpt: Optional[str] = None):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        return self.compatible_algorithms[algo_name](self.root_cfg.algorithm, self.root_cfg.model, model_ckpt)

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            if is_rank_zero:
                print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )

    def _build_trainer_callbacks(self):
        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))

    def _build_training_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        train_dataset = self._build_dataset("training")
        shuffle = (
            False if isinstance(train_dataset, torch.utils.data.IterableDataset) else self.cfg.training.data.shuffle
        )
        if shuffle:
            print("Shuffling data")
        else:
            print("Not shuffling data")
        if train_dataset:
            '''
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_GB = total_memory / (1024 ** 3)
                if gpu_memory_GB <= 41:
                    print("Using smaller batch size")
                    batch_size = self.cfg.training.batch_size
                elif 41 < gpu_memory_GB <= 81:
                    print("Using larger batch size")
                    batch_size = self.cfg.training.batch_size * 2
                else:
                    raise ValueError("GPU memory is too large")
            except:
                print("Could not get GPU memory, using default batch size")
                batch_size = self.cfg.training.batch_size
            print(f"Batch size: {batch_size}")
            '''

            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True if self.cfg.training.data.num_workers > 0 else False,
                drop_last=True,
                pin_memory=False,
                # prefetch_factor=8, # pre-load 8 * 32 data points, equal to 32 batch
            )
        else:
            return None

    def _build_validation_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        validation_dataset = self._build_dataset("validation")
        shuffle = (
            False
            if isinstance(validation_dataset, torch.utils.data.IterableDataset)
            else self.cfg.validation.data.shuffle
        )
        if validation_dataset:
            return torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=self.cfg.validation.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.validation.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True if self.cfg.validation.data.num_workers > 0 else False,
                drop_last=True,
            )
        else:
            return None

    def _build_test_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        test_dataset = self._build_dataset("test")
        shuffle = False if isinstance(test_dataset, torch.utils.data.IterableDataset) else self.cfg.test.data.shuffle
        if test_dataset:
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.test.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.test.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True if self.cfg.test.data.num_workers > 0 else False,
            )
        else:
            return None

    def training(self) -> None:
        """
        All training happens here
        """
        if not self.algo:
            self.algo = self._build_algo(self.cfg.model_ckpt)
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / "checkpoints",
                    **self.cfg.training.checkpointing,
                )
            )
        if torch.cuda.device_count() == 1:
            training_strategy = "auto"
        elif self.cfg.training.strategy == "ddp":
            training_strategy = DDPStrategy(find_unused_parameters=True)
        elif self.cfg.training.strategy == "deepspeed":
            training_strategy = DeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,
                offload_parameters=True,
            )
        else:
            raise ValueError(f"Unknown strategy {self.cfg.training.strategy}")
        trainer = pl.Trainer(
            accelerator="gpu",
            logger=self.logger if self.logger else False,
            devices=self.cfg.training.devices,
            num_nodes=self.cfg.num_nodes,
            strategy=training_strategy,
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=False,  # self.cfg.debug,
            num_sanity_val_steps=int(self.cfg.debug),
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time,
            log_every_n_steps=self.cfg.training.log_every_n_steps,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")
        trainer.fit(
            self.algo,
            train_dataloaders=self._build_training_loader(),
            val_dataloaders=self._build_validation_loader(),
            ckpt_path=self.ckpt_path,
        )

    def validation(self) -> None:
        """
        All validation happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        if torch.cuda.device_count() == 1:
            validation_strategy = "auto"
        elif self.cfg.validation.strategy == "ddp":
            validation_strategy = DDPStrategy(find_unused_parameters=True)
        elif self.cfg.validation.strategy == "deepspeed":
            validation_strategy = DeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,
                offload_parameters=True,
            )
        else:
            raise ValueError(f"Unknown strategy {self.cfg.training.strategy}")

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=validation_strategy,
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.validation.inference_mode,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.validate(
            self.algo,
            dataloaders=self._build_validation_loader(),
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            limit_test_batches=self.cfg.test.limit_batch,
            precision=self.cfg.test.precision,
            detect_anomaly=False,  # self.cfg.debug,
        )

        # Only load the checkpoint if only testing. Otherwise, it will have been loaded
        # and further trained during train.
        trainer.test(
            self.algo,
            dataloaders=self._build_test_loader(),
            ckpt_path=self.ckpt_path,
        )

    def _build_dataset(self, split: str) -> Optional[torch.utils.data.Dataset]:
        if split in ["training", "test", "validation"]:
            return self.compatible_datasets[self.root_cfg.dataset._name](self.root_cfg.dataset, split=split)
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")