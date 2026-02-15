"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""
from typing import Any, Union, Sequence, Optional
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    FrechetVideoDistance, 
    get_validation_metrics_for_videos, 
    log_video, 
    extract, 
    sigmoid_beta_schedule, 
    convert_zero_ckpt_into_state_dict,
)
from transformers import get_cosine_schedule_with_warmup

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch import distributed as dist

class WarmUpScheduler:
    def __init__(self, optimizer, cfg):
        self.optimizer = optimizer
        self.cfg = cfg

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict) -> None:
        self.__dict__.update(state_dict)

    def step(self, step):
        if step < self.cfg.warmup_steps:
            lr_scale = min(1.0, float(step + 1) / self.cfg.warmup_steps)
            assert len(self.optimizer.param_groups) == 2, "Only support two param groups for now."
            self.optimizer.param_groups[0]["lr"] = lr_scale * self.cfg.mem_beta_lr
            self.optimizer.param_groups[1]["lr"] = lr_scale * self.cfg.lr


class AttentionMemoryTrainer(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model_cfg: DictConfig, model_ckpt: str = None):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.x_shape = cfg.x_shape
        self.external_cond_dim = cfg.external_cond_dim
        self.scheduler = cfg.scheduler

        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise
        self.stabilization_level = cfg.diffusion.stabilization_level

        self.cum_snr_decay = self.cfg.diffusion.cum_snr_decay

        self.validation_step_outputs = []
        self.metrics = cfg.metrics
        self.n_frames = cfg.n_frames  # number of max tokens for the model

        self.snr_clip = cfg.diffusion.snr_clip
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.vae_name = cfg.vae_name
        self.scaling_factor = cfg.scaling_factor

        self.stride = model_cfg.stride
        self._build_model(model_ckpt)
        self._build_buffer()

    def register_data_mean_std(
        self, mean: Union[str, float, Sequence], std: Union[str, float, Sequence], namespace: str = "data"
    ):
        """
        Register mean and std of data as tensor buffer.

        Args:
            mean: the mean of data.
            std: the std of data.
            namespace: the namespace of the registered buffer.
        """
        for k, v in [("mean", mean), ("std", std)]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    v = torch.from_numpy(np.load(v))
                elif v.endswith(".pt"):
                    v = torch.load(v)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                v = torch.tensor(v)
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))

    def _build_model(self, model_ckpt):
        from train_oasis.model.attn_mem_dit import DiT
        self.diffusion_model = DiT(
            input_h=self.model_cfg.input_h,
            input_w=self.model_cfg.input_w,
            patch_size=self.model_cfg.patch_size,
            in_channels=self.model_cfg.in_channels,
            hidden_size=self.model_cfg.hidden_size,
            depth=self.model_cfg.depth,
            num_heads=self.model_cfg.num_heads,
            mlp_ratio=self.model_cfg.mlp_ratio,
            external_cond_dim=self.external_cond_dim,
            max_frames=self.model_cfg.max_frames,
            stride=self.model_cfg.stride,
            stabilization_level=self.stabilization_level,
            clip_noise=self.clip_noise,
            timesteps=self.timesteps,
            delta_update=self.model_cfg.delta_update,
            bptt=self.model_cfg.bptt,
            gradient_ckeckpointing=self.gradient_checkpointing,
            dtype=torch.bfloat16 if "bf16" in self.model_cfg.precision else torch.float32,
        )
        
        if model_ckpt:
            print(f"Loading Diffusion model from {model_ckpt}")
            state_dict = convert_zero_ckpt_into_state_dict(model_ckpt)
            self.diffusion_model.load_state_dict(state_dict, strict=True)

        if self.cfg.vae_name == "oasis":
            from train_oasis.model.vae import AutoencoderKL
            from safetensors.torch import load_model
            self.vae = AutoencoderKL(
                latent_dim=16,
                patch_size=20,
                enc_dim=1024,
                enc_depth=6,
                enc_heads=16,
                dec_dim=1024,
                dec_depth=12,
                dec_heads=16,
                input_height=360,
                input_width=640,
            )
            assert self.cfg.vae_ckpt, "VAE checkpoint is required for oasis VAE."
            load_model(self.vae, self.cfg.vae_ckpt)
            self.vae.eval()
        elif self.cfg.vae_name == "sd_vae":
            assert self.cfg.vae_ckpt
            from diffusers.models import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(self.cfg.vae_ckpt)
            self.vae.eval()
        else:
            self.vae = None
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

        self.validation_fid_model = FrechetInceptionDistance(feature=64) if "fid" in self.metrics else None
        self.validation_lpips_model = LearnedPerceptualImagePatchSimilarity() if "lpips" in self.metrics else None
        self.validation_fvd_model = [FrechetVideoDistance()] if "fvd" in self.metrics else None

    def _build_buffer(self):
        global_nan_number = torch.tensor(0, dtype=torch.int32)
        self.register_buffer("global_nan_number", global_nan_number)
        
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        betas = sigmoid_beta_schedule(self.timesteps).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        register_buffer("alphas_cumprod", alphas_cumprod)
        sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()
        register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        # register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        # register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        register_buffer("snr", snr)
        clipped_snr = self.snr.clone()
        clipped_snr.clamp_(max=self.snr_clip)
        register_buffer("clipped_snr", clipped_snr)

    def configure_optimizers(self):
        # params = tuple(self.diffusion_model.parameters())
        mem_beta_group = []
        else_group = []
        for name, param in self.diffusion_model.named_parameters():
            if "mem_beta" in name:
                mem_beta_group.append(param)
            else:
                else_group.append(param)
        if self.cfg.strategy == "ddp":
            # optimizer_dynamics = torch.optim.AdamW(
            #     params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
            # )
            optimizer_dynamics = torch.optim.AdamW(
                [
                    {"params": mem_beta_group, "lr": self.cfg.mem_beta_lr, "weight_decay": 0.0},
                    {"params": else_group},
                ],
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.optimizer_beta,
            )
            if self.scheduler == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer_dynamics,
                    num_warmup_steps=self.cfg.warmup_steps,
                    num_training_steps=self.trainer.estimated_stepping_batches,
                    num_cycles=0.5,
                )
            else:
                scheduler = WarmUpScheduler(optimizer_dynamics, self.cfg)
            return [optimizer_dynamics], [{"scheduler": scheduler, "interval": "step"}]
        elif self.cfg.strategy == "deepspeed":
            optimizer_dynamics = DeepSpeedCPUAdam(
                [
                    {"params": mem_beta_group, "lr": self.cfg.mem_beta_lr, "weight_decay": 0.0},
                    {"params": else_group},
                ],
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.optimizer_beta,
            )
            if self.scheduler == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer_dynamics,
                    num_warmup_steps=self.cfg.warmup_steps,
                    num_training_steps=46000,# self.trainer.estimated_stepping_batches,
                    num_cycles=0.5,
                )
            else:
                scheduler = WarmUpScheduler(optimizer_dynamics, self.cfg)
            return [optimizer_dynamics], [{"scheduler": scheduler, "interval": "step"}]
        else:
            raise ValueError(f"Unsupported strategy {self.cfg.strategy}.")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

    def lr_scheduler_step(self, scheduler, metric):
        if self.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(step=self.trainer.global_step)

    def compute_loss_weights(self, noise_levels: torch.Tensor):
        snr = self.snr[noise_levels]
        clipped_snr = self.clipped_snr[noise_levels]
        normalized_clipped_snr = clipped_snr / self.snr_clip
        normalized_snr = snr / self.snr_clip

        cum_snr = torch.zeros_like(normalized_snr)
        for t in range(0, noise_levels.shape[0]):
            if t == 0:
                cum_snr[t] = normalized_clipped_snr[t]
            else:
                cum_snr[t] = self.cum_snr_decay * cum_snr[t - 1] + (1 - self.cum_snr_decay) * normalized_clipped_snr[t]

        cum_snr = F.pad(cum_snr[:-1], (0, 0, 1, 0), value=0.0)
        clipped_fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_clipped_snr)

        return clipped_fused_snr * self.snr_clip

    def q_sample(self, x_start, t, noise=None):
        # t random(0, timestep)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @torch.no_grad()
    def vae_encode(self, x):
        if not self.vae:
            return x
        elif self.vae_name == "oasis":
            batch_size, n_frames, c, h, w = x.shape # the order of the first two dimensions can be ignored
            x = rearrange(x, "b t ... -> (b t) ...")
            x = self.vae.encode(x).mean * self.scaling_factor
            x = rearrange(x, "(b t) (h w) c -> b t c h w", b=batch_size, t=n_frames, h=18, w=32, c=16)
            return x
        elif self.vae_name == "sd_vae":
            batch_size, n_frames, c, h, w = x.shape # the order of the first two dimensions can be ignored
            x = rearrange(x, "b t ... -> (b t) ...")
            x = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
            x = rearrange(x, "(b t) ... -> b t ...", b=batch_size, t=n_frames)
            return x
        else:
            raise ValueError(f"Unsupported VAE {self.vae_name}.")
    
    @torch.no_grad()
    def vae_decode(self, x):
        # input: (b, t, c, h, w)
        if not self.vae:
            return x
        elif self.vae_name == "oasis":
            batch_size, n_frames, c, h, w = x.shape
            x = rearrange(x, "b t c h w -> (b t) (h w) c")
            x = self.vae.decode(x / self.scaling_factor)
            x = rearrange(x, "(b t) c h w -> b t c h w", b=batch_size, t=n_frames)
            return x
        elif self.vae_name == "sd_vae":
            batch_size, n_frames, c, h, w = x.shape
            x = rearrange(x, "b t ... -> (b t) ...")
            x = self.vae.decode(x / self.vae.config.scaling_factor).sample
            x = rearrange(x, "(b t) ... -> b t ...", b=batch_size, t=n_frames)
            return x
        else:
            raise ValueError(f"Unsupported VAE {self.vae_name}.")
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)
        xs_gt = xs.clone()
        xs = self.vae_encode(xs)
        noise = torch.randn_like(xs)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        noise_levels = self._generate_noise_levels(xs, masks)
        noised_x = self.q_sample(x_start=xs, t=noise_levels, noise=noise)
        model_pred = self.diffusion_model(
            x=rearrange(noised_x, "t b ... -> b t ..."),
            t=rearrange(noise_levels, "t b -> b t"),
            external_cond=rearrange(conditions, "t b ... -> b t ...") if conditions is not None else None,
        )
        model_pred = rearrange(model_pred, "b t ... -> t b ...")
        nan_number = torch.isnan(model_pred).sum()
        dist.all_reduce(nan_number, op=dist.ReduceOp.SUM)
        if nan_number != 0:
            loss = torch.tensor(0.0, dtype=xs_gt.dtype, requires_grad=True, device=self.device)
            self.global_nan_number += 1
            self.log("training/loss", loss, sync_dist=True, prog_bar=True)
            self.log("training/nan", self.global_nan_number, sync_dist=True, prog_bar=True)
            self.log("training/mem_lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True, prog_bar=False)
            self.log("training/lr", self.trainer.optimizers[0].param_groups[1]["lr"], sync_dist=True, prog_bar=False)
            output_dict = {
                "loss": loss,
            }
            return output_dict
        else:
            loss = F.mse_loss(model_pred, xs.detach(), reduction="none")
            loss_weight = self.compute_loss_weights(noise_levels)
            loss_weight = loss_weight.view(*loss_weight.shape, *((1,) * (loss.ndim - 2)))
            loss = loss * loss_weight
            loss = self.reweight_loss(loss, masks)
        
        # log the loss
        self.log("training/loss", loss, sync_dist=True, prog_bar=True)
        self.log("training/nan", self.global_nan_number, sync_dist=True, prog_bar=True)
        self.log("training/mem_lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True, prog_bar=False)
        self.log("training/lr", self.trainer.optimizers[0].param_groups[1]["lr"], sync_dist=True, prog_bar=False)

        output_dict = {
            "loss": loss,
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        return None
    
    def on_validation_epoch_end(self, namespace="validation") -> None:
        if not self.validation_step_outputs:
            return

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        num_frames, batch_size, *_ = xs.shape
        noise_levels = torch.randint(0, self.timesteps, (num_frames // self.stride, batch_size), device=xs.device)
        noise_levels = noise_levels.repeat_interleave(self.stride, dim=0)
        noise_levels = noise_levels[:num_frames]

        if masks is not None:
            # for frames that are not available, treat as full noise
            discard = ~masks.bool()
            noise_levels = torch.where(discard, torch.full_like(noise_levels, self.timesteps - 1), noise_levels)

        return noise_levels

    def reweight_loss(self, loss, weight=None):
        # Note there is another part of loss reweighting (fused_snr) inside the Diffusion class!
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape)
            weight = rearrange(
                weight,
                "t b ... -> t b ..." + " 1" * expand_dim,
            )
            loss = loss * weight

        return loss.mean()

    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames, c, h, w = xs.shape

        masks = torch.ones(n_frames, batch_size).to(self.device)

        if self.external_cond_dim > 0:
            conditions = batch[1]
            conditions = rearrange(conditions, "b t d -> t b d").contiguous()
        else:
            conditions = None

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b t c ... -> t b c ...").contiguous()

        return xs, conditions, masks

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        xs = (xs - mean) / std
        return xs

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        xs = xs * std + mean
        return xs
