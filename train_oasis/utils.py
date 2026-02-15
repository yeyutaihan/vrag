"""
Adapted from https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/utils.py
Action format derived from VPT https://github.com/openai/Video-Pre-Training
"""

import torch
from torch import nn
from torchvision.io import read_image, read_video
from torchvision.transforms.functional import resize
from einops import rearrange
from typing import Mapping, Sequence, Tuple, Optional
import json

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
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]


def one_hot_actions(actions: Sequence[Mapping[str, int]]) -> torch.Tensor:
    actions_one_hot = torch.zeros(len(actions), len(ACTION_KEYS))
    for i, current_actions in enumerate(actions):
        for j, action_key in enumerate(ACTION_KEYS):
            if action_key.startswith("camera"):
                if action_key == "cameraX":
                    value = current_actions["camera"][0]
                elif action_key == "cameraY":
                    value = current_actions["camera"][1]
                else:
                    raise ValueError(f"Unknown camera action key: {action_key}")
                max_val = 20
                bin_size = 0.5
                num_buckets = int(max_val / bin_size)
                value = (value - num_buckets) / num_buckets
                assert -1 - 1e-3 <= value <= 1 + 1e-3, f"Camera action value must be in [-1, 1], got {value}"
            else:
                value = current_actions[action_key]
                assert 0 <= value <= 1, f"Action value must be in [0, 1] got {value}"
            actions_one_hot[i, j] = value

    return actions_one_hot


IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
VIDEO_EXTENSIONS = {"mp4"}


def load_prompt(path, video_offset=None, n_prompt_frames=1):
    if path.lower().split(".")[-1] in IMAGE_EXTENSIONS:
        print("prompt is image; ignoring video_offset and n_prompt_frames")
        prompt = read_image(path)
        # add frame dimension
        prompt = rearrange(prompt, "c h w -> 1 c h w")
    elif path.lower().split(".")[-1] in VIDEO_EXTENSIONS:
        prompt = read_video(path, pts_unit="sec")[0]
        if video_offset is not None:
            prompt = prompt[video_offset:]
        prompt = prompt[:n_prompt_frames]
        prompt = rearrange(prompt, "t h w c -> t c h w")
    else:
        raise ValueError(f"unrecognized prompt file extension; expected one in {IMAGE_EXTENSIONS} or {VIDEO_EXTENSIONS}")
    assert prompt.shape[0] == n_prompt_frames, f"input prompt {path} had less than n_prompt_frames={n_prompt_frames} frames"
    prompt = resize(prompt, (360, 640))
    # add batch dimension
    prompt = rearrange(prompt, "t c h w -> 1 t c h w")
    prompt = prompt.float() / 255.0
    return prompt


def load_actions(path, action_offset=None):
    if path.endswith(".actions.pt"):
        actions = one_hot_actions(torch.load(path))
    elif path.endswith(".one_hot_actions.pt"):
        actions = torch.load(path, weights_only=True)
    elif path.endswith(".jsonl"):
        with open(path, "r") as f:
            lines = f.readlines()
        actions = [parse_VPT_action(line) for line in lines]
        actions = np.array(actions)
        actions = torch.from_numpy(actions).float()
    elif path.endswith(".npz"):
        actions = np.load(path)["actions"]
        actions = actions[1 : ]
        actions = torch.from_numpy(actions).float()
    else:
        raise ValueError("unrecognized action file extension; expected '.jsonl', '*.actions.pt' or '*.one_hot_actions.pt'")
    if action_offset is not None:
        actions = actions[action_offset:]
    actions = torch.cat([torch.zeros_like(actions[:1]), actions], dim=0)
    # add batch dimension
    actions = rearrange(actions, "t d -> 1 t d")
    return actions

from colorama import Fore

def cyan(x: str) -> str:
    return f"{Fore.CYAN}{x}{Fore.RESET}"

import wandb

is_rank_zero = wandb.run is not None

import numpy as np
import scipy

class FrechetVideoDistance(nn.Module):
    def __init__(self, path="models/fvd_detector/i3d_torchscript.pt"):
        super().__init__()
        assert path is not None, "FVD detector path must be provided"
        self.detector = torch.jit.load(path).eval()
        self.detector_kwargs = dict(rescale=False, resize=True, return_features=True)

    def compute_fvd(self, feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
        mu_gen, sigma_gen = self.compute_stats(feats_fake)
        mu_real, sigma_real = self.compute_stats(feats_real)

        m = np.square(mu_gen - mu_real).sum()
        mat = np.dot(sigma_gen, sigma_real)
        s, _ = scipy.linalg.sqrtm(
            mat, disp=False
        )  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

        return float(fid)


    def compute_stats(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = feats.mean(axis=0)  # [d]
        sigma = np.cov(feats, rowvar=False)  # [d, d]

        return mu, sigma

    @torch.no_grad()
    def compute(self, videos_fake: torch.Tensor, videos_real: torch.Tensor):
        """
        :param videos_fake: predicted video tensor of shape (frame, batch, channel, height, width)
        :param videos_real: ground-truth observation tensor of shape (frame, batch, channel, height, width)
        :return:
        """
        n_frames, batch_size, c, h, w = videos_fake.shape
        if n_frames < 2:
            raise ValueError("Video must have more than 1 frame for FVD")

        videos_fake = videos_fake.permute(1, 2, 0, 3, 4)
        videos_real = videos_real.permute(1, 2, 0, 3, 4)

        # detector takes in tensors of shape [batch_size, c, video_len, h, w] with range -1 to 1
        feats_fake = self.detector(videos_fake, **self.detector_kwargs).cpu().numpy()
        feats_real = self.detector(videos_real, **self.detector_kwargs).cpu().numpy()

        return self.compute_fvd(feats_fake, feats_real)

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import (
    structural_similarity_index_measure,
    universal_image_quality_index,
    mean_squared_error,
    peak_signal_noise_ratio
)

def get_validation_metrics_for_videos(
    observation_hat,
    observation_gt,
    lpips_model: Optional[LearnedPerceptualImagePatchSimilarity] = None,
    fid_model: Optional[FrechetInceptionDistance] = None,
    fvd_model: Optional[FrechetVideoDistance] = None,
):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param lpips_model: a LearnedPerceptualImagePatchSimilarity object from algorithm.common.metrics
    :param fid_model: a FrechetInceptionDistance object  from algorithm.common.metrics
    :param fvd_model: a FrechetVideoDistance object  from algorithm.common.metrics
    :return: a tuple of metrics
    """
    frame, batch, channel, height, width = observation_hat.shape
    observation_gt = observation_gt.contiguous()
    observation_hat = observation_hat.contiguous()
    output_dict = {}
    observation_gt = observation_gt.type_as(observation_hat)  # some metrics don't fully support fp16

    if frame < 9:
        fvd_model = None  # FVD requires at least 9 frames

    if fvd_model is not None:
        output_dict["fvd"] = fvd_model.compute(
            torch.clamp(observation_hat, -1.0, 1.0),
            torch.clamp(observation_gt, -1.0, 1.0),
        )

    # reshape to (frame * batch, channel, height, width) for image losses
    observation_hat = observation_hat.view(-1, channel, height, width)
    observation_gt = observation_gt.view(-1, channel, height, width)

    output_dict["mse"] = mean_squared_error(observation_hat, observation_gt)
    output_dict["psnr"] = peak_signal_noise_ratio(observation_hat, observation_gt, data_range=2.0)
    output_dict["ssim"] = structural_similarity_index_measure(observation_hat, observation_gt, data_range=2.0)
    output_dict["uiqi"] = universal_image_quality_index(observation_hat, observation_gt)
    # operations for LPIPS and FID
    observation_hat = torch.clamp(observation_hat, -1.0, 1.0)
    observation_gt = torch.clamp(observation_gt, -1.0, 1.0)

    if lpips_model is not None:
        lpips_model.update(observation_hat, observation_gt)
        lpips = lpips_model.compute().item()
        # Reset the states of non-functional metrics
        output_dict["lpips"] = lpips
        lpips_model.reset()

    if fid_model is not None:
        observation_hat_uint8 = ((observation_hat + 1.0) / 2 * 255).type(torch.uint8)
        observation_gt_uint8 = ((observation_gt + 1.0) / 2 * 255).type(torch.uint8)
        fid_model.update(observation_gt_uint8, real=True)
        fid_model.update(observation_hat_uint8, real=False)
        fid = fid_model.compute()
        output_dict["fid"] = fid
        # Reset the states of non-functional metrics
        fid_model.reset()

    return output_dict


from pathlib import Path
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Union
from typing_extensions import override
import os
from wandb_osh.hooks import TriggerWandbSyncHook
import time
from lightning.pytorch.loggers.wandb import WandbLogger, _scan_checkpoints, ModelCheckpoint, Tensor
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH


if TYPE_CHECKING:
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run


class SpaceEfficientWandbLogger(WandbLogger):
    """
    A wandb logger that by default overrides artifacts to save space, instead of creating new version.
    A variable expiration_days can be set to control how long older versions of artifacts are kept.
    By default, the latest version is kept indefinitely, while older versions are kept for 5 days.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        expiration_days: Optional[int] = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=False,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )

        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self.expiration_days = expiration_days
        self._last_artifacts = []

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        import wandb

        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        artifacts = []
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                checkpoint_callback.__class__.__name__: {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            if not self._checkpoint_name:
                self._checkpoint_name = f"model-{self.experiment.id}"

            artifact = wandb.Artifact(name=self._checkpoint_name, type="model", metadata=metadata)
            artifact.add_file(p, name="model.ckpt")
            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t
            artifacts.append(artifact)

        for artifact in self._last_artifacts:
            if not self._offline:
                artifact.wait()
            artifact.ttl = timedelta(days=self.expiration_days)
            artifact.save()
        self._last_artifacts = artifacts


class OfflineWandbLogger(SpaceEfficientWandbLogger):
    """
    Wraps WandbLogger to trigger offline sync hook occasionally.
    This is useful when running on slurm clusters, many of which
    only has internet on login nodes, not compute nodes.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=False,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self._offline = offline
        communication_dir = Path(".wandb_osh_command_dir")
        communication_dir.mkdir(parents=True, exist_ok=True)
        self.trigger_sync = TriggerWandbSyncHook(communication_dir)
        self.last_sync_time = 0.0
        self.min_sync_interval = 60
        self.wandb_dir = os.path.join(self._save_dir, "wandb/latest-run")

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        out = super().log_metrics(metrics, step)
        if time.time() - self.last_sync_time > self.min_sync_interval:
            self.trigger_sync(self.wandb_dir)
            self.last_sync_time = time.time()
        return out

def is_run_id(run_id: str) -> bool:
    """Check if a string is a run ID."""
    return len(run_id) == 8 and run_id.isalnum()


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])

def download_latest_checkpoint(run_path: str, download_dir: Path) -> Path:
    api = wandb.Api()
    run = api.run(run_path)

    # Find the latest saved model checkpoint.
    latest = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if latest is None or version_to_int(artifact) > version_to_int(latest):
            latest = artifact

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    latest.download(root=root)
    return root / "model.ckpt"

def log_video(
    observation_hat,
    observation_gt=None,
    step=0,
    namespace="train",
    prefix="video",
    context_frames=0,
    color=(255, 0, 0),
    logger=None,
):
    """
    take in video tensors in range [-1, 1] and log into wandb

    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param step: an int indicating the step number
    :param namespace: a string specify a name space this video logging falls under, e.g. train, val
    :param prefix: a string specify a prefix for the video name
    :param context_frames: an int indicating how many frames in observation_hat are ground truth given as context
    :param color: a tuple of 3 numbers specifying the color of the border for ground truth frames
    :param logger: optional logger to use. use global wandb if not specified
    """
    if not logger:
        logger = wandb
    if observation_gt is None:
        observation_gt = torch.zeros_like(observation_hat)
    observation_hat[:context_frames] = observation_gt[:context_frames]
    # Add red border of 1 pixel width to the context frames
    for i, c in enumerate(color):
        c = c / 255.0
        observation_hat[:context_frames, :, i, [0, -1], :] = c
        observation_hat[:context_frames, :, i, :, [0, -1]] = c
        observation_gt[:, :, i, [0, -1], :] = c
        observation_gt[:, :, i, :, [0, -1]] = c
    video = torch.cat([observation_hat, observation_gt], -1).detach().to(torch.float).cpu().numpy()
    video = np.transpose(np.clip(video, a_min=0.0, a_max=1.0) * 255, (1, 0, 2, 3, 4)).astype(np.uint8)
    # video[..., 1:] = video[..., :1]  # remove framestack, only visualize current frame
    n_samples = len(video)
    # use wandb directly here since pytorch lightning doesn't support logging videos yet
    for i in range(n_samples):
        print(i)
        try:
            temp_video = wandb.Video(video[i], fps=20, format="mp4")
        except Exception as e:
            print(f"Error logging video at index {i}: {e}")
            continue
        logger.log(
            {
                f"{namespace}/step_{step}/{prefix}_{i}": temp_video,
                f"trainer/global_step": step,
            }
        )
        print(i)

def extract(a, t, x_shape):
    f, b = t.shape
    out = a[t]
    return out.reshape(f, b, *((1,) * (len(x_shape) - 2)))

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.right.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.right.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

CAMERA_SCALER = 360.0 / 2400.0

def parse_VPT_action(line:str):
    json_action = json.loads(line)
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
    if 1 in mouse_buttons:
        env_action["use"] = 1
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1

    # convert to onehot
    one_hot = np.zeros(len(NOOP_ACTION)+1)
    one_hot[0] = env_action["ESC"]
    one_hot[1] = env_action["back"]
    one_hot[2] = env_action["drop"]
    one_hot[3] = env_action["forward"]
    one_hot[4] = env_action["hotbar.1"]
    one_hot[5] = env_action["hotbar.2"]
    one_hot[6] = env_action["hotbar.3"]
    one_hot[7] = env_action["hotbar.4"]
    one_hot[8] = env_action["hotbar.5"]
    one_hot[9] = env_action["hotbar.6"]
    one_hot[10] = env_action["hotbar.7"]
    one_hot[11] = env_action["hotbar.8"]
    one_hot[12] = env_action["hotbar.9"]
    one_hot[13] = env_action["inventory"]
    one_hot[14] = env_action["jump"]
    one_hot[15] = env_action["left"]
    one_hot[16] = env_action["right"]
    one_hot[17] = env_action["sneak"]
    one_hot[18] = env_action["sprint"]
    one_hot[19] = env_action["swapHands"]
    one_hot[20] = env_action["camera"][0]
    one_hot[21] = env_action["camera"][1]
    one_hot[22] = env_action["attack"]
    one_hot[23] = env_action["use"]
    one_hot[24] = env_action["pickItem"]

    return one_hot

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

def convert_zero_ckpt_into_state_dict(zero_ckpt_path):
    ckpt = get_fp32_state_dict_from_zero_checkpoint(zero_ckpt_path)
    state_dict = {}
    for key, value in ckpt.items():
        if key.startswith("diffusion_model."):
            state_dict[key[16:]] = value
    return state_dict

import yaml
def load_attn_mem_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        attn_mem = yaml.safe_load(f)
    return attn_mem

def parse_flappy_bird_action(act:int):
    if act == 0:
        return np.array([1, 0])
    elif act == 1:
        return np.array([0, 1])
    else:
        raise ValueError(f"Invalid action: {act}. Expected 0 or 1.")

def test_vae():
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(dir_path)
    from train_oasis.model.vae import VAE_models
    from safetensors.torch import load_model
    from torch import autocast
    device = "cuda"
    vae_path = "pretrained_models/vit-l-20.safetensors"
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    load_model(vae, vae_path)
    vae = vae.to(device).eval()
    scaling_factor = 0.07843137255

    video_paths = ["data/minecraft_easy/0/000000.mp4"]

    fid_model = FrechetInceptionDistance(feature=64)
    lpips_model = LearnedPerceptualImagePatchSimilarity()
    fvd_model = FrechetVideoDistance()

    for video_path in video_paths:
        prompt = read_video(video_path, pts_unit="sec")[0].to(device)
        prompt = rearrange(prompt, "t h w c -> t c h w")
        T, C, H, W = prompt.shape
        prompt = prompt.float() / 255.0
        with torch.no_grad():
            with autocast("cuda", dtype=torch.float):
                latent = vae.encode(prompt * 2 - 1).mean * scaling_factor
        # latent = rearrange(latent, "t (h w) c -> t c h w", t=T, h=H // vae.patch_size, w=W // vae.patch_size)

        with torch.no_grad():
            with autocast("cuda", dtype=torch.float):
                obs = (vae.decode(latent / scaling_factor) + 1) / 2

        obs = obs * 2 - 1
        prompt = prompt * 2 - 1
        
        output_dict = get_validation_metrics_for_videos(
            obs.unsqueeze(1).cpu(),
            prompt.unsqueeze(1).cpu(),
            lpips_model=lpips_model,
            fid_model=fid_model,
            fvd_model=None# fvd_model,
        )
        print(output_dict)

def compute_fov(target, candidates, sample_num=1000, r=5, angle=160):
    """
    Compute the overlap between the target and candidates in terms of field of view (FOV).
    target: (B, M, 4)
    candidates: (B, N, 4)
    sample_num: number of samples to use for computing FOV
    r: radius of the FOV
    Returns: (B, M, N)
    """
    # random sample points in the FOV
    # theta = torch.linspace(-angle / 2, angle / 2, sample_num).unsqueeze(0).expand(target.shape[0], sample_num)
    assert target.shape[2] == 4, "Target must have shape (B, M, 4)"
    assert candidates.shape[2] == 4, "Candidates must have shape (B, N, 4)"
    theta = torch.rand(sample_num, device=target.device) * angle - angle / 2
    theta = theta.unsqueeze(0).unsqueeze(0).expand(target.shape[0], target.shape[1], sample_num) # (B, M, sample_num)
    theta = theta + target[:, :, 3].unsqueeze(2)  # add the yaw of the target (B, M, sample_num)
    theta = theta * torch.pi / 180.0  # convert to radians
    sampled_r = torch.sqrt(torch.rand(sample_num, device=target.device)) * r  # (sample_num)
    sampled_r = sampled_r.unsqueeze(0).unsqueeze(0).expand(target.shape[0], target.shape[1], sample_num)  # (B, M, sample_num)
    x = sampled_r * torch.cos(theta) + target[:, :, 0].unsqueeze(2)  # add the x position of the target (B, M, sample_num)
    z = sampled_r * torch.sin(theta) + target[:, :, 2].unsqueeze(2)  # add the z position of the target (B, M, sample_num)

    target_points = torch.stack([x, z], dim=-1)  # (B, M, sample_num, 2)
    length = torch.norm(target_points.unsqueeze(2) - candidates[..., [0, 2]].unsqueeze(1).unsqueeze(3), dim=-1)  # (B, M, N, sample_num)
    alpha = torch.atan2(
        target_points[:, :, :, 1].unsqueeze(2) - candidates[:, :, 2].unsqueeze(1).unsqueeze(-1),
        target_points[:, :, :, 0].unsqueeze(2) - candidates[:, :, 0].unsqueeze(1).unsqueeze(-1)
    ) * 180 / torch.pi  # (B, M, N, sample_num)
    candidates_angle = torch.where(candidates[..., 3] > 180, candidates[..., 3] - 360, candidates[..., 3])  # normalize yaw to [-180, 180]
    alpha = torch.abs(alpha - candidates_angle.unsqueeze(1).unsqueeze(-1)) # (B, M, N, sample_num)
    alpha = torch.where(alpha > 180, 360 - alpha, alpha)  # normalize angle to [0, 180]
    inside = (length < r) & (alpha < angle / 2) # (B, M, N, sample_num)
    inside = inside.sum(dim=-1)  # (B, M, N)
    inside = inside.float() / sample_num  # normalize by the number of samples
    return inside

def test_sd_vae():
    from diffusers.models import AutoencoderKL
    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    # vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    vae.eval()
    path = "data/maze/eval/20220920T045048-1000.npz"
    data = np.load(path, allow_pickle=True)
    x = data["image"][:5]  # (B, T, C, H, W)
    x = torch.from_numpy(x).float().unsqueeze(0) / 255.0
    x = rearrange(x, "b t h w c -> b t c h w")
    print(x.shape)
    x = (x - 0.5) / 0.5
    origin_x = x.clone()

    batch_size, n_frames, c, h, w = x.shape
    x = rearrange(x, "b t ... -> (b t) ...")
    x = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
    x = rearrange(x, "(b t) ... -> b t ...", b=batch_size, t=n_frames)
    print(x.shape)

    x = rearrange(x, "b t ... -> (b t) ...")
    x = vae.decode(x / vae.config.scaling_factor).sample
    x = rearrange(x, "(b t) ... -> b t ...", b=batch_size, t=n_frames)
    print(x.shape)

    mae_error = torch.abs(x - origin_x).mean()
    print("mae error:", mae_error.item())

if __name__ == "__main__":
    test_sd_vae()