"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

import torch
from train_oasis.model.dit_hist_buffer import DiT_models
from train_oasis.model.vae import VAE_models
from torchvision.io import read_video, write_video
from train_oasis.utils import load_prompt, load_actions, sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import argparse
from pprint import pprint
import os
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

# assert torch.cuda.is_available()
# device = "cuda:0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    # model = DiT_models["DiT-S/2"]()
    model = DiT_models[args.model_name]()
    print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(args.oasis_ckpt)}...")
    if args.oasis_ckpt.endswith(".ckpt"):
        ckpt = torch.load(args.oasis_ckpt, map_location="cpu")['state_dict']
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict)
    elif args.oasis_ckpt.endswith(".pt"):
        ckpt = torch.load(args.oasis_ckpt, weights_only=True)
        model.load_state_dict(ckpt, strict=False)
    elif args.oasis_ckpt.endswith(".safetensors"):
        load_model(model, args.oasis_ckpt)
    elif args.oasis_ckpt.endswith(".bin"):
        ckpt = torch.load(args.oasis_ckpt)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    elif os.path.isdir(args.oasis_ckpt):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(args.oasis_ckpt)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"open_oasis_generate: unsupported dit checkpoint format: {args.oasis_ckpt}")
    model = model.to(device).eval()

    # load VAE checkpoint
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(args.vae_ckpt)}...")
    if args.vae_ckpt.endswith(".pt"):
        vae_ckpt = torch.load(args.vae_ckpt, weights_only=True)
        vae.load_state_dict(vae_ckpt)
    elif args.vae_ckpt.endswith(".safetensors"):
        load_model(vae, args.vae_ckpt)
    else:
        raise ValueError(f"open_oasis_generate: unsupported vae checkpoint format: {args.vae_ckpt}")
    vae = vae.to(device).eval()

    # sampling params
    n_prompt_frames = args.n_prompt_frames
    total_frames = args.num_frames
    max_noise_level = 1000
    ddim_noise_steps = args.ddim_steps
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    noise_abs_max = 6 # open oasis use 20
    stabilization_level = 15

    # get prompt image/video
    x = load_prompt(
        args.prompt_path,
        video_offset=args.video_offset,
        n_prompt_frames=n_prompt_frames,
    )
    print(f"Shape of prompt image/video x is: {x.shape}")

    # get input action stream
    actions = load_actions(args.actions_path, 
                           action_offset=args.video_offset
                           )[:, :total_frames] # open oasis

    print(f"Shape of actions is: {actions.shape}")
    assert actions.shape[1] == total_frames, f"generate: {actions.shape[1]} != {total_frames}"

    # sampling inputs
    x = x.to(device)
    actions = actions.to(device)

    # vae encoding
    B = x.shape[0]
    H, W = x.shape[-2:]
    scaling_factor = 0.07843137255
    x = rearrange(x, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        with autocast("cuda", dtype=torch.half):
            x = vae.encode(x * 2 - 1).mean * scaling_factor # input of vae: range [-1, 1]
    x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
    x = x[:, :n_prompt_frames]

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    # TODO: try bigger window size
    # model.max_frames = args.window_size
    model.max_frames = 134
    num_samples, buffer_size = 10, 124

    # sampling loop
    for i in tqdm(range(n_prompt_frames, total_frames)):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        # print(x.shape[1])
        start_frame = max(0, i + 1 - num_samples)
        buffer_start = max(0, i + 1 - num_samples - buffer_size)

        for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
            # set up noise values
            t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=device)
            t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            # sliding window
            x_curr = x.clone()
            x_curr = x_curr[:, buffer_start:]
            t = t[:, buffer_start:]
            t_next = t_next[:, buffer_start:]
            actions_input = actions[:, buffer_start : i + 1]

            # get model predictions
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    v = model(x_curr, t, actions_input)

            if args.predict_v:
                x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
            else:
                x_start = v
            # x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

            # get frame prediction
            alpha_next = alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]

    # vae decoding
    x = rearrange(x, "b t c h w -> (b t) (h w) c")
    with torch.no_grad():
        x = (vae.decode(x / scaling_factor) + 1) / 2 # output of vae: range[-1, 1]
    x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames) # range[0, 1]

    # save video
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    write_video(args.output_path, x[0].cpu(), fps=args.fps)
    print(f"generation saved to {args.output_path}.")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--oasis-ckpt",
        type=str,
        help="Path to Oasis DiT checkpoint.",
        default="/scratch/gpfs/xh4421/outputs/2025-04-18/03-05-46/checkpoints/epoch=4-step=162000.ckpt",
    )
    parse.add_argument(
        "--model-name",
        type=str,
        help="Model name",
        default="self_train_mc",
    )
    parse.add_argument(
        "--predict_v",
        action="store_true",
        help="Whether the model use predict_v.",
        default=False,
    )
    parse.add_argument(
        "--window_size",
        type=int,
        help="Model window size.",
        default=10,
    )
    parse.add_argument(
        "--vae-ckpt",
        type=str,
        help="Path to Oasis ViT-VAE checkpoint.",
        default="models/vit/vit-l-20.safetensors",
    )
    parse.add_argument(
        "--num-frames",
        type=int,
        help="How many frames should the output be?",
        default=300,
    )
    parse.add_argument(
        "--prompt-path",
        type=str,
        help="Path to image or video to condition generation on.",
        default="/scratch/gpfs/xh4421/data_xh/minecraft_easy/7/000008.mp4",
    )
    parse.add_argument(
        "--actions-path",
        type=str,
        help="File to load actions from (.actions.pt or .one_hot_actions.pt)",
        default="/scratch/gpfs/xh4421/data_xh/minecraft_easy/7/000008.npz",
    )
    parse.add_argument(
        "--video-offset",
        type=int,
        help="If loading prompt from video, index of frame to start reading from.",
        default=None,
    )
    parse.add_argument(
        "--n-prompt-frames",
        type=int,
        help="If the prompt is a video, how many frames to condition on.",
        default=1,
    )
    parse.add_argument(
        "--output-path",
        type=str,
        help="Path where generated video should be saved.",
        default="outputs/inference_videos/video.mp4",
    )
    parse.add_argument(
        "--fps",
        type=int,
        help="What framerate should be used to save the output?",
        default=20,
    )
    parse.add_argument("--ddim-steps", type=int, help="How many DDIM steps?", default=10)

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
