"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

import torch
from train_oasis.model.vae import VAE_models
from torchvision.io import write_video
from train_oasis.utils import sigmoid_beta_schedule, load_actions, load_attn_mem_from_yaml
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import argparse
from pprint import pprint
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
from pathlib import Path
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

assert torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
print(f"using device: {device}")

def get_model(yaml_path):
    config = load_attn_mem_from_yaml(yaml_path)
    from train_oasis.model.attn_mem_dit import DiT
    model = DiT(
        input_h=config["input_h"],
        input_w=config["input_w"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        stride=config["stride"],
        max_frames=config["max_frames"],
        delta_update=True,
        dtype=torch.float,
    )
    return model


def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    model = get_model(args.yaml_path)
    print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(args.oasis_ckpt)}...")
    if os.path.isdir(args.oasis_ckpt):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(args.oasis_ckpt)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    elif args.oasis_ckpt.endswith(".pt") or args.oasis_ckpt.endswith(".ckpt"):
        ckpt = torch.load(args.oasis_ckpt, map_location="cpu")['state_dict']
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict)
    elif args.oasis_ckpt.endswith(".safetensors"):
        load_model(model, args.oasis_ckpt)
    else:
        raise ValueError(f"unsupported checkpoint format: {args.oasis_ckpt}")
    model = model.to(device=device, dtype=torch.float).eval()

    # sampling params
    max_frame = model.max_frames # 8
    n_prompt_frames = max_frame // 2 # 4
    total_frames = args.num_frames
    ddim_noise_steps = args.ddim_steps

    video = EncodedVideo.from_path(args.prompt_path, decode_audio=False)
    video = video.get_clip(start_sec=0.0, end_sec=video.duration)["video"]
    video = video.permute(1, 2, 3, 0).numpy()[args.video_offset:args.video_offset+2*max_frame]
    video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
    if args.vae_ckpt is None:
        transform = transforms.Resize((64, 64), antialias=True)
        video = transform(video)[::2]
    else:
        video = video[::2]
    # get input action stream
    actions = load_actions(args.actions_path, action_offset=args.video_offset)[:, :total_frames * 2:2].to(device)
    print("Action shape: ", actions.shape)
    # sampling inputs
    x = video.to(device)
    B = 1
    H, W = x.shape[-2:]
    x = x.reshape(1, -1, 3, H, W)
    x = (x - 0.5) / 0.5

    # vae encoding
    x = x[:, :n_prompt_frames]
    if args.vae_ckpt:
        vae = VAE_models["vit-l-20-shallow-encoder"]()
        print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(args.vae_ckpt)}...")
        if args.vae_ckpt.endswith(".pt"):
            vae_ckpt = torch.load(args.vae_ckpt, weights_only=True)
            vae.load_state_dict(vae_ckpt)
        elif args.vae_ckpt.endswith(".safetensors"):
            load_model(vae, args.vae_ckpt)
        vae = vae.to(device).eval()
        scaling_factor = 0.07843137255
        x = rearrange(x, "b t c h w -> (b t) c h w")
        with torch.no_grad():
            with autocast("cuda", dtype=torch.float):
                x = vae.encode(x).mean * scaling_factor
        x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)

    print("x shape: ", x.shape)
    
    with torch.no_grad():
        x = model.sample(x, n_context_frames=n_prompt_frames, n_frames=total_frames, sampling_timesteps=ddim_noise_steps, external_cond=actions)

    # save video
    if args.vae_ckpt:
        # vae decoding
        x = rearrange(x, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            x = vae.decode(x / scaling_factor)
        x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)
    x = x * 0.5 + 0.5
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte()
    x = x[0].cpu().numpy()
    write_video(args.output_path, x, fps=args.fps)
    print(f"generation saved to {args.output_path}.")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--oasis-ckpt",
        type=str,
        help="Path to Oasis DiT checkpoint.",
        default="outputs/2025-02-08/15-23-01/checkpoints/epoch=0-step=12000.ckpt",
    )
    parse.add_argument(
        "--yaml-path",
        type=str,
        help="path of model yaml file",
        default="config/model/attn_mem_dit.yaml",
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
        default=120,
    )
    parse.add_argument(
        "--prompt-path",
        type=str,
        help="Path to image or video to condition generation on.",
        default="data/VPT/validation/bumpy-pumpkin-dunker-f29a8a62df0a-20220209-203659.mp4",
    )
    parse.add_argument(
        "--actions-path",
        type=str,
        help="File to load actions from (.actions.pt or .one_hot_actions.pt)",
        default="data/VPT/validation/bumpy-pumpkin-dunker-f29a8a62df0a-20220209-203659.jsonl",
    )
    parse.add_argument(
        "--video-offset",
        type=int,
        help="If loading prompt from video, index of frame to start reading from.",
        default=30,
    )
    parse.add_argument(
        "--output-path",
        type=str,
        help="Path where generated video should be saved.",
        default="outputs/video/attn_mem.mp4",
    )
    parse.add_argument(
        "--fps",
        type=int,
        help="What framerate should be used to save the output?",
        default=20,
    )
    parse.add_argument("--ddim-steps", type=int, help="How many DDIM steps?", default=36)

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
