import torch
import random
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import json
from torchvision.io import read_video
import os
from tqdm import tqdm
from fractions import Fraction


class MinecraftPosDataset(torch.utils.data.Dataset):
    """
    Minecraft dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.h = cfg.h
        self.w = cfg.w
        self.external_cond_dim = cfg.external_cond_dim
        self.n_frames = (
            cfg.n_frames * cfg.frame_skip
            if split == "training"
            else cfg.n_frames * cfg.frame_skip * cfg.validation_multiplier
        )
        self.frame_skip = cfg.frame_skip
        self.validation_size = cfg.validation_size
        self.save_dir = Path(cfg.save_dir)
        self.action_type = cfg.action_type
        if self.action_type == "pos" or self.action_type == "pos":
            assert self.external_cond_dim == 4, f"external_cond_dim must be 4 for action_type {self.action_type}"
        elif self.action_type == "both":
            assert self.external_cond_dim == 8, f"external_cond_dim must be 8 for action_type {self.action_type}"
        elif self.action_type == "action":
            assert self.external_cond_dim == 4, f"external_cond_dim must be 4 for action_type {self.action_type}"
        else:
            raise ValueError(f"Invalid action_type: {self.action_type}")
        self.save_dir.mkdir(exist_ok=True, parents=True)
        # self.split_dir = self.save_dir / f"{split}"

        self.metadata_path = self.save_dir / "metadata.json"

        if not self.metadata_path.exists():
            # Build dataset
            print(f"Creating dataset in {self.save_dir}...")
            all_data = self.get_data_lengths()
            json.dump(
                {
                    "training": all_data[self.validation_size:],
                    "validation": all_data[: self.validation_size],
                },
                open(self.metadata_path, "w"),
                indent=4,
            )

        self.metadata = json.load(open(self.metadata_path, "r"))
        # self.data_paths = self.get_data_paths(self.split)
        self.data_paths = [Path(x["file"]) for x in self.metadata[self.split]]
        lengths = [x["length"] for x in self.metadata[self.split]]
        lengths = np.array(lengths)
        self.clips_per_video = np.clip(np.array(lengths) - self.n_frames + 1, a_min=1, a_max=None).astype(
            np.int32
        )
        self.cum_clips_per_video = np.cumsum(self.clips_per_video)

    def __len__(self):
        return self.clips_per_video.sum()

    def get_data_paths(self):
        all_path = []
        for sub_dir in self.save_dir.glob("*/"):
            paths = list(sub_dir.glob("*.npz"))
            all_path.extend(paths)
        return all_path

    def get_data_lengths(self):
        paths = self.get_data_paths()
        total_files = len(paths)

        def process_file(path):
            try:
                data = np.load(path)["actions"]
                line_count = len(data)
                return str(path), line_count
            except Exception as e:
                print(f"Skipping file {path} due to error: {e}")
                return None
        lengths = []
        for path in tqdm(paths, total=total_files, desc="Processing files"):
            result = process_file(path)
            if result is not None:
                lengths.append({
                    "file": result[0],
                    "length": result[1]
                })
        return lengths

    def split_idx(self, idx):
        video_idx = np.argmax(self.cum_clips_per_video > idx)
        frame_idx = idx - np.pad(self.cum_clips_per_video, (1, 0))[video_idx]
        return video_idx, frame_idx

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def getitem(self, idx):
        file_idx, frame_idx = self.split_idx(idx)
        action_path = self.data_paths[file_idx]
        video_path = action_path.with_suffix(".mp4")
        start = Fraction(frame_idx, self.cfg.fps)
        end = Fraction((frame_idx + self.n_frames - 1), self.cfg.fps)
        video, _, _ = read_video(str(video_path), start_pts=start, end_pts=end, pts_unit="sec")
        video = video.contiguous().numpy()
        if self.external_cond_dim > 0:
            data = np.load(action_path)["actions"]
            if self.action_type == "pos":
                actions = data[frame_idx : frame_idx + self.n_frames, 4:]
            elif self.action_type == "action":
                actions = data[frame_idx : frame_idx + self.n_frames, :4]
            elif self.action_type == "both":
                actions = data[frame_idx : frame_idx + self.n_frames]
            assert actions.shape == (self.n_frames, self.external_cond_dim), f"actions.shape={actions.shape} != (self.n_frames - 1, self.external_cond_dim), file_idx={file_idx}, frame_idx={frame_idx}"

        nonterminal = np.ones(self.n_frames)
        video = torch.from_numpy(video).float() / 255.0
        video = video.permute(0, 3, 1, 2).contiguous()
        # print(start, end, video.shape, frame_idx, self.n_frames)
        assert video.shape[0] == self.n_frames, f"video.shape[0]={video.shape[0]} != self.n_frames"

        if self.external_cond_dim > 0:
            return (
                video[:: self.frame_skip],
                torch.tensor(actions[:: self.frame_skip], dtype=torch.float32),
                nonterminal[:: self.frame_skip],
            )
        else:
            return (
                video[:: self.frame_skip],
                nonterminal[:: self.frame_skip],
            )