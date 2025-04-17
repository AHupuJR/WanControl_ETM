import torch, os, imageio, argparse
from torchvision.transforms import v2
import torch.nn as nn
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np

import random
import torch

from typing import List, Tuple, Dict, Any, Union
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor




def read_txt_events(txt_file):
    """
    Read a .txt file of events with lines of the form:
        timestamp x y polarity
    Returns a (N,4) NumPy array of float32: [timestamp, x, y, polarity].
    """
    events_list = []
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                t, x, y, p = parts
                events_list.append([float(t), float(x), float(y), float(p)])
    if len(events_list) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(events_list, dtype=np.float32)


def load_timestamps(timestamps_path):
    """
    Loads timestamps from a file where each line contains a single timestamp.
    Returns a list of floats, sorted ascending.
    """
    if not os.path.isfile(timestamps_path):
        raise FileNotFoundError(f"Cannot find timestamps.txt at {timestamps_path}")
    with open(timestamps_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    timestamps = [float(line) for line in lines]
    return timestamps


def get_txt_files(event_dir):
    """
    Return a sorted list of all .txt event files in `event_dir`.
    """
    if not os.path.isdir(event_dir):
        raise FileNotFoundError(f"No 'event' subfolder found at {event_dir}")
    txt_files = [f for f in os.listdir(event_dir) if f.endswith(".txt")]
    txt_files.sort()
    return txt_files


def process_single_file(index, txt_file, timestamps, event_dir, start_ts, end_ts):
    """
    Worker function for gathering events in [start_ts, end_ts].
    Returns (file_start, subset_events) or None if no overlap or no events.
    """
    file_start = timestamps[index]
    file_end   = timestamps[index + 1] if (index + 1) < len(timestamps) else float('inf')

    # Quick check for overlap
    if file_end < start_ts or file_start > end_ts:
        return None

    txt_path = os.path.join(event_dir, txt_file)
    events = read_txt_events(txt_path)
    if events.shape[0] == 0:
        return None

    # Filter to [start_ts, end_ts]
    mask = (events[:, 0] >= start_ts) & (events[:, 0] <= end_ts)
    subset = events[mask]
    if subset.shape[0] == 0:
        return None

    return (file_start, subset)


def gather_events_in_frame_range(
    txt_files, timestamps, event_dir,
    start_frame_idx, end_frame_idx
):
    """
    Gathers events from the specified [start_frame_idx..end_frame_idx].
    That is, from timestamps[start_frame_idx] to timestamps[end_frame_idx].
    """
    start_ts = timestamps[start_frame_idx]
    end_ts   = timestamps[end_frame_idx] if end_frame_idx < len(timestamps) else timestamps[-1]

    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, txt_file in enumerate(txt_files):
            futures.append(executor.submit(
                process_single_file,
                i, txt_file, timestamps, event_dir,
                start_ts, end_ts
            ))
        for f in futures:
            r = f.result()
            if r is not None:
                results.append(r)

    if len(results) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Sort by file_start
    results.sort(key=lambda x: x[0])
    # Merge
    event_subsets = [r[1] for r in results]
    all_events = np.concatenate(event_subsets, axis=0)
    # Sort by ascending time
    all_events = all_events[np.argsort(all_events[:, 0])]
    return all_events


def get_shift_value(shift_mode: str) -> float:
    """
    Map user-friendly shift mode to a numeric offset in [0..1].
    shift_mode can be:
      - "begin_of_frame" => 0.0
      - "in_the_middle"  => 0.5
      - "end_of_frame"   => 1.0
    """
    mapping = {
        "begin_of_frame": 0.0,
        "in_the_middle":  0.5,
        "end_of_frame":   1.0,
    }
    if shift_mode not in mapping:
        raise ValueError(f"Invalid shift_mode='{shift_mode}'. Must be one of {list(mapping.keys())}.")
    return mapping[shift_mode]


def events_to_voxel_grid(
    events,
    num_bins,
    width,
    height,
    return_format='HWC',
    shift_mode='begin_of_frame',
):
    """
    Build a uniform-time voxel grid with `num_bins`.
    Polarity 0 is mapped to -1. 
    Return shape: 'HWC' => (H, W, num_bins) or 'CHW' => (num_bins, H, W).

    shift_mode controls how we offset the bin index for the earliest event:
      - "begin_of_frame" => no offset
      - "in_the_middle"  => 0.5 offset
      - "end_of_frame"   => 1.0 offset
    """
    if events.shape[0] == 0:
        if return_format == 'CHW':
            return np.zeros((num_bins, height, width), dtype=np.float32)
        else:
            return np.zeros((height, width, num_bins), dtype=np.float32)

    shift_value = get_shift_value(shift_mode)

    events = events[np.argsort(events[:, 0])]   # sort by time
    pol = events[:, 3]
    pol[pol == 0] = -1                         # 0 => -1

    voxel_grid = np.zeros((num_bins, height, width), dtype=np.float32).ravel()

    t0 = events[0, 0]
    t1 = events[-1, 0]
    denom = (t1 - t0) if (t1 > t0) else 1e-9  # avoid divide by zero

    # scaled_ts in [0..(num_bins - 1)] + shift
    scaled_ts = (num_bins - 1) * (events[:, 0] - t0) / denom + shift_value

    # clamp to [0..(num_bins - 1)] to avoid out-of-range
    scaled_ts = np.clip(scaled_ts, 0.0, num_bins - 1 - 1e-9)

    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)

    tis = scaled_ts.astype(int)
    dts = scaled_ts - tis
    vals_left  = pol * (1.0 - dts)
    vals_right = pol * dts

    width_height = width * height
    # Left bin
    valid_left = (tis >= 0) & (tis < num_bins)
    np.add.at(
        voxel_grid,
        xs[valid_left] + ys[valid_left]*width + tis[valid_left]*width_height,
        vals_left[valid_left]
    )
    # Right bin
    valid_right = (tis + 1 < num_bins)
    np.add.at(
        voxel_grid,
        xs[valid_right] + ys[valid_right]*width + (tis[valid_right]+1)*width_height,
        vals_right[valid_right]
    )

    voxel_grid = voxel_grid.reshape(num_bins, height, width)

    if return_format == 'CHW':
        return voxel_grid
    else:
        return voxel_grid.transpose((1, 2, 0))



def collect_subfolders(dataset_root: str) -> List[Dict[str, str]]:
    """
    For each subfolder in dataset_root, check if it has:
     - timestamps.txt
     - event/ subfolder
     - gt/ subfolder
    Return a list of dicts with 'timestamps_path', 'event_dir', 'rgb_dir'.
    """
    subfolders = []
    for entry in sorted(os.listdir(dataset_root)):
        sub_path = os.path.join(dataset_root, entry)
        if not os.path.isdir(sub_path):
            continue
        timestamps_path = os.path.join(sub_path, "timestamps.txt")
        events_dir = os.path.join(sub_path, "event")
        gt_dir     = os.path.join(sub_path, "gt")
        if os.path.isfile(timestamps_path) and os.path.isdir(events_dir) and os.path.isdir(gt_dir):
            subfolders.append({
                "timestamps_path": timestamps_path,
                "event_dir": events_dir,
                "rgb_dir":   gt_dir
            })
    return subfolders



import os
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange

# Reuse your existing helper functions for events
# (read_txt_events, load_timestamps, get_txt_files, etc.)
# You’ll also need events_to_voxel_grid, etc.


class TextEventVideoDataset(torch.utils.data.Dataset):
    """
    Modified so that:
      - "video" is the RGB frames
      - "video_c" is the event voxel
      - We keep the same return dict structure:
         { "text", "video", "video_c", "path", "path_c" }
      - If 'path_c' is not actually an event .txt file, or
        you want it to handle a second video, you can adapt
        the logic as needed.
    """

    def __init__(
        self,
        dataset_root: str,
        frames_per_clip: int = 24,
        num_bins: int = 24,
        shift_mode: str = "begin_of_frame",  # new param
        image_sample_size: Union[int, Tuple[int,int]] = 512,
        #video_sample_size: Union[int, Tuple[int,int]] = 512,
        voxel_channel_mode: str = "repeat",  # "repeat" or "triple_bins"
        load_rgb: bool = True,
        is_i2v : bool = True, 
        out_path : str = "./datasets/event_dataset2/", 
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.frames_per_clip = frames_per_clip
        self.num_bins = num_bins
        self.shift_mode = shift_mode     # new param for shifting event bins
        self.load_rgb = load_rgb
        self.is_i2vs=is_i2v
        self.out_path=out_path

        # unify image_sample_size/video_sample_size into (H,W)
        if isinstance(image_sample_size, int):
            self.image_sample_size = (image_sample_size, image_sample_size)
        else:
            self.image_sample_size = image_sample_size


        # "repeat" => replicate single channel to 3
        # "triple_bins" => 3x bins => shape => (num_bins, 3, H, W)
        if voxel_channel_mode not in ("repeat", "triple_bins"):
            raise ValueError("voxel_channel_mode must be 'repeat' or 'triple_bins'")
        self.voxel_channel_mode = voxel_channel_mode

        # discover subfolders
        subfolders = collect_subfolders(dataset_root)

        # For each subfolder, gather info
        self.folders = []
        self.num_sequences_per_folder = []

        for sf in subfolders:
            ts = load_timestamps(sf["timestamps_path"])
            txt_files = get_txt_files(sf["event_dir"])
            # read how many frames are in 'gt/'
            gt_files = sorted([
                f for f in os.listdir(sf["rgb_dir"]) if f.endswith(".jpg") or f.endswith(".png")
            ])
            if len(gt_files) == 0:
                continue

            # optional check
            if len(gt_files) != len(ts):
                print(f"[WARNING] folder {sf['rgb_dir']} has {len(gt_files)} images but {len(ts)} timestamps")

            num_rgb_frames = len(gt_files)
            num_sequences = num_rgb_frames // self.frames_per_clip
            if num_sequences < 1:
                print(f"[WARNING] folder {sf['rgb_dir']} => not enough frames for a single clip of {self.frames_per_clip}")
                continue

            folder_info = {
                "timestamps": ts,
                "txt_files":  txt_files,
                "event_dir":  sf["event_dir"],
                "rgb_dir":    sf["rgb_dir"],
                "gt_files":   gt_files,  # sorted list of image filenames
                "num_rgb_frames": num_rgb_frames,
                "num_sequences":  num_sequences,
            }
            self.folders.append(folder_info)
            self.num_sequences_per_folder.append(num_sequences)

        # build cumulative sizes
        self.cumulative_sizes = [0]
        for ns in self.num_sequences_per_folder:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + ns)

        # Basic image transforms
        self.image_transform = transforms.Compose([
            #transforms.Resize(min(self.image_sample_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])



    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range (0..{len(self)-1})")

        # figure out folder
        folder_index = None
        for i in range(len(self.folders)):
            if self.cumulative_sizes[i] <= idx < self.cumulative_sizes[i+1]:
                folder_index = i
                break
        folder_info = self.folders[folder_index]
        local_idx = idx - self.cumulative_sizes[folder_index]  # which clip in this folder

        # compute start_frame / end_frame
        start_frame_idx = local_idx * self.frames_per_clip
        end_frame_idx   = start_frame_idx + self.frames_per_clip - 1
        if end_frame_idx >= folder_info["num_rgb_frames"]:
            raise ValueError("Internal indexing error: end_frame_idx out of range")

        # gather events
        all_events = gather_events_in_frame_range(
            txt_files=folder_info["txt_files"],
            timestamps=folder_info["timestamps"],
            event_dir=folder_info["event_dir"],
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx
        )

        if all_events.shape[0] == 0:
            # no events in this range
            return {
                "control_pixel_values": None,
                "pixel_values": None,
                "text": "",
                "data_type": "video",
                "idx": idx
            }

        # figure out max x,y among events
        max_x = int(all_events[:,1].max()) + 1
        max_y = int(all_events[:,2].max()) + 1

        # build voxel
        # We'll do a uniform time approach with num_bins = self.num_bins
        voxel = events_to_voxel_grid(
            events=all_events,
            num_bins=self.num_bins,
            width=max_x,
            height=max_y,
            return_format="HWC",
            shift_mode=self.shift_mode,   # <-- pass shift_mode here
        )  # shape => (H, W, num_bins)

        # shape => (num_bins, H, W) after transpose
        voxel = voxel.transpose((2, 0, 1))  # => (num_bins, H, W)

        if self.voxel_channel_mode == "repeat":
            # => (num_bins, 1, H, W)
            voxel = np.expand_dims(voxel, axis=1)
            # => (num_bins, 3, H, W)
            voxel = np.repeat(voxel, repeats=3, axis=1)
        else:
            # "triple_bins" => we create 3x bins
            triple_num_bins = self.num_bins * 3
            voxel3 = events_to_voxel_grid(
                all_events,
                num_bins=triple_num_bins,
                width=max_x,
                height=max_y,
                return_format="HWC",
                shift_mode=self.shift_mode
            )
            voxel3 = voxel3.transpose((2,0,1))  # (triple_num_bins, H, W)
            # reshape => (self.num_bins, 3, H, W)
            voxel = voxel3.reshape(self.num_bins, 3, voxel3.shape[1], voxel3.shape[2])

        voxel_torch = torch.from_numpy(voxel).float()

        # load frames from start_frame_idx .. end_frame_idx
        frame_paths = folder_info["gt_files"][start_frame_idx : end_frame_idx + 1]
        loaded_frames = []
        if self.load_rgb:
            for fpath in frame_paths:
                full_path = os.path.join(folder_info["rgb_dir"], fpath)
                if not os.path.isfile(full_path):
                    loaded_frames.append(None)
                    continue
                img = Image.open(full_path).convert("RGB")
                img_t = self.image_transform(img)
                loaded_frames.append(img_t)
        else:
            loaded_frames = frame_paths

        # filter out None
        valid_tensors = [x for x in loaded_frames if isinstance(x, torch.Tensor)]
        if len(valid_tensors) == 0:
            return {
                "control_pixel_values": voxel_torch, 
                "pixel_values": None,
                "text": "",
                "data_type": "video",
                "idx": idx
            }

        rgb_frames_4d = torch.stack(valid_tensors, dim=0) # (N, 3, H, W)

        # We'll do a random crop => same approach:
        _, _, H_img, W_img = rgb_frames_4d.shape

        # Possibly resize the voxel to match
        voxel_torch = F.interpolate(voxel_torch, size=(H_img, W_img), mode='bilinear', align_corners=False)

        # random crop
        crop_h, crop_w = self.image_sample_size
        max_top  = H_img - crop_h
        max_left = W_img - crop_w
        if max_top < 0 or max_left < 0:
            raise ValueError(f"Requested crop {crop_h}x{crop_w}, bigger than {H_img}x{W_img}")

        top  = random.randint(0, max_top)
        left = random.randint(0, max_left)

        voxel_cropped = voxel_torch[:, :, top:top+crop_h, left:left+crop_w]
        rgb_cropped   = rgb_frames_4d[:, :, top:top+crop_h, left:left+crop_w]

        # mask logic
        pixel_values = rgb_cropped  # shape => (N, 3, crop_h, crop_w)
        
        if self.is_i2vs:
            # pixel_values: shape (B, C, H, W), dtype=float32 in the range 0‑1
            # pixel_values: (B, C, H, W), float32 in [-1, 1]
            first_frame = pixel_values[0]                     # (C, H, W)

            # ---- denormalize back to [0, 1] ----
            first_frame = (first_frame + 1) * 0.5             # or: first_frame * 0.5 + 0.5
            first_frame = first_frame.clamp(0, 1)

            # ---- convert to uint8 & move channels last ----
            first_frame = (first_frame * 255).round().to(torch.uint8)   # (C, H, W) uint8
            first_frame = rearrange(first_frame, "C H W -> H W C")      # (H, W, C)

            #print("first_frame.shape =", first_frame.shape)   # H × W × C, uint8 0‑255





        pixel_values=  rearrange(pixel_values, "T C H W -> C T H W")
        voxel_cropped=  rearrange(voxel_cropped, "T C H W -> C T H W")

        if self.is_i2vs:
            data = {"text": "", "video": pixel_values, "video_c": voxel_cropped, "path": f"{self.out_path}/video_{idx}.mp4", "path_c": f"{self.out_path}/video_{idx}.mp4", "first_frame": first_frame}
        else:
            data = {"text": "", "video": pixel_values, "video_c": voxel_cropped, "path": f"{self.out_path}/video_{idx}.mp4", "path_c": f"{self.out_path}/video_{idx}.mp4",}
        return data

    # ------------------------------------------------------------
    # The rest is basically your old code for handling videos/images
    # plus a small helper to detect images vs. videos
    # ------------------------------------------------------------
    def is_image(self, file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in [".jpg", ".jpeg", ".png", ".webp"]

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        # CenterCrop & Resize
        frame = self.crop_and_resize(frame)
        frame = self.frame_process(frame)
        # => shape (C, H, W)
        # Reshape to (C, 1, H, W) for consistency
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def load_video(self, file_path):
        # (Same logic as your original code for reading frames with imageio)
        start_frame_id = torch.randint(
            0, self.max_num_frames - (self.num_frames - 1)*self.frame_interval, (1,)
        )[0]

        frames = self.load_frames_using_imageio(
            file_path=file_path,
            max_num_frames=self.max_num_frames,
            start_frame_id=start_frame_id,
            interval=self.frame_interval,
            num_frames=self.num_frames,
            frame_process=self.frame_process
        )
        return frames

    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id,
                                  interval, num_frames, frame_process):
        import imageio
        reader = imageio.get_reader(file_path)
        total_frames = reader.count_frames()
        required_frame_id = start_frame_id + (num_frames - 1)*interval
        if total_frames < max_num_frames:
            reader.close()
            raise ValueError(f"Video {file_path} has {total_frames} frames (< {max_num_frames}).")
        if (total_frames - 1) < required_frame_id:
            reader.close()
            raise ValueError(
                f"Video {file_path} not enough frames: need {required_frame_id}, total {total_frames}."
            )

        frames_list = []
        first_frame = None
        for fidx in range(num_frames):
            frame_idx = start_frame_id + fidx * interval
            frame_np = reader.get_data(frame_idx)
            frame_pil = Image.fromarray(frame_np)
            # Crop & resize
            frame_pil = self.crop_and_resize(frame_pil)
            if first_frame is None:
                first_frame = np.array(frame_pil)  # keep as np array
            # apply transforms => (C, H, W)
            frame_tensor = frame_process(frame_pil)
            frames_list.append(frame_tensor)
        reader.close()

        # => (T, C, H, W)
        frames_tensor = torch.stack(frames_list, dim=0)
        # rearrange => (C, T, H, W)
        frames_tensor = rearrange(frames_tensor, "T C H W -> C T H W")

        if self.is_i2v:
            return (frames_tensor, first_frame)
        else:
            return frames_tensor

    def crop_and_resize(self, image):
        """
        Your custom resizing logic (like in the original).
        """
        w, h = image.size
        scale = max(self.width / w, self.height / h)
        new_h = round(h * scale)
        new_w = round(w * scale)
        image = torchvision.transforms.functional.resize(
            image, (new_h, new_w),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.path_c = [os.path.join(base_path, "train", file_name) for file_name in metadata["control_name"]]
        self.text = metadata["text"].to_list()
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        total_frames = reader.count_frames()
        required_frame_id = start_frame_id + (num_frames - 1) * interval

        if total_frames < max_num_frames:
            reader.close()
            raise ValueError(
                f"Video '{file_path}' has only {total_frames} frames, which is less than the required max_num_frames={max_num_frames}."
            )
        if total_frames - 1 < required_frame_id:
            reader.close()
            raise ValueError(
                f"Video '{file_path}' does not have enough frames: required frame index {required_frame_id}, but only {total_frames} frames available."
            )
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if True:#self.is_i2v:
            print("first_frame.shape=", first_frame.shape)
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        path_c = self.path_c[data_id]
        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
            video = self.load_image(path)
            video_c = self.load_image(path_c)
        else:
            video = self.load_video(path)
            video_c = self.load_video(path_c)
        if self.is_i2v:
            video, first_frame = video
            video_c, _ = video_c
            data = {"text": text, "video": video, "video_c": video_c, "path": path, "path_c": path_c, "first_frame": first_frame}
        else:
            data = {"text": text, "video": video, "video_c": video_c, "path": path, "path_c": path_c}
        print(f'DEBUG: path:{path}, path_c:{path_c}')
        print(f'DEBUG: video.shape:{video.shape}, video_c.shape:{video_c.shape}')


        return data
    

    def __len__(self):
        return len(self.path)

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.path_c = [os.path.join(base_path, "train", file_name) for file_name in metadata["control_name"]]
        self.text = metadata["text"].to_list()
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        total_frames = reader.count_frames()
        required_frame_id = start_frame_id + (num_frames - 1) * interval

        if total_frames < max_num_frames:
            reader.close()
            raise ValueError(
                f"Video '{file_path}' has only {total_frames} frames, which is less than the required max_num_frames={max_num_frames}."
            )
        if total_frames - 1 < required_frame_id:
            reader.close()
            raise ValueError(
                f"Video '{file_path}' does not have enough frames: required frame index {required_frame_id}, but only {total_frames} frames available."
            )
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        path_c = self.path_c[data_id]
        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
            video = self.load_image(path)
            video_c = self.load_image(path_c)
        else:
            video = self.load_video(path)
            video_c = self.load_video(path_c)
        if self.is_i2v:
            video, first_frame = video
            video_c, _ = video_c
            data = {"text": text, "video": video, "video_c": video_c, "path": path, "path_c": path_c, "first_frame": first_frame}
        else:
            data = {"text": text, "video": video, "video_c": video_c, "path": path, "path_c": path_c}
        print(f'DEBUG: path:{path}, path_c:{path_c}')
        print(f'DEBUG: video.shape:{video.shape}, video_c.shape:{video_c.shape}')
        return data
    

    def __len__(self):
        return len(self.path)



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), control_layers=15):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path, control_layers)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        '''batch form TextVideoDataset'''
        # if control, return: {"text": text, "video": video, "video_c": video_c, "path": path, "path_c": path_c}
        # if control & i2v, return: {"text": text, "video": video, "video_c": video_c, "path": path, "path_c": path_c, "first_frame": first_frame}
        text, video, video_c, path, path_c = batch["text"][0], batch["video"], batch["video_c"], batch["path"][0], batch["path_c"][0]
        
        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # control video
            video_c = video_c.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents_c = self.pipe.encode_video(video_c, **self.tiler_kwargs)[0]

            _, _, num_frames, height, width = video.shape
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                
                image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
            else:
                image_emb = {}
            
            data = {
                "latents": latents,
                "latents_c": latents_c,
                "prompt_emb": prompt_emb,
                "image_emb": image_emb,
            }

            save_path = Path(path + ".tensors.pth")
            save_path.parent.mkdir(parents=True, exist_ok=True)  
            torch.save(data, save_path)

            # ---------------- optional metadata.csv ----------------

            print("Path(path)=,", Path(path))
            if Path(path).suffix.lower() == ".mp4" :                   # make sure we’re handling a video
                #@TODO finish here
                root_dir   = Path(path).parent.parent          # …/event_dataset2
                root_dir.mkdir(parents=True, exist_ok=True)
                meta_path  = root_dir / "metadata.csv"
                print("meta_path=,", meta_path)
                file_exists = meta_path.exists()

                file_name     = Path(path).name                # e.g. video_0.mp4
                control_name  = file_name.replace(".mp4", "_c.mp4")
                row           = [file_name, '"video description"', control_name]

                # write header once, then append each new row
                with meta_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["file_name", "text", "control_name"])
                    writer.writerow(row)



class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        print(len(self.path), "videos in metadata.")
        self.path = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")
        # *tensors.pth includes: {"latents": latents, "latents_c": latents_c, "prompt_emb": prompt_emb, "image_emb": image_emb}
        return data
    

    def __len__(self):
        return self.steps_per_epoch



class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None, control_layers=15,
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path], control_layers) # dit init
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path], control_layers)
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            if 1: # control
                model = self.pipe.denoising_model()
                for i in range(control_layers): # copy all expect linear
                    model.control_blocks[i].load_state_dict(model.blocks[i].state_dict(), strict=False) 
                # freeze
                for param in model.parameters():
                    param.requires_grad_(False)
                for name, param in model.named_parameters():
                    if "control_blocks" in name:
                        param.requires_grad_(True)
                print('Copy Done!')
                # self.show_updated_parameters(model)
            else:
                self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    
    def show_updated_parameters(self, model):
        updated_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                module = dict(model.named_modules())[name.rsplit('.', 1)[0]]
                layer_type = "Other"
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    layer_type = "BatchNorm"
                elif "RMSNorm" in str(type(module)):
                    layer_type = "RMSNorm"
                elif "GELU" in str(type(module)) and hasattr(module, "weight"):
                    layer_type = "GELU"
                elif "bias" in name:
                    layer_type = "Bias"
                
                updated_params.append((name, param.shape, layer_type))

        print("| {:<40} | {:<20} | {:<12} |".format("param name", "tensor shape", "op name"))
        print("-" * 80)
        for name, shape, ltype in updated_params:
            print("| {:<40} | {:<20} | {:<12} |".format(name, str(tuple(shape)), ltype))
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bn_params = sum(p.numel() for n,p in model.named_parameters() 
                    if p.requires_grad and isinstance(dict(model.named_modules())[n.rsplit('.',1)[0]], 
                                                    (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)))
        rmsnorm_params = sum(p.numel() for n,p in model.named_parameters() 
                            if p.requires_grad and "RMSNorm" in str(type(dict(model.named_modules())[n.rsplit('.',1)[0]])))
        bias_params = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and "bias" in n)
        print("\narameter update statistics:")
        print(f"Total number of trainable parameters: {total_params}")
        print(f"BatchNorm parameter: {bn_params} ({bn_params/total_params:.2%})")
        print(f"RMSNorm parameter: {rmsnorm_params} ({rmsnorm_params/total_params:.2%})")
        print(f"Bias parameter: {bias_params} ({bias_params/total_params:.2%})")


    def training_step(self, batch, batch_idx):
        # A batch form *tensors.pth includes: {"latents": latents, "latents_c": latents_c, "prompt_emb": prompt_emb, "image_emb": image_emb}
        # Data
        latents = batch["latents"].to(self.device) # video
        latents_c = batch["latents_c"].to(self.device) # control video
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        # control signal
        control_latents = latents_c

        # Compute loss
        noise_pred = self.pipe.denoising_model()( # forward
            noisy_latents, control_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        # noisy_latents: [1, 16, 1, 60, 104], and so is control_latents
        # [1] 812,  
        # prompt_emb['context']: [1, 512, 4096] 
        # extra_input: None
        # image_emb: None

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float()) # noise_pred, training_target: [1, 16, 1, 60, 104]
        loss = loss * self.pipe.scheduler.training_weight(timestep) # timestamp: tensor([688.], device='cuda:0', dtype=torch.bfloat16)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--control_layers",
        type=int,
        default=15,
        help="control_layers.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/work/andrea_alfarano/EventAid-dataset/EvenAid-B",
        help="Path to the main dataset root containing multiple subfolders."
    )
    parser.add_argument(
        "--shift_mode",
        type=str,
        default="in_the_middle",
        choices=["begin_of_frame", "in_the_middle", "end_of_frame"],
        help="Where to align the earliest event in the bins."
    )
    parser.add_argument(
        "--voxel_channel_mode",
        type=str,
        default="repeat",
        choices=["repeat", "triple_bins"],
        help="How to convert voxel bins to 3 channels: "
             "'repeat' (replicate single channel) or 'triple_bins' (3x bins)."
    )
    args = parser.parse_args()
    return args


def data_process(args):
    """
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None
    )
    """

    dataset = TextEventVideoDataset(
        dataset_root= args.dataset_root,# "/work/andrea_alfarano/EventAid-dataset/EvenAid-B",
        frames_per_clip=args.num_frames,
        num_bins=args.num_frames,
        shift_mode= args.shift_mode, #"in_the_middle",
        image_sample_size=[args.height, args.width],
        voxel_channel_mode=  args.voxel_channel_mode,
        load_rgb=True,
        is_i2v=args.image_encoder_path is not None,
        out_path  = args.output_path, 

    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        control_layers=args.control_layers,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
    
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain( # dit init
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        control_layers=args.control_layers,
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
