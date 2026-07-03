import os
import cv2
import numpy as np
from tqdm import tqdm

from research.video_encoders.s3dg import S3D
from decord import VideoReader  # Must be import after S3D, MMPTModel to prevent core dump
from termcolor import cprint

import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"


N_FRAMES_PER_FEATURES = 16


def prepare_input_for_s3d(video, device):
    if isinstance(video, list):
        input_video = np.array(video)   # SxHxWxC
    elif isinstance(video, np.ndarray):
        input_video = video             # SxHxWxC
    else:
        raise TypeError

    assert input_video.shape[0] == 64 and input_video.shape[-1] == 3, f"Wrong format: {input_video.shape}"
    if N_FRAMES_PER_FEATURES == 16:
        used_indices = np.arange(3, 64 + 1, 4, dtype=np.int32)
    else:
        raise ValueError
    input_video = input_video[used_indices]
    input_video = input_video.transpose(3, 0, 1, 2) # SxHxWxC -> CxSxHxW
    input_video = (input_video / 255.0).astype(np.float32)  # 0-255 => 0-1
    input_video = input_video[None, :, :, :, :] # CxSxHxW -> 1xCxSxHxW
    input_video = torch.from_numpy(input_video).to(device)
    return input_video


def main():
    root_path_segments = "datasets/metaworld/segments"

    # Constant
    vifm_model = "s3d"
    RAW_SEGMENT_LEN = 64  # Do not change
    camera_name = "corner2_v2"
    NUM_RAW_SEGMENTS = 20000  # Do not change

    img_size_dict = {
        "s3d": 250,
        # "videoclip": 224,
        # "r3m": 224,
        # "clip": 336,
    }
    img_size = img_size_dict[vifm_model]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if vifm_model in ["s3d"]:
        model_s3d = S3D('research/video_encoders/s3d_dict.npy')
        model_s3d.load_state_dict(torch.load('research/video_encoders/s3d_howto100m.pth'))
        model_s3d.to(device)
        model_s3d = model_s3d.eval()
    else:
        raise NotImplementedError

    envs = ["mw_door-open-v2", "mw_drawer-open-v2", "mw_plate-slide-v2", "mw_sweep-into-v2"]

    all_frame_indices = np.arange(0, RAW_SEGMENT_LEN, 1, dtype=np.int32)
    for env_name in envs:
        sub_path = os.path.join(root_path_segments, "video", env_name)
        cprint(f"Env={env_name}, ViFM={vifm_model}, N_FRAMES_PER_FEATURES={N_FRAMES_PER_FEATURES}, path: {sub_path}", "yellow")

        list_video_feats = []
        for seg_idx in tqdm(range(NUM_RAW_SEGMENTS), desc=env_name):
            clip_segment_url = f"{camera_name}_res256_seg{seg_idx}.mp4"
            video = VideoReader(os.path.join(sub_path, clip_segment_url))
            frames = video.get_batch(all_frame_indices).asnumpy()

            processed_frames = []
            for t in range(len(frames)):
                resized_im = cv2.resize(frames[t], dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC) # 256->250
                processed_frames.append(resized_im)
            processed_frames = np.array(processed_frames)   # SxHxWxC (0-255)

            # Extract features using video embedding
            if vifm_model in ["s3d"]:
                video_input = prepare_input_for_s3d(processed_frames, device)

                with torch.no_grad():
                    assert video_input.shape[2] == N_FRAMES_PER_FEATURES and video_input.shape[3] == img_size
                    video_output = model_s3d(video_input)
                    video_feat = video_output['video_embedding']

                list_video_feats.append(video_feat.cpu().squeeze(0).numpy())

            else:
                raise NotImplementedError

        list_video_feats = np.array(list_video_feats)
        features_dir = os.path.join(root_path_segments, "features")
        os.makedirs(features_dir, exist_ok=True)
        features_filename = os.path.join(features_dir, f"{env_name}_{camera_name}_feat_{vifm_model}.npz")
        np.savez(features_filename, feature=list_video_feats)


if __name__ == "__main__":
    main()
