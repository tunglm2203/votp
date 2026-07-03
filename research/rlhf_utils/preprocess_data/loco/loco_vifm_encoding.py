import os
import cv2
import numpy as np
from tqdm import tqdm

from research.video_encoders.s3dg import S3D
from decord import VideoReader  # Must be import after S3D, MMPTModel to prevent core dump
from termcolor import cprint

import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"


N_FRAMES_PER_FEATURES = 100


def prepare_input_for_s3d(video, device):
    if isinstance(video, list):
        input_video = np.array(video)   # SxHxWxC
    elif isinstance(video, np.ndarray):
        input_video = video             # SxHxWxC
    else:
        raise TypeError

    assert input_video.shape[0] == 100 and input_video.shape[-1] == 3, f"Wrong format: {input_video.shape}"
    if N_FRAMES_PER_FEATURES == 100:
        used_indices = np.linspace(0, 99, 100, endpoint=True, dtype=np.int32)
    else:
        raise ValueError
    input_video = input_video[used_indices]
    input_video = input_video.transpose(3, 0, 1, 2) # SxHxWxC -> CxSxHxW
    input_video = (input_video / 255.0).astype(np.float32)  # 0-255 => 0-1
    input_video = input_video[None, :, :, :, :] # CxSxHxW -> 1xCxSxHxW
    input_video = torch.from_numpy(input_video).to(device)
    return input_video


def main():
    root_path_segments = "datasets/locomotion/segments"

    # Constant
    vifm_model = "s3d"
    RAW_SEGMENT_LEN = 100  # Do not change
    # camera_name = "track"
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

    envs = ["hopper-medium-replay-v2", "hopper-medium-expert-v2", "walker2d-medium-replay-v2", "walker2d-medium-expert-v2"]

    all_frame_indices = np.arange(0, RAW_SEGMENT_LEN, 1, dtype=np.int32)
    for env_name in envs:
        sub_path = os.path.join(root_path_segments, "video", env_name)
        cprint(f"Env={env_name}, ViFM={vifm_model}, N_FRAMES_PER_FEATURES={N_FRAMES_PER_FEATURES}, path: {sub_path}", "yellow")

        list_video_feats = []
        for seg_idx in tqdm(range(NUM_RAW_SEGMENTS), desc=env_name):
            # Consider first 10k features is from left of a pair, last 10k from right of a pair
            if seg_idx < NUM_RAW_SEGMENTS // 2:
                clip_segment_url = f"res256_seg{seg_idx}_1.mp4"
            else:
                clip_segment_url = f"res256_seg{seg_idx - NUM_RAW_SEGMENTS // 2}_2.mp4"
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
        features_filename = os.path.join(features_dir, f"{env_name}_feat_{vifm_model}.npz")
        np.savez(features_filename, feature=list_video_feats)


if __name__ == "__main__":
    main()
