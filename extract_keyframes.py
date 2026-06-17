import os
import time
import io
import cv2
import decord
import torch
from typing import Tuple, List
import logging

logger = logging.getLogger()

def read_video(video: str | bytes) -> Tuple[torch.Tensor, List[int]]:
    if isinstance(video, bytes):
        fp = io.BytesIO(video)
        vr = decord.VideoReader(fp)
    else:
        vr = decord.VideoReader(video)
    nframes, video_fps = len(vr), vr.get_avg_fps()
    timestamps = torch.FloatTensor([(1 / video_fps) * i for i in range(nframes)])

    indices = torch.linspace(0, nframes - 1, nframes).round().long()
    frames = vr.get_batch(indices.tolist()).asnumpy()
    frames = torch.tensor(frames).permute(0, 3, 1, 2)  
    timestamps = timestamps[indices]
    return frames, timestamps


def save_keyframes(video_path, m2v_model, memory_latents, prompt=None,
                   split_learnable_query=False, global_query_num=0,
                   output_prefix=None, skip_video_io=False):
    """
    直接从已经筛选好的 memory_latents 中解码出关键帧并保存。
    不再进行重复的 query 模型前向传播，彻底解决 OOM 问题！

    当 split_learnable_query=True 时，会在每帧图像左上角叠加标签：
      - 前 (M - global_query_num) 帧标注 "local"
      - 后 global_query_num 帧标注 "global"
    同时生成一张横向拼接的全览图 *_keyframes_grid.jpg。
    """
    st = time.time()
    # output_prefix: 关键帧图片的路径前缀（不含 .mp4），默认与 video_path 相同
    out_base = output_prefix if output_prefix is not None else video_path

    # IO 操作和 VAE 解码只交给 Rank 0 去做，防止死锁
    if m2v_model.rank == 0:
        if memory_latents is None:
            logger.info(f"save_keyframes: {video_path=}, memory_latents is None, skipping latent decoding")
        else:
            M = memory_latents.shape[1]
            logger.info(f"save_keyframes: {video_path=}, Decoding {M} Top-K frames "
                        f"(split_learnable_query={split_learnable_query}, global_query_num={global_query_num}), "
                        f"time={time.time() - st:.3f}s")

            device = m2v_model.device
            # pool 结构：前 (M - global_query_num) 帧 = local，后 global_query_num 帧 = global
            # 与 filter_memory_by_query / _forward_single_item 的约定一致
            local_count = M - global_query_num if (split_learnable_query and global_query_num > 0) else M

            decoded_frames = []
            for i in range(M):
                # 直接按顺序取出 Latent 解码，因为它本身就是 Top-K
                lat = memory_latents[:, i:i+1, :, :].unsqueeze(0).to(device).float()
                pixel = m2v_model.vae.decode(lat)
                pixel = pixel.squeeze().clamp(-1, 1)
                pixel = ((pixel + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
                pixel = cv2.cvtColor(pixel, cv2.COLOR_RGB2BGR)

                # 在图像左上角叠加类型标签
                if split_learnable_query and global_query_num > 0:
                    is_global = i >= local_count
                    label = "global" if is_global else "local"
                    color = (0, 200, 0) if is_global else (200, 200, 0)  # 绿色=global, 黄色=local
                    cv2.putText(pixel, f"KF{i} [{label}]", (8, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(pixel, f"KF{i} [{label}]", (8, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                else:
                    cv2.putText(pixel, f"KF{i}", (8, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(pixel, f"KF{i}", (8, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

                save_path = out_base.replace(".mp4", f"_keyframe{i}.jpg")
                cv2.imwrite(save_path, pixel)
                decoded_frames.append(pixel)

            # 生成横向拼接总览图（所有 Keyframe 一张图）
            if decoded_frames:
                grid = cv2.hconcat(decoded_frames)
                grid_path = out_base.replace(".mp4", "_keyframes_grid.jpg")
                cv2.imwrite(grid_path, grid)
                logger.info(f"save_keyframes: saved grid to {grid_path}")

        # 保存 last_frame.jpg 和 motion_frames.mp4（high noise 调用时跳过）
        if not skip_video_io:
            frames, _ = read_video(video_path)

            last_frame = frames[-1].permute(1, 2, 0).numpy()
            last_frame = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(os.path.dirname(video_path), "last_frame.jpg"), last_frame)

            last_frames = frames[-5:]
            motion_frames_path = os.path.join(os.path.dirname(video_path), "motion_frames.mp4")
            _, _c, h, w = last_frames.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(motion_frames_path, fourcc, 5, (w, h))
            for frame in last_frames:
                frame = frame.permute(1, 2, 0).numpy()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
            writer.release()