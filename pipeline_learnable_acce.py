import os
import gc
import sys
import glob
import json5
import argparse
import subprocess
import logging
import torch
import torch.distributed as dist
import multiprocessing as mp
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video, str2bool
from extract_keyframes import save_keyframes
from wan.memory2video_learnable_rope_fine_acce import WanM2V_Acce as WanM2V_Learnable

def _parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--story_script_path", type=str, default="./story/neo.json")
    parser.add_argument("--t2v_model_path", type=str, default="./models/Wan2.2-T2V-A14B")
    parser.add_argument("--i2v_model_path", type=str, default="./models/Wan2.2-I2V-A14B")
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--max_memory_size", type=int, default=10)
    parser.add_argument("--input_dir", type=str, default="./input")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_file", type=str, default="./log.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
    parser.add_argument("--convert_model_dtype", action="store_true", default=False, help="Whether to convert model paramerters dtype.")
    parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'], help="The solver used to sample.")
    parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument("--sample_guide_scale", type=float, default=3.5, help="Classifier free guidance scale.")
    parser.add_argument("--frame_num", type=int, default=None, help="How many frames of video are generated. The number should be 4n+1")
    parser.add_argument("--offload_model", action="store_true", help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.")
    parser.add_argument("--t2v_first_shot", action="store_true", help="Whether to generate the first shot with T2V model.")
    parser.add_argument("--m2v_first_shot", action="store_true", help="Whether to generate the first shot with M2V model.")
    parser.add_argument("--mi2v", action="store_true", help="Whether to start from last frame of last video shot with MI2V")
    parser.add_argument("--mm2v", action="store_true", help="Whether to start from last frame of last video shot with MM2V")
    parser.add_argument("--fix", type=int, default=0, help="Whether to fix the first n keyframes.")
    parser.add_argument("--finetune_checkpoint_dir", type=str, default=None, help="The path to the finetune checkpoint.")
    parser.add_argument("--lora_weight_path", type=str, default=None, help="The path to the LoRA weight.")
    parser.add_argument("--lora_rank", type=int, default=None, help="The rank of LoRA weight.")
    parser.add_argument("--storymem_mode", action="store_true", default=False, help="Use baseline fixed MEMORY_ROPE_SHIFT=5 spacing for RoPE.")
    parser.add_argument("--max_memory_frames", type=int, default=10,
                        help="Max number of memory frames per clip (controls last-shot sampling, memory size, and query top-K)")
    parser.add_argument("--split_identity_attn", action="store_true",
                        help="Enable split identity+memory / memory+generation self-attention")
    parser.add_argument("--split_learnable_query", action="store_true",
                        help="Split queries into local + global (must match training config)")
    parser.add_argument("--global_query_num", type=int, default=0,
                        help="Number of global queries (must match training config)")
    parser.add_argument("--use_subject_recon", action="store_true", default=False,
                        help="Enable subject reconstruction: prepend identity frames from story reconstruct_target to enforce character consistency.")
    parser.add_argument("--idt_back_mode", action="store_true", default=False,
                        help="为true则重建拼在后面")
    parser.add_argument("--use_both_query", action="store_true", default=False,
                        help="Use each model's own keyframe_query for memory selection (match training behavior)")
    parser.add_argument("--compile_dit", action="store_true", default=False,
                        help="Apply torch.compile to DiT models for ~10-20%% speedup (first step slower due to compilation)")

    args = parser.parse_args()
    return args

def _init_logging(rank, log_file):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(stream=sys.stdout),
                logging.FileHandler(log_file, mode='a', encoding='utf-8')
            ])
    else:
        logging.basicConfig(
            level=logging.ERROR,
            handlers=[logging.StreamHandler(stream=sys.stdout)]
        )

def main(args):
    ###### Init ######
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank, args.log_file)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if dist.is_initialized():
        base_seed = [args.seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.seed = base_seed[0]

    story_script = json5.load(open(args.story_script_path, "r", encoding="utf-8"))
    os.makedirs(args.output_dir, exist_ok=True)

    # 主题重建：从 story script 读取 reconstruct_target 描述
    recon_caption = None
    if args.use_subject_recon:
        recon_caption = story_script.get("reconstruct_target", None)
        if not recon_caption:
            logging.warning("--use_subject_recon enabled but no 'reconstruct_target' field found in story script, disabling.")
            recon_caption = None
        else:
            logging.info(f"[SubjectRecon] reconstruct_target: {recon_caption[:120]}...")

    ###### Generate first-shot videos ######
    if args.t2v_first_shot:
        t2v_config = WAN_CONFIGS["t2v-A14B"]

        logging.info("Loading T2V model...")
        t2v_model = wan.WanT2V(
            config=t2v_config,
            checkpoint_dir=args.t2v_model_path,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        prompt = story_script["scenes"][0]["video_prompts"][0]
        logging.info(f"Generating Scene 1 / Shot 1: {prompt}")

        video = t2v_model.generate(
            prompt, # batch["raw_text"]
            size=SIZE_CONFIGS[args.size], # batch["t_h_w_list"]
            frame_num=t2v_config.frame_num,
            shift=t2v_config.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=t2v_config.sample_steps, # longbin debug
            # sampling_steps=1,
            guide_scale=args.sample_guide_scale,
            seed=args.seed,
            offload_model=args.offload_model
        )

        prev_shot_video_cpu = None
        if rank == 0:
            save_video(
                tensor=video[None],
                save_file=f"{args.output_dir}/01_01.mp4",
                fps=t2v_config.sample_fps,
                nrow=1, normalize=True, value_range=(-1, 1)
            )
            prev_shot_video_cpu = video.detach().cpu().clone()
            torch.save(prev_shot_video_cpu.squeeze(0), f"{args.output_dir}/01_01_tensor.pt")
            del video
            torch.cuda.empty_cache()

        shared_text_encoder = t2v_model.text_encoder
        shared_vae = t2v_model.vae

        # 显式删除 T2V 的 DiT 模型，释放 GPU 显存
        if hasattr(t2v_model, 'low_noise_model'):
            del t2v_model.low_noise_model
        if hasattr(t2v_model, 'high_noise_model'):
            del t2v_model.high_noise_model
        del t2v_model

        gc.collect()
        torch.cuda.empty_cache()

        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
            # dist.destroy_process_group()
            obj_list = [prev_shot_video_cpu]
            dist.broadcast_object_list(obj_list, src=0)
            prev_shot_video = obj_list[0]
        else:
            prev_shot_video = prev_shot_video_cpu
    else:
        shared_text_encoder = None
        shared_vae = None
        prev_shot_video = None

    ###### Generate next-shot videos ######
    m2v_config = WAN_CONFIGS["m2v-A14B"]
    if args.lora_weight_path is not None:
        # 检查权重文件格式（支持.pth和.safetensors）
        low_noise_path = os.path.join(args.lora_weight_path, "backbone_low_noise.safetensors")
        if not os.path.exists(low_noise_path):
            low_noise_path = os.path.join(args.lora_weight_path, "backbone_low_noise.pth")
            if not os.path.exists(low_noise_path):
                raise FileNotFoundError(f"Cannot find LoRA low noise weights: {low_noise_path}")

        high_noise_path = os.path.join(args.lora_weight_path, "backbone_high_noise.safetensors")
        if not os.path.exists(high_noise_path):
            high_noise_path = os.path.join(args.lora_weight_path, "backbone_high_noise.pth")
            if not os.path.exists(high_noise_path):
                raise FileNotFoundError(f"Cannot find LoRA high noise weights: {high_noise_path}")

        m2v_config.low_noise_lora.weight = low_noise_path
        m2v_config.high_noise_lora.weight = high_noise_path
    if args.lora_rank is not None:
        m2v_config.low_noise_lora.r = m2v_config.low_noise_lora.lora_alpha = args.lora_rank
        m2v_config.high_noise_lora.r = m2v_config.high_noise_lora.lora_alpha = args.lora_rank

    logging.info("Loading M2V model...")
    m2v_model = WanM2V_Learnable(
        config=m2v_config,
        checkpoint_dir=args.i2v_model_path,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        finetune_checkpoint_dir=args.finetune_checkpoint_dir,
        text_encoder=shared_text_encoder, # <--- 传入复用的 T5
        vae=shared_vae,
        split_identity_attn=args.split_identity_attn,
        split_learnable_query=args.split_learnable_query,
        global_query_num=args.global_query_num,
        use_both_query=args.use_both_query,
    )

    if args.compile_dit:
        m2v_model.compile_dit()

    # memory_dir：统一的 memory 持久化目录，generate() 每次从中读取并写回
    memory_dir = os.path.join(args.output_dir, "memory")
    if rank == 0:
        os.makedirs(memory_dir, exist_ok=True)

    t2v_shot_path = f"{args.output_dir}/01_01.mp4"
    if args.t2v_first_shot and os.path.exists(t2v_shot_path) and rank == 0:
        first_shot_prompt = story_script["scenes"][0]["video_prompts"][0]
        save_keyframes(t2v_shot_path, m2v_model, None, first_shot_prompt,
                       split_learnable_query=args.split_learnable_query,
                       global_query_num=args.global_query_num)

    if dist.is_initialized():
        dist.barrier()   # 确保所有 rank 都等 rank 0 写完再进循环

    prev_shot_file = None   # 上一 shot 保存的 .mp4 路径

    # 如果 t2v_first_shot 已经生成了 01_01.mp4，将其作为第二 shot 的 prev_shot_file，
    # generate() 会以空 pool 追加 01_01.mp4 的 10 帧
    if args.t2v_first_shot and not getattr(args, 'm2v_first_shot', False):
        t2v_tensor_path = f"{args.output_dir}/01_01_tensor.pt"
        if os.path.exists(t2v_tensor_path):
            prev_shot_file = t2v_tensor_path

    for scene in story_script["scenes"]:
        scene_num = scene["scene_num"]

        for i, prompt in enumerate(scene["video_prompts"]):
            shot_num = i + 1
            if (not args.m2v_first_shot) and scene_num == 1 and shot_num == 1:
                continue
            logging.info(f"Generating Scene {scene_num} / Shot {shot_num}: {prompt}")
            guide_scale = args.sample_guide_scale
            if args.mi2v and not scene["cut"][i]:
                guide_scale = args.sample_guide_scale
                print(f"current cfg: {guide_scale}")
                _candidate = f"{args.output_dir}/last_frame.jpg"
                first_frame_file = _candidate if os.path.exists(_candidate) else None
                if first_frame_file is None:
                    logging.warning(f"mi2v enabled but {_candidate} not found, falling back to no first-frame conditioning")
            else:
                guide_scale = args.sample_guide_scale + 1.5
                print(f"current cfg: {guide_scale}")
                first_frame_file = None
            if args.mm2v and not scene["cut"][i]:
                _candidate = f"{args.output_dir}/motion_frames.mp4"
                motion_frames_file = _candidate if os.path.exists(_candidate) else None
                if motion_frames_file is None:
                    logging.warning(f"mm2v enabled but {_candidate} not found, falling back to no motion-frame conditioning")
            else:
                motion_frames_file = None

            video, used_memory, used_memory_high, video_full, recon_frames = m2v_model.generate(
                input_prompt=prompt,
                memory_dir=memory_dir,              # 从文件夹读取 memory pool
                prev_shot_video_path=prev_shot_file,# 上一 shot 视频路径，用于更新 memory
                first_frame_file=first_frame_file,
                motion_frames_file=motion_frames_file,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=m2v_config.frame_num,
                shift=m2v_config.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=m2v_config.sample_steps, # longbin debug
                # sampling_steps=1,
                guide_scale=guide_scale,
                seed=args.seed+i,
                offload_model=args.offload_model,
                max_memory_size=args.max_memory_size,
                fix=args.fix,
                num_sample_frames=args.max_memory_frames,
                storymem_mode=args.storymem_mode,
                reconstruct_caption=recon_caption,
                idt_back_mode = args.idt_back_mode
            )

            if rank == 0:
                video_to_save = video

                save_video(
                    tensor=video_to_save[None],
                    save_file=f"{args.output_dir}/{scene_num:02d}_{shot_num:02d}.mp4",
                    fps=m2v_config.sample_fps,
                    nrow=1, normalize=True, value_range=(-1, 1)
                )
                save_video(tensor=video_full[None], save_file=f"{args.output_dir}/{scene_num:02d}_{shot_num:02d}_full.mp4", fps=m2v_config.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
                tensor_save_path = f"{args.output_dir}/{scene_num:02d}_{shot_num:02d}_tensor.pt"
                torch.save(video_to_save.clone().cpu(), tensor_save_path)

            # 必须加一个 Barrier，等 rank 0 存完视频 mp4，因为所有 rank 等下都要读取它
            if dist.is_initialized():
                dist.barrier()

            # 用 generate() 直接返回的 memory（即生成本 shot 时实际使用的 memory），
            # 而非从文件重新加载（避免候选帧混入）
            # low noise keyframes
            save_keyframes(
                f"{args.output_dir}/{scene_num:02d}_{shot_num:02d}.mp4",
                m2v_model, used_memory, prompt,
                split_learnable_query=args.split_learnable_query,
                global_query_num=args.global_query_num)

            # high noise keyframes（use_both_query 模式下才有独立的 high noise pool）
            if used_memory_high is not None:
                save_keyframes(
                    f"{args.output_dir}/{scene_num:02d}_{shot_num:02d}.mp4",
                    m2v_model, used_memory_high, prompt,
                    split_learnable_query=args.split_learnable_query,
                    global_query_num=args.global_query_num,
                    output_prefix=f"{args.output_dir}/{scene_num:02d}_{shot_num:02d}_high.mp4",
                    skip_video_io=True)

            # save_keyframes 中 VAE decode 占用 GPU 显存，需要释放
            torch.cuda.empty_cache()

            # 等待 rank0 写完 last_frame.jpg / motion_frames.mp4，
            # 防止下一 shot 读到未写完的文件导致脸接不上
            if dist.is_initialized():
                dist.barrier()

            if rank == 0:
                del video, video_to_save, video_full, recon_frames
            del used_memory, used_memory_high
            gc.collect()
            torch.cuda.empty_cache()

            # 记录本 shot 路径...
            # prev_shot_file = f"{args.output_dir}/{scene_num:02d}_{shot_num:02d}.mp4"
            ## 42调试
            prev_shot_file = f"{args.output_dir}/{scene_num:02d}_{shot_num:02d}_tensor.pt"

            if dist.is_initialized():
                dist.barrier()


    if rank == 0:
        # 只拼接 XX_YY.mp4 格式的 shot 视频，排除 *_full.mp4 / motion_frames.mp4 等
        import re
        all_mp4 = sorted(glob.glob(f"{args.output_dir}/*.mp4"))
        videos = [v for v in all_mp4 if re.match(r".*/\d+_\d+\.mp4$", v)]
        list_path = os.path.join(args.output_dir, "concat_list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for v in videos:
                f.write(f"file '{os.path.abspath(v)}'\n")
        out = os.path.join(args.output_dir, f"{os.path.basename(args.output_dir)}.mp4")
        ret = subprocess.run(
            ["ffmpeg", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", "-y", out]
        )
        if ret.returncode != 0:
            subprocess.run([
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_path,
                "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-r", "30", "-y", out
            ], check=True)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    logging.info("Finished.")

if __name__ == "__main__":
    args = _parse_args()
    main(args)
