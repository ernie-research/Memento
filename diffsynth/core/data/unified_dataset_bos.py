import warnings
warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated",
    category=UserWarning
)

import torch, torchvision, imageio, os, subprocess, random, shutil
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

try:
    from baidubce.bce_client_configuration import BceClientConfiguration
    from baidubce.auth.bce_credentials import BceCredentials
    from baidubce.services.bos.bos_client import BosClient
    HAS_BOS = True
except ImportError:
    HAS_BOS = False


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators = [] if operators is None else operators

    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width, blur_radius=0):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = F.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=InterpolationMode.BILINEAR
        )
        image = F.center_crop(image, (target_height, target_width))
        if blur_radius > 0:
            kernel_size = int(2 * np.ceil(2 * blur_radius) + 1)
            kernel_size = max(kernel_size if kernel_size % 2 == 1 else kernel_size + 1, 3)
            image = F.gaussian_blur(image, kernel_size=(kernel_size, kernel_size), sigma=blur_radius)
        return image

    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def __call__(self, data, blur_radius=0):
        return self.crop_and_resize(data, *self.get_height_width(data), blur_radius=blur_radius)


class LoadVideoHttp(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1,
                 frame_processor=lambda x: x, ffmpeg_bin=None, ffprobe_bin=None):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.high_fps = 16
        self.ffmpeg_bin = ffmpeg_bin or shutil.which('ffmpeg') or 'ffmpeg'
        self.ffprobe_bin = ffprobe_bin or shutil.which('ffprobe') or 'ffprobe'

    def get_num_frames(self, real_frames):
        num_frames = self.num_frames
        if int(real_frames) < num_frames:
            num_frames = int(real_frames)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def get_unresize_frames(self, http_url):
        def get_video_info(video_path):
            command = [
                self.ffprobe_bin, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "format=duration:stream=width,height",
                "-of", "csv=p=0", video_path,
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            output = result.stdout.strip().replace("\n", ",")
            width, height, duration = output.split(",")
            return int(width), int(height), float(duration)

        expected_path = None
        try:
            width, height, duration = get_video_info(http_url)
            random_suffix = str(random.randint(0, 1000000))
            expected_path = f"/tmp/tmp_video_{random_suffix}_{os.path.basename(http_url)[-20:]}.mp4"
            start_index = (duration - 10) / 2 if duration >= 10 else 0
            cut_duration = 11

            command = [
                self.ffmpeg_bin, '-loglevel', 'quiet', '-ss', str(start_index), '-i', http_url,
                '-t', str(cut_duration), '-r', f'{self.high_fps}',
                '-c:v', 'h264', '-preset', 'fast', '-tune', 'zerolatency',
                '-crf', '18', '-threads', '8', '-an',
                expected_path, '-y'
            ]
            subprocess.run(command, check=True, timeout=120)

            cutted_clip, _, _ = torchvision.io.read_video(
                filename=expected_path, pts_unit='sec', output_format='TCHW')
            cutted_clip = cutted_clip.permute(0, 2, 3, 1)
        except Exception as e:
            print(f"LoadVideoHttp error: {e}")
            cutted_clip = torch.zeros(50, 256, 256, 3)

        num_frames = self.get_num_frames(len(cutted_clip))
        cutted_clip = cutted_clip[:num_frames]
        if expected_path is not None and os.path.exists(expected_path):
            os.remove(expected_path)
        if len(cutted_clip) == 0:
            cutted_clip = torch.zeros(1, 256, 256, 3)
        return cutted_clip

    def get_local_frames(self, video_path):
        """Load frames from a local video file."""
        try:
            cutted_clip, _, _ = torchvision.io.read_video(
                filename=video_path, pts_unit='sec', output_format='TCHW')
            cutted_clip = cutted_clip.permute(0, 2, 3, 1)
        except Exception as e:
            print(f"LoadVideoHttp local error: {e}")
            cutted_clip = torch.zeros(50, 256, 256, 3)

        num_frames = self.get_num_frames(len(cutted_clip))
        cutted_clip = cutted_clip[:num_frames]
        if len(cutted_clip) == 0:
            cutted_clip = torch.zeros(1, 256, 256, 3)
        return cutted_clip

    def __call__(self, data: str):
        if "?" in data:
            cutted_frames = self.get_unresize_frames(data)
        elif data.startswith("http://") or data.startswith("https://"):
            cutted_frames = self.get_unresize_frames(data)
        else:
            cutted_frames = self.get_local_frames(data)

        frame_np = cutted_frames.cpu().numpy().astype(np.uint8)
        frames = []
        for frame_id in range(len(cutted_frames)):
            frame = Image.fromarray(frame_np[frame_id])
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames
