# don't push DEBUG_MODE = True to Replicate!
DEBUG_MODE = False

MAX_PIXELS = 1024 * 1024
# MAX_PIXELS = 768 * 768

from cog import BasePredictor, BaseModel, Path, Input
import os
from PIL import Image
from typing import Iterator, Optional
import tempfile
import utils
import cv2

os.environ["TORCH_HOME"] = "/src/.torch"

DATA_DIR = Path('data')

class CogOutput(BaseModel):
    files: list[Path]
    name: Optional[str] = None
    thumbnails: Optional[list[Path]] = None
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False


def download_face_file(face_url):
    if not face_url:  
        raise Exception("Missing face or speech file")
    try:
        #ext = os.path.splitext(face_url)[1]
        face_file = utils.download(face_url, DATA_DIR)
    except Exception as e:
        raise Exception(f"Error downloading image file: {e}")
    return face_file


def download_speech_file(speech_url):
    if not speech_url:  
        raise Exception("Missing face or speech file")
    try:
        speech_file = utils.download(speech_url, DATA_DIR)
    except Exception as e:
        raise Exception(f"Error downloading speech file: {e}")
    return speech_file


def average_aspect_ratio(images):
    total_aspect_ratio = 0.0
    for img in images:
        w, h = img.size
        total_aspect_ratio += w / h
    return total_aspect_ratio / len(images)


def smallest_image_size(images):
    min_w, min_h = float('inf'), float('inf')
    for img in images:
        w, h = img.size
        min_w, min_h = min(min_w, w), min(min_h, h)
    return min_w, min_h


def resize_and_center_crop(img, target_width, target_height):
    img_w, img_h = img.size
    
    # Scale down the image such that the smallest dimension fits the target size
    scale_factor = min(target_width / img_w, target_height / img_h)
    new_w = int(img_w * scale_factor)
    new_h = int(img_h * scale_factor)
    img = img.resize((new_w, new_h), Image.ANTIALIAS)
    
    # Center-crop
    left = (new_w - target_width) / 2
    top = (new_h - target_height) / 2
    right = (new_w + target_width) / 2
    bottom = (new_h + target_height) / 2
    img = img.crop((left, top, right, bottom))

    return img


def run_wav2lip(face_file, speech_file, gfpgan, gfpgan_upscale, intro_text=None):

    # setup file paths
    out_dir = Path(tempfile.mkdtemp())
    temp_video_file = out_dir / 'wav2lip_temp.avi'
    temp_frames_dir = out_dir / 'frames'
    output_file = out_dir / 'output.mp4'
    fps = 25
    output_mode = 'frames' if gfpgan else 'audiovideo'

    # if it's a video, switch fps to video fps
    ext = str(face_file).split('.')[-1]
    if ext.lower() in ['.mp4', '.mov', '.avi', '.wmv', '.mkv']:
        cap = cv2.VideoCapture(str(face_file))
        fps = cap.get(cv2.CAP_PROP_FPS)

    # run wav2lip
    cmd = f'python Wav2Lip/inference.py \
            --output "{output_mode}" \
            --checkpoint_path "wav2lip_files/wav2lip_gan.pth" \
            --fps {fps} \
            --face "{face_file}" \
            --audio "{speech_file}" \
            --temp_frames_dir "{temp_frames_dir}" \
            --temp_video_file "{temp_video_file}" \
            --outfile "{output_file}"'
    
    result = os.system(cmd)
    if result != 0:
        #raise Exception("Wav2Lip failed")
        print("Wav2Lip failed, let's just superimpose the image on the audio")
        cmd = f'ffmpeg -y -loop 1 -i {face_file} -i {speech_file} \
                -c:v libx264 -tune stillimage -pix_fmt yuv420p -c:a aac -b:a 128k -shortest {output_file}'
        result = os.system(cmd)
        gfpgan = False
        if result != 0:
            raise Exception("ffmpeg failed")
        print("ffmpeg finished")

    if gfpgan:
        temp_gfpgan_frames_dir = out_dir / 'frames_gfpgan'
        os.makedirs(temp_gfpgan_frames_dir, exist_ok=True)

        cmd = f'python /GFPGAN/inference_gfpgan.py \
                --model_path gfpgan \
                -o {temp_gfpgan_frames_dir} \
                -v 1.3 -s {gfpgan_upscale} --bg_upsampler none \
                -i "{temp_frames_dir}" '

        result = os.system(cmd)
        if result != 0:
            raise Exception("GFPGAN failed")

        cmd = f'ffmpeg -y -i {speech_file} -framerate {fps} \
                -i {temp_gfpgan_frames_dir}/restored_imgs/f%05d.png \
                -r {fps} -c:v libx264 -pix_fmt yuv420p {output_file}'

        result = os.system(cmd)
        if result != 0:
            raise Exception("ffmpeg failed")
        print("ffmpeg finished")

        # cleanup
        utils.try_delete(temp_gfpgan_frames_dir)

    if intro_text:
        output_file_prepended = out_dir / 'output_prepended.mp4'
        width, height, fps = utils.get_video_dimensions(output_file)
        duration = 5
        temp_black_frame_file = out_dir / 'black_frame.mp4'
        utils.make_black_frame_video(width, height, duration, fps, intro_text, temp_black_frame_file)
        utils.concatenate_videos(temp_black_frame_file, output_file, output_file_prepended)
        output_file = output_file_prepended
        utils.try_delete(temp_black_frame_file)

    # cleanup
    utils.try_delete(temp_video_file)
    utils.try_delete(temp_frames_dir)

    return output_file


class Predictor(BasePredictor):

    GENERATOR_OUTPUT_TYPE = Path if DEBUG_MODE else CogOutput

    def setup(self):
        print("cog:setup")
        pass

    def predict(
        self,
        mode: str = Input(
            description="Mode",
            choices=["wav2lip"],
            default="wav2lip",
        ),
        face_url: str = Input(
            description="Image of the face to render", 
            default=None,
        ),
        speech_url: str = Input(
            description="The audio file containing speech to be lip-synced",
            default=None,
        ),
        width: int = Input(
            description="Override width for output (same as face_url if left blank)",
            default=None,
        ),
        height: int = Input(
            description="Override height for output (same as face_url if left blank)",
            default=None,
        ),
        gfpgan: bool = Input(
            description="Whether to apply GFPGAN to the Wav2Lip output",
            default=False,
        ),
        gfpgan_upscale: int = Input(
            description="Upscale factor (only used if GFPGAN is enabled)",
            default=1,
            choices=[1, 2],
        ),
        intro_text: str = Input(
            description="Text for introduction screen (optional)",
            default=None,
        )

    ) -> Iterator[GENERATOR_OUTPUT_TYPE]:
        print("cog:predict")
        print(f"Face_url: {face_url}, speech_url: {speech_url}")

        if mode != "wav2lip":
            raise Exception(f"Unsupported mode: {mode}")

        # download face and speech files
        face_downloads, speech_downloads = {}, {}
        face_urls, speech_urls = str(face_url).split("|"), str(speech_url).split("|")
        assert len(face_urls) == len(speech_urls), "Need same number of face and speech urls"
        for face_url in face_urls:
            if face_url not in face_downloads:
                face_downloads[face_url] = download_face_file(face_url)
        for speech_url in speech_urls:
            if speech_url not in speech_downloads:
                speech_downloads[speech_url] = download_speech_file(speech_url)

        # Resize all faces to the same size
        target_width = width
        target_height = height
        if not target_width or not target_height:
            images = [Image.open(i) for i in list(face_downloads.values())]
            avg_aspect_ratio = average_aspect_ratio(images)
            min_w, min_h = smallest_image_size(images)
            if min_w / min_h > avg_aspect_ratio:
                target_height = min_h
                target_width = int(target_height * avg_aspect_ratio)
            else:
                target_width = min_w
                target_height = int(target_width / avg_aspect_ratio)
            if target_width * target_height > MAX_PIXELS:
                ratio = (target_width * target_height) / MAX_PIXELS
                ratio = ratio ** 0.5
                target_width = int(target_width / ratio)
                target_height = int(target_height / ratio)
            target_width = target_width - (target_width % 2) # make sure even numbers
            target_height = target_height - (target_height % 2)
        for face_download in face_downloads:
            pil_img = Image.open(face_downloads[face_download]).convert('RGB')
            resized_img = resize_and_center_crop(pil_img, int(target_width), int(target_height))
            resized_img_location = tempfile.mktemp(suffix='.png')
            resized_img.save(resized_img_location)
            face_downloads[face_download] = resized_img_location

        # Run wav2lip on each segment
        segments = []
        for face_url, speech_url in zip(face_urls, speech_urls):
            face_file = face_downloads[face_url]
            speech_file = speech_downloads[speech_url]
            output_file = run_wav2lip(face_file, speech_file, gfpgan, gfpgan_upscale, intro_text)
            segments.append(output_file)
            intro_text = False  # only do intro text on first segment

        # Concatenate segments
        if len(segments) > 1:
            output_file = Path(tempfile.mktemp(suffix='.mp4'))
            utils.concatenate_videos(*segments, output_file)
        else:
            output_file = segments[0]
        
        # Return results
        if DEBUG_MODE:
            yield output_file
        else:
            face_file = Path(face_file)
            name = intro_text if intro_text else "wav2lip"
            yield CogOutput(files=[output_file], name=name, thumbnails=[face_file], attributes=None, progress=1.0, isFinal=True)

