from cog import BasePredictor, BaseModel, Path, Input
import os
import tempfile
import requests
from PIL import Image

from gpt3 import complete

os.environ["TORCH_HOME"] = "/src/.torch"

DATA_DIR = Path('data')


def download(url, folder, ext):
    filename = url.split('/')[-1]+ext
    filepath = folder / filename
    if filepath.exists():
        return filepath
    raw_file = requests.get(url, stream=True).raw
    with open(filepath, 'wb') as f:
        f.write(raw_file.read())
    return filepath


class CogOutput(BaseModel):
    file: Path
    thumbnail: Path
    attributes: dict


def run_wav2lip(face_url, speech_url, gfpgan, gfpgan_upscale):
    if not face_url or not speech_url:
        raise Exception("Missing face or speech file")

    try:
        face_file = download(face_url, DATA_DIR, '.jpg')
    except Exception as e:
        raise Exception(f"Error downloading image file: {e}")

    try:
        speech_file = download(speech_url, DATA_DIR, '.wav')
    except Exception as e:
        raise Exception(f"Error downloading speech file: {e}")

    # setup file paths
    out_dir = Path(tempfile.mkdtemp())
    temp_video_file = out_dir / 'wav2lip_temp.avi'
    temp_frames_dir = out_dir / 'frames'
    output_file = out_dir / 'output.mp4'
    fps = 25
    output_mode = 'frames' if gfpgan else 'audiovideo'

    # run wav2lip
    cmd = f'python /Wav2Lip/inference.py \
            --output "{output_mode}" \
            --checkpoint_path "wav2lip_files/wav2lip_gan.pth" \
            --fps {fps} \
            --face "{face_file}" \
            --static 1 \
            --audio "{speech_file}" \
            --temp_frames_dir "{temp_frames_dir}" \
            --temp_video_file "{temp_video_file}" \
            --outfile "{output_file}"'

    result = os.system(cmd)
    if result != 0:
        raise Exception("Wav2Lip failed")

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

        cmd = f'ffmpeg -y -i {speech_file} \
                -i {temp_gfpgan_frames_dir}/restored_imgs/f%05d.png \
                -r {fps} -c:v libx264 -pix_fmt yuv420p {output_file}'

        result = os.system(cmd)
        if result != 0:
            raise Exception("ffmpeg failed")

    # cleanup
    os.system(f'rm {temp_video_file}')
    os.system(f'rm -rf {temp_frames_dir}')
    os.system(f'rm -rf {temp_gfpgan_frames_dir}')

    return output_file


def run_complete(prompt, max_tokens, temperature):
    if not prompt:
        raise Exception("Question must be provided")

    out_dir = Path(tempfile.mkdtemp())
    output_file = out_dir / 'output.txt'

    stops = ['\nQuestion', '\nAnswer', '\n', 'Question:']
    completion = complete(prompt, stops, max_tokens=max_tokens, temperature=temperature)

    with open(output_file, 'w') as f:
        f.write(completion)

    return output_file, completion


class Predictor(BasePredictor):

    def setup(self):
        pass

    def predict(
        self,
        mode: str = Input(
            description="Mode",
            choices=["wav2lip", "complete"]
        ),
        face_url: str = Input(
            description="Image of the face to render", 
            default=None,
        ),
        speech_url: str = Input(
            description="The audio file containing speech to be lip-synced",
            default=None,
        ),
        gfpgan: bool = Input(
            description="Whether to apply GFPGAN to the Wav2Lip output",
            default=True,
        ),
        gfpgan_upscale: int = Input(
            description="Upscale factor (only used if GFPGAN is enabled)",
            default=1,
            choices=[1, 2],
        ),
        prompt: str = Input(
            description="GPT-3 prompt",
            default=None,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate with GPT-3",
            default=150,
        ),
        temperature: float = Input(
            description="Temperature for GPT-3",
            default=0.9,
        ),

    ) -> CogOutput:

        if mode == "wav2lip":
            output_file = run_wav2lip(face_url, speech_url, gfpgan, gfpgan_upscale)
            return CogOutput(file=output_file, thumbnail=None, attributes={})

        elif mode == "complete":
            output_file, completion = run_complete(prompt, max_tokens, temperature)
            attributes = {"completion": completion}
            return CogOutput(file=output_file, thumbnail=None, attributes=attributes)

        else:
            raise Exception("Invalid mode")
