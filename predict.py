from cog import BasePredictor, Path, Input
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


def run_wav2lip(face_url, speech_url):
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
    output_file = out_dir / 'output.mp4'
    
    # run wav2lip
    cmd = f'python /Wav2Lip/inference.py \
            --checkpoint_path "wav2lip_files/wav2lip_gan.pth" \
            --face "{face_file}" \
            --static 1 \
            --audio "{speech_file}" \
            --temp_video_file "{temp_video_file}" \
            --outfile "{output_file}"'

    result = os.system(cmd)
    
    if result != 0:
        raise Exception("Wav2Lip failed")

    return output_file


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

    ) -> Path:

        if mode == "wav2lip":
            output_file = run_wav2lip(face_url, speech_url)
            return output_file

        elif mode == "complete":
            if not prompt:
                raise Exception("Question must be provided")
            stops = ['\nQuestion', '\nAnswer', '\n', 'Question:']
            completion = complete(prompt, stops, max_tokens=max_tokens, temperature=temperature)
            output_file = Path(tempfile.mkdtemp()) / 'output.txt'
            with open(output_file, 'w') as f:
                f.write(completion)
            return output_file

        else:
            raise Exception("Invalid mode")
