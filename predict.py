from cog import BasePredictor, Path, Input
import os
import tempfile
import requests
from PIL import Image

from gpt3 import get_gpt3_answer

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
            choices=["wav2lip", "gpt3"]
        ),
        face_url: str = Input(
            description="Image of the face to render", 
            default=None,
        ),
        speech_url: str = Input(
            description="The audio file containing speech to be lip-synced",
            default=None,
        ),
        question: str = Input(
            description="Question to ask GPT-3",
            default=None,
        ),

    ) -> Path:

        if mode == "wav2lip":
            output_file = run_wav2lip(face_url, speech_url)
            return output_file

        elif mode == "gpt3":
            if not question:
                raise Exception("Question must be provided")
            completion = get_gpt3_answer(question)
            output_file = Path(tempfile.mkdtemp()) / 'output.txt'
            with open(output_file, 'w') as f:
                f.write(completion)
            return output_file

        else:
            raise Exception("Invalid mode")
