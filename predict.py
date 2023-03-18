from cog import BasePredictor, BaseModel, Path, Input
import os
from PIL import Image
from typing import Iterator, Optional
import tempfile
import utils

os.environ["TORCH_HOME"] = "/src/.torch"

DATA_DIR = Path('data')

class CogOutput(BaseModel):
    file: Path
    name: Optional[str] = None
    thumbnail: Optional[Path] = None
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False

def run_wav2lip(face_url, speech_url, gfpgan, gfpgan_upscale, intro_text=None):
    if not face_url or not speech_url:  
        raise Exception("Missing face or speech file")

    try:
        ext = os.path.splitext(face_url)[1]
        face_file = utils.download(face_url, DATA_DIR, ext)
    except Exception as e:
        raise Exception(f"Error downloading image file: {e}")

    try:
        speech_file = utils.download(speech_url, DATA_DIR, '.wav')
    except Exception as e:
        raise Exception(f"Error downloading speech file: {e}")

    # setup file paths
    out_dir = Path(tempfile.mkdtemp())
    # out_dir = Path('results')
    temp_video_file = out_dir / 'wav2lip_temp.avi'
    temp_frames_dir = out_dir / 'frames'
    output_file = out_dir / 'output.mp4'
    fps = 25
    output_mode = 'frames' if gfpgan else 'audiovideo'

    # if it's a video, switch fps to video fps
    if ext.lower() in ['.mp4', '.mov', '.avi', '.wmv', '.mkv']:
        cap = cv2.VideoCapture(str(face_file))
        fps = cap.get(cv2.CAP_PROP_FPS)

    # run wav2lip
    cmd = f'python /Wav2Lip/inference.py \
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

    face_file = Path(face_file)

    return output_file, face_file


class Predictor(BasePredictor):

    def setup(self):
        print("cog:setup")
        pass

    def predict(
        self,
        mode: str = Input(
            description="Mode",
            choices=["wav2lip"]
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
        ),

    ) -> Iterator[CogOutput]:

        print("cog:predict")
        print(f"Running in {mode} mode")

        if mode == "wav2lip":
            print(f"face_url: {face_url}, speech_url: {speech_url}")
            output_file, face_file = run_wav2lip(face_url, speech_url, gfpgan, gfpgan_upscale, intro_text)
            name = intro_text if intro_text else "wav2lip"
            yield CogOutput(file=output_file, name=name, thumbnail=face_file, attributes=None, progress=1.0, isFinal=True)
                        
        else:
            raise Exception("Invalid mode")
