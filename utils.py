import os
import shutil
import requests
from pathlib import Path
import cv2
import numpy as np
import moviepy.editor as mp
import gizeh as gz


def download(url, folder, ext):
    filename = url.split('/')[-1]
    if not filename.endswith(ext):
        filename += ext
    filepath = folder / filename
    if filepath.exists():
        return filepath
    raw_file = requests.get(url, stream=True).raw
    with open(filepath, 'wb') as f:
        f.write(raw_file.read())
    return filepath


def try_delete(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def get_video_dimensions(video_file):
    cap = cv2.VideoCapture(str(video_file))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps


def make_black_frame_video(width, height, duration, fps, text, output_file):
    black_clip = mp.ColorClip((width, height), color=(0,0,0), duration=duration)
    surface = gz.Surface(width=width, height=height)

    font = "Arial"
    fontsize = width//20
    num_words_per_line = 5

    # Split the text into lines of 8 words    
    words = text.split()
    groups = [words[i:i+num_words_per_line] for i in range(0, len(words), num_words_per_line)]
    lines = [ " ".join(group) for group in groups ]

    # Draw each line of text on the surface
    for l, line in enumerate(lines):
        y_offset = height//2 + (l - len(lines)//2) * fontsize * 1.25
        text_elem = gz.text(line, 
                            fontfamily=font, fontsize=fontsize, fill=(1,1,1), 
                            xy=(width//2, y_offset))
        text_elem.draw(surface)
    
    text_clip = mp.VideoClip(lambda t: surface.get_npimage(), duration=duration)
    final_clip = mp.CompositeVideoClip([black_clip, text_clip])
    audio_clip = mp.AudioClip(lambda t: [0], duration=0)
    final_clip = final_clip.set_audio(audio_clip)
    final_clip.set_fps(fps).write_videofile(str(output_file), audio_fps=44100)


def concatenate_videos(intro_video, content_video, output_file):
    cmd = f'ffmpeg -i {intro_video} -i {content_video} -filter_complex "[0:v] [0:a] [1:v] [1:a] concat=n=2:v=1:a=1 [v] [a]" -map "[v]" -map "[a]" {output_file}'
    result = os.system(cmd)
    if result != 0:
        raise Exception("ffmpeg failed")


