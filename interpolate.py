import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip, concatenate_videoclips
import time
import subprocess

def interpolate_frames(prev_frame, next_frame, factor=2):
    t = np.linspace(0, 1, factor+2)[1:-1]
    interpolated_frames = []
    for i in range(factor):
        alpha = t[i]
        beta = 1 - alpha
        interpolated_frame = cv2.addWeighted(prev_frame, beta, next_frame, alpha, 0) # 图像融合：addWeighted
        interpolated_frames.append(interpolated_frame)
    return interpolated_frames

def interpolate_video(video_path, output_path, factor=2):
    clip = VideoFileClip(video_path)
    fps = clip.fps
    width, height = clip.size

    new_frames = []
    prev_frame = None
    for frame in clip.iter_frames(dtype='uint8'):
        if prev_frame is not None:
            interpolated_frames = interpolate_frames(prev_frame, frame, factor)
            new_frames.extend(interpolated_frames)

        prev_frame = frame

    new_clip = ImageSequenceClip(new_frames, fps=factor * fps)
    new_clip = new_clip.resize((width, height))
    new_clip = new_clip.set_audio(clip.audio)

    new_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

def optical_flow_interpolation(input_path, output_path, factor):  # optical_flow method
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    new_fps = int(fps * factor)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []  # List to store interpolated frames

    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            h, w = gray.shape[:2]
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_flow = x + flow[..., 0]
            y_flow = y + flow[..., 1]

            for i in range(new_fps // int(fps)):
                alpha = (i + 1) / (new_fps // int(fps) + 1)

                x_interpolated = x + alpha * flow[..., 0]
                y_interpolated = y + alpha * flow[..., 1]

                # Interpolate the frame using remap
                interpolated_frame = cv2.remap(prev_frame, x_interpolated.astype(np.float32), y_interpolated.astype(np.float32), cv2.INTER_LINEAR)

                # Convert the frame back to BGR color
                interpolated_frame_bgr = cv2.cvtColor(interpolated_frame, cv2.COLOR_RGB2BGR)

                # Append the interpolated frame to the list of frames
                frames.append(interpolated_frame_bgr)

        prev_frame = frame

    cap.release()
    cv2.destroyAllWindows()

    # Create an ImageSequenceClip from the list of interpolated frames
    clip = ImageSequenceClip(frames, fps=new_fps)

    # Load the original audio
    audio_clip = VideoFileClip(input_path).audio

    # Set the audio for the final video clip
    final_clip = clip.set_audio(audio_clip)

    # Write the final video to the output file
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', remove_temp=True)


def interpolate_frames_by_ffmpeg(video_path, output_path, factor=2):
    clip = VideoFileClip(video_path)
    fps = clip.fps
    out_fps = fps * factor
    command = f"""ffmpeg -i {video_path} -filter_complex "minterpolate='fps={out_fps}'" {output_path}"""
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    fps = 8
    factor = 4
    video_path = "./videos/fps_{}.mp4".format(fps)
    output_path = "./TVFIResults/fps_{}_{}X.mp4".format(fps, factor)
    start = time.time()
    interpolate_video(video_path, output_path, factor=factor)
    end = time.time()
    print('TVFI Time: {}'.format(end - start))