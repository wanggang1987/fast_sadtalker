from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from src.gradio_demo import SadTalker
from datetime import datetime
from shutil import copyfile
from interpolate import interpolate_video, optical_flow_interpolation, interpolate_frames_by_ffmpeg

app = FastAPI()

def _get_current_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    return formatted_time

# 场景1：输入 语音+图片 ， 一次性等待返回 生成的 mp4视频
@app.post("/create_video")
async def create_video(image_file: UploadFile = File(...),
                       audio_file: UploadFile = File(...),
                       preprocess: str = Form(default='crop'),
                       still_mode: bool = Form(default=False),
                       use_enhancer: bool = Form(default=False),
                       fps: float = Form(default=12.),
                       interpolate_factor: int = Form(default=2),
                       add_silent_both_slides: bool = Form(default=False),
                       silent_seconds: float = Form(default=0.5)
                       ):
    print("DEBUG create_video 001 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    global sad_talker_crop 
    global sad_talker_full
    
    full, crop, resize = None, None, None # e.g. eval("full") == full
    if isinstance(eval(preprocess), str):
        preprocess = eval(preprocess)
    sad_talker = sad_talker_full if preprocess == 'full' else sad_talker_crop

    print("DEBUG create_video 002 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    # user input files
    currentTime = _get_current_time()
    imageFilePath = f"./image_uploaded_files/{currentTime}_{image_file.filename}"
    audioFilePath = f"./audio_uploaded_files/{currentTime}_{audio_file.filename}"

    # save image
    with open(imageFilePath, "wb") as f:
        f.write(await image_file.read())
    with open(audioFilePath, "wb") as f:
        f.write(await audio_file.read())

    print("DEBUG create_video 003 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    # other default params, TODO as params
    # dynamic
    pose_style: int = 0
    exp_scale: float = 1.0
    use_ref_video: bool = False
    ref_video: str = None
    input_yaw_list: list = None # [5, -1, 1]
    input_pitch_list: list = None # [-3, 2, 0]
    input_roll_list: list = None # [1, -1, 0]
    result_dir: str = './results/'
    batch_size: int = 32
    fps: float = 12. # origin:25.
    interpolate_dir: str = './InterpolatedVideo/'
    interpolate_factor: int = 2 # origin: 1
    add_silent_both_slides: bool = False
    silent_seconds: float = 0.5

    print("DEBUG create_video 004 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

    print(f"The fps of generated video is: {fps}!")
    resultVideoFilePath = sad_talker.test(imageFilePath, audioFilePath, preprocess=preprocess, still_mode=still_mode,
                                          use_enhancer=use_enhancer, batch_size=batch_size, fps=fps,
                                          result_dir=result_dir, exp_scale=exp_scale,
                                          input_yaw_list=input_yaw_list, input_pitch_list=input_pitch_list,
                                          input_roll_list=input_roll_list, add_silent_both_slides = add_silent_both_slides,
                                          silent_seconds = silent_seconds)

    print("DEBUG create_video 005 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

    # Interpolate Video
    InterpolateVideoFilePath = interpolate_dir + str(currentTime) + '.mp4'
    interpolate_video(resultVideoFilePath, InterpolateVideoFilePath, factor=interpolate_factor)
    # optical_flow_interpolation(resultVideoFilePath, InterpolateVideoFilePath, factor=interpolate_factor)
    
    # def generate_video():
    #     with open(resultVideoFilePath, mode="rb") as file_like:
    #         print("DEBUG create_video 006 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    #         yield from file_like
    #         print("DEBUG create_video 007 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

    def generate_video():
        with open(InterpolateVideoFilePath, mode="rb") as file_like:
            print("DEBUG create_video 006 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            yield from file_like
            print("DEBUG create_video 007 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

    print("DEBUG create_video 008 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    # 返回生成的视频
    return StreamingResponse(generate_video(), media_type="video/mp4")

@app.post("/create_idle_video")
async def create_idle_video(image_file: UploadFile = File(...),
                       preprocess: str = Form(default='crop'),
                       still_mode: bool = Form(default=True),
                       use_enhancer: bool = Form(default=False),
                       fps: float = Form(default=12.),
                       interpolate_factor: int = Form(default=4),
                       ):
    print("DEBUG create_video 001 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    global sad_talker_crop 
    global sad_talker_full 
    
    full, crop, resize = None, None, None # e.g. eval("full") == full
    if isinstance(eval(preprocess), str):
        preprocess = eval(preprocess)
    sad_talker = sad_talker_full if preprocess == 'full' else sad_talker_crop

    print("DEBUG create_video 002 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    # user input files
    currentTime = _get_current_time()
    imageFilePath = f"./image_uploaded_files/{currentTime}_{image_file.filename}"
    audioFilePath = f"./audio_uploaded_files/{currentTime}_silent.wav"
    defaultAudioFilePath = f"./silent.wav"
    copyfile(defaultAudioFilePath, audioFilePath)

    # save image
    with open(imageFilePath, "wb") as f:
        f.write(await image_file.read())

    print("DEBUG create_video 003 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    # other default params, TODO as params
    # static
    pose_style: int = 0
    exp_scale: float = 1.0
    use_ref_video: bool = False
    ref_video: str = None
    input_yaw_list: list = None # [5, -1, 1]
    input_pitch_list: list = None # [-3, 2, 0]
    input_roll_list: list = None # [1, -1, 0]
    result_dir: str = './results/'
    batch_size: int = 32
    # fps: float = 16.
    interpolate_dir: str = './InterpolatedVideo/'

    print("DEBUG create_video 004 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))


    resultVideoFilePath = sad_talker.test(imageFilePath, audioFilePath, preprocess=preprocess, still_mode=still_mode,
                                          use_enhancer=use_enhancer, result_dir=result_dir, exp_scale=exp_scale, 
                                          input_yaw_list=input_yaw_list, input_pitch_list=input_pitch_list,
                                          input_roll_list=input_roll_list, batch_size=batch_size, fps=fps)

    print("DEBUG create_video 005 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

    # Interpolate Video
    InterpolateVideoFilePath = interpolate_dir + str(currentTime) + '.mp4'
    interpolate_video(resultVideoFilePath, InterpolateVideoFilePath, factor=interpolate_factor)
    

    # def generate_video():
    #     with open(resultVideoFilePath, mode="rb") as file_like:
    #         print("DEBUG create_video 006 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    #         yield from file_like
    #         print("DEBUG create_video 007 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

    def generate_video():
        with open(InterpolateVideoFilePath, mode="rb") as file_like:
            print("DEBUG create_video 006 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            yield from file_like
            print("DEBUG create_video 007 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

    print("DEBUG create_video 008 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    # 返回生成的视频
    return StreamingResponse(generate_video(), media_type="video/mp4")

# 场景2：输入 语音+图片 ， 流式得到 多个mp4视频片段 目录,
# TODO 当然可以改造返回逐个mp4视频流，暂时未调通客户端怎么播放这些分配数据，先返回mp4片段文件名

# @app.post("/create_video_for_flow")
# async def create_video_for_flow(image_file: UploadFile = File(...), audio_file: UploadFile = File(...)):
#     print("DEBUG create_video_for_flow 001 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
#     global sad_talker
#
#     print("DEBUG create_video_for_flow 002 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
#     # user input files
#     imageFilePath = f"./image_uploaded_files/{image_file.filename}"
#     audioFilePath = f"./audio_uploaded_files/{audio_file.filename}"
#
#     # save image
#     with open(imageFilePath, "wb") as f:
#         f.write(await image_file.read())
#     with open(audioFilePath, "wb") as f:
#         f.write(await audio_file.read())
#
#     print("DEBUG create_video_for_flow 003 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
#     # other default params, TODO as params
#     preprocess: str = 'crop'
#     still_mode: bool = False
#     use_enhancer: bool = False
#     result_dir: str = './results/'
#
#     print("DEBUG create_video_for_flow 004 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
#     resultVideoSegmentPath = sad_talker.test_flow(imageFilePath, audioFilePath, preprocess=preprocess, still_mode=still_mode,
#                                           use_enhancer=use_enhancer, result_dir=result_dir)
#
#     print("DEBUG create_video_for_flow 005")
#     #
#     def generate_video_fragments():
#         for segFilePath in resultVideoSegmentPath:
#             print("DEBUG create_video_for_flow 006 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
#             yield segFilePath + "\n"
#             print("DEBUG create_video_for_flow 007 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
#
#     print("DEBUG create_video_for_flow 008 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
#     # 返回生成的视频
#     return StreamingResponse(generate_video_fragments(), media_type="text/plain")

# server demo
if __name__ == "__main__":
    sad_talker_crop = SadTalker(lazy_load=False, preprocess='crop')
    sad_talker_full = SadTalker(lazy_load=False, preprocess='full')
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9009, workers=1)

# client demo: create_video
# curl -X POST -F "image_file=@/Users/weizy/PycharmProjects/pythonProject/media/art_5_256x256.png" -F "audio_file=@/Users/weizy/PycharmProjects/pythonProject/media/RD_Radio31_000.wav" http://localhost:9009/create_video -o output_test.mp4
# curl -X POST -F "audio_file=@/home/redhat/AiModels/SadTalker/examples/driven_audio/RD_Radio31_000.wav" -F "image_file=@/home/redhat/AiModels/SadTalker/examples/source_image/art_5.png" http://localhost:8008/create_video -o output_test.mp4

# client demo: create_video_for_flow
# curl -X POST -F "image_file=@/Users/weizy/PycharmProjects/pythonProject/media/art_5_256x256.png" -F "audio_file=@/Users/weizy/PycharmProjects/pythonProject/media/RD_Radio31_000.wav" http://localhost:8000/create_video_for_flow
# curl -X POST -F "audio_file=@/home/redhat/AiModels/SadTalker/examples/driven_audio/RD_Radio31_000.wav" -F "image_file=@/home/redhat/AiModels/SadTalker/examples/source_image/art_5.png" http://localhost:8000/create_video_for_flow
