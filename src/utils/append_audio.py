from pydub import AudioSegment

def append_silent_audio(audio_path, seconds, output_path = None):
    """
    给指定路径的语音前后添加空语音
    参数说明：
    audio_path_source: 源语音路径, 对应前后添加的空语音
    audio_path_des: 目标语音路径, 对应要被添加的语音段
    output_path: 拼接语音的输出路径
    seconds: 空语音要添加的秒数
    """
    silent_audio = AudioSegment.silent(duration=1000 * seconds)
    music = AudioSegment.from_mp3(file=audio_path)
    out_audio = silent_audio + music + silent_audio
    if output_path is not None:
        out_audio.export(out_f=f"{output_path}", format='wav')
    return out_audio