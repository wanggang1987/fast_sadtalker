import torch, uuid
import os, sys, shutil
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from src.utils.append_audio import append_silent_audio
from pydub import AudioSegment


os.environ ['CUDA_VISIBLE_DEVICES'] = '0, 1' 

def mp3_to_wav(mp3_filename,wav_filename,frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename,format="wav")


class SadTalker():

    def __init__(self, checkpoint_path='checkpoints', config_path='src/config', size = 256, preprocess = 'crop', lazy_load=False):

        if torch.cuda.is_available() :
            device = "cuda"
        else:
            device = "cpu"
        
        self.device = device
        self.size = size
        self.preprocess = preprocess

        os.environ['TORCH_HOME']= checkpoint_path

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path

        self.sadtalker_paths = init_path(self.checkpoint_path, self.config_path, self.size, False, self.preprocess)
        print(self.sadtalker_paths)
            
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)
      

    def test(self, source_image, driven_audio, preprocess='crop', 
        still_mode=False,  use_enhancer=False, batch_size=16, size=256, 
        pose_style = 0, exp_scale=1.0, 
        use_ref_video = False,
        ref_video = None,
        ref_info = None,
        use_idle_mode = False,
        length_of_audio = 0, use_blink=True,
        input_yaw_list = None, input_pitch_list = None,
        input_roll_list = None,
        result_dir='./results/',
        add_silent_both_slides = False, silent_seconds = 0.5,
        fps=25):           

        if add_silent_both_slides:
            # replace the audio path to processed file path
            output_path = "./tmp_audio.wav"  # remove the tmp audio finally
            append_silent_audio(audio_path, silent_seconds, output_path)  # seconds: 0.5s
            audio_path = output_path

        time_tag = str(uuid.uuid4())
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)

        input_dir = os.path.join(save_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        print(source_image)
        pic_path = os.path.join(input_dir, os.path.basename(source_image)) 
        shutil.move(source_image, input_dir)

        if driven_audio is not None and os.path.isfile(driven_audio):
            audio_path = os.path.join(input_dir, os.path.basename(driven_audio))  

            #### mp3 to wav
            if '.mp3' in audio_path:
                mp3_to_wav(driven_audio, audio_path.replace('.mp3', '.wav'), 16000)
                audio_path = audio_path.replace('.mp3', '.wav')
            else:
                shutil.move(driven_audio, input_dir)

        elif use_idle_mode:
            audio_path = os.path.join(input_dir, 'idlemode_'+str(length_of_audio)+'.wav') ## generate audio from this new audio_path
            from pydub import AudioSegment
            one_sec_segment = AudioSegment.silent(duration=1000*length_of_audio)  #duration in milliseconds
            one_sec_segment.export(audio_path, format="wav")
        else:
            print(use_ref_video, ref_info)
            assert use_ref_video == True and ref_info == 'all'

        if use_ref_video and ref_info == 'all': # full ref mode
            ref_video_videoname = os.path.basename(ref_video)
            audio_path = os.path.join(save_dir, ref_video_videoname+'.wav')
            print('new audiopath:',audio_path)
            # if ref_video contains audio, set the audio from ref_video.
            cmd = r"ffmpeg -y -hide_banner -loglevel error -i %s %s"%(ref_video, audio_path)
            os.system(cmd)        

        os.makedirs(save_dir, exist_ok=True)
        
        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(pic_path, first_frame_dir, preprocess, True, size)
        
        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        if use_ref_video:
            print('using ref video for genreation')
            ref_video_videoname = os.path.splitext(os.path.split(ref_video)[-1])[0]
            ref_video_frame_dir = os.path.join(save_dir, ref_video_videoname)
            os.makedirs(ref_video_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_video_coeff_path, _, _ =  self.preprocess_model.generate(ref_video, ref_video_frame_dir, preprocess, source_image_flag=False)
        else:
            ref_video_coeff_path = None

        if use_ref_video:
            if ref_info == 'pose':
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = None
            elif ref_info == 'blink':
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'pose+blink':
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'all':            
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = None
            else:
                raise('error in refinfo')
        else:
            ref_pose_coeff_path = None
            ref_eyeblink_coeff_path = None

        #audio2ceoff
        if use_ref_video and ref_info == 'all':
            coeff_path = ref_video_coeff_path # self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
        else:
            batch = get_data(fps, first_coeff_path, audio_path, self.device, ref_eyeblink_coeff_path=ref_eyeblink_coeff_path, still=still_mode, idlemode=use_idle_mode, length_of_audio=length_of_audio, use_blink=use_blink) # longer audio?
            coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

        #coeff2video
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, still_mode=still_mode, preprocess=preprocess, size=size, expression_scale = exp_scale,
                                   input_yaw_list = input_yaw_list, input_pitch_list= input_pitch_list,
                                   input_roll_list = input_roll_list
                                   )
        return_path = self.animate_from_coeff.generate(data, save_dir,  pic_path, crop_info, fps, enhancer='gfpgan' if use_enhancer else None, preprocess=preprocess, img_size=size)
        video_name = data['video_name']
        print(f'The generated video is named {video_name} in {save_dir}')

        # del self.preprocess_model
        # del self.audio_to_coeff
        # del self.animate_from_coeff

        if add_silent_both_slides:
            # remove tmp_file
            os.remove(output_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        import gc; gc.collect()
        
        return return_path

    