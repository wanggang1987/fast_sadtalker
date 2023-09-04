from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from src.utils.append_audio import append_silent_audio
from interpolate import interpolate_video, optical_flow_interpolation, interpolate_frames_by_ffmpeg

import os
os.environ ['CUDA_VISIBLE_DEVICES'] = '0, 1' 

def main(args):

    #torch.backends.cudnn.enabled = False
    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose
    add_silent_both_slides = args.add_silent_both_slides
    silent_seconds = args.silent_seconds

    if add_silent_both_slides:
        # replace the audio path to processed file path
        output_path = "./tmp_audio.wav"  # remove the tmp audio finally
        append_silent_audio(audio_path, silent_seconds, output_path)  # seconds: 1s
        audio_path = output_path

    torch.cuda.synchronize()
    extract_start = time.time()
    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info = args.preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,\
                                                                             source_image_flag=True, pic_size=args.size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  args.preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  args.preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None
    
    torch.cuda.synchronize()
    extract_end = time.time()
    
    # del preprocess_model
    # torch.cuda.empty_cache()

    torch.cuda.synchronize()
    gen_start = time.time()
        
    #audio2ceoff
    batch = get_data(args.fps, first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = args.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    torch.cuda.synchronize()
    gen_end = time.time()

    # del audio_to_coeffnvidia
    # torch.cuda.empty_cache()

    # 3dface render
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    torch.cuda.synchronize()
    render_start = time.time()
    # coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)
    
    result = args.animate_from_coeff.generate(data, save_dir, pic_path, crop_info, args.fps, \
                                enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
    torch.cuda.synchronize()
    render_end = time.time()


    torch.cuda.synchronize() 
    interpolate_start = time.time()
    # add interpolate to final generated video
    interpolate_result = "./interpolate_videos.mp4"
    interpolate_video(result, interpolate_result, factor=2)  # Interpolate Time: 0.8898725509643555
    # optical_flow_interpolation(result, interpolate_result, factor=2) # Interpolate Time: 4.715899467468262
    # interpolate_frames_by_ffmpeg(result, interpolate_result, factor=2)
    torch.cuda.synchronize() 
    interpolate_end = time.time()
    print(f"Interpolated result is located in {interpolate_result}.")

    shutil.move(interpolate_result, save_dir+'.mp4')
    # shutil.move(result, save_dir+'.mp4')
    print('The generated video is named:', save_dir+'.mp4')

    # del animate_from_coeff
    # torch.cuda.empty_cache()

    if not args.verbose:
        shutil.rmtree(save_dir)
    extract_list.append(extract_end - extract_start)
    gen_list.append(gen_end - gen_start)
    render_list.append(render_end - render_start)
    Interpolate_list.append(interpolate_end - interpolate_start)
    print('Extract Time: {}'.format(extract_end - extract_start))
    print('Gen Coeff Time: {}'.format(gen_end - gen_start))
    print('Render Time: {}'.format(render_end - render_start))
    print('Interpolated Time: {}'.format(interpolate_end - interpolate_start))

    if add_silent_both_slides:
        # remove tmp_file
        os.remove(output_path)


if __name__ == '__main__':
    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/people_0.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=32,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose", action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 


    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    # fps
    parser.add_argument('--fps', type=float, default=12.)

    # add silent audio to both slides
    parser.add_argument('--add_silent_both_slides', action="store_true", help="add silent audio (default:0.5s) on both slides of the initial audio")
    parser.add_argument("--silent_seconds", type=float, default=0.5,  help="the silent seconds to add")

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    #init model
    torch.cuda.synchronize()
    init_start = time.time()
    args.preprocess_model = CropAndExtract(sadtalker_paths, args.device)
    args.audio_to_coeff = Audio2Coeff(sadtalker_paths,  args.device)
    args.animate_from_coeff = AnimateFromCoeff(sadtalker_paths, args.device)
    torch.cuda.synchronize()
    init_end = time.time()
    print('Init Time: {}'.format(init_end - init_start))
    
    extract_list, gen_list, render_list, Interpolate_list = [], [], [], []
    all_list = []
    main(args)
    # for i in range(10):
    #     # torch.cuda.empty_cache()
    #     torch.cuda.synchronize()
    #     start = time.time()
    #     main(args)
    #     torch.cuda.synchronize()
    #     end = time.time()
    #     all_list.append(end - start)
    #     print('All Time: {}'.format(end - start))
    # import numpy as np
    # print(f"Average time:  Extract Time: {np.mean(extract_list)}, Gen Time: {np.mean(gen_list)}, Render Time: {np.mean(render_list)}, Interpolate Time: {np.mean(Interpolate_list)}")
    # print(f"Average All Time: {np.mean(all_list)}")
        

