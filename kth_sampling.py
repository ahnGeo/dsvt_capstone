import os
import subprocess
import ffmpeg

with open("/data/ahngeo11/svt/datasets/annotations/00sequences.txt", 'r') as f :
    data = f.readlines()
    
ann_dict = dict()
for line in data :
    if line == '\n' or "missing" in line :
        continue
    line = line.strip('\n')
    vid_name = line.split()[0]
    valid_interval = line.split("frames\t")[1].split(", ")
    ann_dict[vid_name] = valid_interval

FPS = 25

def get_ss_and_t(start_frame_idx, end_frame_idx):
    start_second = round(start_frame_idx / FPS)   #* input of -ss should be second, not fraction
    duration_as_sec = round(end_frame_idx / FPS) - start_second 
    ss = "%02d:%02d:%02d" % (0, start_second // 60, start_second % 60)
    t = "%02d:%02d:%02d" % (0, duration_as_sec // 60, duration_as_sec % 60)
    return ss, t

root_dir = "/local_datasets/KTH"

#! cut every interval and merge all clips == only remove no-human frames
# for vid_name in ann_dict.keys() :
#     vid_label = vid_name.split("_")[1]
#     input_vid_path = os.path.join(root_dir, vid_label, vid_name + "_uncomp.avi")
    
#     trimmed_videos = []
#     for i, start_and_end in enumerate(ann_dict[vid_name]) :    
#         #* if you use subprocess
#         # ss, t = get_ss_and_t(start_and_end[0], start_and_end[1])
#         # command = ['ffmpeg', '-i', input_vid_path, 
#         #         '-ss', ss, '-t', t, 
#         #             '-c', 'copy',
#         #             '-threads', '16',
#         #             output_vid_path
#         #         ]
#         # command = ' '.join(command)
#         # try:
#         #     subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
#         # except subprocess.CalledProcessError:
#         #     print(f'Trimming of the Event {vid_name} of Video {vid_name} Failed', flush=True)

#         input_video = ffmpeg.input(input_vid_path)
#         start_frame, end_frame = int(start_and_end.split('-')[0]), int(start_and_end.split('-')[1])
#         trimmed_video = input_video.trim(start_frame=start_frame, end_frame=end_frame).setpts('PTS-STARTPTS')
#         trimmed_videos.append(trimmed_video)
    
#     merged_video = ffmpeg.concat(*trimmed_videos)
    
#     output_vid_path =  os.path.join(root_dir, "processed", vid_label, vid_name + ".avi")
#     ffmpeg.output(merged_video, output_vid_path).run()
    
#! repeat action once
for vid_name in ann_dict.keys() :
    vid_label = vid_name.split("_")[1]
    input_vid_path = os.path.join(root_dir, "raw-videos", vid_label, vid_name + "_uncomp.avi")
    
    input_video = ffmpeg.input(input_vid_path)
    
    if "d2" in vid_name :    #* scale variant version
        start_frame, end_frame = int(ann_dict[vid_name][-1].split('-')[0]), int(ann_dict[vid_name][-1].split('-')[1])
    else :
        start_frame, end_frame = int(ann_dict[vid_name][0].split('-')[0]), int(ann_dict[vid_name][0].split('-')[1])
    
    trimmed_video = input_video.trim(start_frame=start_frame, end_frame=end_frame).setpts('PTS-STARTPTS')
    
    output_vid_path = os.path.join(root_dir, "cut-one", vid_label, vid_name + ".avi")
    ffmpeg.output(trimmed_video, output_vid_path).run()