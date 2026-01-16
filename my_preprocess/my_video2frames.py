## my video to frame
#done!

import json
import os
import subprocess

from PIL import Image


class Video2Frames:
    def __init__(self,input_path,output_path,ffprobe_path,ffmpeg_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.input_path=input_path
        self.output_path=output_path
        self.ffprobe_path=ffprobe_path
        self.ffmpeg_path=ffmpeg_path

    def Preprocess(self):
        videos=[]
        i=0
        for video in os.listdir(self.input_path):
            if(video.endswith(".mp4")):
                videos.append(video)
                i=i+1
        print("Total number of videos:",i)
        for video in videos:
            input_video = os.path.join(self.input_path, video)
            self.extract_frames(input_video, output_path,video)

    def extract_frames(self, input_video, output_path, video_name):
        duration = self.get_video_duration(input_video)
        video_id = video_name.split(".")[0]  # 获取视频ID
        video_output_folder = os.path.join(output_path, video_id)  # 为每个视频创建文件夹

        # 检查是否已处理
        if os.path.exists(video_output_folder):
            existing_frames = len([f for f in os.listdir(video_output_folder) if f.endswith(".jpg")])
            if existing_frames == 16:
                print(f"Skip {video_id}: already has 16 frames")
                return

        os.makedirs(video_output_folder, exist_ok=True)

        # 计算均匀采样时间点
        interval = duration / 16
        timestamps = [i * interval for i in range(16)]

        for i, timestamp in enumerate(timestamps):
            output_frame = os.path.join(video_output_folder, f"frame_{i:03d}.jpg")  # 修正路径
            ffmpeg_cmd = [
                self.ffmpeg_path,
                "-i", input_video,
                "-ss", str(timestamp),
                "-frames:v", "1",
                "-loglevel", "error",
                output_frame
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True, shell=True)
            except subprocess.CalledProcessError:
                # 如果提取失败，创建黑色占位图
                print(f"Failed to extract frame {i} for {video_id}, using placeholder")
                img = Image.new("RGB", (224, 224), color="black")
                img.save(output_frame)

    def get_video_duration(self, input_video):
        get_duration= [
            self.ffprobe_path,
            '-v',
            'error',
            '-show_entries',
            'format=duration',
            '-of', 'json',
            input_video
        ]
        result = subprocess.run(get_duration, capture_output=True, text=True)
        # print(f"命令返回值: {result.returncode}")
        # print(f"标准输出: '{result.stdout}'")
        # print(f"错误输出: '{result.stderr}'")
        info = json.loads(result.stdout)
        return float(info['format']['duration'])


if __name__ == '__main__':
    input_path=r"D:\Desktop\videos"
    output_path=r"D:\Desktop\frame"
    ffprobe_path=r"D:\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffprobe.exe"
    ffmpeg_path=r"D:\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
    preprocess_frame=Video2Frames(input_path,output_path,ffprobe_path,ffmpeg_path)
    preprocess_frame.Preprocess()
    print("done")