# video to wav

import os
import subprocess

class Preprocessor:
    def __init__(self,input_path,output_path,ffmpeg_path):
        self.ffmpeg_path = ffmpeg_path
        if(not os.path.exists(output_path)):
            os.makedirs(output_path)
        self.input_path=input_path
        self.output_path=output_path

    def preprocess(self):
        videos=[]
        i=0
        for video in os.listdir(self.input_path):
            if(video.endswith(".mp4")):
                videos.append(video)
                i=i+1
        print("total videos",i)
        for video in videos:
            if self.has_audio_stream(os.path.join(self.input_path,video)):
                video_path=os.path.join(self.input_path,video)
                output_video=os.path.splitext(video)[0]+".wav"
                output_path=os.path.join(self.output_path,output_video)
                ffmpeg_command=[
                    self.ffmpeg_path,
                    "-i",video_path,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "44100",
                    "-ac", "2",
                    output_path
                ]
                subprocess.run(ffmpeg_command,check=True)

    def has_audio_stream(self, video_path):
        try:
            cmd = [
                str(self.ffmpeg_path),
                '-i', str(video_path),
                '-t', '0.1',  # 只检查很短的时间
                '-map', '0:a',  # 只映射音频流
                '-f', 'null',  # 输出到空设备
                '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False


if __name__=="__main__":
    input_path=r"D:\Desktop\videos"
    output_path=r"D:\Desktop\audio"
    ffmpeg_path=r"D:\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
    preprocessor=Preprocessor(input_path,output_path,ffmpeg_path)
    preprocessor.preprocess()
    print("done")



