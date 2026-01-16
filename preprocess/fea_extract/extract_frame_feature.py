# 这个脚本专门从视频抽帧用视觉模型提取特征。
# 通过vit-base-patch16-224生成帧特征
# 生成的特征文件：
# fea_frames_16_google-vit-base-16-224.pt (标准采样16帧)
#将帧.jpg直接转换为特征文件.pt

import os
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

class Frame_extract:
    def __init__(self,input_path,output_path,model_path):
        if os.path.exists(output_path):
            self.output_path = output_path
        else:
            os.makedirs(output_path)
        self.input_path=input_path
        self.model_path=model_path

    def pre_extract(self):
        print("downloading...")
        model=AutoModel.from_pretrained(self.model_path,device_map="cpu",local_files_only=True)
        processor=AutoProcessor.from_pretrained(self.model_path)
        model.eval()
        return model, processor

    def extract(self):
        model, processor = self.pre_extract()
        features_dict = {}

        # 修复：通过文件夹结构获取视频ID
        video_folders = [f for f in os.listdir(self.input_path)
                         if os.path.isdir(os.path.join(self.input_path, f))]

        for video_id in tqdm(video_folders, desc="Extracting features"):
            frame_folder = os.path.join(self.input_path, video_id)
            frame_files = [f for f in os.listdir(frame_folder) if f.endswith(".jpg")]
            frame_files.sort()  # 确保按顺序

            # 修复：检查并补齐帧数
            if len(frame_files) < 16:
                print(f"Warning: {video_id} has only {len(frame_files)} frames, padding to 16")
                frame_files = self._pad_frames(frame_files, frame_folder, 16)
            elif len(frame_files) > 16:
                frame_files = frame_files[:16]  # 取前16帧

            video_features = []

            # 批量处理提高效率
            frames_batch = []
            for frame_file in frame_files:
                frame_path = os.path.join(frame_folder, frame_file)
                img = Image.open(frame_path).convert("RGB")
                frames_batch.append(img)

            # 批量处理
            processed_frames = processor(images=frames_batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in processed_frames.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                for i in range(len(frames_batch)):
                    frame_feature = outputs.last_hidden_state[i, 0, :].cpu()
                    video_features.append(frame_feature)

            video_tensor = torch.stack(video_features)  # [16, 768]
            features_dict[video_id] = video_tensor

        # 修复：使用标准输出文件名
        output_file = os.path.join(self.output_path, "fea_frames_16_google-vit-base-16-224.pt")
        torch.save(features_dict, output_file)
        print(f"Features saved: {output_file}")
        print(f"Total videos: {len(features_dict)}, Feature shape: [16, 768]")

    def _pad_frames(self, frame_files, frame_folder, target_count):
        """用黑色占位图补齐帧数"""
        current_count = len(frame_files)
        for i in range(current_count, target_count):
            placeholder_path = os.path.join(frame_folder, f"frame_{i:03d}.jpg")
            img = Image.new("RGB", (224, 224), color="black")
            img.save(placeholder_path)
            frame_files.append(f"frame_{i:03d}.jpg")
        return frame_files

if __name__ == '__main__':
    input_path=r"D:\codeC\my_MoRE\my_MoRE\data\MultiHateClip\en\frames_16"
    output_path=r"D:\codeC\my_MoRE\my_MoRE\data\MultiHateClip\en\fea"
    model_path=r"D:\models\vit-base-patch16-224"
    frame_extract=Frame_extract(input_path,output_path,model_path)
    frame_extract.extract()
    print("done")