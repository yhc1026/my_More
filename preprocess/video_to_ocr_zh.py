# my frame to text translator
# 将frame帧转换为文本，.jpg->ocr_results.jsonl
# done!

import json
from collections import defaultdict

import easyocr
import os

import pandas as pd
from tqdm import tqdm


class Frame_to_OCR:
    def __init__(self,input_path,output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.reader = easyocr.Reader(['zh'])  # 中文

    def translator(self):
        """提取OCR文本 - 按视频文件夹处理"""

        # 获取所有视频文件夹
        video_folders = [f for f in os.listdir(self.input_path)
                         if os.path.isdir(os.path.join(self.input_path, f))]
        print(f"找到 {len(video_folders)} 个视频文件夹")

        # 按视频ID处理OCR
        ocr_results = []

        for video_id in tqdm(video_folders, desc="Processing videos"):
            video_folder_path = os.path.join(self.input_path, video_id)

            # 获取该视频的所有帧文件
            frame_files = [f for f in os.listdir(video_folder_path) if f.endswith(".jpg")]
            frame_files.sort()  # 按文件名排序确保顺序

            # 对每个视频的所有帧提取OCR文本
            all_texts = []

            for frame_file in frame_files:
                frame_path = os.path.join(video_folder_path, frame_file)

                try:
                    # 提取该帧的OCR文本
                    result = self.reader.readtext(frame_path)
                    frame_texts = [item[1] for item in result if item[1].strip()]
                    texts=[text for text in frame_texts]
                    all_texts.extend(texts)

                except Exception as e:
                    print(f"处理帧 {frame_file} 时出错: {e}")
                    continue

            # 合并该视频的所有OCR文本
            if all_texts:
                unique_texts = list(set(all_texts))  # 简单去重
                ocr_text = "\n".join(unique_texts)
            else:
                ocr_text = ""  # 如果没有检测到文本，使用空字符串

            # 添加到结果列表
            ocr_results.append({
                "vid": video_id,
                "ocr": ocr_text
            })

        # 保存为标准格式
        output_file = os.path.join(self.output_path, "ocr.jsonl")
        df = pd.DataFrame(ocr_results)
        df.to_json(output_file, orient="records", lines=True, force_ascii=False)

        print(f"processed videos: {len(ocr_results)} ")

        self.clean_up()

    def clean_up(self):
        if hasattr(self, 'reader'):
            del self.reader

if __name__ == '__main__':
    input_path = r"D:\Desktop\frame"
    output_path = r"D:\Desktop\text"
    frame_to_ocr = Frame_to_OCR(input_path,output_path)
    frame_to_ocr.translator()
    print("done")
