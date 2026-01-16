#将jsonl格式文件转换为可后续使用的speech.jsonl格式文件
import os
import json
import torch
import pandas as pd
import speech_recognition as sr

# done
def convert_to_OCR_jsonl(ocr_file, output_path):
    df = pd.read_json(ocr_file, lines=True)

    df['vid'] = df['frame'].str[:-2]    #去除最后两个字符
    df['vid'] = df['vid'].apply(lambda x: x[:-1] if x.endswith('_') else x)   # 如果最后一个字符是_，则删去_

    grouped = df.groupby("vid")
    applied = grouped['text'].apply(' '.join)
    speech_df = applied.reset_index()
    speech_df = speech_df.rename(columns={'text': 'transcript'})
    speech_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Successfully converted to speech, generated {len(speech_df)} video records")
    return speech_df

# done
def voice_to_text(input_path):
    r=sr.Recognizer()
    with sr.AudioFile(input_path) as source:
        audio = r.record(source)
    try:
        text=r.recognize_sphinx(audio,language="en-US")
        return text
    except Exception as e:
        print(e)

# done
def convert_to_speech_jsonl(input_path, output_path):
    audios=[]
    for audio_file in os.listdir(input_path):
        if audio_file.endswith('.wav'):
            audios.append(audio_file)
    output_file = os.path.join(output_path, "speech.jsonl")
    with open(output_file, 'a', encoding="utf-8") as outfile:
        for audio in audios:
            audio_path = os.path.join(input_path, audio)
            text=voice_to_text(audio_path)
            data = {
                "vid": audio.split(".")[0],
                "transcript": text
            }
            data = json.dumps(data, ensure_ascii=False)
            outfile.write(data)
            outfile.write("\n")


def create_empty_title_jsonl(csv_file,output_path):
    """
    为数据集创建空的title.jsonl文件
    """

    if not os.path.exists(csv_file):
        print("doesn't exist")
        return False

    with open(csv_file, 'r') as f:
        vids = [line.strip() for line in f if line.strip()]

    # 创建title.jsonl文件
    title_file = os.path.join(output_path, 'title.jsonl')

    with open(title_file, 'w', encoding='utf-8') as f:
        for vid in vids:
            data = {
                "vid": vid,
                "text": ""  # 空标题
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("done")
    return True


if __name__ == '__main__':
    # 帧内容转文字(.jpg->OCR.jsonl)
    # OCR_file = r"D:\Desktop\text\ocr_results.jsonl"
    # output_file = r"D:\Desktop\text\OCR.jsonl"
    # speech_df = convert_to_OCR_jsonl(ocr_file=OCR_file, output_path=output_file)

    #视频内人声转文字(.wav->speech.jsonl)
    input_path = r"D:\Desktop\audio"
    output_path = r"D:\Desktop\text"
    convert_to_speech_jsonl(input_path, output_path)
    print("done")

    # 创建(空)标题文件(title.jsonl)
    # csv_file = r"D:\Desktop\vids.csv"
    # output_path = r"D:\Desktop\text"
    # create_empty_title_jsonl(csv_file,output_path)