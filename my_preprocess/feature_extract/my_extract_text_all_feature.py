#拼接title.jsonl, OCR.jsonl, speech.jsonl拼接为一整个文本，并用bert提取
#done

import torch
import os
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

class Extract_text_all_feature():
    def __init__(self,title_path,speech_path,OCR_path,csv_path,model_path,output_path):
        self.title_path = title_path
        self.speech_path = speech_path
        self.OCR_path = OCR_path
        self.csv_path = csv_path
        self.model_path = model_path
        self.output_path = output_path
        self.vids = pd.read_csv(csv_path, header=None, names=['vid'])
        self.ocr_df = pd.read_json(OCR_path, lines=True)
        self.trans_df = pd.read_json(speech_path, lines=True)
        self.title_df = pd.read_json(title_path, lines=True)

    def extractor(self):
        combined_texts=[]
        for vid in self.vids['vid']:
            ocr_text = self.ocr_df[self.ocr_df['vid'] == vid]['transcript'].values[0]
            trans_text = self.trans_df[self.trans_df['vid'] == vid]['transcript'].values[0]
            title_text = self.title_df[self.title_df['vid'] == vid]['text'].values[0]
            text_ocr_trans = f"{ocr_text}\n{trans_text}\n{title_text}".strip()
            combined_texts.append({
                "vid": vid,
                "text": text_ocr_trans,
            })

        model = AutoModel.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        features_dict = {}
        model.eval()
        with torch.no_grad():
            for item in tqdm(combined_texts):
                vid = item["vid"]
                text = item["text"]
                if not text.strip():
                    print(f"跳过空文本: {vid}")
                    continue
                inputs = tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                outputs = model(**inputs)
                text_features = outputs.last_hidden_state[:, 0, :]  # [1, 768]
                features_dict[vid] = text_features.squeeze(0).detach().cpu()
        output_file = os.path.join(self.output_path, "text_features_all.pt")
        torch.save(features_dict, output_file)

if __name__ == '__main__':
    speech_path = r"D:\Desktop\text\speech.jsonl"
    title_path = r"D:\Desktop\text\title.jsonl"
    OCR_path = r"D:\Desktop\text\OCR.jsonl"
    output_path = r"D:\Desktop\features"
    model_path = r"D:\models\bert\bert-base-uncased"
    csv_path = r"D:\Desktop\vids.csv"
    extract_text_trans = Extract_text_all_feature(title_path, speech_path, OCR_path, csv_path, model_path, output_path)
    extract_text_trans.extractor()
    print("done")