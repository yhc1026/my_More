# 把三个jsonl文件中的数据按条拼接，然后用bert提取文本特征
from transformers import ViTImageProcessor, ViTModel, CLIPVisionModel, CLIPImageProcessor
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, ChineseCLIPImageProcessor, ChineseCLIPVisionModel, ChineseCLIPTextModel, ChineseCLIPFeatureExtractor
from transformers import BertModel, BertTokenizer,AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import torch



class MyDataset(Dataset):
    def __init__(self,input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path
        vids_file=os.path.join(input_path, 'vids.csv')
        with open(vids_file, 'r', encoding='utf-8') as f:
            self.vids = [line.strip() for line in f]
        ocr_file = os.path.join(input_path, 'ocr.jsonl')
        trans_file = os.path.join(input_path, 'speech.jsonl')
        title_file = os.path.join(input_path, 'title.jsonl')
        self.ocr_df = pd.read_json(ocr_file, lines=True)
        self.trans_df = pd.read_json(trans_file, lines=True)
        self.title_df = pd.read_json(title_file, lines=True)
        print(f"title_df 列名: {self.title_df.columns.tolist()}")

    def __getitem__(self, index):
        vid = self.vids[index]
        ocr = self.ocr_df[self.ocr_df['vid'] == vid]['ocr'].values[0]
        trans = self.trans_df[self.trans_df['vid'] == vid]['transcript'].values[0]
        title = self.title_df[self.title_df['vid'] == vid]['text'].values[0]
        text=f"{ocr}\n{title}\n{trans}"
        return vid, text

    def __len__(self):
        return len(self.vids)


class ExtractAllFeature:
    def __init__(self, input_path, output_path, model_path):
        self.output_path = output_path
        self.model_path = model_path
        self.input_file = input_path
        self.model=AutoModel.from_pretrained(model_path)
        self.processor=AutoTokenizer.from_pretrained(model_path)

    def customed_collate_fn(self, batch):
        vids, texts = zip(*batch)
        inputs = self.processor(texts, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
        return vids, inputs

    def extract_text_title_trans_feature(self):
        save_dict = {}
        my_dataset=MyDataset(input_path=self.input_file, model_path=self.model_path)
        dataloader = DataLoader(
            dataset=my_dataset,
            batch_size=1,
            collate_fn=self.customed_collate_fn,
            num_workers=0,
            shuffle=False
        )
        self.model.eval()
        for batch in tqdm(dataloader):
            vids, inputs = batch
            pooler_output = self.model(**inputs)['last_hidden_state'][:, 0, :]
            for i, vid in enumerate(vids):
                save_dict[vid] = pooler_output[i]
        torch.save(save_dict, self.output_path)



