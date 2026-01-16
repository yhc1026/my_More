# 先对指定text进行拼接，再bert特征提取，最后输出pt特征文件

import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


class MyDataset(Dataset):
    def __init__(self, dataset_dir, output_file, text_type):
        self.dataset_dir = dataset_dir
        self.output_file = output_file
        self.text_type = text_type
        vid_file = os.path.join(dataset_dir, 'vids.csv')
        with open(vid_file, 'r',encoding='utf-8') as f:
            self.vids = [line.strip() for line in f]
        ocr_file = os.path.join(dataset_dir, 'ocr.jsonl')
        trans_file = os.path.join(dataset_dir, 'speech.jsonl')
        title_file = os.path.join(dataset_dir, 'title.jsonl')
        self.ocr_df = pd.read_json(ocr_file, lines=True)
        self.trans_df = pd.read_json(trans_file, lines=True)
        self.title_df = pd.read_json(title_file, lines=True)
        print(f"title_df 列名: {self.title_df.columns.tolist()}")

    def __getitem__(self, index):
        vid = self.vids[index]
        ocr = self.ocr_df[self.ocr_df['vid'] == vid]['ocr'].values[0]
        trans = self.trans_df[self.trans_df['vid'] == vid]['transcript'].values[0]
        title = self.title_df[self.title_df['vid'] == vid]['text'].values[0]
        if self.text_type == 'title_trans':
            text = f'{title}\n{trans}'
        elif self.text_type == 'ocr_trans':
            text = f'{ocr}\n{trans}'
        else:
            raise ValueError("Invalid text_type. Choose 'title_trans' or 'ocr_trans'.")
        return vid, text

    def __len__(self):
        return len(self.vids)


class ExtractTextTitleTransFeature:
    def __init__(self, dataset_dir, output_file, model_id, text_type):
        self.output_file = output_file
        self.text_type = text_type
        self.dataset_dir = dataset_dir
        self.model_id = model_id
        self.model=AutoModel.from_pretrained(model_id)
        self.processor = AutoTokenizer.from_pretrained(model_id)
        self.device = torch.device('cuda')
        self.model=self.model.to(self.device)


    def customed_collate_fn(self, batch):
        vids, texts = zip(*batch)
        inputs = self.processor(texts, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return vids, inputs

    def extract_text_title_trans_feature(self):
        save_dict = {}
        my_dataset=MyDataset(dataset_dir=self.dataset_dir, output_file=self.output_file, text_type='title_trans')
        dataloader = DataLoader(
            dataset=my_dataset,
            batch_size=8,
            collate_fn=self.customed_collate_fn,
            num_workers=0,
            shuffle=False
        )
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                try:
                    vids, inputs = batch
                    outputs = self.model(**inputs)
                    pooler_output = outputs['last_hidden_state'][:, 0, :]
                    pooler_output = pooler_output.cpu()
                    for i, vid in enumerate(vids):
                        save_dict[vid] = pooler_output[i].clone()
                    del outputs, pooler_output, inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"处理批次时出错: {e}")
                    continue
        print("saving")
        torch.save(save_dict, self.output_file)



if __name__ == '__main__':
    # ch
    # dataset_dir = r""
    # model_id = r""
    # output_file = r"\fea_title_trans_bert-base-chinese.pt"
    # text_type = r"title_trans"
    # extract = ExtractTextTitleTransFeature(dataset_dir, output_file, model_id, text_type)

    # en
    dataset_dir=r"D:\codeC\my_MoRE\my_MoRE\data\MultiHateClip\en"
    model_id=r"D:\models\bert\bert-base-uncased"
    output_file=r"D:\codeC\my_MoRE\my_MoRE\data\MultiHateClip\en\fea\fea_title_trans_bert-base-uncased.pt"
    text_type=r"ocr_trans"
    extract = ExtractTextTitleTransFeature(dataset_dir, output_file, model_id, text_type)
    extract.extract_text_title_trans_feature()

    # hateMM
    # dataset_dir = r""
    # model_id = r""
    # output_file = r""
    # text_type = r"title_trans"
    # extract = ExtractTextTitleTransFeature(dataset_dir, output_file, model_id, text_type)