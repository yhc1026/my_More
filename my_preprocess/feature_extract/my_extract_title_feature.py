#对应extract_text_title_desc_feature.py，但是源码好像只提取了title，因此在此也只提取title
# 将title.jsonl转化为.pt文件
import json
import os
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer


class Extract_text_title():
    def __init__(self,title_path,model_path,csv_path,output_path):
        self.title_path=title_path
        self.model_path=model_path
        self.csv_path=csv_path
        self.output_path=output_path
        self.data = []
        with open(title_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                item = json.loads(line.strip())
                self.data.append({'vid': item['vid'],
                                  'text': item['text']
                                  })

    def extractor(self,batch_size=1,max_length=512):
        try:
            model = AutoModel.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None
        device = torch.device("cpu")
        model = model.to(device)
        model.eval()
        features_dict = {}
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(self.data), batch_size)):
                batch_data = self.data[i:i + batch_size]
                batch_texts = [item['text'] for item in batch_data]
                batch_vids = [item['vid'] for item in batch_data]
                inputs = tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                batch_features = outputs.last_hidden_state[:, 0, :]  # CLS token
                batch_features = batch_features.cpu()
                for j, vid in enumerate(batch_vids):
                    features_dict[vid] = batch_features[j]
        output_file = os.path.join(self.output_path, "text_features_title.pt")
        torch.save(features_dict, output_file)

if __name__ == '__main__':
    title_path=r"D:\Desktop\text\title.jsonl"
    output_path=r"D:\Desktop\features"
    model_path=r"D:\models\bert\bert-base-uncased"
    vid_file=r"D:\Desktop\vids.csv"
    extract_text=Extract_text_title(title_path,model_path,vid_file,output_path)
    extract_text.extractor()
    print("done")