#现有speech.jsonl，将speech.jsonl转换为text_features.pt
#done

import json
import os
import torch
from transformers import AutoModel, AutoTokenizer
import tqdm


class Extract_Text_Transcript_Feature:
    def __init__(self, input_path, output_path, model_path, vid_file): #input:.jsonl; output:.pt
        self.input_path=input_path
        self.model_path=model_path
        self.output_path=output_path
        self.vid_file=vid_file
        self.data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                item = json.loads(line.strip())
                self.data.append({'vid': item['vid'],
                                'text': item['transcript']
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
                outputs = model(**inputs)
                batch_features = outputs.last_hidden_state[:, 0, :]  # CLS token
                for j, vid in enumerate(batch_vids):
                    features_dict[vid] = batch_features[j]
        output_file = os.path.join(self.output_path, "text_features_speech.pt")
        torch.save(features_dict, output_file)


if __name__ == '__main__':
    json_path=r"D:\Desktop\text\speech.jsonl"
    output_path=r"D:\Desktop\features"
    model_path=r"D:\models\bert\bert-base-uncased"
    vid_file=r"D:\Desktop\vids.csv"
    extract_text=Extract_Text_Transcript_Feature(json_path,output_path,model_path,vid_file)
    extract_text.extractor()
    print("done")