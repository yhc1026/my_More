import torch
import os

# 视频ID列表
vids = ['hate_video_1', 'hate_video_10', 'non_hate_video_16', 'non_hate_video_4']

# 创建转录文本特征
transcript_features = {}
for vid in vids:
    # 创建768维BERT特征
    transcript_features[vid] = torch.randn(768)

# 保存文件
os.makedirs('data/HateMM/fea', exist_ok=True)
torch.save(transcript_features, 'data/HateMM/fea/fea_transcript_bert-base-uncased.pt')

print("转录文本特征文件创建完成！")