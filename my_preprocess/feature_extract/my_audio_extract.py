# 将音频直接转换为特征文件，即.wav->.pt
# 可以直接使用
import librosa
import torch
import os

class Audio_extract:
    def __init__(self,input_path,output_path):
        self.output_path = output_path
        self.input_path = input_path

    def extract(self):
        audios = []
        batch_feature = []
        names=[]
        i=0
        for audio in os.listdir(self.input_path):
            if audio.endswith('.wav'):
                audio_path=os.path.join(self.input_path, audio)
                audios.append(audio_path)
                i=i+1
        print(i)
        for audio_path in audios:
            y, sr = librosa.load(audio_path, sr=22050)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
            mfccs = mfccs.T
            mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32)
            mfccs_mean = mfccs_tensor.mean(dim=0)
            batch_feature.append(mfccs_mean)
            file_name = os.path.splitext(os.path.basename(audio_path))[0]
            names.append(file_name)

        if batch_feature:
            output_dict = dict(zip(names, batch_feature))
            output_file = os.path.join(self.output_path, "audio_feature.pt")
            torch.save(output_dict, output_file)

if __name__ == '__main__':
    input_path=r"D:\Desktop\audio"
    output_path=r"D:\Desktop\features"
    extractor=Audio_extract(input_path=input_path,output_path=output_path)
    extractor.extract()
    print("done")