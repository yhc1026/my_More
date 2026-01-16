import os

class Csv_Generator:
    def __init__(self,input_path,output_path):
        self.input_path=input_path
        self.output_path=output_path

    def generate_csv(self):
        names=[]
        i=0
        for video in os.listdir(self.input_path):
            if video.endswith(".mp4"):
                name=video.split(".")[0]
                names.append(name)
                i=i+1
        print("Number of videos:",i)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for name in names:
                f.write(name)
                f.write('\n')
            f.close()

if __name__ == '__main__':
    input_path=r"D:\Desktop\videos"
    output_path=r"D:\Desktop\vids.csv"
    csv_generator=Csv_Generator(input_path,output_path)
    csv_generator.generate_csv()
    print("done")

