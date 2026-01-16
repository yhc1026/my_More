# 输入数据，输入第几集为测试集，然后输入需要测试,验证还是训练集，然后输入具体任务，
# 函数读取数据后，将所有数据分为5集，然后找到测试集，将详细测试数据赋给 train_val_data，

# 前置知识：
# 训练集：用于模型权重更新
# 验证集：用于超参数调优、模型选择、早停
# 测试集：用于最终性能评估，确保无偏估计

from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


class HateMM_Dataset(Dataset):  #被HateMM_MoRE_Dataset集成，本身不被调用
    def __init__(self):
        super(HateMM_Dataset, self).__init__()

    def _get_data(self, fold: int, split: str, task: str):
        data = pd.read_csv(r"D:\codeC\my_MoRE\my_MoRE\data\HateMM\HateMM_annotation.csv")
        if task == 'binary':
            pass
        else:
            raise NotImplementedError(f"Invalid task: {task}")
        replace_vaule = {
            'Hate': 1,
            'Non Hate': 0,
        }
        data['label'] = data['label'].replace(replace_vaule)
        data['Video_ID'] = data['video_file_name'].str.split('.').str[0]
        data['vid'] = data['Video_ID']
        if fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(data, fold, split)
        elif fold in ['default']:
            data = self._get_default_data(data, split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data

    def _get_default_data(self, data, split):  # 科研人员提前创造test,val,train三个csv文件，需要哪个自己拿
        vid_file = fr"D:\codeC\my_MoRE\my_MoRE\data\HateMM\vids\{split}.csv"
        vids = pd.read_csv(vid_file, header=None)[0].tolist()
        data = data[data['vid'].isin(vids)]
        return data

    def _get_fold_data(self, data, fold: int, split: str):  # 假设fold=1，split=train，那么就是将fold1设置为test，这样train就在剩下四个fold中，
        train_size, val_size, test_size = 0.7, 0.1, 0.2  # 因为split=train，因此返回剩下四个folds中的train集
        seed = 2024
        target_column = 'label'
        data_split = {}
        X = data.drop(columns=[target_column])
        y = data[target_column]
        y = y.astype('category')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold = fold - 1
        for i, (train_val_idx, test_idx) in enumerate(
                skf.split(X, y)):  # 因为skf的设计特点限制（只能分离训练验证集和测试集），需要在后文手动从训练验证集中分离出训练集和验证集
            if i == fold:
                data_split['test'] = data.iloc[test_idx]  # 验证集被自动分离，直接赋值

                train_val_data = data.iloc[train_val_idx]  # 手动从训练验证集中分离出训练集和验证集
                data_split['train'], data_split['valid'] = train_test_split(
                    train_val_data,
                    test_size=val_size / (1 - test_size),
                    stratify=train_val_data[target_column],
                    random_state=seed
                )
                break
        data = data_split[split]
        return data
