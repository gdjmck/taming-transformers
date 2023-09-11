from osgeo import gdal
import torch
import numpy as np
import json
from torch.utils.data import Dataset


class StreetMap(Dataset):
    def __init__(self, src_file, tgt_file, anno_file, index_file, crop_size=256, phase='train', get_type='target'):
        """
        data_file: 栅格数据
        anno_file: 掩码数据
        """
        super().__init__()
        assert crop_size % 2 == 0
        self.get_type = get_type
        self.crop_size = crop_size
        # 主干道路栅格
        self.data_src = gdal.Open(src_file).ReadAsArray()  # 2d matrix
        # 生成路网栅格
        self.data_tgt = gdal.Open(tgt_file).ReadAsArray()  # 2d matrix
        # 可采样位置
        self.sample_position = np.load(anno_file)  # dict(xs, ys)
        # 采样位置的索引
        with open(index_file, 'r') as f:
            self.index_list = json.load(f)['{}_index'.format(phase)]

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, i):
        index = self.index_list[i]
        x = self.sample_position['xs'][index]
        y = self.sample_position['ys'][index]
        # 以(x, y)为中心，裁剪尺寸为(crop_size, crop_size)的矩阵
        patch_label = self.data_src[y - self.crop_size // 2: y + self.crop_size // 2,
                      x - self.crop_size // 2: x + self.crop_size // 2]
        patch_target = self.data_tgt[y - self.crop_size // 2: y + self.crop_size // 2,
                       x - self.crop_size // 2: x + self.crop_size // 2]
        if self.get_type == 'both':
            return {'label': torch.tensor(patch_label[..., None], dtype=torch.float32),
                    'image': torch.tensor(patch_target[..., None], dtype=torch.float32)}
        elif self.get_type == 'label':
            return {'image': torch.tensor(patch_label[..., None], dtype=torch.float32)}
        elif self.get_type == 'target':
            return {'image': torch.tensor(patch_target[..., None], dtype=torch.float32)}