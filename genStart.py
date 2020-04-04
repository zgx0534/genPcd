import os
import provider
import numpy as np
from pcdService import points2pcd

# 存放路径
PCD_DIR_PATH = os.path.join(os.path.abspath('.'), 'pcds')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
train_file_idxs = np.arange(0, len(TRAIN_FILES))
# 对每一个点云集合个数
for fn in range(len(TRAIN_FILES)):
    index = 0
    # 获得一个点云集合
    current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])

    # current_data = provider.jitter_point_cloud(current_data)
    PCD_DIR_DETAIL_PATH = os.path.join(PCD_DIR_PATH, str(fn))
    os.mkdir(PCD_DIR_DETAIL_PATH)
    fileNum = current_data.shape[0]
    for index in range(fileNum):
        PCD_FILE_PATH = os.path.join(PCD_DIR_DETAIL_PATH, str(fn) + '_' + str(index) + '.pcd')
        points2pcd(current_data[index], PCD_FILE_PATH)
        index = index + 1
