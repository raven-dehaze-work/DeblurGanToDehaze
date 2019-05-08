"""
数据处理工具脚本
"""
import glob
from PIL import Image
import os
import numpy as np

img_width = 256
img_height = 256


def convert_img2npy(img_dir, npy_dir):
    """
    将图片转换为npy文件，方便后续训练使用 仅转换.jpg后缀
    :param img_dir: 图片目录
    :param npy_dir: 保存的npy文件目录
    :return:
    """
    # 得到所有clear img 的文件名
    imgs_path = glob.glob(os.path.join(img_dir, "*.jpg"))

    for filepath in imgs_path:
        file_name = os.path.basename(filepath)
        print('convert %s' % file_name)

        # 打开图片
        img = Image.open(filepath).convert('RGB')
        img = img.resize((img_width, img_height))

        # 转为np
        img_np = np.array(img)
        # 保存
        np.save(os.path.join(npy_dir, file_name.replace("jpg", "npy")), img_np)



if __name__ == '__main__':
    mode = 'test'       # 'test' or 'train'

    if mode == 'train':
        # 清晰图片目录
        clear_img_dir = './train_datasets/image/clear'
        # 清晰图片的npy存储目录
        clear_npy_dir = './train_datasets/npy/clear'
        # 雾图目录
        haze_img_dir = './train_datasets/image/haze'
        # 雾图npy文件存储目录
        haze_npy_dir = './train_datasets/npy/haze'
        # 转换清晰图
        convert_img2npy(clear_img_dir, clear_npy_dir)
        # 转换雾图
        convert_img2npy(haze_img_dir, haze_npy_dir)
    elif mode == 'test':
        haze_img_dir = './test_datasets/image'
        haze_npy_dir = './test_datasets/npy'
        # 转换雾图
        convert_img2npy(haze_img_dir,haze_npy_dir)


