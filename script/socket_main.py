"""
使用Socket通信，完成:
手机上传雾图，服务器分析后回传
"""

from dehazegan.model import generator_model
import os
from PIL import Image, ImageFile
import numpy as np
import socket
import sys

haze_path = './img_from_socket/haze.jpg'
dehaze_path = './img_from_socket/dehaze.jpg'

g = None
def init_net_module():
    """
    初始话网络模块
    :return: 生成器
    """
    # 模型保存目录
    model_save_dir = './model_save_res_block_256_vgg16'
    def load_saved_weight(g, d=None):
        """
        加载已训练好的权重
        :param g: 生成器
        :param d: 判别器
        :return:
        """
        # TODO: 这里需要做细化处理。判定文件是否存在。多个权重文件找到最新的权重文件
        g.load_weights(os.path.join(model_save_dir, 'generator_49_33.h5'))
        if d is None:
            return
        d.load_weights(os.path.join(model_save_dir, 'discriminator_49.h5'))

    # 构建网络模型
    global g
    g = generator_model()
    # 加载模型权重
    load_saved_weight(g)

def dehaze():
    """
    实现去雾
    :return:
    """
    haze_img = np.array(Image.open(haze_path).convert('RGB'))
    generated_img = g.predict(haze_img.reshape((1, 256, 256, 3)) / 127.5 - 1)
    generated_img = (generated_img + 1) * 127.5
    dehazed_img = Image.fromarray(generated_img[0].astype('uint8'))

    # 保存下来
    dehazed_img.save(dehaze_path)

def socket_service():
    """
    开启socket服务
    :return:
    """
    # 开启socket服务
    try:
        s = socket.socket()
        s.bind(('127.0.0.1', 6666))
        s.listen(10)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    print("Wait")

    def save_haze_file(sock):
        """
        从sock中获取数据，并保存下来
        :param sock:
        :return:
        """
        with open(haze_path, 'wb') as f:
            print('file opened')
            while True:
                data = sock.recv(1024)
                # print()
                if not data:
                    break
                elif 'EOF' in str(data):
                    f.write(data[:-len('EFO')])
                    break
                # write data to a file
                f.write(data)
        # sock.close()
        print('pic received finished')

    def send_img(sock):
        # 发送处理后的图片
        with open(dehaze_path, 'rb') as f:
            for data in f:
                sock.send(data)
        print('send finished')

    # 等待连接并处理
    while True:
        sock, _ = s.accept()
        try:
            save_haze_file(sock)
            dehaze()
            send_img(sock)
        except Exception as reason:
            print(reason)
        finally:
            sock.close()


if __name__ == '__main__':
    init_net_module()
    # 设定可读TRUNCATED 图片
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    socket_service()