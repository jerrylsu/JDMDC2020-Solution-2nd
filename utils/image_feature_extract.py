# -*- coding: utf-8 -*-
"""
# 使用resnet18作为特征提取器, 对图像进行特征提取
"""
import os
import argparse
import PIL
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

# set default path for data and test data
project_dir = Path(__file__).resolve().parent.parent

# temporarily use resent18 image statistics
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class Res18ImgFeatureExtractor(object):
    def __init__(self):
        self.feature_extractor = self._build_feature_extractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor.to(self.device)  # 将模型拷贝到相应设备
        self.feature_extractor.eval()  # 设置模式, 此处只进行推理

        self.images_feature = {}  # 键值对形式存储image feature

    def _build_feature_extractor(self):
        """构建特征提取器"""
        model_ft = torchvision.models.resnet18(pretrained=True)
        # 剔除last全连接
        feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])

        return feature_extractor

    def save_image_feature(self, image_feature_path='./images_feature.pkl'):
        """ 存储image feature
        @param image_feature_path:
        @return:
        """
        torch.save(self.images_feature, f=image_feature_path)

    def get_image_to_feature(self, image_dir, data_type='val'):
        """ 获取图像feature
        @param image_dir: 图像的存储路劲
        @param data_type
        @return:
        """
        if not os.path.isdir(image_dir):
            raise FileExistsError("Image Directory No Exist.")

        image_files = os.listdir(image_dir)  # 获取路径下所有文件名字
        for image_file in tqdm(image_files):
            image_path = os.path.join(image_dir, image_file)
            if os.path.isdir(image_path):
                # 列表不进行处理, 其实可以递归的, self.get
                continue
            image_data = self.read_image(image_path, data_type=data_type)
            # 0维扩充
            image_data = image_data.unsqueeze(0).to(self.device)
            image_feature = self.feature_extractor(image_data)
            # [1, 512, 1, 1] -> [512, ]
            image_feature = torch.flatten(image_feature, 1).cpu().data.numpy().squeeze()

            # 存储数据
            self.images_feature[image_file] = image_feature  # image name is key

    def get_image_to_feature_from_dirs(self, image_dir_list, data_type='val'):
        """ 获取图像feature从多个文件夹下
        @param image_dir_list: 图像的存储路劲
        @param data_type
        @return:
        """
        for image_dir in image_dir_list:
            if not os.path.isdir(image_dir):
                raise FileExistsError("Image Directory No Exist.")

            image_files = os.listdir(image_dir)  # 获取路径下所有文件名字
            for image_file in tqdm(image_files):
                image_path = os.path.join(image_dir, image_file)
                if os.path.isdir(image_path):
                    # 列表不进行处理, 其实可以递归的, self.get
                    continue
                image_data = self.read_image(image_path, data_type=data_type)
                # 0维扩充
                image_data = image_data.unsqueeze(0).to(self.device)
                image_feature = self.feature_extractor(image_data)
                # [1, 512, 1, 1] -> [512, ]
                image_feature = torch.flatten(image_feature, 1).cpu().data.numpy().squeeze()

                # 存储数据
                self.images_feature[image_file] = image_feature  # image name is key

    def read_image(self, image_path, data_type='val'):
        """ 读取图片
        @param image_path:
        @param data_type:
        @return:
        """
        image = torch.zeros(3, 224, 224)
        try:
            image_tmp = PIL.Image.open(image_path)
            image = data_transforms[data_type](image_tmp)
        except Exception as err:
            print(err)

        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="设置图像特征提取参数")

    parser.add_argument('--img_dir', default=os.path.join(project_dir, 'data/images_train'))
    parser.add_argument('--img_dev_dir', default=os.path.join(project_dir, 'data/images_dev'))
    parser.add_argument('--data_type', default='train')
    parser.add_argument('--img_feature_path', default=os.path.join(project_dir, 'data/train_images_feature.pkl'))

    args = parser.parse_args()

    main = Res18ImgFeatureExtractor()
    if args.img_dev_dir:
        img_dir_list = [args.img_dir, args.img_dev_dir]
    else:
        img_dir_list = [args.img_dir]
    # main.get_image_to_feature(image_dir=args.img_dir, data_type=args.data_type)
    main.get_image_to_feature_from_dirs(img_dir_list, data_type=args.data_type)
    main.save_image_feature(image_feature_path=args.img_feature_path)
