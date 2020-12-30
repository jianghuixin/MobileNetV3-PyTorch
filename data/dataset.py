import logging
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class CatDog(Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        """
        获取所有目标图片
        :param root: 包含 test1 和 train 子文件夹
        :param transforms: 数据转换
        :param train: 是否为训练集
        :param test: 是否为测试集
        """

        assert not (train and test), "train and test are true"

        self.root = root
        self.test = test

        if test:
            img_folder = os.path.join(root, "test1")
        else:
            img_folder = os.path.join(root, "train")

        img_names = sorted(img_name for img_name in os.listdir(img_folder))
        img_num = len(img_names)

        if test:
            self.imgs = img_names
        elif train:
            # 训练集 70%, 验证集 30%
            num = int(img_num * 0.7)
            self.imgs = img_names[:num]
        else:
            num = int(img_num * 0.7)
            self.imgs = img_names[num:]

        # 获取数据转换操作

        if transforms is None:
            normalize = T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )

            if train:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
        else:
            self.transforms = transforms

    def __getitem__(self, idx):
        img_name = self.imgs[idx]

        if self.test:
            img_path = os.path.join(self.root, "test1", img_name)
        else:
            img_path = os.path.join(self.root, "train", img_name)

        img = Image.open(img_path)

        img_data = self.transforms(img)

        if self.test:
            return img_data
        else:
            animal = img_name.split(".", maxsplit=1)[0]

            if animal == "cat":
                label = 0
            elif animal == "dog":
                label = 1
            else:
                logging.warning(f"{img_name}")
                return self[idx + 1]

            return img_data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    transforms = T.Compose([])
    dataset_train = CatDog("/home/jianghuixin/Datasets/CatDog", transforms=transforms, train=False)

    img, label = dataset_train[2]
    print(label)
    img.show()
