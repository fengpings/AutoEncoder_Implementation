from typing import Tuple, Any

import numpy as np
from PIL import Image
from torchvision import datasets


class NoiseAEMNIST(datasets.MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img1 = img.numpy().copy()
        img = Image.fromarray(img1, mode='L')

        h, w = img1.shape
        img2 = img1.copy().reshape(-1)
        noise_index = np.random.randint(0, h * w, 100)
        img2[noise_index] = 255  # 白噪声
        img2 = img2.reshape(h, w)
        img2 = Image.fromarray(img2, mode="L")

        # 噪声信息增加
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # 真实图像，加噪声后的图像，标签值
        return img, img2, target
