from torch.utils.data import Dataset
import random
import h5py
import time
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class CsrNetDataset(Dataset):
    def __init__(
            self,
            root,
            shape=None,
            shuffle=True,
            transform=None,
            train=False,
            seen=0,
            batch_size=1,
            num_workers=4):
        if train:
            root = root * 4
        random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ground_truth_dir = 'csrnetplus_ground_truth'

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]

        img, target, locations = self.load_data(img_path)

        if self.transform is not None:
            img_transformed = self.transform(img)
        else:
            img_transformed = transforms.ToTensor()(img)

        img = transforms.ToTensor()(img)
        return img_transformed, img, target, locations

    @staticmethod
    def load_gt_path(gt_path: str) -> h5py.File:
        r"""Loads an h5py file into memory. In case of an error in loading, execution will pause for 30 seconds before
        trying again. After 8 retries, the error will be rethrown.

        Args:
            gt_path: str. The path to the data to load.

        Returns
        -------
        The data in the file.
        """
        retry_count = 8
        for i in range(retry_count + 1):
            try:
                gt_file = h5py.File(gt_path)
                return gt_file
            except OSError:
                if i >= retry_count:
                    print('OS Error. Giving Up')
                    raise
                else:
                    print('OS Error. Waiting 30 seconds and trying again...')
                    time.sleep(30)

    def load_data(self, img_path):
        gt_path = img_path.replace('.jpg', '.h5').replace('images', self.ground_truth_dir)
        img = Image.open(img_path).convert('RGB')
        gt_file = self.load_gt_path(gt_path)
        target = np.asarray(gt_file['static_density_1'])
        locations = np.asarray(gt_file['locations'])

        target = cv2.resize(target, (target.shape[1]//8, target.shape[0]//8), interpolation=cv2.INTER_CUBIC)*64

        return img, target, locations
