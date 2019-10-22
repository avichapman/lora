import argparse
import h5py
import scipy.io as io
import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import scipy

parser = argparse.ArgumentParser(description='Make CSRNet Static Dataset')

parser.add_argument('--page_number', dest='page_number', metavar='PAGE_NUMBER', default=-1, type=int,
                    help='If provided, this and page_size are used to work out subset of images to process')

parser.add_argument('--page_size', dest='page_size', metavar='PAGE_SIZE', default=-1, type=int,
                    help='If provided, this and page_number are used to work out subset of images to process')

parser.add_argument('--dataset_loc', dest='dataset_loc', metavar='DATASET_LOC', default='C:\\Datasets\\ShanghaiTech',
                    type=str, help='Path to the dataset root directory')


# this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt, scale_factor: int, enforce_minimum: bool):
    print("Generating density with factor", scale_factor)
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    print('gt_count: ', gt_count)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    print('pts.shape: ', pts.shape)
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    min_sigma = 5.85410196624968 / 4  # Obtained by eyeballing the output for one that is just over 8 pixels wide

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
        sigma = sigma / scale_factor
        if enforce_minimum:
            sigma = max(min_sigma, sigma)
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


def gaussian_static_filter(gt, scale_factor: int):
    print("Generating static density")
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    print('gt_count: ', gt_count)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    print('pts.shape: ', pts.shape)

    enforce_minimum = True
    min_sigma = 5.85410196624968 / 4  # Obtained by eyeballing the output for one that is just over 8 pixels wide

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        sigma = 4.881709115  # Median of sigmas from original CSRNet target density
        sigma = sigma / scale_factor
        if enforce_minimum:
            sigma = max(min_sigma, sigma)
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


def main():
    args = parser.parse_args()

    # set the root to the Shanghai dataset you download
    root = args.dataset_loc

    output_dir = 'csrnetplus_ground_truth'

    # now generate the ShanghaiA's ground truth
    part_a_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_a_test = os.path.join(root, 'part_A_final/test_data', 'images')
    part_b_train = os.path.join(root, 'part_B_final/train_data', 'images')
    part_b_test = os.path.join(root, 'part_B_final/test_data', 'images')
    path_sets = [part_a_train, part_a_test, part_b_train, part_b_test]

    # Work out which images to process..
    if args.page_number > -1 and args.page_size > -1:
        first_image = args.page_number * args.page_size
        last_image = args.page_number * args.page_size + args.page_size - 1
    else:
        first_image = 1
        last_image = float('inf')

    print('first_image:', first_image)
    print('last_image:', last_image)

    img_paths = []
    current_image = -1
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            current_image += 1
            if first_image <= current_image <= last_image:
                img_paths.append(img_path)

    current_image = 0
    image_count = len(img_paths)
    for img_path in img_paths:
        current_image += 1
        print(current_image, '/', image_count, ':', img_path)

        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', output_dir).replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path)
        locations = np.zeros((img.shape[0], img.shape[1]))
        gt = mat["image_info"][0, 0][0, 0][0]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                locations[int(gt[i][1]), int(gt[i][0])] = 1
        static_density_1 = gaussian_static_filter(locations, 1)
        k1 = gaussian_filter_density(locations, 1, False)
        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', output_dir), 'w') as hf:
                hf['density'] = k1
                hf['locations'] = locations
                hf['static_density_1'] = static_density_1


if __name__ == '__main__':
    main()
