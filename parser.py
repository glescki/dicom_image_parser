#  https://github.com/ben-heil/DICOM-CNN/blob/master/utilityFunctions.py
#  https://www.researchgate.net/post/Deep_Learning_What_is_the_best_way_to_to_feed_dicom_files_into_object_detection_algorithm
# for unpacking data
import os.path
import sys
import pickle
import pydicom
import nrrd
import scipy.ndimage
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

from pathlib import Path
from datetime import datetime

from multiprocessing import Pool
from scipy.io import loadmat
from medpy.io import load
import SimpleITK as sitk
from PIL import Image


def load_dcm(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    #  try:
    #      slice_thickness = np.abs(
    #          slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    #  except:
    #      slice_thickness = np.abs(
    #          slices[0].SliceLocation - slices[1].SliceLocation)
    #
    #  for s in slices:
    #      s.SliceThickness = slice_thickness

    return slices


def load_nrrd(path):
    slices, header = nrrd.read(path)
    return np.array(slices, dtype=np.int16), header


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    image = slope * image + intercept

    return np.array(image, dtype=np.int16)


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image


# Resample voxel space to 1x1x1mm
def resample(image, spacing, new_spacing=[1, 1, 1], binary=False):
    # Determine current pixel spacing
    #  spacing = map(float, ([scan[0].PixelSpacing[0], scan[0].PixelSpacing[1],
    #                         scan[0].SliceThickness]))
    #
    spacing = np.array(spacing)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    if binary:
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=0)
    else:
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image

def sample_pair(image, label, c):
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("Image")
    ax[0].imshow(image, cmap='gray')
    ax[0].axis('off')
    ax[1].set_title("Label")
    ax[1].imshow(label, cmap='gray')
    ax[1].axis('off')
    #  plt.show()
    plt.savefig('out'+str(c)+'.png')
    plt.close('all')

def unpack_hc(patient_path):
    label_parts = list(patient_path.parts)
    label_parts[-2] = 'labels'
    label_path = Path(*label_parts).with_suffix('.nii.gz')

    label_data = nib.load(label_path)
    label = label_data.get_fdata()
    label = label.transpose(1, 0, 2)

    patient = load_dcm(str(patient_path))
    img = get_pixels_hu(patient)
    img = img.transpose(1, 2, 0)
    # Invert the z-axis
    img = img[:,:,::-1]

    spacing_array = [patient[0].PixelSpacing[0], patient[0].PixelSpacing[1], patient[0].SliceThickness]
    new_spacing_array = [patient[0].PixelSpacing[0], patient[0].PixelSpacing[1], 1]

    img, spacing = resample(img,
                            spacing_array,
                            new_spacing=new_spacing_array)

    label, spacing = resample(label,
                              spacing_array,
                              new_spacing=new_spacing_array, binary=True)

    n_channels = label.shape[2]

    for channel in range(n_channels):
        if np.any(label[:,:,channel]):
            if np.sum(label[:,:,channel]) <= 5:
                label[:,:,channel] = np.zeros_like(label[:,:,channel])
            #  sample_pair(img[:,:,channel], label[:,:,channel], channel)

    float_sa = list(map(float, new_spacing_array))
    return img, label, float_sa

def get_meta_data(split, unpack_data):
    concat_data = None
    for p in split:
        image, mask, _ = unpack_data(p)
        image_flat = image.flatten()

        if concat_data is None:
            concat_data = image_flat
        else:
            concat_data = np.concatenate(([concat_data, image_flat]), axis=0)

    return [np.mean(concat_data), np.std(concat_data), np.max(concat_data), np.min(concat_data)]


def multi_processing_create_image(inputs):
    out_dir, in_dir, unpack_data = inputs
    out_id = in_dir.name
    print("processing {}/{}".format(out_dir, out_id))
    out_id = out_id.split(".")[0]
    #  p_id = str(''.join(filter(str.isdigit, out_id)))
    #  p_id = out_id.replace('PAT', '')
    p_id = out_id.replace('PAT', '')

    out_pat = os.path.join(out_dir, p_id)

    if not os.path.exists(out_pat):
        os.makedirs(out_pat)

    c_image, c_label, metadata = unpack_data(in_dir)

    spacing_path = os.path.join(out_pat, 'spacing.npy')
    np.save(spacing_path, metadata)

    meta_lst = []
    for c in range(c_image.shape[2]):
        s_id = str(c+1)
        slice_filename = s_id
        s_image = window_image(c_image[..., c], window_center, window_width)

        norm_image = cv2.normalize(s_image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        norm_image = norm_image.astype(np.uint8)
        s_label = c_label[..., c]
        s_label = s_label.astype(np.uint8)

        out_image_path = os.path.join(out_pat, '{}.jpeg'.format(slice_filename))
        im = Image.fromarray(norm_image)
        im.save(out_image_path)

        out_mask_path = os.path.join(out_pat, '{}_mask.jpeg'.format(slice_filename))
        mask = Image.fromarray(s_label)
        mask.save(out_mask_path)

        class_id = []
        num_labels = 0
        if np.any(s_label == 1):
            im_label = s_label.astype(np.uint8)
            num_labels, _ = cv2.connectedComponents(im_label)
            num_labels -= 1

        meta_lst.append([p_id, s_id, num_labels, out_path])

    df = pd.DataFrame(meta_lst, columns=['p_id', 's_id', 'num_labels', 'path'])
    meta_filepath = out_dir+'/'+p_id+'.csv'
    df.to_csv(meta_filepath, sep=',', index=False)

def main(args):

    root_dir = str(Path.cwd())
    input_dir = Path(args.dataset)

    output_dir = os.path.join(root_dir, 'jpeg_images')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    patients = [p for p in sorted(input_dir.iterdir())]
    patients = sorted(patients, key=lambda i: int(str(os.path.splitext(os.path.basename(i))[0])[2:]))

    info = [[output_dir, pat, unpack_hc] for pat in patients]

    cpus = 1
    if args.cpus is None or args.cpus == 'arg_was_not_given':
        cpus = 1
    else:
        cpus = int(args.cpus)

    pool = Pool(processes=cpus)
    pool.map(multi_processing_create_image, info, chunksize=1)
    pool.close()
    pool.join()


def parse_args(args_lst):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="Dataset directory location")
    parser.add_argument('-c','--cpus', nargs='?', const='arg_was_not_given',
                        help="Number of cpus for multi processing")
    parser.add_argument('-wc', '--windowcenter', nargs='?', const=150, help="Window Center value")
    parser.add_argument('-ww', '--windowwidth', nargs='?', const=700, help="Window Width value")

    return parser.parse_args(args_lst)


window_center = 150
window_width = 700

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    main(args)

