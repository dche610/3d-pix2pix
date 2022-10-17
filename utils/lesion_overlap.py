# imports
import argparse
from pathlib import Path 
import numpy as np 
from tqdm import tqdm
import nibabel as nib
import shutil
import os


def load_data(input_path, flip_path):

    input = nib.load(input_path)
    flip = nib.load(flip_path)
    assert(input.shape == flip.shape)

    return input, flip

def overlap_calc(input, flip, t): 

    input_data = input.get_fdata()
    flip_data = flip.get_fdata()
    overlap = input_data * flip_data
    overlap_pct = np.sum(overlap) / np.sum(input_data) 
    if overlap_pct > t:
        return overlap_pct
    else: 
        return False


def main():
    parser = argparse.ArgumentParser(description='data, destination')
    parser.add_argument('-i')
    parser.add_argument('-d')

    args = parser.parse_args()
    input_masks = sorted(p for p in Path(args.i+'\\masks').glob('*.nii.gz*'))
    flipped_masks = sorted(p for p in Path(args.i+'\\flipped_masks').glob('*.nii.gz*'))
    input_images = sorted(p for p in Path(args.i+'\\images').glob('*.nii.gz*'))
    label_images = sorted(p for p in Path(args.i+'\\labels').glob('*.nii.gz*'))
    input_names = os.listdir(path=Path(args.i+'\\images'))

    if not os.path.exists(args.d):
        os.makedirs(args.d)
        print("Created ouput directory: " + args.d)
        os.makedirs(Path(args.d) / 'image')
        os.makedirs(Path(args.d) / 'mask')
        os.makedirs(Path(args.d) / 'label')

    t = 0.05 # threshold for overlap

    assert len(input_masks) == len(flipped_masks) == len(input_images)

    with open(Path(args.d) / 'overlap.txt', 'w') as txt: 
        for nfile in tqdm(range(len(input_masks))): 
            input, flip = load_data(input_masks[nfile], flipped_masks[nfile])
            overlap_pct = overlap_calc(input, flip, t)   
            if overlap_pct: 
                txt.write(input_names[nfile]+'  '+str(overlap_pct)+'\n')
                shutil.move(input_images[nfile], Path(args.d) / 'image')
                shutil.move(input_masks[nfile], Path(args.d) / 'mask')
                shutil.move(label_images[nfile], Path(args.d) / 'label')


if __name__ == '__main__': 
    main()