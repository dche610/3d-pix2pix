import nibabel as nib
import argparse
from pathlib import Path 
from tqdm import tqdm
import os
import torch


def load_data(input_path, output_path, mask):

    input = nib.load(input_path)
    output = nib.load(output_path)
    mask = nib.load(mask)
    assert(input.shape == output.shape == mask.shape)

    return input, output, mask


def blender(input, output, mask, name, destination): 

    input_data = input.get_fdata()
    output_data = output.get_fdata()
    mask_data = mask.get_fdata()
    blended = input_data.copy()
    inv_mask = 1 - mask_data 
    masked_input = blended * inv_mask
    cropped_region = output_data * mask_data 
    blended = masked_input + cropped_region

    blended_nii = nib.Nifti1Image(blended, input.affine, input.header)
    nib.save(blended_nii, Path(destination+'/blended_'+name))


def main():

    parser = argparse.ArgumentParser(description='input, output, mask, destination')
    parser.add_argument('-i')
    parser.add_argument('-o')
    parser.add_argument('-m')
    parser.add_argument('-d')

    args = parser.parse_args()
    input_files = sorted(p for p in Path(args.i).glob('*.nii.gz*'))
    output_files = sorted(p for p in Path(args.o).glob('*.nii.gz*'))
    mask_files = sorted(p for p in Path(args.m).glob('*.nii.gz*'))
    input_names = os.listdir(path=args.i)

    assert len(input_files) == len(output_files) == len(mask_files)
    
    for nfile in tqdm(range(len(input_files))): 
        input, output, mask = load_data(input_files[nfile], output_files[nfile], mask_files[nfile])
        blender(input, output, mask, input_names[nfile], args.d)   


if __name__ == "__main__": 
    main()