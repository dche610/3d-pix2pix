import nibabel as nib
import argparse
from pathlib import Path 
from tqdm import tqdm
import os


def load_data(input_path, label_path, mask):

    input = nib.load(input_path)
    label = nib.load(label_path)
    mask = nib.load(mask)
    assert(input.shape == mask.shape == label.shape)

    return input, label, mask


def mask_filter(input, label, mask, name, destination): 
    input_data = input.get_fdata()
    mask_data = mask.get_fdata()
    label_data = label.get_fdata()

    for i in range(mask_data.shape[-1]): 
        if 1 in mask_data[:, :, i]: 
            img_nii = nib.Nifti1Image(input_data[:, :, i], input.affine, input.header)
            label_nii = nib.Nifti1Image(label_data[:, :, i], label.affine, label.header)
            new_mask = nib.Nifti1Image(mask_data[:, :, i], mask.affine, mask.header)
            nib.save(img_nii, Path(destination+'/'+'images'+'/'+name[:-7]+'_image2d_'+str(i)+'.nii.gz'))
            nib.save(label_nii, Path(destination+'/'+'labels'+'/'+name[:-7]+'_label2d_'+str(i)+'.nii.gz'))
            nib.save(new_mask, Path(destination+'/'+'masks'+'/'+name[:-7]+'_mask2d_'+str(i)+'.nii.gz'))
    

def main():

    parser = argparse.ArgumentParser(description='input, label, mask, destination')
    parser.add_argument('-i')
    parser.add_argument('-d')

    args = parser.parse_args()
    input_files = sorted(p for p in Path(args.i+'\\images').glob('*.nii.gz*'))
    label_files = sorted(p for p in Path(args.i+'\\labels').glob('*.nii.gz*'))
    mask_files = sorted(p for p in Path(args.i+'\\masks').glob('*.nii.gz*'))
    input_names = os.listdir(path=Path(args.i+'\\images'))
    print(input_names[0])

    assert len(input_files) == len(mask_files) == len(label_files)
    
    for nfile in tqdm(range(len(input_files))): 
        input, label, mask = load_data(input_files[nfile], label_files[nfile], mask_files[nfile])
        mask_filter(input, label, mask, input_names[nfile], args.d)   


if __name__ == "__main__": 
    main()