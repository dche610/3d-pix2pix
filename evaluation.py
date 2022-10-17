from skimage import metrics
from pathlib import Path
from tqdm import tqdm
import numpy as np 
import nibabel as nib
import os

def eval(input_path, output_path):
    input = nib.load(input_path)
    output = nib.load(output_path)
    input_data = input.get_fdata()
    output_data = output.get_fdata()
    mse = metrics.mean_squared_error(input_data, output_data)
    psnr = metrics.peak_signal_noise_ratio(input_data, output_data, data_range=255)
    ssim = metrics.structural_similarity(input_data, output_data, data_range=255)

    return mse, psnr, ssim

def main():
    input_files = sorted(p for p in Path('Data_folder/test/pred').glob('*.nii*'))
    output_files = sorted(p for p in Path('Data_folder/test/new_labels').glob('*.nii*'))
    # input_names = os.listdir(path='Data_folder/test/new_labels')
    # print(input_names)
    assert len(input_files) == len(output_files)

    all_mse = []
    all_psnr = []
    all_ssim = []
    for nfile in tqdm(range(len(input_files))):
        mse, psnr, ssim = eval(input_files[nfile], output_files[nfile])
        all_mse.append(mse)
        all_psnr.append(psnr)
        all_ssim.append(ssim)

    print(f'Avg. MSE: {np.mean(all_mse)}, Avg. PSNR: {np.mean(all_psnr)}, Avg. SSIM: {np.mean(all_ssim)}')

if __name__ == '__main__':
    main()