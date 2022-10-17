import numpy, shutil, os, nibabel
import argparse
import imageio
from pathlib import Path 
from tqdm import tqdm
import numpy as np 


def main():

    base_path=os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description='Arguments for input and output files')
    parser.add_argument('-i', type=str, default = base_path, help='Path of the input files')
    parser.add_argument('-o', type=str, default = base_path, help='Place to save images')
    args = parser.parse_args()
    input_path = args.i
    outputfile = args.o

    if not os.path.exists(outputfile):
        os.makedirs(outputfile)
        print("Created ouput directory: " + outputfile)

    source_files = sorted(p for p in Path(input_path).glob('*.nii.gz*'))

    for file in tqdm(source_files):
        image_array = np.fliplr(nibabel.load(file).get_fdata())
        data = numpy.rot90(image_array, 3)
        fname = os.path.basename(file)
        image_name = fname[:-7] + ".png"
        imageio.imwrite(image_name, data)

        src = image_name
        shutil.move(src, outputfile)


if __name__ == "__main__": 
    main()