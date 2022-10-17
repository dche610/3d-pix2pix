# imports 
import glob 
import shutil 
import argparse
import pathlib 


def image_label_split(train_path, test_path):

    with open('images.txt', 'w') as images, open('labels.txt', 'w') as labels:
        for file in glob.glob(train_path+'\\**\\*.nii.gz', recursive=True): 
            if 'T1lesion_mask' in file: 
                labels.write(file+'\n')
            elif 'T1w' in file: 
                images.write(file+'\n')
    
    with open('test_images.txt', 'w') as images: 
        for file in glob.glob(test_path+'\\**\\*.nii.gz', recursive=True): 
            images.write(file+'\n')

def image_mover(destination):
    
    image_dest = pathlib.Path(destination+'\\inputs')
    image_dest.mkdir(exist_ok=True)
    label_dest = pathlib.Path(destination+'\\label_maps')
    label_dest.mkdir(exist_ok=True)
    test_dest = pathlib.Path(destination+'\\test')
    test_dest.mkdir(exist_ok=True)
    
    with open('images.txt', 'r') as images, open('labels.txt', 'r') as labels: 
        for line in images: 
            shutil.move(line.strip('\n'), image_dest)
        for line in labels: 
            shutil.move(line.strip('\n'), label_dest)
    
    with open('test_images.txt') as images: 
        for line in images:
            shutil.move(line.strip('\n'), test_dest)
            

def main(): 
    parser = argparse.ArgumentParser(description='splits image and label from training data')
    parser.add_argument('-train_path')
    parser.add_argument('-test_path')
    parser.add_argument('-dest')
    args = parser.parse_args()
    image_label_split(args.train_path, args.test_path)
    image_mover(args.dest)


if __name__ == "__main__": 
    main()