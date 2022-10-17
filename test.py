import os
from options.test_options import TestOptions
import sys
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset_testing
from torch.utils.data import DataLoader
from models import create_model
import math
from torch.autograd import Variable
from tqdm import tqdm
import datetime

def from_numpy_to_itk(image_np, image_itk):
    image_np = np.transpose(image_np, (2, 1, 0))
    image = sitk.GetImageFromArray(image_np)
    image.SetOrigin(image_itk.GetOrigin())
    image.SetDirection(image_itk.GetDirection())
    image.SetSpacing(image_itk.GetSpacing())
    return image


if __name__ == '__main__':

    opt = TestOptions().parse()

    test_set = NifitDataSet(opt.val_path, which_direction='AtoB', transforms=None, shuffle_labels=False, test=True)
    print('length test list:', len(test_set))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=True)  # Here are then fed to the network with a defined batch size

    model = create_model(opt)
    model.setup(opt)

    directory = './Data_folder/test/images'
    image_paths = [os.path.join(directory, file) for file in os.listdir(directory)]

    result_path = opt.result

    if opt.save_labels: 
        label_path = './Data_folder/test/new_labels'
        for i, data in enumerate(test_loader):
            reader = sitk.ImageFileReader()
            reader.SetFileName(image_paths[i])
            image = reader.Execute() 
            label = data[1].squeeze().data.cpu().numpy()
            label = (label * 127.5) + 127.5
            label = from_numpy_to_itk(label, image)
            writer = sitk.ImageFileWriter()
            writer.SetFileName(label_path+'/'+str(i)+'.nii')
            writer.Execute(label)
            print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), result_path+'/'+str(i)+'.nii'))

    else:
        for i, data in enumerate(test_loader):
            reader = sitk.ImageFileReader()
            reader.SetFileName(image_paths[i])
            image = reader.Execute()
            model.set_input(data)
            model.test()
            pred = model.get_current_visuals()
            pred = pred['fake_B']
            pred = pred.squeeze().data.cpu().numpy()
            pred = (pred * 127.5) + 127.5
            label = from_numpy_to_itk(pred, image)
            writer = sitk.ImageFileWriter()
            writer.SetFileName(result_path+'/'+str(i)+'.nii')
            writer.Execute(label)
            print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), result_path+'/'+str(i)+'.nii'))