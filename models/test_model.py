from models.pix2pix3D_model import Pix2Pix3DModel
from .base_model import BaseModel
from . import networks3D
from .cycle_gan_model import CycleGANModel
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser = Pix2Pix3DModel.modify_commandline_options(parser, is_train=False)
        parser.set_defaults(dataset_mode='single')

        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [which_epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix]

        self.netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.gated)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_A = input[0].to(self.device)  # the torch tensor patch in the inference function
        self.real_B = input[1].to(self.device)
        self.mask_A = input[2].to(self.device)
        # self.image_paths = input['A_paths']

    def forward(self):
        mask_A_down = (self.mask_A - 0.5) / 0.5
        self.fake_B = self.netG(torch.cat((mask_A_down, self.real_A), 1)) 
        self.fake_B = self.fake_B * (self.mask_A) + self.real_A * (1 - self.mask_A)  
