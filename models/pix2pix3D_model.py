import torch
from .base_model import BaseModel
from . import networks3D


class Pix2Pix3DModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='aligned')  # CHANGED 
        if is_train:
            # parser.set_defaults(pool_size=0, gan_mode='vanilla')  gan_mode should be equivalent to the no_lsgan option
            parser.set_defaults(pool_size=0)
            parser.add_argument('--lambda_L1', type=float, default=100000.0, help='weight for L1 loss')

        return parser

    def name(self):
        return 'Pix2Pix3D'

    def initialize(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # self.loss_names = ['G_GAN', 'G_GAN2', 'G_L1', 'D_real', 'D_fake', 'D_real2', 'D_fake2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain and not self.coarse:
            self.model_names = ['G', 'D']
            # self.model_names = ['G', 'D', 'D2']
        else:  # during test time, only load G (also during coarse network training)
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.gated)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if opt.gan == 'nsgan':
                use_sigmoid = True
            else: 
                use_sigmoid = False
            use_linear = opt.global_disc
            spectral = opt.spectral
            if not self.coarse:
                self.netD = networks3D.define_D(opt.input_nc, opt.ndf, opt.netD, # removed adding output_nc to input_nc
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids, spectral, use_linear)
                # self.netD2 = networks3D.define_D(opt.input_nc, opt.ndf, opt.netD, # patchgan
                #                          opt.n_layers_D2, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids, spectral, not use_linear)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks3D.GANLoss(type=opt.gan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if not self.coarse: 
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                # self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            if not self.coarse:
                self.optimizers.append(self.optimizer_D)
                # self.optimizers.append(self.optimizer_D2)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)
        self.mask_A = input[2].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        mask_A_down = (self.mask_A - 0.5) / 0.5
        self.fake_B = self.netG(torch.cat((mask_A_down, self.real_A), 1))  # G(A)

    def backward_D(self):  # rename in the future to GD if using 2 discriminators? 
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.mask_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False, is_disc=True)
        # Real
        real_AB = torch.cat((self.mask_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True, is_disc=True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_D2(self):
        fake_AB2 = torch.cat((self.mask_A, self.raw_fake_B), 1)
        pred_fake_patch = self.netD2(fake_AB2.detach())
        self.loss_D_fake2 = self.criterionGAN(pred_fake_patch, False, is_disc=True)
        real_AB2 = torch.cat((self.mask_A, self.real_B), 1)
        pred_real_patch = self.netD2(real_AB2)
        self.loss_D_real2 = self.criterionGAN(pred_real_patch, True, is_disc=True)
        self.loss_D_patch = (self.loss_D_fake2 + self.loss_D_real2) * 0.5
        self.loss_D_patch.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if not self.coarse:
            fake_AB = torch.cat((self.mask_A, self.fake_B), 1)
            # fake_AB_patch = torch.cat((self.mask_A, self.raw_fake_B), 1)
            pred_fake = self.netD(fake_AB)
            # pred_fake_patch = self.netD2(fake_AB_patch)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True, is_disc=False)
            # self.loss_G_GAN2 = self.criterionGAN(pred_fake_patch, True, is_disc=False)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        if not self.coarse:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
            # self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1
        else: 
            self.loss_G = self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # self.raw_fake_B = self.fake_B.clone() # raw output
        self.fake_B = self.fake_B * (self.mask_A) + self.real_A * (1 - self.mask_A)  # blended image
        if not self.coarse:
            self.mask_A = (self.mask_A - 0.5)/0.5 
            # self.fake_B = self.real_A * (1 - (self.mask_A/255))  # blended image
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            # self.set_requires_grad(self.netD2, False)
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights

            # self.set_requires_grad(self.netD2, True)  # enable backprop for D2
            # self.set_requires_grad(self.netD, False)
            # self.optimizer_D2.zero_grad()     # set D2's gradients to zero
            # self.backward_D2()                # calculate gradients for D2
            # self.optimizer_D2.step()          # update D2's weights
        # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            # self.set_requires_grad(self.netD2, False)  # D2 requires no gradients when optimizing G 
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
