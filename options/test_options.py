from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--image", type=str, default='./Data_folder/test/images/0.nii')
        parser.add_argument("--mask", type=str, default='./Data_folder/test/masks/0.nii')
        parser.add_argument("--true_label", type=str, default='./Data_folder/test/labels/0.nii')
        parser.add_argument("--result", type=str, default='./Data_folder/test/pred', help='path to the folder to save predictions')
        parser.add_argument('--phase', type=str, default='test', help='test')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument("--stride_inplane", type=int, nargs=1, default=32, help="Stride size in 2D plane")
        parser.add_argument("--stride_layer", type=int, nargs=1, default=32, help="Stride size in z direction")
        parser.add_argument("--save_labels", type=bool, default=False, help="Save normalized labels (255)")

        parser.set_defaults(model='test')
        self.isTrain = False
        return parser