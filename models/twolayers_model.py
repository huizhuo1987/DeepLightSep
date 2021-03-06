import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class TwoLayersModel(BaseModel):
    def name(self):
        return 'TwoLayersModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser
    def compute_gamma(self, chrom):
        #norm_chrom = chrom/(1e-6 + torch.sum(chrom, 1, keepdim=True).repeat(1, 3, 1, 1))
        chrom = torch.max(chrom, torch.zeros_like(chrom))
        self.no_albedo_nf = torch.log(self.rgb_img + 1e-6) - torch.log(1e-6 + chrom)
        sum_albedo = torch.sum(torch.exp(self.no_albedo_nf), dim=1, keepdim=True)
        gamma = self.no_albedo_nf - torch.log(sum_albedo.repeat(1, 3, 1, 1) + 1e-6)
        return torch.exp(gamma)

    def compute_whitebalance(self, chrom):
        chrom = torch.max(chrom, torch.zeros_like(chrom))
        self.no_albedo_nf = torch.log(self.rgb_img + 1e-6) - torch.log(1e-6 + chrom)
        sum_albedo = torch.sum(torch.exp(self.no_albedo_nf), dim=1, keepdim=True)
        gamma = self.no_albedo_nf - torch.log(sum_albedo.repeat(1, 3, 1, 1) + 1e-6)
        img_wb = torch.log(self.rgb_img + 1e-6) - gamma
        return torch.exp(img_wb)

    def compute_shading(self, gamma):
        shading1 = torch.cuda.FloatTensor(gamma.size(0), 1, gamma.size(2), gamma.size(3))
        shading2 = torch.cuda.FloatTensor(gamma.size(0), 1, gamma.size(2), gamma.size(3))
        for i in range(gamma.size(0)):
            gamma_slice = gamma[i, :, :, :]
            gamma_vec = gamma_slice.view(3, -1)

            lightT = self.lightColors[i, :, :].t()
            light = self.lightColors[i, :, :]
            A = torch.mm(lightT, light)
            B = torch.mm(lightT, gamma_vec)

            shadings, _ = torch.gesv(B, A)
            shading1[i, :, :, :] = shadings[0, :].view(1, 1, gamma.size(2), gamma.size(3))
            shading2[i, :, :, :] = shadings[1, :].view(1, 1, gamma.size(2), gamma.size(3))

        s1 = torch.mul(shading1.repeat(1, 3, 1, 1),
                       self.l1.view(self.im1.size(0), 3, 1, 1).repeat(1, 1, shading1.size(2), shading1.size(3)))
        s2 = torch.mul(shading2.repeat(1, 3, 1, 1),
                       self.l2.view(self.im1.size(0), 3, 1, 1).repeat(1, 1, shading2.size(2), shading2.size(3)))
        #s1 = shading1.repeat(1, 3, 1, 1)
        #s2 = shading2.repeat(1, 3, 1, 1)
        return s1, s2
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_A', 'G_B']#['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['rgb_img', 'chrom', 'predication', 'shading1', 'shading2']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks



        if self.isTrain:
            self.model_names = ['G_A', 'G_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        # load/define networks
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, "mnet_256", opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        """
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
        """
        if self.isTrain:
            self.image_pool = ImagePool(opt.pool_size)
            # define loss functions
            #self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            #self.criterionL1 = torch.nn.L1Loss()
            self.loss = networks.JointLoss()
            self.sloss = networks.ShadingLoss()
            self.rloss = networks.ReconstructionLoss()
            self.gloss = networks.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lrA, betas=(opt.beta1, 0.999))

            self.optimizer_G_B = torch.optim.Adam(self.netG_B.parameters(),
                                                lr=opt.lrB, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_G_B)

    def set_input(self, input):

        self.rgb_img = input['rgb_img'].to(self.device)
        self.chrom = input['chrom'].to(self.device)
        self.gamma = input['gamma'].to(self.device)

        self.image_paths = input['A_paths']
        self.mask = input['mask'].to(self.device)

        self.im1 = input['im1'].to(self.device)
        self.im2 = input['im2'].to(self.device)
        self.img_wb = input['img_wb'].to(self.device)


    def forward(self):
        self.predication = self.netG_A(self.rgb_img)
        self.est_gamma = self.compute_gamma(self.predication)
        self.shading1, self.shading2 = self.netG_B(self.est_gamma)

    def backward_G_B(self):
        input_G_B = self.image_pool.query(self.predication)

        self.est_gamma = self.compute_gamma(input_G_B.detach())
        self.shading1, self.shading2 = self.netG_B(self.est_gamma)
        self.gt_shading1, self.gt_shading2 = self.netG_B(self.gamma)

        #self.shading1, self.shading2 = self.netG_B(input_G_B.detach())

        self.loss_G_B = .25 * self.sloss(self.im1, self.im2, self.shading1, self.shading2, self.mask) \
                        + .25 * self.sloss(self.im1, self.im2, self.gt_shading1, self.gt_shading2, self.mask) \
                        + .25*self.rloss(self.shading1, self.shading2, self.est_gamma, self.mask) \
                        + .25 * self.rloss(self.gt_shading1, self.gt_shading2, self.gamma, self.mask)\
                        +.5 * self.gloss(self.gamma, self.est_gamma, self.mask)
        self.loss_G_B.backward()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        self.loss_G_A = self.loss(self.chrom, self.predication, self.mask)
        self.loss_G = self.loss_G_A
        self.loss_G.backward()



    def optimize_parameters(self):
        """
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        """
        self.forward()
        # update G_B
        self.set_requires_grad(self.netG_B, True)
        self.optimizer_G_B.zero_grad()
        self.backward_G_B()
        self.optimizer_G_B.step()

        self.set_requires_grad(self.netG_B, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()




