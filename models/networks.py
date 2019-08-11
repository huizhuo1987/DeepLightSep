import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.autograd import Variable

import scipy.io as sio
import numpy as np

import torchvision.transforms as transforms
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'mnet_256':
        net = MultiUnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'tnet_256':
        net = MultiUnetGenerator(input_nc * 2, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'connector':
        net = ConnectorGenerator(input_nc * 2, output_nc, ngf)
    elif netG == 'upunet_256':
        net = UpsampleUnetGenerator(input_nc * 2, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'mgunet_256':
        net = MergeUnetGenerator(input_nc * 2, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'render':
        net = RenderGenerator(input_nc, output_nc, ngf)
    elif netG == 'onenet_256':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
        net = netWrapper(net, input_nc * 2, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        #MultiUnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.cuda())

class L1Loss(nn.Module):
    def __call__(self, gt, prediction, mask):
        num_valid = torch.sum( mask )
        diff = torch.mul(mask, torch.abs(prediction - gt))
        return torch.sum(diff)/num_valid


class ReconstructionLoss(nn.Module):
    def L1Loss(self, prediction, gt, mask):
        num_valid = torch.sum( mask )
        diff = torch.mul(mask, torch.abs(prediction - gt))
        return torch.sum(diff)/num_valid

    def __call__(self, im1, im2, predication1, predication2, mask):
        self.loss = torch.min(self.L1Loss(predication1, im1, mask) \
                               + self.L1Loss(predication2, im2, mask),
                               self.L1Loss(predication2, im1, mask) \
                               + self.L1Loss(predication1, im2, mask))
        return self.loss

class ShadingLoss(nn.Module):
    def __init__(self):
        super(ShadingLoss, self).__init__()
        self.loss = None

    def L2Loss(self, prediction_n, mask, gt):
        num_valid = torch.sum( mask )

        diff = torch.mul(mask, torch.pow(prediction_n - gt,2))
        return torch.sum(diff)/num_valid

    def L2GradientMatchingLoss(self, log_prediction, mask, log_gt):
        N = torch.sum(mask)
        log_diff = log_prediction - log_gt
        log_diff = torch.mul(log_diff, mask)

        v_gradient = torch.pow(log_diff[:,:,0:-2,:] - log_diff[:,:,2:,:],2)
        v_mask = torch.mul(mask[:,:,0:-2,:], mask[:,:,2:,:])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.pow(log_diff[:,:,:,0:-2] - log_diff[:,:,:,2:],2)
        h_mask = torch.mul(mask[:,:,:,0:-2], mask[:,:,:,2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = (torch.sum(h_gradient) + torch.sum(v_gradient))
        gradient_loss = gradient_loss/N

        return gradient_loss
    def L1GradientMatchingLoss(self, prediction, mask, gt):
        N = torch.sum( mask )
        diff = prediction - gt
        diff = torch.mul(diff, mask)

        v_gradient = torch.abs(diff[:,:,0:-2,:] - diff[:,:,2:,:])
        v_mask = torch.mul(mask[:,:,0:-2,:], mask[:,:,2:,:])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(diff[:,:,:,0:-2] - diff[:,:,:,2:])
        h_mask = torch.mul(mask[:,:,:,0:-2], mask[:,:,:,2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = (torch.sum(h_gradient) + torch.sum(v_gradient))/2.0
        gradient_loss = gradient_loss/N

        return gradient_loss

    def Data_Loss(self, log_prediction, mask, log_gt):
        N = torch.sum(mask)
        log_diff = log_prediction - log_gt
        log_diff = torch.mul(log_diff, mask)
        s1 = torch.sum( torch.pow(log_diff,2) )/N
        s2 = torch.pow(torch.sum(log_diff),2)/(N*N)
        data_loss = s1 - s2
        return data_loss

    def ScaleInvarianceFramework(self, prediction, gt, mask):
        #prediction[prediction < 0] = 0
        #prediction[prediction > 1] = 1
        assert(prediction.size(1) == gt.size(1))
        assert(prediction.size(1) == mask.size(1))

        final_loss = self.L2Loss(prediction, mask, gt)
        final_loss += self.L1GradientMatchingLoss(prediction , mask, gt)

        # level 0
        prediction_1 = prediction[:,:,::2,::2]
        prediction_2 = prediction_1[:,:,::2,::2]
        prediction_3 = prediction_2[:,:,::2,::2]

        mask_1 = mask[:,:,::2,::2]
        mask_2 = mask_1[:,:,::2,::2]
        mask_3 = mask_2[:,:,::2,::2]

        gt_1 = gt[:,:,::2,::2]
        gt_2 = gt_1[:,:,::2,::2]
        gt_3 = gt_2[:,:,::2,::2]

        final_loss +=  self.L1GradientMatchingLoss(prediction_1, mask_1, gt_1)
        final_loss +=  self.L1GradientMatchingLoss(prediction_2, mask_2, gt_2)
        final_loss +=  self.L1GradientMatchingLoss(prediction_3, mask_3, gt_3)

        return final_loss
    def ReconstructionLoss(self, predict1, predict2, gt, mask):
        predict = predict1 + predict2
        return self.L2Loss(predict, gt, mask)

    def __call__(self, im1, im2, predication1, predication2, mask):
        self.loss =  torch.min(self.ScaleInvarianceFramework(predication1, im1, mask) \
                    + self.ScaleInvarianceFramework(predication2, im2, mask), \
                    self.ScaleInvarianceFramework(predication1, im2, mask) \
                    + self.ScaleInvarianceFramework(predication2, im1, mask))
        return self.loss

class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.loss = None

    def L1Loss(self, prediction, mask, gt):
        prediction_n = prediction
        num_valid = torch.sum( mask )
        diff = torch.mul(mask, torch.abs(prediction - gt))
        return torch.sum(diff)/num_valid

    def L2Loss(self, prediction_n, mask, gt):
        num_valid = torch.sum( mask )

        diff = torch.mul(mask, torch.pow(prediction_n - gt,2))
        return torch.sum(diff)/num_valid

    def L1GradientMatchingLoss(self, prediction, mask, gt):
        N = torch.sum( mask )
        diff = prediction - gt
        diff = torch.mul(diff, mask)

        v_gradient = torch.abs(diff[:,:,0:-2,:] - diff[:,:,2:,:])
        v_mask = torch.mul(mask[:,:,0:-2,:], mask[:,:,2:,:])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(diff[:,:,:,0:-2] - diff[:,:,:,2:])
        h_mask = torch.mul(mask[:,:,:,0:-2], mask[:,:,:,2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = (torch.sum(h_gradient) + torch.sum(v_gradient))/2.0
        gradient_loss = gradient_loss/N

        return gradient_loss

    def LinearScaleInvarianceFramework(self, prediction, gt, mask):

        assert(prediction.size(1) == gt.size(1))
        assert(prediction.size(1) == mask.size(1))
        #w_data = 1.0
        # w_grad = 0.5
        #gt_vec = gt[mask > 0.1]
        #pred_vec = prediction[mask > 0.1]
        #gt_vec = gt_vec.unsqueeze(1).float().cpu()
        #pred_vec = pred_vec.unsqueeze(1).float().cpu()

        #scale, _ = torch.gels(gt_vec.data, pred_vec.data)
        #scale = scale[0,0]

        # print("scale" , scale)
        # sys.exit()
        prediction_scaled = prediction #* scale
        final_loss =  self.L1Loss(prediction_scaled, mask, gt)

        prediction_1 = prediction_scaled[:,:,::2,::2]
        prediction_2 = prediction_1[:,:,::2,::2]
        prediction_3 = prediction_2[:,:,::2,::2]

        mask_1 = mask[:,:,::2,::2]
        mask_2 = mask_1[:,:,::2,::2]
        mask_3 = mask_2[:,:,::2,::2]

        gt_1 = gt[:,:,::2,::2]
        gt_2 = gt_1[:,:,::2,::2]
        gt_3 = gt_2[:,:,::2,::2]

        final_loss += self.L1GradientMatchingLoss(prediction_scaled , mask, gt)
        final_loss += self.L1GradientMatchingLoss(prediction_1, mask_1, gt_1)
        final_loss += self.L1GradientMatchingLoss(prediction_2, mask_2, gt_2)
        final_loss += self.L1GradientMatchingLoss(prediction_3, mask_3, gt_3)

        return final_loss


    def ReconstructionLoss(self, rgb_img, predication, gt, lightColors, mask):
        total_loss = Variable(torch.FloatTensor(1))
        total_loss[0] = 0

        predication = predication/(1e-6 + torch.sum(predication, 1, keepdim=True).repeat(1, 3, 1, 1))

        rgb_img = Variable(rgb_img.cpu(), requires_grad = False)
        gt = Variable(gt.cpu(), requires_grad=False)
        lightColors = Variable(lightColors.cpu(), requires_grad=False)
        mask = Variable(mask.cpu(), requires_grad=False)

        no_albedo_nf = rgb_img / (1e-6 + predication)
        sum_albedo = torch.sum(no_albedo_nf, 1, keepdim=True)
        gamma_p = no_albedo_nf / (sum_albedo.repeat(1, 3, 1, 1) + 1e-6)
        img_wb_p = rgb_img / (3 * gamma_p + 1e-6)

        no_albedo_nf = rgb_img / (1e-6 + gt)
        sum_albedo = torch.sum(no_albedo_nf, 1, keepdim=True)
        gamma = no_albedo_nf / (sum_albedo.repeat(1, 3, 1, 1) + 1e-6)
        img_wb = rgb_img / (3 * gamma + 1e-6)

        image_numpy = img_wb[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)))

        im_numpy = rgb_img[0].cpu().float().numpy()
        im_numpy = (np.transpose(im_numpy, (1, 2, 0)))

        mask_numpy = mask[0].cpu().float().numpy()
        mask_numpy = (np.transpose(mask_numpy, (1, 2, 0)))

        gamma_numpy = gamma[0].cpu().float().numpy()
        gamma_numpy = (np.transpose(gamma_numpy, (1, 2, 0)))

        sio.savemat('testWb.mat', {'image': image_numpy, 'input': im_numpy, 'mask': mask_numpy, 'gamma': gamma_numpy})

        total_loss = self.L2Loss(img_wb_p, mask, img_wb)
        return total_loss/rgb_img.size(0)

    def __call__(self, gt, predication, mask):
        #mask = Variable(mask.cuda(), requires_grad=False)
        #mask_R = mask[:, 0, :, :].unsqueeze(1).repeat(1, A.size(1), 1, 1)
        #gt_R = Variable(A.cuda(), requires_grad=False)
        #print(A.shape)
        self.loss =  self.LinearScaleInvarianceFramework(predication, gt, mask)
        return self.loss

    # Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class RenderGenerator(nn.Module):
    def __init__(self, input_nc, output_nc,  ngf=128):
        super(RenderGenerator, self).__init__()
        self.model = ConnectorBlock(output_nc, ngf, input_nc=input_nc)
    def forward(self, input):
        return self.model(input)

class RenderBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc):
        super(RenderBlock, self).__init__()
        ndf = 64
        model = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * 2, inner_nc, kernel_size=1, stride=1, padding=0),
                nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)



class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.Upsample(scale_factor=2, mode='nearest'),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            down = [downconv]
            up = [uprelu,
                  nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1),
                  nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            down = [downrelu, downconv]
            up = [uprelu,
                  nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1),
                  upnorm]
            model = down + up
        else:
            down = [downrelu, downconv, downnorm]
            up = [uprelu,
                  nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1),
                  upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class ConnectorGenerator(nn.Module):
    def __init__(self, input_nc, output_nc,  ngf=16):
        super(ConnectorGenerator, self).__init__()
        self.model = ConnectorBlock(output_nc, ngf, input_nc=input_nc)
    def forward(self, input):
        return self.model(input)

class ConnectorBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc):
        super(ConnectorBlock, self).__init__()
        model = [nn.Conv2d(input_nc, inner_nc, kernel_size=1),
                 nn.Conv2d(inner_nc, outer_nc, kernel_size=1)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class MultiUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(MultiUnetGenerator, self).__init__()

        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = MultiUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = MultiUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                                      use_dropout=use_dropout)
        unet_block = MultiUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = MultiUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = MultiUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = MultiUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block
    def forward(self, input):
        return self.model(input)
            # self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class MultiUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(MultiUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        # print("we are in mutilUnet")
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:

            # upconv = nn.ConvTranspose2d(inner_nc * 2, n_output_dim,
            # kernel_size=4, stride=2,
            # padding=1)
            # downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
            # stride=2, padding=1)
            # conv1 = nn.Conv2d(inner_nc, 1, kernel_size=5,
            #                  stride=1, padding=2)
            # conv2 = nn.Conv2d(inner_nc, 3, kernel_size=5,
            #                  stride=1, padding=2)
            # down = [downconv]
            down = [nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)]
            # upconv_model_1 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, 1,
            #                             kernel_size=4, stride=2, padding=1)]

            # upconv_model_2 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, 1,
            #                             kernel_size=4, stride=2, padding=1)]

            # upconv_model_u = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, inner_nc,
            # kernel_size=4, stride=2, padding=1), nn.ReLU(False),
            # nn.Conv2d(inner_nc, 1, kernel_size=1) , nn.Sigmoid()]

            # self.upconv_model_u = nn.Sequential(*upconv_model_u)
            upconv_model_1 = [nn.ReLU(False),
                              nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1),
                              norm_layer(inner_nc, affine=True), nn.ReLU(False),
                              nn.Conv2d(inner_nc, 3, kernel_size=1, bias=True)]

            upconv_model_2 = [nn.ReLU(False),
                              nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1),
                              norm_layer(inner_nc, affine=True), nn.ReLU(False),
                              nn.Conv2d(inner_nc, 3, kernel_size=1, bias=True)]


        elif innermost:
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            down = [downrelu, downconv]
            upconv_model_1 = [nn.ReLU(False),
                              nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1),
                              norm_layer(outer_nc, affine=True)]

            upconv_model_2 = [nn.ReLU(False),
                              nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1),
                              norm_layer(outer_nc, affine=True)]
        else:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            down = [downrelu, downconv, downnorm]
            up_1 = [nn.ReLU(False),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1),
                    norm_layer(outer_nc, affine=True)]
            up_2 = [nn.ReLU(False),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1),
                    norm_layer(outer_nc, affine=True)]

            if use_dropout:
                upconv_model_1 = up_1 + [nn.Dropout(0.5)]
                upconv_model_2 = up_2 + [nn.Dropout(0.5)]
                # model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                upconv_model_1 = up_1
                upconv_model_2 = up_2

            # model = down + [submodule]

        self.downconv_model = nn.Sequential(*down)
        self.submodule = submodule
        self.upconv_model_1 = nn.Sequential(*upconv_model_1)
        self.upconv_model_2 = nn.Sequential(*upconv_model_2)

    def forward(self, x):

        if self.outermost:
            down_x = self.downconv_model(x)
            y_1, y_2 = self.submodule.forward(down_x)
            # y_u = self.upconv_model_u(y_1)
            y_1 = self.upconv_model_1(y_1)

            y_2 = self.upconv_model_2(y_2)

            return y_1, y_2
            # return self.model(x)
        elif self.innermost:
            down_output = self.downconv_model(x)

            y_1 = self.upconv_model_1(down_output)
            y_2 = self.upconv_model_2(down_output)
            y_1 = torch.cat([y_1, x], 1)
            y_2 = torch.cat([y_2, x], 1)

            return y_1, y_2
        else:
            down_x = self.downconv_model(x)
            y_1, y_2 = self.submodule.forward(down_x)
            y_1 = self.upconv_model_1(y_1)
            y_2 = self.upconv_model_2(y_2)
            y_1 = torch.cat([y_1, x], 1)
            y_2 = torch.cat([y_2, x], 1)

            return y_1, y_2


#
class UpsampleUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UpsampleUnetGenerator, self).__init__()

        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UpsampleUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UpsampleUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                                      use_dropout=use_dropout)
        unet_block = UpsampleUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UpsampleUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UpsampleUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UpsampleUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block
    def forward(self, input):
        return self.model(input)
            # self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UpsampleUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UpsampleUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        # print("we are in mutilUnet")
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:

            # upconv = nn.ConvTranspose2d(inner_nc * 2, n_output_dim,
            # kernel_size=4, stride=2,
            # padding=1)
            # downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
            # stride=2, padding=1)
            # conv1 = nn.Conv2d(inner_nc, 1, kernel_size=5,
            #                  stride=1, padding=2)
            # conv2 = nn.Conv2d(inner_nc, 3, kernel_size=5,
            #                  stride=1, padding=2)
            #down = [downconv]
            #        norm_layer(16, affine=True),
            #        downrelu,
            #        norm_layer(16, affine=True),
            #        downrelu,

            down = [nn.Conv2d(input_nc, 16, kernel_size=1, stride=1, padding=0),
                    nn.LeakyReLU(0.2, False),
                    nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(16),
                    nn.LeakyReLU(0.2, False),
                    nn.Conv2d(16, inner_nc, kernel_size=4, stride=2, padding=1)]
            # upconv_model_1 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, 1,
            #                             kernel_size=4, stride=2, padding=1)]

            # upconv_model_2 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, 1,
            #                             kernel_size=4, stride=2, padding=1)]

            # upconv_model_u = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, inner_nc,
            # kernel_size=4, stride=2, padding=1), nn.ReLU(False),
            # nn.Conv2d(inner_nc, 1, kernel_size=1) , nn.Sigmoid()]

            # self.upconv_model_u = nn.Sequential(*upconv_model_u)
            upconv_model_1 = [nn.ReLU(False),
                              nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1),
                              norm_layer(inner_nc, affine=True), nn.ReLU(False),
                              nn.Conv2d(inner_nc, 3, kernel_size=1, bias=True),
                              nn.Tanh()]
            upconv_model_2 = [nn.ReLU(False),
                              nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1),
                              norm_layer(inner_nc, affine=True), nn.ReLU(False),
                              nn.Conv2d(inner_nc, 3, kernel_size=1, bias=True),
                              nn.Tanh()]


        elif innermost:
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            down = [downrelu, downconv]
            upconv_model_1 = [nn.ReLU(False),
                              nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1),
                              norm_layer(outer_nc, affine=True)]

            upconv_model_2 = [nn.ReLU(False),
                              nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1),
                              norm_layer(outer_nc, affine=True)]
        else:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            down = [downrelu, downconv, downnorm]
            up_1 = [nn.ReLU(False),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1),
                    norm_layer(outer_nc, affine=True)]
            up_2 = [nn.ReLU(False),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1),
                    norm_layer(outer_nc, affine=True)]

            if use_dropout:
                upconv_model_1 = up_1 + [nn.Dropout(0.5)]
                upconv_model_2 = up_2 + [nn.Dropout(0.5)]
                # model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                upconv_model_1 = up_1
                upconv_model_2 = up_2

            # model = down + [submodule]

        self.downconv_model = nn.Sequential(*down)
        self.submodule = submodule
        self.upconv_model_1 = nn.Sequential(*upconv_model_1)
        self.upconv_model_2 = nn.Sequential(*upconv_model_2)

    def forward(self, x):

        if self.outermost:
            down_x = self.downconv_model(x)
            y_1, y_2 = self.submodule.forward(down_x)
            # y_u = self.upconv_model_u(y_1)
            y_1 = self.upconv_model_1(y_1)

            y_2 = self.upconv_model_2(y_2)

            return y_1, y_2
            # return self.model(x)
        elif self.innermost:
            down_output = self.downconv_model(x)

            y_1 = self.upconv_model_1(down_output)
            y_2 = self.upconv_model_2(down_output)
            y_1 = torch.cat([y_1, x], 1)
            y_2 = torch.cat([y_2, x], 1)

            return y_1, y_2
        else:
            down_x = self.downconv_model(x)
            y_1, y_2 = self.submodule.forward(down_x)
            y_1 = self.upconv_model_1(y_1)
            y_2 = self.upconv_model_2(y_2)
            y_1 = torch.cat([y_1, x], 1)
            y_2 = torch.cat([y_2, x], 1)

            return y_1, y_2


#
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

class MergeUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(MergeUnetGenerator, self).__init__()

        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UpsampleUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UpsampleUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                                      use_dropout=use_dropout)
        unet_block = MergeUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = MergeUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = MergeUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = MergeUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block
    def forward(self, input):
        return self.model(input)
            # self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class MergeUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(MergeUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        # print("we are in mutilUnet")
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:

            # upconv = nn.ConvTranspose2d(inner_nc * 2, n_output_dim,
            # kernel_size=4, stride=2,
            # padding=1)
            # downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
            # stride=2, padding=1)
            # conv1 = nn.Conv2d(inner_nc, 1, kernel_size=5,
            #                  stride=1, padding=2)
            # conv2 = nn.Conv2d(inner_nc, 3, kernel_size=5,
            #                  stride=1, padding=2)
            #down = [downconv]
            down = [nn.Conv2d(input_nc, 16, kernel_size=1, stride=1, padding=0),
                    nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
                    nn.Conv2d(16, inner_nc, kernel_size=4, stride=2, padding=1)]
            # upconv_model_1 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, 1,
            #                             kernel_size=4, stride=2, padding=1)]

            # upconv_model_2 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, 1,
            #                             kernel_size=4, stride=2, padding=1)]

            # upconv_model_u = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, inner_nc,
            # kernel_size=4, stride=2, padding=1), nn.ReLU(False),
            # nn.Conv2d(inner_nc, 1, kernel_size=1) , nn.Sigmoid()]

            # self.upconv_model_u = nn.Sequential(*upconv_model_u)
            upconv_model_1 = [nn.ReLU(False),
                              nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1),
                              norm_layer(inner_nc, affine=True)
                              ]
            upconv_model_2 = [nn.ReLU(False),
                              nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1),
                              norm_layer(inner_nc, affine=True)
                             ]

            merge_model_1 =  [nn.Conv2d(inner_nc + 6, inner_nc, kernel_size=1, stride=1, padding=0),
                              nn.Conv2d(inner_nc, inner_nc, kernel_size=1, stride=1, padding=0),
                              nn.Conv2d(inner_nc, 3, kernel_size=1, stride=1, padding=0)]

            merge_model_2 =  [nn.Conv2d(inner_nc + 6, inner_nc, kernel_size=1, stride=1, padding=0),
                              nn.Conv2d(inner_nc, inner_nc, kernel_size=1, stride=1, padding=0),
                              nn.Conv2d(inner_nc, 3, kernel_size=1, stride=1, padding=0)]

            self.merge_model_1  = nn.Sequential(*merge_model_1)
            self.merge_model_2 = nn.Sequential(*merge_model_2)
        elif innermost:
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            down = [downrelu, downconv]
            upconv_model_1 = [nn.ReLU(False),
                              nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1),
                              norm_layer(outer_nc, affine=True)]

            upconv_model_2 = [nn.ReLU(False),
                              nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1),
                              norm_layer(outer_nc, affine=True)]
        else:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            down = [downrelu, downconv, downnorm]
            up_1 = [nn.ReLU(False),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1),
                    norm_layer(outer_nc, affine=True)]
            up_2 = [nn.ReLU(False),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1),
                    norm_layer(outer_nc, affine=True)]

            if use_dropout:
                upconv_model_1 = up_1 + [nn.Dropout(0.5)]
                upconv_model_2 = up_2 + [nn.Dropout(0.5)]
                # model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                upconv_model_1 = up_1
                upconv_model_2 = up_2

            # model = down + [submodule]

        self.downconv_model = nn.Sequential(*down)
        self.submodule = submodule
        self.upconv_model_1 = nn.Sequential(*upconv_model_1)
        self.upconv_model_2 = nn.Sequential(*upconv_model_2)

    def forward(self, x):

        if self.outermost:
            down_x = self.downconv_model(x)
            y_1, y_2 = self.submodule.forward(down_x)
            # y_u = self.upconv_model_u(y_1)
            y_1 = self.upconv_model_1(y_1)

            y_2 = self.upconv_model_2(y_2)

            y_1 = torch.cat([y_1, x], 1)
            y_1 = self.merge_model_1(y_1)

            y_2 = torch.cat([y_2, x], 1)
            y_2 = self.merge_model_2(y_2)
            #y_1, y_2 = output[:, :3, :, :], output[:, 3:, :, :]
            return y_1, y_2
            # return self.model(x)
        elif self.innermost:
            down_output = self.downconv_model(x)

            y_1 = self.upconv_model_1(down_output)
            y_2 = self.upconv_model_2(down_output)
            y_1 = torch.cat([y_1, x], 1)
            y_2 = torch.cat([y_2, x], 1)

            return y_1, y_2
        else:
            down_x = self.downconv_model(x)
            y_1, y_2 = self.submodule.forward(down_x)
            y_1 = self.upconv_model_1(y_1)
            y_2 = self.upconv_model_2(y_2)
            y_1 = torch.cat([y_1, x], 1)
            y_2 = torch.cat([y_2, x], 1)

            return y_1, y_2


class netWrapper(nn.Module):
    def __init__(self, net, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(netWrapper, self).__init__()

        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UpsampleUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UpsampleUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                                      use_dropout=use_dropout)
        unet_block = UpsampleUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UpsampleUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UpsampleUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UpsampleUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block
        self.net = net
    def forward(self, input):
        output = self.net(input)
        input = torch.cat((output, input), 1)
        return self.model(input)



