import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import scipy.io as sio
from PIL import Image
import numpy as np

import skimage
from skimage import io
from skimage.transform import resize
from skimage.morphology import square
from scipy.ndimage import gaussian_filter

from scipy.misc import imread, imsave, imresize


class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.image_paths = make_dataset(self.dir_AB)
        self.image_paths = sorted(self.image_paths)
        self.isTrain = opt.isTrain
        # self.albedo_paths = sorted(self.albedo_paths)
        # self.mask_paths = sorted(self.mask_paths)
        assert (opt.resize_or_crop == 'resize_and_crop')

        self.l2Matrix = np.array([
            [49.2180748926073, 95.1918695015005, 110.590055605892],
            [59.8210540410088,	102.319349811969,	92.8595961470224],
            [52.0690518693229,	48.7959094258630,	154.135038704814],
            [47.1725592158890,	103.481827402417,	104.345613381694],
            [34.2438709828394,	94.6610859648431,	126.095043052318],
            [49.4695849857719,	100.658820118230,	104.871594895998],
            [42.7656219746282,	97.7882203333240,	114.446157692048],
            [10.5071554938084,	104.192033757628,	140.300810748563]
        ])

        self.l1Matrix = np.array([
            [164.725258574570,	79.3805344244429,	10.8942070009866],
            [164.965219814927,	81.5305678843789,	8.50421230069432],
            [152.433831660239,	80.0901827617372,	22.4759855780239],
            [141.148844417699,	82.2257789519073,	31.6253766303939],
            [114.793288740708,	93.1434031296364,	47.0633081296560],
            [166.435893274431,	85.3447183606299,	3.21938836493893],
            [173.996930461557,	80.9981152619878,	0.00495427645500340],
            [168.121663355009,	75.9110572408919,	10.9672794040988]
        ])

    def produceColor(self, img1, img2, arr_id, light_id1, light_id2, l1, l2):
        if light_id1 == 8 or light_id2 == 8:
            for i in range(3):
                img1[:,:,i] = img1[:,:,i]*l1[i]
                img2[:,:,i] = img2[:,:,i]*l2[i]
            return img1, img2, l1, l2
        else:
            if arr_id == 0:
                l1 = self.l1Matrix[light_id1]/255
                l2 = self.l2Matrix[light_id2]/255
            else:
                l2 = self.l1Matrix[light_id1]/255
                l1 = self.l2Matrix[light_id2]/255
            l1 = l1/np.sum(l1)
            l2 = l2/np.sum(l2)

        for i in range(3):
            img1[:,:,i] = img1[:,:,i]*l1[i]
            img2[:,:,i] = img2[:,:,i]*l2[i]

        l1 = l1.reshape(3, -1)
        l2= l2.reshape(3, -1)
        return img1, img2, l1, l2

    def DA(self, img, random_id):
        new_img = np.zeros_like(img)
        if random_id == 0:
            new_img = img.copy()
        elif random_id == 1:
            new_img = np.flip(img, axis=1).copy()
        elif random_id == 2:
            new_img = np.flip(img, axis=0).copy()
        elif random_id == 3:
            new_img = np.flip(np.flip(img, axis=0), axis=1).copy()
        return new_img

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # albedo_path = self.albedo_paths[index]
        # mask_path = self.mask_paths[index]

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        content = sio.loadmat(image_path)

        if self.isTrain:
            rgb_img = content['imag']
            chrom = content['chrom']
            mask = content['mask']

            mask = resize(mask, [384, 512], 1)
            mask[mask > 0] = 1

            mask = np.mean(mask, axis=2)
            #mask = skimage.morphology.binary_erosion(mask, square(1))
            mask = np.expand_dims(mask, axis=2)
            mask = np.repeat(mask, 3, axis=2)

            l1 = content['l1']
            l2 = content['l2']
            
            rand_id = random.randint(0, 3)
            light_id1 = random.randint(0, 8)
            light_id2 = random.randint(0, 8)
            arr_id = random.randint(0, 1)


            img1 = content['im1']
            img2 = content['im2']

            img1 = np.nan_to_num(img1)
            img2 = np.nan_to_num(img2)

            img1[img1 > 1.0] = 1.0
            img1[img1 < 0.0] = 0.0

            img2[img2 > 1.0] = 1.0
            img2[img2 < 0.0] = 0.0

            [img1, img2, l1, l2] = self.produceColor(img1, img2, arr_id, light_id1, light_id2, l1, l2)
            # img2 = self.DA(img2, rand_id)
            #img1 = img1/2
            #img2 = img2/2
            chrom = np.nan_to_num(chrom)
            #rgb_img = np.nan_to_num(rgb_img)

            #l1 = np.nan_to_num(l1)
            #l2 = np.nan_to_num(l2)
            # img1[img1 != img1] = 0.0
            # img2[img2 != img2] = 0.0
            # chrom[chrom != chrom] = 0.0
            # rgb_img[rgb_img != rgb_img] = 0.0

            #for i in range(3):
            #    img1[:, :, i] = img1[:, :, i] * l1[i]
            #    img2[:, :, i] = img2[:, :, i] * l2[i]

            rgb_img = img1 + img2
            #rgb_img[rgb_img > 1.0] = 1.0
            #rgb_img[rgb_img < 0.0] = 0.0

            chrom[chrom > 1.0] = 1.0
            chrom[chrom < 0.0] = 0.0

            rgb_img = resize(rgb_img, [384, 512], 1)
            img1 = resize(img1, [384, 512], 1)
            img2 = resize(img2, [384, 512], 1)
            chrom = resize(chrom, [384, 512], 1)

            rgb_img = self.DA(rgb_img, rand_id)
            chrom = self.DA(chrom, rand_id)
            mask = self.DA(mask, rand_id)

            img1 = self.DA(img1, rand_id)
            img2 = self.DA(img2, rand_id)
            # if arr_id:
            #    l1 = self.l1Matrix[light_id1]/255
            #    l2 = self.l2Matrix[light_id2]/255
            #    l1 = np.reshape(l1, (3, -1))
            #    l2 = np.reshape(l2, (3, -1))
            # else:
            #    l1 = self.l2Matrix[light_id2]/255
            #    l2 = self.l1Matrix[light_id1]/255
            #    l1 = np.reshape(l1, (3, -1))
            #    l2 = np.reshape(l2, (3, -1))

            # rgb_img = img1 + img2
            lightColor = np.concatenate((l1, l2), axis=1)
            lightColor = torch.from_numpy(lightColor).contiguous().float()

            rgb_img = torch.from_numpy(np.transpose(rgb_img, (2, 0, 1))).contiguous().float()
            chrom = torch.from_numpy(np.transpose(chrom, (2, 0, 1))).contiguous().float()
            mask = torch.from_numpy(np.transpose(mask.astype(float), (2, 0, 1))).contiguous().float()

            no_albedo_nf = rgb_img / (1e-6 + chrom)
            sum_albedo = torch.sum(no_albedo_nf, 0, keepdim=True)
            gamma = no_albedo_nf / (sum_albedo.repeat(3, 1, 1) + 1e-6)
            gamma = gamma.view(3, -1)

            lightT = lightColor.t()
            light = lightColor
            B = torch.mm(lightT, gamma)
            A = torch.mm(lightT, light)
            shadings, _ = torch.gesv(B, A)
            # shadings[0, :] = (shadings[0:, ] - torch.min(shadings[0, :]))/(torch.max(shadings[0:, ]) - torch.min(shadings[0, :]))
            # shadings[1, :] = (shadings[1:, ] - torch.min(shadings[1, :]))/(torch.max(shadings[1:, ]) - torch.min(shadings[1, :]))
            shadings[shadings != shadings] = 0.0
            im1 = shadings[0, :].repeat(3, 1).view(3, rgb_img.size(1), rgb_img.size(2))
            im2 = shadings[1, :].repeat(3, 1).view(3, rgb_img.size(1), rgb_img.size(2))

            im1 = (im1 - torch.min(im1[mask > 0]))/(torch.max(im1[mask > 0]) - torch.min(im1[mask > 0]))
            im2 = (im2 - torch.min(im2[mask > 0])) / (torch.max(im2[mask > 0]) - torch.min(im2[mask > 0]))

            # remove nan values
            im1[im1 != im1] = 0.0
            im2[im2 != im2] = 0.0

            im1[mask == 0] = 0.0
            im2[mask == 0] = 0.0

            im1[0, :, :] *= lightColor[0, 0]
            im1[1, :, :] *= lightColor[1, 0]
            im1[2, :, :] *= lightColor[2, 0]

            im2[0, :, :] *= lightColor[0, 1]
            im2[1, :, :] *= lightColor[1, 1]
            im2[2, :, :] *= lightColor[2, 1]

            # normalize the data
            # im1 = torch.mul(shadings[0, :].repeat(3, 1), lightColor[:, 0].view(3, 1).repeat(1, shadings.size(1)))
            # im2 = torch.mul(shadings[1, :].repeat(3, 1), lightColor[:, 1].view(3, 1).repeat(1, shadings.size(1)))

            # im1 = im1.view(3, rgb_img.size(1), rgb_img.size(2))
            # im2 = im2.view(3, rgb_img.size(1), rgb_img.size(2))

            im1[im1 > 1] = 1.0
            im2[im2 > 1] = 1.0

            im1[im1 < 0] = 0.0
            im2[im2 < 0] = 0.0

            # rgb_img = 2*rgb_img - 1.0

            img1 = torch.from_numpy(np.transpose(img1, (2, 0, 1))).contiguous().float()
            img2 = torch.from_numpy(np.transpose(img2, (2, 0, 1))).contiguous().float()

            return {'rgb_img': rgb_img, 'chrom': chrom, 'im1': im1, 'im2': im2, 'A_paths': image_path, 'mask': mask,
                    'img1': img1, 'img2': img2}

        else:
            rgb_img = content['imag']
            #chrom = content['chrom']
            #mask = content['chrom']
            #im1 = content['im1']
            #im2 = content['im2']

            rgb_img[rgb_img > 1] = 1
            rgb_img[rgb_img < 0] = 0
            # rgb_img = 2*rgb_img - 1.0

            #img1 = content['im1']
            #img2 = content['im2']

            rgb_img = torch.from_numpy(np.transpose(rgb_img, (2, 0, 1))).contiguous().float()
            #chrom = torch.from_numpy(np.transpose(chrom, (2, 0, 1))).contiguous().float()
            #mask = torch.from_numpy(np.transpose(mask.astype(float), (2, 0, 1))).contiguous().float()
            #im1 = torch.from_numpy(np.transpose(im1, (2, 0, 1))).contiguous().float()
            #im2 = torch.from_numpy(np.transpose(im2, (2, 0, 1))).contiguous().float()
            return {'rgb_img': rgb_img, 'A_paths': image_path}


    def __len__(self):
        return len(self.image_paths)

    def name(self):
        return 'AlignedDataset'
