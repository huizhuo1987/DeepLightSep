# Learning to Separate Multiple Illuminants in a Single Image

This is the implementation described in the paper "Learning to Separate Multiple Illuminants in a Single Image, Zhuo Hui, Ayan Chakrabarti, Kalyan Sunkavalli, Aswin C. Sankaranarayanan, CVPR 2019" .

Website: https://huizhuo1987.github.io/learningIllum.html

The code skeleton is based on "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix" and "https://github.com/lixx2938/CGIntrinsics". If you use our code for academic purposes, please consider citing:

    @inproceedings{hui2019learning,
	  	title={Learning to Separate Multiple Illuminants in a Single Image},
	  	author={Hui, Zhuo and Chakrabarti, Ayan and Sunkavalli, Kalyan and Sankaranarayanan, Aswin C},
	  	booktitle={Computer Vision and Pattern Recognition (CVPR 2019)},
	  	year={2019}
	}
  

### Training dataset:
Download the dataset from Google drive: [Comming soon]
### Test images:
Download the test image dataset from Google drive: 
#### Pretrained model:
Download the pretrained the model: https://www.dropbox.com/s/cn1xylahysyqmnr/pretrained_models.zip?dl=0

### Train the network
To train your network, run the following command
```bash
    python train.py --dataroot {path_to_training_data} --model threelayers --name {your_training_name} 
    --lrA 0.0001 --lrB 0.0001 --niter 100 --niter_decay 100 --display_id -1 --gpu_ids {your_gpu_ids}
```

### Test image
To test the performance, run the following command
```bash
    python test.py --dataroot {path_to_test_data} --model threelayers --name {your_training_name} 
    --gpu_ids {your_gpu_ids}
```
