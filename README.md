# EWT-PyTorch

![](/figs/main.png)

This repository is an official PyTorch implementation of the paper **"EWT: Efficient Wavelet-Transformer for
Single Image Denoising"** from **NN 2024**.

Transformer-based image denoising methods have
achieved encouraging results in the past year. However, Transformers must use linear operations to model long-range dependencies, which greatly increases model inference time and
consumes GPU storage space. Compared with convolutional neural network-based methods, current Transformer-based image
denoising methods cannot achieve a balance between performance improvement and resource consumption. In this paper,
we propose an Efficient Wavelet Transformer (EWT) for image
denoising. Specifically, we use Discrete Wavelet Transform (DWT)
and Inverse Wavelet Transform (IWT) for downsampling and
upsampling, respectively. This method can fully preserve the image features while reducing the image resolution, thereby greatly
reducing the device resource consumption of the Transformer
model. Furthermore, we propose a novel Two-Stream Feature
Extraction Block (DFEB) to extract image features at different
levels, which can further reduce model inference time and GPU
memory usage. Experiments show that our method speeds up the
original Transformer by more than 80%, reduces GPU memory
usage by more than 60%, and achieves excellent denoising results.
All code will be public.


We provide scripts for reproducing all the results from our paper. You can train your model from scratch, or use a pre-trained model to enlarge your images.

## Dependencies
* Python 3.8
* PyTorch >= 1.7.1
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm

## Dataset
We use DIV2K dataset as clear images to train our model. Please download it from <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">here</a>  or  <a href="https://cv.snu.ac.kr/research/EDSR/DIV2K.tar">SNU_CVLab</a>. Put all clear images into the dataset/DIV2K/DIV2K_train_HR.
As for noisy images, we use Matlab/generate_noise.m to generate noisy images and put them into the dataset/DIV2K/DIV2K_train_LR_bicubic/x1.

When testing, you can put the clear images and noisy images of the test set into dataset/DIV2K/DIV2K_train_HR and dataset/DIV2K/DIV2K_train_LR_bicubic/x1 respectively

##Training

Using --ext sep_reset argument on your first running. 

You can skip the decoding part and use saved binaries with --ext sep argument in second time.

```python
## train
python main.py --scale 1 --patch_size 176 --save ewt --ext sep_reset
```

##Testing
All pre-trained model should be put into experiment/ first.
```python
## test
python main.py --data_test DIV2K --data_range 1-24 --scale 1 --pre_train your_path/EWT/experiment/model_name/model/model_best.pt --test_only --save_results --ext sep_reset
```
After the above command is run, a file named test will be generated in experiment/, where you can view the noise-removed image.

## Performance