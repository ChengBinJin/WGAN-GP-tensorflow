# WGAN-GP-tensorflow
This repository is a Tensorflow implementation of [WGAN-GP](https://arxiv.org/abs/1704.00028).

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/46613989-a53ea080-cb4f-11e8-83c1-a8b99dc7bc5b.png" width=800)
</p>  
  
* *All samples in README.md are genearted by neural network except the first image for each row.*

## Requirements

## Applied GAN Structure

## Generated Images
### 1. Toy Dataset
Results from 2-dimensional of the 8 Gaussian Mixture Models, 25 Gaussian Mixture Models, and Swiss Roll data. [Ipython Notebook](https://github.com/ChengBinJin/WGAN-GP-tensorflow/tree/master/src/jupyter).  

**Note:** To demonstrate following experiment, we held the generator distribution Pg fixed at the real distribution plus unit-variance Gaussian noise.
- **Top:** GAN discriminator  
- **Middle:** WGAN critic with weight clipping  
- **Bottom:** WGAN critic with weight penalty  
<p align = 'center'>
  <a>
    <img src = 'https://user-images.githubusercontent.com/37034031/46775237-55bfc680-cd41-11e8-84ee-21d793f56631.gif' width=1000>
  </a>
</p>

**Note:** For the next experiment, we did not fix generator and showed generated points by the generator.
- **Top:** GAN discriminator     
- **Middle:** WGAN critic with weight clipping  
- **Bottom:** WGAN critic with weight penalty  
<p align = 'center'>
  <a>
    <img src = 'https://user-images.githubusercontent.com/37034031/46775252-6a9c5a00-cd41-11e8-8646-8778b9561519.gif' width=1000>
  </a>
</p>

### 2. MNIST Dataset
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/47339737-1dbc6a00-d6d7-11e8-83bc-d499d06912fc.png" width=900>
</p>

<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/47339786-3f1d5600-d6d7-11e8-98ec-bc19c9532168.png" width=900>
</p>

### 3. CIFAR-10

## Documentation
### Download Dataset

### Directory Hierarchy

### Implementation Details

### Training WGAN-GP

### WGAN-GP During Training
1. **MNIST**
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/47340155-56107800-d6d8-11e8-8ec2-46ef2d8af096.png" width=900>
</p>

### Test WGAN-GP

### Citation
```
  @misc{chengbinjin2018wgan-gp,
    author = {Cheng-Bin Jin},
    title = {WGAN-GP-tensorflow},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/WGAN-GP-tensorflow}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project borrowed some code from [igul222](https://github.com/igul222/improved_wgan_training).
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer).

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.

## Related Projects
- [Vanilla GAN](https://github.com/ChengBinJin/VanillaGAN-TensorFlow)
- [DCGAN](https://github.com/ChengBinJin/DCGAN-TensorFlow)
- [WGAN](https://github.com/ChengBinJin/WGAN-TensorFlow)
- [pix2pix](https://github.com/ChengBinJin/pix2pix-tensorflow)
- [DiscoGAN](https://github.com/ChengBinJin/DiscoGAN-TensorFlow)
