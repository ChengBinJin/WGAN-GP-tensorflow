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
    <img src = 'https://user-images.githubusercontent.com/37034031/46774996-29f01100-cd40-11e8-92c4-3caff671b926.gif' width=1000>
  </a>
</p>

**Note:** For the next experiment, we did not fix generator and showed generated points by the generator.
- **Top:** GAN discriminator     
- **Middle:** WGAN critic with weight clipping  
- **Bottom:** WGAN critic with weight penalty  
<p align = 'center'>
  <a>
    <img src = 'https://user-images.githubusercontent.com/37034031/46775078-90752f00-cd40-11e8-9c9b-0e8a59f15687.gif' width=1000>
  </a>
</p>
