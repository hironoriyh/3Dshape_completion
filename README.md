# 3D Shape Completion 

## Introduction

This is a Keras Implementation of 3D Encoder-Decoder Generative Adversarial Network (3D-ED-GAN) for 3D shape Inpainting. The 3D-ED-GAN is a 3D convolutional neural network trained with a generative adversarial paradigm to fill missing 3D data. The cuboid data is generated through Python code and a random masking is applied to each instance to create corrupted shapes. 

In this experiment, since the shape is pretty simple, the network gives excellent performance after 10 mins of training on GPU.  

## Requirement

Tensorflow-gpu==1.7.0

Keras==2.20

Matplotlib>=2.2

## Example

1. epxort pattern_1 and 2 text files from Rhino + Grasshopper
2. extract voxel files by running  `python text2numpy.py`


~~~
python EncoderDecoderGAN3D.py
~~~

## ToDos
- [ ] data structure (data-tree3-images)