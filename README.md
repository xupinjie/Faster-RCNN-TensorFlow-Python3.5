# tf-faster-rcnn
Tensorflow Faster R-CNN for Windows by using Python 3.5 

This is the branch to compile Faster R-CNN on Windows and resolve some problem in the [original branch](https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3.5). It is heavily inspired by the great work done [here](https://github.com/smallcorgi/Faster-RCNN_TF) and [here](https://github.com/rbgirshick/py-faster-rcnn). I have not implemented anything new but I fixed the implementations for Windows and Python 3.5. 

# How To Use This Branch
1- Install tensorflow, preferably GPU version. Follow [instructions]( https://www.tensorflow.org/install/install_windows). If you do not install GPU version, you need to comment out all the GPU calls inside code and replace them with relavent CPU ones.

2- Install python packages (cython, python-opencv, easydict)

3- Checkout this branch

4- Go to  ./data/coco/PythonAPI

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run `python setup.py build_ext --inplace`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run `python setup.py build_ext install`

5- Follow this instruction to download PyCoco database. [Link]( https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)
  
6- Download pre-trained VGG16 from [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) and place it as "data\imagenet_weights\vgg16.ckpt"
 
 For rest of the models, please check [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
 
7- Run `train.py` to train the model
  
8- Run `test.py` to get the result

&nbsp;&nbsp;&nbsp;&nbsp;Run `PR.py` to get the mAP and PR curve


&nbsp;&nbsp;&nbsp;&nbsp;Run `draw.py` to draw the results in test set
