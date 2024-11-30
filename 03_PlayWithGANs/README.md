# Assignment 3 - Play with GANs

### In this assignment, you will implement two methods for using GANs in digital image processing.

### Resources:
- [DragGAN](https://vcai.mpi-inf.mpg.de/projects/DragGAN/): [Implementaion 1](https://github.com/XingangPan/DragGAN) & [Implementaion 2](https://github.com/OpenGVLab/DragGAN)
- [Facial Landmarks Detection](https://github.com/1adrianb/face-alignment)

---
### Task 1: Increment hw2 with Discriminative Loss



### Task 2: Automatically Edit Faces


#### Setup
1. Download the [DragGAN](https://github.com/OpenGVLab/DragGAN) codebase and the [Facial Landmarks Detection](https://github.com/1adrianb/face-alignment) codebase. 

To use the codebase:
```setup
python setup.py build
python setup.py install
```

2. You can also run the demo.
```demo
python draggan_example.py
python facealingment_example.py
```
For draggan, the generated pictures and videos are in the folder [draggan_tmp](https://github.com/GrowLaugh/zuoye/tree/main/03_PlayWithGANs/Automatically%20Edit%20Faces/draggan_tmp).

For face alignment, you can see a picture:

We want to use the DragGAN to edit the face and use the face alignment to get the facial landmarks.

## Running

To run Automatically Edit Faces, run:

```
python final_editor.py
```

## Results 
The set_expressions.py includes 'enlarge eyes', 'close eyes', 'close mouth', 'smile mouth' and 'slim face'. You can add more expressions in the file.

 The generated pictures and videos are in the folder [final_tmp](https://github.com/GrowLaugh/zuoye/tree/main/03_PlayWithGANs/Automatically%20Edit%20Faces/final_tmp).

#### Expression: enlarge eyes

#### Expression: close eyes

![enlarge eyes](./images/enlarge_eyes.png)

#### Expression: close mouth

![close mouth](./images/close_mouth.png)

#### Expression: slim face

#### Expression: smile mouth


### Basic Transformation
<img src="pics/global_demo.gif" alt="alt text" width="800">

### Point Guided Deformation:
<img src="pics/point_demo.gif" alt="alt text" width="800">

