# Assignment 3 - Play with GANs

### In this assignment, you will implement two methods for using GANs in digital image processing.

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

<div style="display: flex; justify-content: center; align-items: center;">
    <img src="https://github.com/user-attachments/assets/71f26f1c-2653-4d31-be3b-004a768b5024"  style="width: 200px;">
    <img src="https://github.com/user-attachments/assets/ff3b99e5-6cbd-450e-a0d5-62e6b625651e"  style="width: 200px;">
</div>

<div style="display: flex; justify-content: center; align-items: center;">
     <img src="https://github.com/user-attachments/assets/7ccdf2fc-c10c-49a2-9fbe-28ba59c3ff57"  style="width: 200px;">
     <img src="https://github.com/user-attachments/assets/10f007a7-056c-411a-aaba-2d06a9b2390b"  style="width: 200px;">
</div>

For face alignment, you can see a picture:
<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/f2cf0b5b-6eca-4864-9d99-5fd6c6b76938" alt="Figure_1" style="width: 500px;">
</div>

#### Running

To run Automatically Edit Faces, run:

```
python final_editor.py
```

### Results 

The set_expressions.py includes expressions like 'enlarge eyes', 'close eyes', 'close mouth', 'smile mouth' and 'slim face'. The final_editor.py can automatically edit faces.

The Gradio interface may be unstable, but the images and videos generated after the iteration will be located in the folder [final_tmp](https://github.com/GrowLaugh/zuoye/tree/main/03_PlayWithGANs/Automatically%20Edit%20Faces/final_tmp).



##### Expression: enlarge eyes

<div style="display: flex; justify-content: center; align-items: center;">
     <img src="https://github.com/user-attachments/assets/03dd0d24-0f9f-4b98-aa59-89e72c592af2"  style="width: 200px;">
     <img src="https://github.com/user-attachments/assets/52f2c7bd-127b-4573-9e82-06e42706028d"  style="width: 200px;">
</div>



##### Expression: close eyes   

<div style="display: flex; justify-content: center; align-items: center;">
     <img src="https://github.com/user-attachments/assets/65cf169b-53e9-4bec-815c-178480890c39"  style="width: 200px;">
     <img src="https://github.com/user-attachments/assets/02ad8cf3-5f7c-4ace-ad8a-af0bfb6e0975"  style="width: 200px;">
</div>


##### Expression: slim face   

<div style="display: flex; justify-content: center; align-items: center;">
     <img src="https://github.com/user-attachments/assets/b56fb384-70bd-4227-b309-065ba7c04806"  style="width: 200px;">
     <img src="https://github.com/user-attachments/assets/59b16d37-725c-4f79-91e5-bfea91682c4e"  style="width: 200px;">
</div>


##### Expression: smile mouth  

<div style="display: flex; justify-content: center; align-items: center;">
     <img src="https://github.com/user-attachments/assets/c40a955e-abfc-4bbb-a28c-1e884547f798"  style="width: 200px;">
     <img src="https://github.com/user-attachments/assets/ac147e79-6b6f-4ebb-af87-64434f93475e"  style="width: 200px;">
</div>


### Acknowledgement

>📋 Thanks for the algorithms proposed by [GAN-Pytorch](https://github.com/growvv/GAN-Pytorch) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [DragGAN](https://github.com/autonomousvision/draggan) and [Face-Alingment](https://github.com/1adrianb/face-alignment)

