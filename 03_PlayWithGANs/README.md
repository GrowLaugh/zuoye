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
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="https://github.com/user-attachments/assets/ff3b99e5-6cbd-450e-a0d5-62e6b625651e" alt="image_b0fb85a1-2650-45ea-aac7-d8589836bb0f" style="width: 200px;">
    <img src="https://github.com/user-attachments/assets/10f007a7-056c-411a-aaba-2d06a9b2390b" alt="image_64f18675-fd26-430e-9af0-deef7ab7b208" style="width: 200px;">
</div>

<div style="display: flex; justify-content: center; align-items: center;">
    <video controls width="200" src="https://github.com/user-attachments/assets/85767f85-1748-4def-badd-164834f69837">Your browser does not support the video tag.</video>
    <video controls width="200" src="https://github.com/user-attachments/assets/9fc342b2-267b-45c5-90bb-46b489525da6">Your browser does not support the video tag.</video>
</div>


For face alignment, you can see a picture:
<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/f2cf0b5b-6eca-4864-9d99-5fd6c6b76938" alt="Figure_1" style="width: 300px;">
</div>

### Running

To run Automatically Edit Faces, run:

```
python final_editor.py
```

### Results 

The set_expressions.py includes expressions like 'enlarge eyes', 'close eyes', 'close mouth', 'smile mouth' and 'slim face'. The final_editor.py can automatically edit faces.

The Gradio interface may be unstable, but the images and videos generated after the iteration will be located in the folder [final_tmp](https://github.com/GrowLaugh/zuoye/tree/main/03_PlayWithGANs/Automatically%20Edit%20Faces/final_tmp).



##### Expression: enlarge eyes
<div style="display: flex; flex-direction: column; align-items: center;">
    <!-- 第一行 -->
    <div style="display: flex; justify-content: center; width: 100%;">
        <video controls width="300" src="https://github.com/user-attachments/assets/52f2c7bd-127b-4573-9e82-06e42706028d">Your browser does not support the video tag.</video>
        <img src="https://github.com/user-attachments/assets/68a4fdf0-fe33-45dc-bf5e-0b69aac98de7" alt="image_f30b59e6-0838-436e-9254-b67f7dc1c975" style="width: 300px;">
    </div>
##### Expression: close eyes       
    <!-- 第二行 -->
    <div style="display: flex; justify-content: center; width: 100%;">
        <video controls width="300" src="https://github.com/user-attachments/assets/02ad8cf3-5f7c-4ace-ad8a-af0bfb6e0975">Your browser does not support the video tag.</video>
        <img src="https://github.com/user-attachments/assets/a5025c45-69db-4052-a4a3-dd87bbdf1a98" alt="image_87fa318a-ca5b-42a1-b879-ab2356596e85" style="width: 300px;">
    </div>
##### Expression: slim face     
    <!-- 第三行 -->
    <div style="display: flex; justify-content: center; width: 100%;">
        <video controls width="300" src="https://github.com/user-attachments/assets/59b16d37-725c-4f79-91e5-bfea91682c4e">Your browser does not support the video tag.</video>
        <img src="https://github.com/user-attachments/assets/6655d524-7e58-4f5d-b4fa-c577571bf783" alt="image_d69ddc51-7f80-4c2d-891c-97d23b4ad0ee" style="width: 300px;">
    </div>
##### Expression: smile mouth  
    <!-- 第四行 -->
    <div style="display: flex; justify-content: center; width: 100%;">
        <video controls width="300" src="https://github.com/user-attachments/assets/ac147e79-6b6f-4ebb-af87-64434f93475e">Your browser does not support the video tag.</video>
        <img src="https://github.com/user-attachments/assets/cdecfe89-2b6c-469f-a0a9-320e6774692d" alt="image_e71baabd-dba4-49e7-a98d-219feaa7e131" style="width: 300px;">
    </div>


