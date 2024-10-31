# Assignment 2 - DIP with PyTorch





## Results 
### 1. Implement Poisson Image Editing with PyTorch.
Fill the [Polygon to Mask function](run_blending_gradio.py#L95) and the [Laplacian Distance Computation](run_blending_gradio.py#L115) of 'run_blending_gradio.py'.

<img src="results_image/01.png" alt="alt text" width="400"><img src="results_image/02.png" alt="alt text" width="400">
<img src="results_image/03.png" alt="alt text" width="400">

---
### 2. Pix2Pix implementation.
See more in [results_image subfolder](results_image/).

#### facades train results
<img src="results_image/facades_epoch_795/result_1.png" alt="alt text" width="480">  <img src="results_image/facades_epoch_795/result_2.png" alt="alt text" width="480">
<img src="results_image/facades_epoch_795/result_3.png" alt="alt text" width="480">  <img src="results_image/facades_epoch_795/result_4.png" alt="alt text" width="480">
<img src="results_image/facades_epoch_795/result_5.png" alt="alt text" width="480">  

#### facades val results
<img src="results_image/facades_val_epoch_795/result_1.png" alt="alt text" width="480">  <img src="results_image/facades_val_epoch_795/result_2.png" alt="alt text" width="480">
<img src="results_image/facades_val_epoch_795/result_3.png" alt="alt text" width="480">  <img src="results_image/facades_val_epoch_795/result_4.png" alt="alt text" width="480">
<img src="results_image/facades_val_epoch_795/result_5.png" alt="alt text" width="480">  

#### cityscapes train results
<img src="results_image/cityscapes/cityscapes_epoch_650/result_1.png" alt="alt text" width="480">  <img src="results_image/cityscapes/cityscapes_epoch_650/result_2.png" alt="alt text" width="480">
<img src="results_image/cityscapes/cityscapes_epoch_650/result_3.png" alt="alt text" width="480">  <img src="results_image/cityscapes/cityscapes_epoch_650/result_4.png" alt="alt text" width="480">
<img src="results_image/cityscapes/cityscapes_epoch_650/result_5.png" alt="alt text" width="480">  

#### cityscapes val results
<img src="results_image/cityscapes/cityscapes_val_epoch_650/result_1.png" alt="alt text" width="480">  <img src="results_image/cityscapes/cityscapes_val_epoch_650/result_2.png" alt="alt text" width="480">
<img src="results_image/cityscapes/cityscapes_val_epoch_650/result_3.png" alt="alt text" width="480">  <img src="results_image/cityscapes/cityscapes_val_epoch_650/result_4.png" alt="alt text" width="480">
<img src="results_image/cityscapes/cityscapes_val_epoch_650/result_5.png" alt="alt text" width="480">  

---
### Requirements:
- 请自行环境配置，推荐使用[conda环境](https://docs.anaconda.com/miniconda/)
-or  :
```setup
python -m pip install -r requirements.txt
```
## Running

To run poisson image editing , run:

```basic
python run_blending_gradio.py
```

To run Pix2Pix, run:
put [train_list.txt](datasets/facades/train_list/) and [val_list.txt](datasets/facades/val_list/) into [Pix2Pix] .
```point
python train.py
```


## Acknowledgement
- [Assignment Slides](https://rec.ustc.edu.cn/share/705bfa50-6e53-11ef-b955-bb76c0fede49)  
- [Paper: Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)
- [Paper: Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)
- [Paper: Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- [PyTorch Installation & Docs](https://pytorch.org/)
