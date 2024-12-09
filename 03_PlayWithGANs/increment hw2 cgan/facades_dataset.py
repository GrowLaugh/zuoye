import torch
from torch.utils.data import Dataset
import cv2

class FacadesDataset(Dataset):
    def __init__(self, list_file):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # Read the list of image filenames
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]#将每个图像文件名存储在列表中
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)#获取数据集大小
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]#获取图像文件名
        img_color_semantic = cv2.imread(img_name)#读取图像文件，返回一个NumPy数组
        # Convert the image to a PyTorch tensor
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        '''
        torch.from_numpy(img_color_semantic).permute(2, 0, 1)
        将图像的通道维度（H, W, C）转换为通道维度在前（C, H, W）。/255.0 * 2.0 -1.0 将像素值从 [0, 255] 范围缩放到 [-1, 1] 范围。
        /255.0 * 2.0 -1.0 将像素值从 [0, 255] 范围缩放到 [-1, 1] 范围。
        '''
        image_rgb = image[:, :, :256]#宽度上，前256是rgb标签，后256是语义标签
        image_semantic = image[:, :, 256:]
        # image_semantic = image[:, :, :256]#宽度上，前256是语义标签，后256是rgb标签
        # image_rgb = image[:, :, 256:]
        return image_rgb, image_semantic