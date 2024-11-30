# import face_alignment
# from skimage import io
# import torch
# import numpy as np

# # 初始化FaceAlignment对象，这里选择检测二维关键点
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda:0', dtype=torch.bfloat16 )

# input = io.imread('test/assets/aflw-test.jpg')#读取图片
# preds = fa.get_landmarks(input)# 获取人脸地标点
# print("Detected landmarks:", preds)


###
import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections


# Optionally set detector and some additional detector parameters
face_detector = 'sfd'
face_detector_kwargs = {
    "filter_threshold" : 0.8
}

# Run the 2D face alignment on a test image, with CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda:0',
                                  face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

try:
    input_img = io.imread('../assets/aflw-test.jpg')
except FileNotFoundError:
    input_img = io.imread('test/assets/aflw-test.jpg')

preds = fa.get_landmarks(input_img)[-1]
print("Detected landmarks:", preds)


# 2D-Plot
plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
# 定义一个字典，用于存储预测类型
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),  # 预测类型为face，预测范围为0-17，颜色为(0.682, 0.780, 0.909, 0.5)
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),  # 预测类型为eyebrow1，预测范围为17-22，颜色为(1.0, 0.498, 0.055, 0.4)
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),  # 预测类型为eyebrow2，预测范围为22-27，颜色为(1.0, 0.498, 0.055, 0.4)
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),  # 预测类型为nose，预测范围为27-31，颜色为(0.345, 0.239, 0.443, 0.4)
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),  # 预测类型为nostril，预测范围为31-36，颜色为(0.345, 0.239, 0.443, 0.4)
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),  # 预测类型为eye1，预测范围为36-42，颜色为(0.596, 0.875, 0.541, 0.3)
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),  # 预测类型为eye2，预测范围为42-48，颜色为(0.596, 0.875, 0.541, 0.3)
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),  # 预测类型为lips，预测范围为48-60，颜色为(0.596, 0.875, 0.541, 0.3)
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))  # 预测类型为teeth，预测范围为60-68，颜色为(0.596, 0.875, 0.541, 0.4)
              }

fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(input_img)

for pred_type in pred_types.values():
    ax.plot(preds[pred_type.slice, 0],
            preds[pred_type.slice, 1],
            color=pred_type.color, **plot_style)

ax.axis('off')

plt.show()
