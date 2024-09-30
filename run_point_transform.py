import cv2
import numpy as np
import gradio as gr
from PIL import Image


# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = Image.fromarray(img)
    return image

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    if image is None:
        return image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标

    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点

    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    image_array=np.array(image)

    marked_image_array = image_array.copy()

    for pt in points_src:
        cv2.circle(marked_image_array, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image_array, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点

    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image_array, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射

    marked_image = Image.fromarray(marked_image_array)
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    np.seterr(divide='ignore', invalid='ignore')
    q = np.ascontiguousarray(target_pts.astype(np.int16))
    p = np.ascontiguousarray(source_pts.astype(np.int16))

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p
    # 创建图像的坐标网格
    image_array = np.array(image)
    height, width = image_array.shape[:2]
    gridX = np.arange(width, dtype=np.int16)  # 创建一个包含从0到width-1的整数序列的一维数组，用于表示图像的每一列
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridY, gridX)  # vx 矩阵中的每个元素 (i, j) 表示第 i 行第 j 列像素的x坐标，而 vy 矩阵中的每个元素 (i, j) 表示相应的y坐标。
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute

    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # 控制点坐标的数组，形状为[ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # 坐标网格的数组，形状为[2, grow, gcol]
    # reshape(1,grow,gcol)将 vx 和 vy 数组重新整形为形状为 (1, grow, gcol) 的三维数组。grow 是图像的高度，gcol 是图像的宽度。reshape 方法确保每个数组的第一维是 1，这允许它们在垂直方向上堆叠。
    # np.vstack这个函数将输入数组垂直（沿着第一个轴）堆叠在一起。在这里，它将 vx 和 vy 的重新整形版本堆叠成一个形状为 (2, grow, gcol) 的新数组。

    w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha  # [ctrls, grow, gcol]
    w /= np.sum(w, axis=0, keepdims=True)  # [ctrls, grow, gcol]
    # reshaped_p - reshaped_v 代表每个控制点与网格中每个像素点之间的差异，astype(np.float32)将差异数组转换为 float32 类型，以进行后续的浮点数运算。
    # ** 2对差异数组的每个元素进行平方运算，np.sum(..., axis=1)沿着第一个轴（控制点的坐标轴）对平方差异求和。这将减少数组的维度，得到形状为 [ctrls, grow, gcol]的数组。
    # 将一个小的正数 eps 加到求和结果中，以避免除以零.** alpha,将上述结果提升到 alpha 次幂。alpha 是一个参数，用于控制权重的平滑度。
    # 1.0 / ...:取上述结果的倒数，得到权重 w。权重与控制点到像素点的距离成反比。

    pstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        pstar += w[i] * reshaped_p[i]  # [2, grow, gcol]

    phat = reshaped_p - pstar  # 每个控制点相对于加权中心的偏差，形状为[ctrls, 2, grow, gcol]
    phat = phat.reshape(ctrls, 2, 1, grow, gcol)  # 将 phat 重新整形为[ctrls, 2, 1, grow, gcol]
    phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)  # 将 phat 重新整形为[ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # 将权重 w 重新整形为[ctrls, 1, 1, grow, gcol]
    pTwp = np.zeros((2, 2, grow, gcol), np.float32)
    for i in range(ctrls):
        pTwp += phat[i] * reshaped_w[i] * phat1[i]
        # pTwp += ... 将每个控制点的贡献累加到局部变形矩阵pTwp 中
        # phat[i]为第i个控制点相对于加权中心pstar的偏差，reshaped_w[i]为第 i 个控制点的权重
    del phat1  # 循环结束后，phat1 不再需要

    try:
        inv_pTwp = np.linalg.inv(pTwp.transpose(2, 3, 0, 1))  # [grow, gcol, 2, 2],np.linalg.inv函数用于计算方阵的逆
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        # pTwp.transpose(2, 3, 0, 1)将四维数组pTwp的维度重新排列。假设pTwp的原始形状是(a, b, c, d)，转置后的形状将是(c, d, a, b)。
        # np.linalg.inv(pTwp.transpose(2, 3, 0, 1))尝试计算转置矩阵的逆。如果矩阵是奇异的，将抛出LinAlgError。
        det = np.linalg.det(pTwp.transpose(2, 3, 0, 1))  # [grow, gcol]
        det[det < 1e-8] = np.inf  # 处理接近0的行列式，将行列式值小于1e-8的元素设置为无穷大。这通常用于处理数值计算中的奇异矩阵，其中行列式接近零表明矩阵接近奇异。
        reshaped_det = det.reshape(1, 1, grow, gcol)  # [1, 1, grow, gcol]
        adjoint = pTwp[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]  # [2, 2, grow, gcol] 从pTwp中提取伴随矩阵，伴随矩阵是通过交换每个2x2矩阵的非对角线元素并改变它们的符号来得到的
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]  # [2, 2, grow, gcol]
        inv_pTwp = (adjoint / reshaped_det).transpose(2, 3, 0, 1)  # [grow, gcol, 2, 2]将伴随矩阵的每个元素除以对应的行列式值来计算逆矩阵，然后进行转置操作，以得到最终的逆矩阵形状[grow, gcol, 2, 2]

    mul_left = reshaped_v - pstar  # 从reshaped_v中减去pstar，得到一个形状为[2, grow, gcol]的数组
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)  # [grow, gcol, 1, 2]首先将mul_left重塑为形状[1, 2, grow, gcol]的数组，然后转置得到形状[grow, gcol, 1, 2]的数组。
    mul_right = np.multiply(reshaped_w, phat, out=phat)  # [ctrls, 2, 1, grow, gcol] 将reshaped_w和phat逐元素相乘，结果存储在phat中。结果数组的形状为[ctrls, 2, 1, grow, gcol]。
    reshaped_mul_right = mul_right.transpose(0, 3, 4, 1, 2)  # [ctrls, grow, gcol, 2, 1]将mul_right转置，得到形状[ctrls, grow, gcol, 2, 1]的数组。
    out_A = mul_right.reshape(2, ctrls, grow, gcol, 1, 1)[0]  # [ctrls, grow, gcol, 1, 1]首先将mul_right重塑为形状[2, ctrls, grow, gcol, 1, 1]的数组，然后提取第一个元素（索引为0），得到形状[ctrls, grow, gcol, 1, 1]的数组。
    A = np.matmul(np.matmul(reshaped_mul_left, inv_pTwp), reshaped_mul_right, out=out_A)  # [ctrls, grow, gcol, 1, 1]首先，reshaped_mul_left和inv_pTwp相乘，然后结果与reshaped_mul_right相乘。最终结果存储在out_A中，形状为[ctrls, grow, gcol, 1, 1]。
    A = A.reshape(ctrls, 1, grow, gcol)  # [ctrls, 1, grow, gcol]将A重塑为形状[ctrls, 1, grow, gcol]的数组。
    del mul_right, reshaped_mul_right, phat  # 删除不再需要的变量，以释放内存。

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    qstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        qstar += w[i] * reshaped_q[i]  # [2, grow, gcol]
    del w, reshaped_w

    # Get final image transfomer -- 3-D array
    transformers = np.zeros((2, grow, gcol), np.float32)  # 初始化一个形状为[2, grow, gcol]的数组transformers，所有元素为零。
    for i in range(ctrls):
        transformers += A[i] * (reshaped_q[i] - qstar)
    transformers += qstar
    del A

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf  # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > grow - 1] = 0
    transformers[1][transformers[1] > gcol - 1] = 0


    affine_maps = transformers.astype(np.float32)  # 形状为[2, grow, gcol]
    # 使用 cv2.remap 进行变形
    warped_image_array = cv2.rotate(cv2.flip(cv2.remap(image_array, affine_maps[0], affine_maps[1], cv2.INTER_LINEAR),1), cv2.ROTATE_90_COUNTERCLOCKWISE)
    warped_image = Image.fromarray(warped_image_array)

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables
    if image is None or len(points_src) < 4 or len(points_dst) < 4:
        return image
    warped_image = point_guided_deformation(image, np.array(points_src,  dtype=np.float32), np.array(points_dst,  dtype=np.float32), alpha=1.0, eps=1e-8)

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
