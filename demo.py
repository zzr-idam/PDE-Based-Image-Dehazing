import torch
import torch.nn.functional as F
import numpy as np
from skimage import io, img_as_float32
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cv2
import torchvision.transforms.functional as TF


def calculate_dark_channel(image, patch_size=15):
    """改进的暗通道计算"""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # 使用最小值池化代替简单的最小值计算
    min_channel = torch.min(image, dim=1, keepdim=True)[0]
    pad = patch_size // 2
    padded = F.pad(min_channel, (pad, pad, pad, pad), mode='reflect')
    
    # 使用平均池化代替最大值池化，减少噪声影响
    dark_channel = F.avg_pool2d(padded, kernel_size=patch_size, stride=1)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel, top_percent=0.1):
    """改进的大气光照估计"""
    b, c, h, w = image.shape
    
    # 找到暗通道中最亮的0.1%像素
    num_pixels = max(1, int(h * w * top_percent))
    _, indices = torch.topk(dark_channel.view(b, -1), num_pixels, dim=1)
    
    # 使用这些位置从原图中提取像素
    indices = indices.unsqueeze(1).expand(-1, c, -1)
    flat_image = image.view(b, c, -1)
    selected_pixels = torch.gather(flat_image, 2, indices)
    
    # 取最亮的前1%像素的平均值
    top_pixels = selected_pixels.topk(k=max(1, num_pixels//100), dim=2)[0]
    atmospheric_light = top_pixels.mean(dim=2, keepdim=True)
    
    return atmospheric_light.view(b, c, 1, 1)

def estimate_transmission_map(dark_channel, omega=0.75, t_min=0.2):
    """改进的传输图估计"""
    # 调整传输图计算公式
    transmission = 1.0 - omega * dark_channel
    # 增加最小值限制，避免除零错误
    return torch.clamp(transmission, min=t_min, max=0.99)

def compute_gradient(image):
    """使用Scharr算子计算更精确的梯度"""
    scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], 
                           dtype=image.dtype, device=image.device) / 32.0
    scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], 
                           dtype=image.dtype, device=image.device) / 32.0
    
    scharr_x = scharr_x.view(1, 1, 3, 3)
    scharr_y = scharr_y.view(1, 1, 3, 3)
    
    if image.shape[1] > 1:
        scharr_x = scharr_x.repeat(image.shape[1], 1, 1, 1)
        scharr_y = scharr_y.repeat(image.shape[1], 1, 1, 1)
    
    grad_x = F.conv2d(image, scharr_x, padding=1, groups=image.shape[1])
    grad_y = F.conv2d(image, scharr_y, padding=1, groups=image.shape[1])
    
    return grad_x, grad_y

def compute_gradient_norm(grad_x, grad_y, epsilon=1e-3):
    """计算梯度范数（添加稳定性）"""
    return torch.sqrt(grad_x**2 + grad_y**2 + epsilon**2)

def compute_diffusion_coefficient(gradient_norm, k=0.1):
    """改进的扩散系数计算（Perona-Malik）"""
    # 使用指数函数避免过度平滑
    return torch.exp(-(gradient_norm / k)**2)

def compute_nonlocal_term(image, sigma=1.0, kernel_size=3):
    """改进的非局部项计算"""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # 使用高斯滤波代替双边滤波（避免OpenCV错误）
    return TF.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)

def compute_divergence_term(diffusion_coeff, grad_x, grad_y):
    """使用Scharr算子计算散度项"""
    scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], 
                           dtype=grad_x.dtype, device=grad_x.device) / 32.0
    scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], 
                           dtype=grad_x.dtype, device=grad_x.device) / 32.0
    
    scharr_x = scharr_x.view(1, 1, 3, 3)
    scharr_y = scharr_y.view(1, 1, 3, 3)
    
    channels = grad_x.shape[1]
    scharr_x = scharr_x.repeat(channels, 1, 1, 1)
    scharr_y = scharr_y.repeat(channels, 1, 1, 1)
    
    # 计算扩散系数加权梯度
    Vx = diffusion_coeff * grad_x
    Vy = diffusion_coeff * grad_y
    
    # 计算散度
    div_x = F.conv2d(Vx, scharr_x, padding=1, groups=channels)
    div_y = F.conv2d(Vy, scharr_y, padding=1, groups=channels)
    
    return div_x + div_y

def dehaze_pde(
    hazy_image,
    lambda0=0.1,  # 减小正则化权重
    beta=2.0,      # 调整自适应参数
    max_iter=100,  # 减少迭代次数
    tol=1e-4,
    patch_size=15,
    omega=0.75,    # 减小omega值
    t_min=0.2,     # 增加传输图最小值
    sigma=1.0,
    kernel_size=3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """改进的PDE图像去雾方法"""
    hazy_image = hazy_image.to(device)
    
    if hazy_image.dim() == 3:
        hazy_image = hazy_image.unsqueeze(0)
    
    if hazy_image.shape[1] == 1:
        hazy_image = torch.cat([hazy_image] * 3, dim=1)
    
    b, c, h, w = hazy_image.shape
    
    # 步骤1: 计算暗通道
    dark_channel = calculate_dark_channel(hazy_image, patch_size)
    
    # 步骤2: 估计大气光照
    atmospheric_light = estimate_atmospheric_light(hazy_image, dark_channel)
    
    # 步骤3: 估计传输图
    transmission = estimate_transmission_map(dark_channel, omega, t_min)
    
    # 步骤4: 初始化去雾图像
    with torch.no_grad():
        dehazed_image = (hazy_image - atmospheric_light * (1 - transmission)) / transmission
        dehazed_image = torch.clamp(dehazed_image, 0, 1)
    
    # 计算自适应正则化参数
    lambda_t = lambda0 * torch.exp(-beta * (1 - transmission))
    
    # 迭代求解PDE
    prev_dehazed = dehazed_image.clone()
    for i in tqdm(range(max_iter), desc="Dehazing"):
        # 计算梯度
        grad_x, grad_y = compute_gradient(prev_dehazed)
        
        # 计算梯度范数
        gradient_norm = compute_gradient_norm(grad_x, grad_y)
        
        # 计算扩散系数
        diffusion_coeff = compute_diffusion_coefficient(gradient_norm)
        
        # 计算非局部项
        nonlocal_term = compute_nonlocal_term(prev_dehazed, sigma, kernel_size)
        
        # 计算重建项（保真项）
        reconstruction = (hazy_image - atmospheric_light * (1 - transmission)) / transmission
        
        # 计算散度项
        divergence_term = compute_divergence_term(diffusion_coeff, grad_x, grad_y)
        
        # 更新去雾图像（减小步长）
        update = divergence_term - lambda_t * nonlocal_term + 0.1 * (reconstruction - prev_dehazed)
        dehazed_image = prev_dehazed + 0.05 * update
        
        # 限制像素值范围
        dehazed_image = torch.clamp(dehazed_image, 0, 1)
        
        # 检查收敛性
        residual = torch.norm(dehazed_image - prev_dehazed)
        if residual < tol:
            print(f"Converged at iteration {i} with residual {residual:.6f}")
            break
            
        prev_dehazed = dehazed_image.clone()
        
        if (i+1) % 10 == 0:
            print(f"Iteration {i+1}/{max_iter}, Residual: {residual:.6f}")
    
    return dehazed_image

def enhance_image_contrast(image):
    """使用自适应直方图均衡化增强对比度（修复合并错误）"""
    # 确保图像在0-255范围内
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:  # RGB图像
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
        
        # 分离通道
        l, a, b = cv2.split(lab)
        
        # 对L通道进行CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # 合并通道前确保所有通道数据类型一致
        lab = cv2.merge([l, a.astype(l.dtype), b.astype(l.dtype)])
        
        # 转换回RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:  # 灰度图像
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image_uint8)
    
    # 转换回浮点并归一化
    return enhanced.astype(np.float32) / 255.0

def process_image(image_path, output_path=None, show_result=True):
    """处理单张图像"""
    img = img_as_float32(io.imread(image_path))
    
    # 保存原始图像尺寸
    original_shape = img.shape
    
    # 转换为PyTorch张量
    if len(img.shape) == 2:  # 灰度图
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:  # RGB图
        img_tensor = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    
    # 执行去雾
    start_time = time.time()
    dehazed_tensor = dehaze_pde(img_tensor)
    end_time = time.time()
    print(f"Dehazing completed in {end_time - start_time:.2f} seconds")
    
    # 转换回numpy数组
    dehazed_img = dehazed_tensor.squeeze(0).detach().cpu().numpy()
    if dehazed_img.shape[0] == 3:  # RGB图
        dehazed_img = np.transpose(dehazed_img, (1, 2, 0))
    
    # 确保尺寸匹配
    if dehazed_img.shape[:2] != original_shape[:2]:
        dehazed_img = cv2.resize(dehazed_img, (original_shape[1], original_shape[0]))
    
    # 应用对比度增强
    dehazed_img = enhance_image_contrast(dehazed_img)
    
    # 保存结果
    if output_path:
        plt.imsave(output_path, np.clip(dehazed_img, 0, 1))
        print(f"Dehazed image saved to {output_path}")
    
    # 显示结果
    if show_result:
        plt.figure(figsize=(15, 8))
        
        plt.subplot(131)
        plt.imshow(img)
        plt.title('Original Hazy Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(np.clip(dehazed_img, 0, 1))
        plt.title('Dehazed Image')
        plt.axis('off')
        
        # 显示直方图比较
        plt.subplot(133)
        plt.hist(img.flatten(), bins=256, color='blue', alpha=0.5, label='Original')
        plt.hist(dehazed_img.flatten(), bins=256, color='red', alpha=0.5, label='Dehazed')
        plt.title('Histogram Comparison')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return dehazed_img

if __name__ == "__main__":
    try:
        image_path = "1.jpg"
        output_path = "dehazed_image.jpg"
        dehazed_image = process_image(image_path, output_path)
    except Exception as e:
        print(f"An error occurred: {e}")
