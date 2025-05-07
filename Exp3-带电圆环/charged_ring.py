import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import quad # 如果需要精确计算单点，可以取消注释

# --- 常量定义 ---
a = 1.0  # 圆环半径 (单位: m)
# q = 1.0  # 可以定义 q 参数，或者直接在 C 中体现
# V(x,y,z) = q/(2*pi) * integral(...)
# C 对应 q/(2*pi)，这里设 q=1
C = 1.0 / (2 * np.pi)

# --- 计算函数 ---

def calculate_potential_on_grid(y_coords, z_coords):
    """
    在 yz 平面 (x=0) 的网格上计算电势 V(0, y, z)。
    使用 numpy 的向量化和 trapz 进行数值积分。
    参数:
        y_coords (np.ndarray): y 坐标数组
        z_coords (np.ndarray): z 坐标数组
    返回:
        V (np.ndarray): 在 (y, z) 网格上的电势值 (z 维度优先)
        y_grid (np.ndarray): 绘图用的二维 y 网格坐标
        z_grid (np.ndarray): 绘图用的二维 z 网格坐标
    """
    print("开始计算电势...")
    # 1. 创建 y, z, phi 网格
    z_grid, y_grid = np.meshgrid(z_coords, y_coords, indexing='ij')
    phi = np.linspace(0, 2*np.pi, 1000)  # phi 的积分点数
    
    # 2. 计算场点到圆环上各点的距离 R
    x_s = a * np.cos(phi)  # 圆环上点的 x 坐标
    y_s = a * np.sin(phi)  # 圆环上点的 y 坐标
    z_s = 0                # 圆环上点的 z 坐标
    
    # 计算距离 R，添加维度以便广播
    R = np.sqrt((0 - x_s)**2 + (y_grid[..., np.newaxis] - y_s)**2 + (z_grid[..., np.newaxis] - z_s)**2)
    
    # 3. 处理 R 可能为零或非常小的情况，避免除零错误
    R = np.maximum(R, 1e-10)
    
    # 4. 计算电势微元 dV = C / R
    dV = C / R
    
    # 5. 对 phi 进行积分
    V = np.trapz(dV, phi, axis=-1)
    
    print("电势计算完成.")
    return V, y_grid, z_grid

def calculate_electric_field_on_grid(V, y_coords, z_coords):
    """
    根据电势 V 计算 yz 平面上的电场 E = -∇V。
    使用 np.gradient 进行数值微分。
    参数:
        V (np.ndarray): 电势网格 (z 维度优先)
        y_coords (np.ndarray): y 坐标数组
        z_coords (np.ndarray): z 坐标数组
    返回:
        Ey (np.ndarray): 电场的 y 分量
        Ez (np.ndarray): 电场的 z 分量
    """
    print("开始计算电场...")
    # 1. 计算 y 和 z 方向的网格间距
    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1
    dz = z_coords[1] - z_coords[0] if len(z_coords) > 1 else 1
    
    # 2. 使用 np.gradient 计算电势的负梯度
    grad_z, grad_y = np.gradient(-V, dz, dy)
    Ez = grad_z
    Ey = grad_y
    
    # 处理电场在圆环中心的奇异性
    y_grid, z_grid = np.meshgrid(y_coords, z_coords, indexing='ij')
    distance_to_ring = np.abs(np.sqrt(y_grid**2 + z_grid**2) - a)
    mask = distance_to_ring < 0.1 * a  # 圆环附近区域
    
    # 限制电场最大值，避免数值不稳定
    max_field = 100.0
    field_magnitude = np.sqrt(Ey**2 + Ez**2)
    scale = np.minimum(1.0, max_field / (field_magnitude + 1e-10))
    
    Ey *= scale
    Ez *= scale
    
    print("电场计算完成.")
    return Ey, Ez

# --- 可视化函数 ---

def plot_potential_and_field(y_coords, z_coords, V, Ey, Ez, y_grid, z_grid):
    """
    绘制 yz 平面上的等势线和电场线。
    参数:
        y_coords, z_coords: 定义网格的坐标范围
        V: 电势网格
        Ey, Ez: 电场分量网格
        y_grid, z_grid: 绘图用的二维网格坐标
    """
    print("开始绘图...")
    fig = plt.figure('Potential and Electric Field of Charged Ring (yz plane, x=0)', figsize=(14, 6))

    # 1. 绘制等势线图 (左侧子图)
    plt.subplot(1, 2, 1)
    contourf_plot = plt.contourf(y_grid, z_grid, V, levels=50, cmap='viridis')
    plt.colorbar(contourf_plot, label='电势 (V)')
    plt.contour(y_grid, z_grid, V, levels=10, colors='white', linewidths=0.5)
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('带电圆环的电势分布')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. 绘制电场线图 (右侧子图)
    plt.subplot(1, 2, 2)
    E_magnitude = np.sqrt(Ey**2 + Ez**2)
    stream_plot = plt.streamplot(y_grid, z_grid, Ey, Ez, color=E_magnitude, 
                               cmap='plasma', linewidth=1, density=1.5, arrowstyle='->')
    plt.colorbar(stream_plot.lines, label='电场强度 (V/m)')
    
    # 标记圆环位置
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(a*np.cos(theta), a*np.sin(theta), 'r-', linewidth=2, label='圆环截面')
    
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('带电圆环的电场分布')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()
    print("绘图完成.")

# --- 主程序 ---
if __name__ == "__main__":
    # 定义计算区域 (yz 平面, x=0)
    num_points_y = 60  # y 方向点数
    num_points_z = 60  # z 方向点数
    range_factor = 2   # 计算范围是半径的多少倍
    y_range = np.linspace(-range_factor * a, range_factor * a, num_points_y)
    z_range = np.linspace(-range_factor * a, range_factor * a, num_points_z)

    # 1. 计算电势
    V, y_grid, z_grid = calculate_potential_on_grid(y_range, z_range)

    # 2. 计算电场
    Ey, Ez = calculate_electric_field_on_grid(V, y_range, z_range)

    # 3. 可视化
    if V is not None and Ey is not None:
        plot_potential_and_field(y_range, z_range, V, Ey, Ez, y_grid, z_grid)
    else:
        print("计算未完成，无法绘图。请先实现计算函数。")
