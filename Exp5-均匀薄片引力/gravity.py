"""
均匀薄片引力计算 - 学生模板

本模板用于计算方形薄片在垂直方向上的引力分布，学生需要完成以下部分：
1. 实现高斯-勒让德积分方法
2. 计算不同高度处的引力值
3. 绘制引力随高度变化的曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# 物理常数
G = 6.67430e-11  # 万有引力常数 (单位: m^3 kg^-1 s^-2)

def calculate_sigma(length, mass):
    """
    计算薄片的面密度
    
    参数:
        length: 薄片边长 (m)
        mass: 薄片总质量 (kg)
        
    返回:
        面密度 (kg/m^2)
    """
    # TODO: 实现面密度计算公式
    return mass / (length**2)

def integrand(x, y, z):
    """
    被积函数，计算引力积分核
    
    参数:
        x, y: 薄片上点的坐标 (m)
        z: 测试点高度 (m)
        
    返回:
        积分核函数值
    """
    # TODO: 实现积分核函数
    return 1.0 / ((x**2 + y**2 + z**2)**1.5)

def gauss_legendre_integral(length, z, n_points=100):
    """
    使用高斯-勒让德求积法计算二重积分
    
    参数:
        length: 薄片边长 (m)
        z: 测试点高度 (m)
        n_points: 积分点数 (默认100)
        
    返回:
        积分结果值
        
    提示:
        1. 使用np.polynomial.legendre.leggauss获取高斯点和权重
        2. 将积分区间从[-1,1]映射到[-L/2,L/2]
        3. 实现双重循环计算二重积分
    """
    # TODO: 实现高斯-勒让德积分
    # 获取高斯点和权重
    xi, wi = np.polynomial.legendre.leggauss(n_points)
    # 区间映射 [-1,1] -> [-L/2, L/2]
    a, b = -length/2, length/2
    x = 0.5 * (b - a) * xi + 0.5 * (b + a)
    y = 0.5 * (b - a) * xi + 0.5 * (b + a)
    wx = wi * 0.5 * (b - a)
    wy = wi * 0.5 * (b - a)
    integral = 0.0
    for i in range(n_points):
        for j in range(n_points):
            integral += wx[i] * wy[j] * integrand(x[i], y[j], z)
    return integral

def calculate_force(length, mass, z, method='gauss'):
    """
    计算给定高度处的引力
    
    参数:
        length: 薄片边长 (m)
        mass: 薄片质量 (kg)
        z: 测试点高度 (m)
        method: 积分方法 ('gauss'或'scipy')
        
    返回:
        引力值 (N)
    """
    # TODO: 调用面密度计算函数
    # TODO: 根据method选择积分方法
    # TODO: 返回最终引力值
    sigma = calculate_sigma(length, mass)
    
    if method == 'gauss':
        integral = gauss_legendre_integral(length, z)
    elif method == 'scipy':
        L_half = length / 2
        # 调整参数顺序以适配scipy的dblquad
        integral, _ = dblquad(lambda y, x: integrand(x, y, z), 
                             -L_half, L_half, 
                             lambda x: -L_half, lambda x: L_half)
    else:
        raise ValueError("Invalid method. Use 'gauss' or 'scipy'.")
    
    force = G * sigma * z * integral
    return force

def plot_force_vs_height(length, mass, z_min=0.1, z_max=10, n_points=100):
    """
    绘制引力随高度变化的曲线
    
    参数:
        length: 薄片边长 (m)
        mass: 薄片质量 (kg)
        z_min: 最小高度 (m)
        z_max: 最大高度 (m)
        n_points: 采样点数
    """
    # TODO: 生成高度点数组
    # TODO: 计算各高度点对应的引力
    # TODO: 绘制曲线图
    # TODO: 添加理论极限线
    # TODO: 设置图表标题和标签
    z_vals = np.linspace(z_min, z_max, n_points)
    F_gauss = [calculate_force(length, mass, z, method='gauss') for z in z_vals]
    F_scipy = [calculate_force(length, mass, z, method='scipy') for z in z_vals]

    plt.figure(figsize=(8,6))
    plt.plot(z_vals, F_gauss, label='Gauss-Legendre Quadrature', color='blue')
    plt.plot(z_vals, F_scipy, label='SciPy Integration', linestyle='--', color='red')
    plt.xlabel('Height z (m)')
    plt.ylabel('Gravitational Force F_z (N)')
    plt.title('Gravitational Force vs Height above Square Plate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gravity_vs_z.png")
    plt.show()  # 可取消注释在交互环境中查看

# 示例使用
if __name__ == '__main__':
    # 参数设置 (边长10m，质量1e4kg)
    length = 10
    mass = 1e4
    
    # 计算并绘制引力曲线
    plot_force_vs_height(length, mass)
    
    # 打印几个关键点的引力值
    for z in [0.1, 1, 5, 10]:
        F = calculate_force(length, mass, z)
        print(f"高度 z = {z:.1f}m 处的引力 F_z = {F:.3e} N")
