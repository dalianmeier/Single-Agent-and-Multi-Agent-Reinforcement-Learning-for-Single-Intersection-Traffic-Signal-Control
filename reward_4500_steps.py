import matplotlib.pyplot as plt
import numpy as np

# 数据
algorithms = ['DDPG', 'TD3', '2MADDPG', '3MADDPG']
mean_steps = [3600.00, 3599.70, 2509.63, 2000.38]
std_devs = [0.00, 1.99, 42.05, 29.30]
# 设置条形的位置和宽度
x_pos = np.arange(len(algorithms))
width = 0.5

# 创建条形图
fig, ax = plt.subplots()

# 绘制条形并添加图例
bars = ax.bar(x_pos, mean_steps, width, yerr=std_devs, capsize=5, color=['skyblue', 'orange', 'limegreen', 'pink'])

# 添加标签和标题
ax.set_xlabel('Algorithms')
ax.set_ylabel('Mean Steps to reach 4500 reward')
ax.set_title('Comparison of Mean Steps to reach 4500 reward with Std')
ax.set_xticks(x_pos)
ax.set_xticklabels(algorithms)

# 设置图例
legend_labels = [f'{algorithms[i]} (Mean: {mean_steps[i]:.2f}, Std: {std_devs[i]:.2f})' for i in range(len(algorithms))]
ax.legend(bars, legend_labels, loc='best')

# 显示数据标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}', ha='center', va='bottom')

add_labels(bars)

# 显示图像
plt.tight_layout()
plt.show()