import matplotlib.pyplot as plt
import numpy as np

# 数据
episodes = [i for i in range(1, 101)]

DDPG = [7513, 7410, 6143, 4514, 4519, 4565, 4515, 4815, 4545, 4520, 4520, 4971, 4523, 4514, 4500, 4516, 4520, 4510, 4519, 4536, 4663, 4570, 4512, 4537, 4653, 4605, 4614, 4522, 4506, 4566, 4523, 4522, 5215, 4516, 4523, 4572, 4523, 4601, 4518, 4538, 4805, 4703, 4551, 4522, 4524, 4620, 4520, 4658, 4637, 4521, 4780, 4521, 4572, 4616, 4518, 4523, 4513, 4704, 4520, 4815, 4521, 4599, 4506, 4531, 4512, 4527, 4783, 4525, 4872, 4516, 4582, 4702, 4518, 5054, 4801, 4518, 4674, 4518, 4524, 5144, 4513, 4523, 4573, 4521, 4518, 4802, 4520, 4768, 4521, 4677, 4550, 4664, 4607, 4588, 4593, 5008, 4954, 4518, 4573, 4704]
TD3 = [7417, 7351, 6588, 4639, 4765, 4517, 4528, 5074, 4523, 4508, 4526, 4507, 4544, 4514, 4697, 4711, 4521, 4561, 4551, 4931, 4535, 4531, 5005, 4534, 4694, 4608, 4500, 4522, 5353, 4526, 4977, 4519, 4521, 4527, 4533, 4627, 4615, 4615, 4866, 4519, 4550, 4498, 4523, 4521, 4532, 4863, 4521, 4757, 4693, 4620, 4516, 4661, 5046, 4524, 4520, 4517, 4516, 5012, 4520, 4528, 4510, 4515, 4573, 4519, 4524, 4519, 4524, 5103, 4502, 4520, 4681, 4510, 4578, 4517, 4516, 4522, 4538, 4514, 4599, 4599, 4521, 4744, 4515, 4613, 4520, 4520, 4525, 4825, 4521, 4525, 4669, 4542, 4525, 4566, 4527, 4520, 4738, 4514, 4515, 4545]
MADDPG_2 = [7518, 7479, 6449, 4510, 4558, 4498, 4775, 4614, 4507, 4501, 4684, 4493, 4495, 4515, 4551, 4508, 4508, 4510, 4615, 4695, 4499, 4986, 4545, 5167, 4501, 4527, 4973, 4569, 4855, 4525, 5286, 4512, 5073, 4526, 4587, 4506, 4590, 4506, 4502, 4486, 4570, 4741, 4495, 4511, 4496, 5153, 4501, 4519, 4493, 4503, 4536, 4490, 4507, 4908, 4616, 4704, 4495, 4505, 4499, 4505, 4725, 4984, 4509, 4682, 4493, 4749, 4602, 4513, 4501, 4504, 4507, 4503, 4506, 4843, 4706, 4497, 5094, 4920, 4885, 4781, 4500, 4510, 4487, 5295, 4725, 4813, 4806, 5158, 4946, 4942, 5310, 4888, 5096, 5213, 5186, 5070, 4887, 5311, 5429, 5360]
MADDPG_3 = [7460, 7387, 6986, 4485, 4485, 4485, 4633, 4641, 4496, 4526, 4710, 4483, 4489, 4708, 4564, 4561, 4543, 4534, 4522, 4472, 4543, 4661, 4482, 4474, 4746, 4871, 4479, 4628, 4521, 4506, 4503, 4481, 4485, 4474, 4485, 4481, 4566, 4485, 4467, 4476, 4554, 4491, 4520, 4487, 4481, 4484, 4466, 4508, 4495, 4480, 4744, 4600, 4494, 4484, 4482, 4482, 4594, 4703, 4874, 4496, 4486, 4656, 4488, 4597, 4485, 4469, 5079, 4494, 4549, 4858, 4738, 4475, 4574, 4497, 4493, 4722, 4495, 4497, 4475, 4469, 4808, 4796, 5047, 4691, 4917, 5103, 5096, 4804, 4809, 4960, 5020, 5063, 5101, 4757, 4970, 4985, 5122, 5497, 5096, 5544]

# 计算平均值和标准差
def compute_stats(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std

# 绘图
fig, ax = plt.subplots()

mean_DDPG, std_DDPG = compute_stats(DDPG)
mean_TD3, std_TD3 = compute_stats(TD3)
mean_MADDPG_2, std_MADDPG_2 = compute_stats(MADDPG_2)
mean_MADDPG_3, std_MADDPG_3 = compute_stats(MADDPG_3)

# 画DDPG的曲线和填充区域
ax.plot(episodes, DDPG, label='DDPG', color='#1f77b4',linewidth=1)
ax.fill_between(episodes, np.array(DDPG)-std_DDPG, np.array(DDPG)+std_DDPG, color='#1f77b4', alpha=0.2, edgecolor='#1f77b4')

# 画TD3的曲线和填充区域
ax.plot(episodes, TD3, label='TD3', color='#ff7f0e',linewidth=1)
ax.fill_between(episodes, np.array(TD3)-std_TD3, np.array(TD3)+std_TD3, color='#ff7f0e', alpha=0.2, edgecolor='#ff7f0e')

# 画MADDPG_2的曲线和填充区域
ax.plot(episodes, MADDPG_2, label='MADDPG_2', color='#2ca02c',linewidth=1)
ax.fill_between(episodes, np.array(MADDPG_2)-std_MADDPG_2, np.array(MADDPG_2)+std_MADDPG_2, color='#2ca02c', alpha=0.2, edgecolor='#2ca02c')

# 画MADDPG_3的曲线和填充区域
ax.plot(episodes, MADDPG_3, label='MADDPG_3', color='#d62728',linewidth=1)
ax.fill_between(episodes, np.array(MADDPG_3)-std_MADDPG_3, np.array(MADDPG_3)+std_MADDPG_3, color='#d62728', alpha=0.2, edgecolor='#d62728')

# 设置图例和标签
ax.legend()
ax.set_xlabel('Episodes')
ax.set_ylabel('Rewards')
ax.set_title('Average Reward and Standard Deviation over Episodes')

# 显示图像
plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 数据
# metrics = [
#     'reward', 'pedwaittime (10^9 s)', 'peddelaytime (10^9 s)', 'pedwaitnumber',
#     'vehwaittime (10^6 s)', 'vehdelaytime (10^6 s)', 'vehwaitnumber'
# ]
#
# # 调整数据，缩放单位以便更易于理解
# means = {
#     'DDPG': [4682.52, 168523.30, 300902.23, 407.38, 3.04, 571.97, 16695.06],
#     'TD3': [4684.11, 171046.11, 303328.78, 409.83, 3.11, 582.89, 16709.27],
#     '2MADDPG': [4778.87, 126634.36, 225146.69, 203.71, 1.45, 270.69, 8186.10],
#     '3MADDPG': [4726.85, 113615.30, 200564.27, 136.29, 0.99, 183.74, 5470.42]
# }
#
# stds = {
#     'DDPG': [449.36, 24.86, 40.01, 50.30, 0.58, 87.12, 1702.82],
#     'TD3': [461.55, 24.54, 40.86, 51.30, 0.52, 88.57, 1696.41],
#     '2MADDPG': [497.84, 18.30, 30.43, 24.85, 0.35, 48.01, 901.97],
#     '3MADDPG': [505.07, 16.77, 26.46, 16.38, 0.24, 33.20, 640.25]
# }
#
# # 创建数据框
# mean_df = pd.DataFrame(means, index=metrics)
# std_df = pd.DataFrame(stds, index=metrics)
#
# # 保留两位小数
# mean_df = mean_df.round(2)
# std_df = std_df.round(2)
#
# # 合并均值和标准差数据框
# result_df = mean_df.copy()
# for col in mean_df.columns:
#     result_df[col] = mean_df[col].astype(str) + " ± " + std_df[col].astype(str)
#
# # 在数据框中插入“Algorithm”列
# result_df.insert(0, 'Algorithm', metrics)
#
# # 创建表格图片
# fig, ax = plt.subplots(figsize=(12, 6))  # 调整figsize以适应表格的大小
# ax.axis('tight')
# ax.axis('off')
#
# # 创建表格
# table = ax.table(
#     cellText=result_df.values,
#     colLabels=result_df.columns,
#     cellLoc='center',
#     loc='center'
# )
#
# # 设置表格样式
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1.2, 1.2)  # 调整表格大小
#
# # 添加表头
# plt.title('Performance Metrics', fontsize=14, fontweight='bold')
#
# # 保存为图片
# plt.savefig('table.png', bbox_inches='tight', dpi=300)
# plt.show()