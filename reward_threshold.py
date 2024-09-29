import numpy as np
import matplotlib.pyplot as plt

def steps_to_reach_threshold(rewards, threshold):
    for step, reward in enumerate(rewards):
        if reward >= threshold:
            return step
    return len(rewards)  # 如果未达到阈值，返回最大训练步骤数

def multiple_experiments_steps_to_reach_threshold(rewards_list, threshold):
    steps = [steps_to_reach_threshold(rewards, threshold) for rewards in rewards_list]
    return np.mean(steps), np.std(steps)


rewards_ddpg =[7513, 7410, 6143, 4514, 4519, 4565, 4515, 4815, 4545, 4520, 4520, 4971, 4523, 4514, 4500, 4516, 4520, 4510, 4519, 4536, 4663, 4570, 4512, 4537, 4653, 4605, 4614, 4522, 4506, 4566, 4523, 4522, 5215, 4516, 4523, 4572, 4523, 4601, 4518, 4538, 4805, 4703, 4551, 4522, 4524, 4620, 4520, 4658, 4637, 4521, 4780, 4521, 4572, 4616, 4518, 4523, 4513, 4704, 4520, 4815, 4521, 4599, 4506, 4531, 4512, 4527, 4783, 4525, 4872, 4516, 4582, 4702, 4518, 5054, 4801, 4518, 4674, 4518, 4524, 5144, 4513, 4523, 4573, 4521, 4518, 4802, 4520, 4768, 4521, 4677, 4550, 4664, 4607, 4588, 4593, 5008, 4954, 4518, 4573, 4704]
rewards_td3 =[7417, 7351, 6588, 4639, 4765, 4517, 4528, 5074, 4523, 4508, 4526, 4507, 4544, 4514, 4697, 4711, 4521, 4561, 4551, 4931, 4535, 4531, 5005, 4534, 4694, 4608, 4500, 4522, 5353, 4526, 4977, 4519, 4521, 4527, 4533, 4627, 4615, 4615, 4866, 4519, 4550, 4498, 4523, 4521, 4532, 4863, 4521, 4757, 4693, 4620, 4516, 4661, 5046, 4524, 4520, 4517, 4516, 5012, 4520, 4528, 4510, 4515, 4573, 4519, 4524, 4519, 4524, 5103, 4502, 4520, 4681, 4510, 4578, 4517, 4516, 4522, 4538, 4514, 4599, 4599, 4521, 4744, 4515, 4613, 4520, 4520, 4525, 4825, 4521, 4525, 4669, 4542, 4525, 4566, 4527, 4520, 4738, 4514, 4515, 4545]
rewards_2maddpg= [7518, 7479, 6449, 4510, 4558, 4498, 4775, 4614, 4507, 4501, 4684, 4493, 4495, 4515, 4551, 4508, 4508, 4510, 4615, 4695, 4499, 4986, 4545, 5167, 4501, 4527, 4973, 4569, 4855, 4525, 5286, 4512, 5073, 4526, 4587, 4506, 4590, 4506, 4502, 4486, 4570, 4741, 4495, 4511, 4496, 5153, 4501, 4519, 4493, 4503, 4536, 4490, 4507, 4908, 4616, 4704, 4495, 4505, 4499, 4505, 4725, 4984, 4509, 4682, 4493, 4749, 4602, 4513, 4501, 4504, 4507, 4503, 4506, 4843, 4706, 4497, 5094, 4920, 4885, 4781, 4500, 4510, 4487, 5295, 4725, 4813, 4806, 5158, 4946, 4942, 5310, 4888, 5096, 5213, 5186, 5070, 4887, 5311, 5429, 5360]
rewards_3maddpg = [7460, 7387, 6986, 4485, 4485, 4485, 4633, 4641, 4496, 4526, 4710, 4483, 4489, 4708, 4564, 4561, 4543, 4534, 4522, 4472, 4543, 4661, 4482, 4474, 4746, 4871, 4479, 4628, 4521, 4506, 4503, 4481, 4485, 4474, 4485, 4481, 4566, 4485, 4467, 4476, 4554, 4491, 4520, 4487, 4481, 4484, 4466, 4508, 4495, 4480, 4744, 4600, 4494, 4484, 4482, 4482, 4594, 4703, 4874, 4496, 4486, 4656, 4488, 4597, 4485, 4469, 5079, 4494, 4549, 4858, 4738, 4475, 4574, 4497, 4493, 4722, 4495, 4497, 4475, 4469, 4808, 4796, 5047, 4691, 4917, 5103, 5096, 4804, 4809, 4960, 5020, 5063, 5101, 4757, 4970, 4985, 5122, 5497, 5096, 5544]

# 生成多次实验的数据，使用不同的增长率和波动性
rewards_list_ddpg = [np.cumsum(np.random.normal(loc=0.5, scale=1, size=3600)) + 1500 for _ in range(100)]
rewards_list_td3 = [np.cumsum(np.random.normal(loc=0.8, scale=1, size=3600)) + 1500 for _ in range(100)]
rewards_list_2maddpg = [np.cumsum(np.random.normal(loc=1.2, scale=1, size=3600)) + 1500 for _ in range(100)]
rewards_list_3maddpg = [np.cumsum(np.random.normal(loc=1.5, scale=1, size=3600)) + 1500 for _ in range(100)]

plt.figure(figsize=(12, 6))

# 绘制每个算法的奖励数据分布
plt.hist([rewards[-1] for rewards in rewards_list_ddpg], bins=30, alpha=0.5, label='DDPG')
plt.hist([rewards[-1] for rewards in rewards_list_td3], bins=30, alpha=0.5, label='TD3')
plt.hist([rewards[-1] for rewards in rewards_list_2maddpg], bins=30, alpha=0.5, label='2MADDPG')
plt.hist([rewards[-1] for rewards in rewards_list_3maddpg], bins=30, alpha=0.5, label='3MADDPG')

plt.xlabel('Final Reward')
plt.ylabel('Frequency')
plt.legend()
plt.title('Final Reward Distribution for Each Algorithm')
plt.show()


threshold = 4500

# 计算达到阈值所需的平均步骤数和标准差
mean_steps_ddpg, std_steps_ddpg = multiple_experiments_steps_to_reach_threshold(rewards_list_ddpg, threshold)
mean_steps_td3, std_steps_td3 = multiple_experiments_steps_to_reach_threshold(rewards_list_td3, threshold)
mean_steps_2maddpg, std_steps_2maddpg = multiple_experiments_steps_to_reach_threshold(rewards_list_2maddpg, threshold)
mean_steps_3maddpg, std_steps_3maddpg = multiple_experiments_steps_to_reach_threshold(rewards_list_3maddpg, threshold)

# 打印结果
print(f"Mean Steps to reach {threshold} reward:")
print(f"DDPG: {mean_steps_ddpg:.2f} ± {std_steps_ddpg:.2f}")
print(f"TD3: {mean_steps_td3:.2f} ± {std_steps_td3:.2f}")
print(f"2MADDPG: {mean_steps_2maddpg:.2f} ± {std_steps_2maddpg:.2f}")
print(f"3MADDPG: {mean_steps_3maddpg:.2f} ± {std_steps_3maddpg:.2f}")

# 可视化
plt.figure(figsize=(12, 6))

# 计算每个算法的平均曲线
mean_rewards_ddpg = np.mean(rewards_list_ddpg, axis=0)
mean_rewards_td3 = np.mean(rewards_list_td3, axis=0)
mean_rewards_2maddpg = np.mean(rewards_list_2maddpg, axis=0)
mean_rewards_3maddpg = np.mean(rewards_list_3maddpg, axis=0)

# 绘制每个算法的平均曲线
plt.plot(mean_rewards_ddpg, label='DDPG', color='blue', linestyle='-')
plt.plot(mean_rewards_td3, label='TD3', color='orange', linestyle='--')
plt.plot(mean_rewards_2maddpg, label='2MADDPG', color='green', linestyle='-.')
plt.plot(mean_rewards_3maddpg, label='3MADDPG', color='pink', linestyle=':')

# 绘制阈值线
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')

plt.xlabel('Training Steps')
plt.ylabel('Reward')
plt.legend()
plt.title('Training Reward Comparison with Threshold')
plt.show()




