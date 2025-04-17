# longlonglonglong
import numpy as np
import config as conf
from ga import Ga
import matplotlib.pyplot as plt
#钱和路程，换算比例0.5元：1000米

config = conf.get_config()

ff=0
def build_dist_mat(input_list):#计算距离矩阵
    n = config.city_num
    dist_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            d = input_list[i, :] - input_list[j, :]
            ff=input_list[j, :]
            # 计算点积
            dist_mat[i, j] = np.dot(d, d)
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat
#####################################################################
# 统一坐标，把钱换算成距离或者把距离换算成钱。
# 城市坐标加，删除
city_pos_list = np.array([[0.0637446 , 0.31560375],
 [0.23725847 ,0.16989026],
 [0.64013563 ,0.39146139],
 [0.81340981 ,0.19076131],
 [0.039784   ,0.7227136 ],
 [0.78406733 ,0.15264327],
 [0.39135416 ,0.35978683],
 [0.5427002  ,0.1133714 ],
 [0.93530276 ,0.17315227],
 [0.49321454 ,0.39323142],
 [0.20645743 ,0.85424208],
 [0.31852136 ,0.60586346],
 [0.55241582 ,0.11759948],
 [0.13422259 ,0.97527681],
 [0.25519623 ,0.29631646]])
#####################################################################
# 城市距离矩阵
#city_pos_list = np.random.rand(config.city_num, config.pos_dimension)
city_dist_mat = build_dist_mat(city_pos_list)

print(city_pos_list)
print(city_dist_mat)

# 遗传算法运行
ga = Ga(city_dist_mat)
result_list, fitness_list = ga.train()
result = result_list[-1]
print(result)
result_pos_list = city_pos_list[result, :]
# print(result_pos_list)
# print(result_pos_list[:,1])
# 绘图
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
fig = plt.figure()

for i in range(len(result)-1):
  plt.arrow(result_pos_list[i,0], result_pos_list[i,1], -result_pos_list[i,0]+result_pos_list[i+1,0],  -result_pos_list[i,1]+result_pos_list[i+1,1], width=0.01, head_starts_at_zero=True, length_includes_head=True)
plt.scatter(city_pos_list[:, 0], city_pos_list[:, 1],marker='o')
for i in range(len(result)-1):
  plt.text(city_pos_list[i,0],city_pos_list[i,1],i)
#plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r')
plt.title(u"Cruise robot route")
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(fitness_list)
plt.title(u"Fitness curve")
plt.legend()
fig.show()
