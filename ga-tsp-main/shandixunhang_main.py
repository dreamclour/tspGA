# longlonglonglong
import numpy as np
import config as conf
from shandixunhang import Ga
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
city_pos_list = np.array([[0.6292745 , 0.99547315],
 [0.21321984 ,0.45482849],
 [0.50367519, 0.85195591],
 [0.86722766 ,0.67427923],
 [0.6918603  ,0.25528757],
 [0.4033128  ,0.7181223 ],
 [0.98178831 ,0.69506441],
 [0.18115401 ,0.31721647],
 [0.08463564 ,0.03737757],
 [0.99295723 ,0.96877693],
 [0.18263214 ,0.61471626],
 [0.66581514 ,0.6486331 ],
 [0.51635642 ,0.54161892],
 [0.48324469 ,0.04747629],
 [0.02157062 ,0.96565334]])
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

biaoji1=0
biaoji2=0
for i in range(15):
  if result[i] == 4:
    biaoji1 = i
  if result[i] == 12:
    biaoji2=i
#print(result_pos_list)
plt.arrow(result_pos_list[biaoji1,0], result_pos_list[biaoji1,1], -result_pos_list[biaoji1,0]+result_pos_list[biaoji2,0],  -result_pos_list[biaoji1,1]+result_pos_list[biaoji2,1], width=0.02,fc='red', ec='black', head_starts_at_zero=True, length_includes_head=True)


plt.scatter(city_pos_list[:, 0], city_pos_list[:, 1],marker='o')

for i in range(len(result)-1):
  plt.text(city_pos_list[i,0],city_pos_list[i,1],i)
#plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r')
plt.title(u"Cruise robot route")
plt.xlabel('KM')
plt.ylabel('KM')
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(fitness_list)
plt.title(u"Fitness curve")
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.legend()
fig.show()
