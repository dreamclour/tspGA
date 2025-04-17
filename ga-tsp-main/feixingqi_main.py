import numpy as np
import config as conf
from ga import Ga
import matplotlib.pyplot as plt
import time
from feixingqi import Ga

config = conf.get_config()


start=time.time()
print(start)


def build_dist_mat(input_list):
    n = config.city_num
    dist_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            d = input_list[i, :] - input_list[j, :]
            # 计算点积
            dist_mat[i, j] =  pow(np.dot(d, d),0.5)
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat

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
# 城市距离矩阵
#city_pos_list = np.random.rand(config.city_num, config.pos_dimension)

city_dist_mat = build_dist_mat(city_pos_list)

# print(city_pos_list)
# print(city_dist_mat)

# 遗传算法运行
ga = Ga(city_dist_mat)
result_list, fitness_list = ga.train()
result = result_list[-1]
result_pos_list = city_pos_list[result, :]

end=time.time()
print(end)

# 绘图
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
fig = plt.figure()
# x = [0, 1]
# y = [0, 1]
# plt.plot(x, y, linewidth=2)
# plt.axvline(2)

x1 = np.linspace(0.01, 0.06, 50)#第一横
mt1=np.ones((50,1))
y1 = 0.65*mt1
plt.plot(x1, y1,linewidth=2,color='blue')


x2 = np.linspace(0.01, 0.06, 50)#第二横
mt1=np.ones((50,1))
y2 = 0.6*mt1
plt.plot(x2, y2,linewidth=2,color='blue')

y3 = np.linspace(0.6, 0.65, 50)#第一竖
mt1=np.ones((50,1))
x3 = 0.01*mt1
plt.plot(x3, y3,linewidth=2,color='blue')

y4 = np.linspace(0.6, 0.65, 50)#第二竖
mt1=np.ones((50,1))
x4 = 0.06*mt1
plt.plot(x4, y4,linewidth=2,color='blue')

plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-b')
plt.xlabel('KM')
plt.ylabel('KM')
plt.title(u"Cruise robot route")
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(fitness_list)
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title(u"Fitness curve")
plt.legend()
fig.show()
