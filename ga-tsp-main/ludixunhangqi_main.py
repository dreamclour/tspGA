import numpy as np
import config as conf
from ga import Ga
import matplotlib.pyplot as plt
import time
from ludixunhangqi import Ga

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
city_pos_list = np.array([[0.32049452 ,0.67712423],
 [0.30159215 ,0.05555383],
 [0.44028489 ,0.99575702],
 [0.28691876 ,0.150718  ],
 [0.99826564, 0.03947411],
 [0.30188408, 0.13806913],
 [0.45630897, 0.75596295],
 [0.96518372, 0.46981834],
 [0.39878515, 0.54586346],
 [0.73193743 ,0.19545278],
 [0.74376063 ,0.60936822],
 [0.30999021 ,0.28504798],
 [0.60217688 ,0.88076488],
 [0.27416236 ,0.08453904],
 [0.98670279 ,0.69776317],
 [0.15962319 ,0.43875982],
 [0.50321003 ,0.24204551],
 [0.42936463 ,0.12238503],
 [0.75653798, 0.38888992],
 [0.31119416, 0.83672505],
 [0.11872901, 0.38732859],
 [0.33093899, 0.89183565],
 [0.27488876 ,0.38960117],
 [0.84771265 ,0.18618894],
 [0.78821761 ,0.46308555],
 [0.57581166 ,0.71276657],
 [0.48426562 ,0.02254245],
 [0.03372926 ,0.69844222],
 [0.59756087 ,0.90327394],
 [0.71972813 ,0.1463084 ],
 [0.09844389 ,0.66770439],
 [0.38207951 ,0.87706836],
 [0.39168465 ,0.92979978],
 [0.39695246 ,0.53104058],
 [0.84365388 ,0.65713566],
 [0.63533432 ,0.91392048],
 [0.23992893 ,0.58391577],
 [0.60384095 ,0.55736808],
 [0.896318   ,0.41643641],
 [0.07046391, 0.76905616],
 [0.2071473  ,0.41653072],
 [0.48394258, 0.10685631],
 [0.82527105 ,0.79704542],
 [0.80950552 ,0.7630145 ],
 [0.28410973 ,0.02593513],
 [0.02256002 ,0.62830436],
 [0.77761043 ,0.29675669],
 [0.54664069 ,0.15736746],
 [0.12086224 ,0.95662839],
 [0.29025816 ,0.57108414],
 [0.12938235 ,0.48032634],
 [0.31974212 ,0.49850555],
 [0.40108129 ,0.69601215],
 [0.87330051 ,0.40019826],
 [0.46561237 ,0.38036861],
 [0.63263496 ,0.62423676],
 [0.12979482 ,0.33010163],
 [0.33001032 ,0.05525514],
 [0.54071141 ,0.42529072],
 [0.27687891 ,0.31047782],
 [0.69609917 ,0.33028093],
 [0.5021334  ,0.10184565],
 [0.74757644 ,0.36565322],
 [0.22187418 ,0.07784169],
 [0.84031668 ,0.15559562],
 [0.1469265  ,0.72499465],
 [0.33068374 ,0.03139533],
 [0.45205844 ,0.10176948],
 [0.33102637, 0.19711627],
 [0.0934555 , 0.23426304],
 [0.363366   ,0.79186583],
 [0.4353718  ,0.87699838],
 [0.96901193 ,0.93825654],
 [0.48178694 ,0.73254669],
 [0.17261003 ,0.099181  ],
 [0.65799669 ,0.13250784],
 [0.67597145 ,0.0777891 ],
 [0.0991898  ,0.03936412],
 [0.25988338 ,0.78592753],
 [0.68758885 ,0.96432346],
 [0.62769567 ,0.27829763],
 [0.55117317 ,0.71463832],
 [0.62356187, 0.46653436],
 [0.63806905 ,0.97737545],
 [0.97163997 ,0.67638645],
 [0.00476341 ,0.87254007],
 [0.82855716, 0.99156786],
 [0.54621993, 0.91959346],
 [0.94589381 ,0.28865255],
 [0.61784883, 0.7951146 ],
 [0.54270068 ,0.20350674],
 [0.88802198, 0.79730769],
 [0.92226601, 0.45860003],
 [0.68574166 ,0.83554939],
 [0.23258367 ,0.00346971],
 [0.96017944 ,0.94729931],
 [0.7057942  ,0.6663922 ],
 [0.40392723, 0.80155113],
 [0.46625315, 0.8990185 ],
 [0.37234394, 0.79939663]])
# 城市距离矩阵
#city_pos_list = np.random.rand(config.city_num, config.pos_dimension)

city_dist_mat = build_dist_mat(city_pos_list)

print(city_pos_list)
print(city_dist_mat)

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
#
# x1 = np.linspace(0.01, 0.06, 50)#第一横
# mt1=np.ones((50,1))
# y1 = 0.65*mt1
# plt.plot(x1, y1,linewidth=2,color='blue')
#
#
# x2 = np.linspace(0.01, 0.06, 50)#第二横
# mt1=np.ones((50,1))
# y2 = 0.6*mt1
# plt.plot(x2, y2,linewidth=2,color='blue')
#
# y3 = np.linspace(0.6, 0.65, 50)#第一竖
# mt1=np.ones((50,1))
# x3 = 0.01*mt1
# plt.plot(x3, y3,linewidth=2,color='blue')
#
# y4 = np.linspace(0.6, 0.65, 50)#第二竖
# mt1=np.ones((50,1))
# x4 = 0.06*mt1
# plt.plot(x4, y4,linewidth=2,color='blue')

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
