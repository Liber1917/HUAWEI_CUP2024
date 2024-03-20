#  -*-  codeing  =  utf-8  -*-
import matplotlib.pyplot as plt
import numpy as np
import time




# 读取地图文件
with open('maps/map2.txt', 'r') as f:
    map_data = [line.strip() for line in f]  # 逐行读取文件内容，并去除每行末尾的换行符，并将每行字符串组成的列表赋值给变量map_data




def flood_fill(matrix, x, y): # 对通路实施洪水填充算法
    if x < 0 or y < 0 or x >= len(matrix[0]) or y >= len(matrix):
        return

    table1 = []  # 下一步要遍历的
    row = len(matrix[0])
    col = len(matrix)
    value_table = [[0 if j != '#' and j != '*' else -1 for j in i] for i in matrix]
    # for i in value_table:
    #     print(i)
    table1.append((x, y))  # 从本位置开始
    value_table[x][y] = 1
    while table1:
        coordx = table1[0]
        table1.remove(coordx)

        if value_table[coordx[0]-1][coordx[1]] == 0 and coordx[0]-1 > 0:
            table1.append((coordx[0]-1, coordx[1]))
            value_table[coordx[0]-1][coordx[1]] = 1 + value_table[coordx[0]][coordx[1]]

        if value_table[coordx[0]+1][coordx[1]] == 0 and coordx[0]+1 < col:
            table1.append((coordx[0]+1, coordx[1]))
            value_table[coordx[0]+1][coordx[1]] = 1 + value_table[coordx[0]][coordx[1]]

        if value_table[coordx[0]][coordx[1]-1] == 0 and coordx[0]-1 > 0:
            table1.append((coordx[0], coordx[1]-1))
            value_table[coordx[0]][coordx[1]-1] = 1 + value_table[coordx[0]][coordx[1]]

        if value_table[coordx[0]][coordx[1]+1] == 0 and coordx[0]-1 < row:
            table1.append((coordx[0], coordx[1]+1))
            value_table[coordx[0]][coordx[1]+1] = 1 + value_table[coordx[0]][coordx[1]]
    return value_table


# 将map_data转换为矩阵
map_matrix = np.array([list(row) for row in map_data])
# 复制地图矩阵
map_cpy = np.copy(map_matrix)

# 从港口向外灌水
A_indices = np.where(map_matrix == 'A')
A_coordinates = list(zip(A_indices[0], A_indices[1]))

B_coordinates = np.argwhere(map_matrix == 'B')
selected_B_coordinates = []
for i, coord in enumerate(B_coordinates):
    if i%16 == 5:
        selected_B_coordinates.append(coord)
B_coordinates = selected_B_coordinates

# print(B_coordinates)


filled_coords_list = []  # 用于存储所有的填充坐标

test = [] # 用于存储所有的填充坐标的值表


# 洪水
for coord in B_coordinates:
    x, y = coord
    # start=time.time()
    filled_coords = flood_fill(map_cpy, x, y)

    # print(len(filled_coords))
    # test.append(filled_coords)
    A_in_filled_coords = [coord for coord in A_coordinates if filled_coords[coord[0]][coord[1]] > 0]
    filled_coords_list.append(A_in_filled_coords)
    # end=time.time()
    # print(end-start)



############################NSGA2接口测试#######################################
import random
import math

n = 200
robot_num = 10
berth_num = 10  # 泊位
boat_num = 5
N = 210


class Robot:
    def __init__(self, startX=0, startY=0, goods=0, status=0, mbx=0, mby=0):
        self.x = startX
        self.y = startY
        self.goods = goods # 携带货物
        self.status = status # 状态
        self.mbx = mbx # 目标位置
        self.mby = mby

class Berth:
    def __init__(self, x=0, y=0, transport_time=0, loading_speed=0, flood_table=None):
        self.x = x
        self.y = y
        self.transport_time = transport_time
        self.loading_speed = loading_speed
        self.flood_table = flood_table


class Cargo:
    def __init__(self, x=-1, y=-1, worth=-1):
        self.targeted = 0
        self.x = x
        self.y = y
        self.worth = worth

##########################统一初始化##################################
n = 200
robot_num = 10
berth_num = 10
cargoNum = 50

# 实例init
robot = [Robot() for _ in range(robot_num)]
berth = [Berth() for _ in range(berth_num)]
cargo = [Cargo() for _ in range(cargoNum)]

sample_robot = [[] for _ in range(5)]
sample_berth = []
sample_cargo = []
# gene_list = [robot, berth, cargo]
# 指定区域内的坐标范围
min_x = 0
max_x = 200
min_y = 0
max_y = 200

for i in range(5):
    num_selected_robots = random.randint(1, robot_num)
    selected_coordinates = [(random.randint(min_x, max_x), random.randint(min_y, max_y)) for _ in range(num_selected_robots)]
    selected_robots = [r for r in robot if (r.x, r.y) in selected_coordinates]
    sample_robot[i] = (selected_robots)

print(sample_robot)

selected_B = random.sample(B_coordinates, 5)

for i, coord in enumerate(selected_B):
    berth[i].x = coord[0]
    berth[i].y = coord[1]
    berth[i].flood_table = filled_coords_list[i]
    sample_berth.append(berth[i])




'''
判题器回传样本获得cargo信息
'''
class back_info:
    def __init__(self) -> None:
        self.tickInfo = []
        self.cargoNum = 0
        self.cargoInfo = []
        self.robotInfo = []
        self.shipInfo = []

back_info = back_info()
# # 打开文件
# with open('output.txt', 'r') as file:
#     # 读取第一行
#     first_line = next(file)
#     back_info.tickInfo.append(first_line.strip())  # 将第一行添加到 tickInfo 中

#     # 逐行读取文件内容
#     for line in file:
#         line = line.strip()  # 移除行首和行尾的空白字符

#         if line == 'OK':
#             break

#         space_count = line.count(' ')  # 计算空格的数量

#         if space_count == 0:
#             back_info.cargoNum = int(line)  # 直接将整行转换为整数并赋给 cargoNum
#         elif space_count == 2:
#             # 将每个 item 转换为整数或浮点数并添加到 cargoInfo 中
#             back_info.cargoInfo.append([int(x) if x.isdigit() else float(x) for x in line.split()])
#         elif space_count == 3:
#             back_info.robotInfo.append([int(x) if x.isdigit() else float(x) for x in line.split()])
#         elif space_count == 1:
#             back_info.shipInfo.append([int(x) if x.isdigit() else float(x) for x in line.split()])
