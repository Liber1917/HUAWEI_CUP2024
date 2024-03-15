import matplotlib.pyplot as plt
import numpy as np
import time

def visualize_map(map_data):
    # 定义地图字符与颜色的映射关系
    color_map = {'#': 'black', '.': 'white', '*': 'blue', 'A': 'green', 'B': 'orange'}

    # 创建一个新的图形
    plt.figure(figsize=(8, 8))

    # 循环遍历地图数据并绘制每个单元格
    for y in range(len(map_data)):
        for x in range(len(map_data[y])):
            # 获取当前单元格的字符
            cell = map_data[y][x]

            # 获取当前单元格的颜色
            color = color_map.get(cell, 'white')

            # 绘制一个矩形作为当前单元格，并设置颜色
            plt.fill([x, x+1, x+1, x], [len(map_data) - y, len(map_data) - y, len(map_data) - (y+1), len(map_data) - (y+1)], color=color)

    # 设置图形的标题和轴标签
    plt.title('Map Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 设置坐标轴刻度范围
    plt.xlim(0, len(map_data[0]))
    plt.ylim(0, len(map_data))

    # 显示图形
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



# 读取地图文件
with open('./maps/map2.txt', 'r') as f:
    map_data = [line.strip() for line in f]  # 逐行读取文件内容，并去除每行末尾的换行符，并将每行字符串组成的列表赋值给变量map_data

# 可视化地图数据
# visualize_map(map_data)

def visualize_value_table(value_table):
    plt.imshow(value_table, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.show()


def flood_fill(matrix, x, y, target_sign='.'):
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
# print(map_cpy[4][1]=='.')

# 使用np.argwhere()一次性找到字符'A'和'B'的坐标
# A_coordinates = np.argwhere(map_matrix == 'A')

# B_indices = np.where(map_matrix == 'B')
# B_coordinates = list(zip(B_indices[0], B_indices[1]))


# 从港口向外灌水
A_indices = np.where(map_matrix == 'A')
A_coordinates = list(zip(A_indices[0], A_indices[1]))

B_coordinates = np.argwhere(map_matrix == 'B')
selected_B_coordinates = []
for i, coord in enumerate(B_coordinates):
    if i%16 == 5:
        selected_B_coordinates.append(coord)
B_coordinates = selected_B_coordinates

print(len(B_coordinates))
# print(B_coordinates)

filled_coords_list = []  # 用于存储所有的填充坐标

test = []


for coord in B_coordinates:
    x, y = coord
    # start=time.time()
    filled_coords = flood_fill(map_cpy, x, y)
    # visualize_value_table(filled_coords)
    # test.append(filled_coords)
    A_in_filled_coords = [coord for coord in filled_coords if coord in A_coordinates]
    filled_coords_list.append(A_in_filled_coords)
    # end=time.time()
    # print(end-start)


# for index, filled_coords in enumerate(filled_coords_list):
#     print(f"Filled Area {index + 1} Size: {len(filled_coords)}")

def visualize_map_with_filled_areas(map_data, filled_coords_list):
    # 定义地图字符与颜色的映射关系
    color_map = {'#': 'black', '.': 'white', '*': 'blue', 'A': 'green', 'B': 'orange', 'x': 'red'}  # 添加 'x': 'red'

    # 创建一个新的图形
    plt.figure(figsize=(8, 8))

    # 循环遍历地图数据并绘制每个单元格
    for y in range(len(map_data)):
        for x in range(len(map_data[y])):
            # 获取当前单元格的字符
            cell = map_data[y][x]

            # 获取当前单元格的颜色
            color = color_map.get(cell, 'white')

            # 绘制一个矩形作为当前单元格，并设置颜色
            plt.fill([x, x+1, x+1, x], [len(map_data) - y, len(map_data) - y, len(map_data) - (y+1), len(map_data) - (y+1)], color=color)

    # 标记填充的区域为红色
    for filled_coords in filled_coords_list:
        for coord in filled_coords:
            x, y = coord
            plt.fill([x, x+1, x+1, x], [len(map_data) - y, len(map_data) - y, len(map_data) - (y+1), len(map_data) - (y+1)], color='red')

    # 设置图形的标题和轴标签
    plt.title('Map Visualization with Filled Areas')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 设置坐标轴刻度范围
    plt.xlim(0, len(map_data[0]))
    plt.ylim(0, len(map_data))

    # 显示图形
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()







# 可视化地图数据及填充区域
# visualize_map_with_filled_areas(map_data, test)
