import matplotlib.pyplot as plt
import numpy as np

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

def flood_fill(matrix, x, y, target_sign='.'):
    if x < 0 or y < 0 or x >= len(matrix[0]) or y >= len(matrix):
        return

    table0 = set()  # 已经遍历过的
    table1 = set()  # 下一步要遍历的
    table1.add((x, y))  # 从本位置开始

    while table1:
        coordx = table1.pop()
        # print(coordx)
        if matrix[coordx[1]][coordx[0]] == target_sign and coordx not in table0:
            table0.add(coordx)

            table1.add((coordx[0]-1, coordx[1]))
            table1.add((coordx[0]+1, coordx[1]))
            table1.add((coordx[0], coordx[1]-1))
            table1.add((coordx[0], coordx[1]+1))

    return table0



# 将map_data转换为矩阵
map_matrix = np.array([list(row) for row in map_data])
# 复制地图矩阵
map_cpy = np.copy(map_matrix)
# print(map_cpy[4][1]=='.')

# 使用np.argwhere()一次性找到字符'A'和'B'的坐标
A_coordinates = np.argwhere(map_matrix == 'A')

B_indices = np.where(map_matrix == 'B')
B_coordinates = list(zip(B_indices[0], B_indices[1]))


filled_coords_list = []  # 用于存储所有的填充坐标

# 对所有字符'A'使用flood_fill函数将位置填充为'x'，并将填充的坐标存储到filled_coords_list中
for coord in A_coordinates:
    x, y = coord
    filled_coords = flood_fill(map_cpy, x, y)

    B_in_filled_coords = [coord for coord in filled_coords if coord in B_coordinates]
    filled_coords_list.append(B_in_filled_coords)


for index, filled_coords in enumerate(filled_coords_list):
    print(f"Filled Area {index + 1} Size: {len(filled_coords)}")
