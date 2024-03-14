import matplotlib.pyplot as plt

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
            plt.fill([x, x+1, x+1, x], [200-y, 200-y, 199-y, 199-y], color=color)

    # 设置图形的标题和轴标签
    plt.title('Map Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 设置坐标轴刻度范围
    plt.xlim(0, 200)
    plt.ylim(0, 200)

    # 显示图形
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# 读取地图文件
with open('./maps/map2.txt', 'r') as f:
    map_data = [line.strip() for line in f]

# 可视化地图数据
visualize_map(map_data)
