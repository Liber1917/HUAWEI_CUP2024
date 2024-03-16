import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_map_with_filled_areas(map_data, filled_coords_list, **kwargs):
    fig, ax = plt.subplots()

    # 自定义颜色映射
    cmap = ListedColormap(['blue', 'red', 'green'])

    # 绘制地图
    ax.imshow(map_data, cmap=cmap, interpolation='nearest')

    # 填充矩形格子
    for coord, value in filled_coords_list:
        x, y = coord
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=cmap(value), alpha=0.5)
        ax.add_patch(rect)
        ax.text(x, y, str(value), color='white', ha='center', va='center')

    plt.axis('off')
    plt.show()


# 示例调用
map_data = [[0, 0, 0],
            [0, -1, 0],
            [0, 0, 0]]

filled_coords = [((0, 0), 0), ((1, 1), -1), ((2, 2), 10)]  # 填充的坐标和对应的数字

visualize_map_with_filled_areas(map_data, filled_coords)