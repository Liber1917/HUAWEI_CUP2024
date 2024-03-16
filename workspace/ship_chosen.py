def ship_chosen(flood_fill_map_datas, robot_locations):
    total_berth_distance = []
    for map_data in flood_fill_map_datas:
        distances = [map_data[robot[0]][robot[1]] for robot in robot_locations]
        total_berth_distance.append(sum(distances))

    return total_berth_distance[:5]


def country_road(flood_fill_map_data, start_location, targeted_location):
    #返回路径表
    path_data = [targeted_location]
    while path_data[-1] != start_location:
        current_location = path_data[-1]
        if current_location[0]-1>=0 and flood_fill_map_data[current_location[0]-1][current_location[1]] == flood_fill_map_data[current_location[0]][current_location[1]]-1:
            path_data.append([current_location[0]-1, current_location[1]])
            continue
        if current_location[0]+1<len(flood_fill_map_data) and flood_fill_map_data[current_location[0]+1][current_location[1]] == flood_fill_map_data[current_location[0]][current_location[1]]-1:
            path_data.append([current_location[0]+1, current_location[1]])
            continue
        if current_location[1]-1>=0 and flood_fill_map_data[current_location[0]][current_location[1]-1] == flood_fill_map_data[current_location[0]][current_location[1]]-1:
            path_data.append([current_location[0], current_location[1]-1])
            continue
        if current_location[1]+1<len(flood_fill_map_data[0]) and flood_fill_map_data[current_location[0]][current_location[1]+1] == flood_fill_map_data[current_location[0]][current_location[1]]-1:
            path_data.append([current_location[0], current_location[1]+1])
            continue
    return path_data