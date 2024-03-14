import sys
import random

n = 200
robot_num = 10
berth_num = 10  # 泊位
boat_num = 5
N = 210


class Robot:
    def __init__(self, startX=0, startY=0, goods=0, status=0, mbx=0, mby=0):
        self.x = startX
        self.y = startY
        self.goods = goods
        self.status = status
        self.mbx = mbx
        self.mby = mby


robot = [Robot() for _ in range(robot_num)]


class Berth:
    def __init__(self, x=0, y=0, transport_time=0, loading_speed=0):
        self.x = x
        self.y = y
        self.transport_time = transport_time
        self.loading_speed = loading_speed


berth = [Berth() for _ in range(berth_num)]


class Boat:
    def __init__(self, num=0, pos=0, status=0):
        self.num = num
        self.pos = pos
        self.status = status

    capacity = 0


boat = [Boat() for _ in range(5)]


class Cargo:
    def __init__(self, x, y, worth):
        self.targeted = 0
        self.x = x
        self.y = y
        self.worth = worth


money = 0
id = 0
ch = []
gds = [[0 for _ in range(N)] for _ in range(N)]
all_Cargo = []


def Init():
    for i in range(0, n):
        line = input()
        ch.append([c for c in line.split(sep=" ")])
    for i in range(berth_num):
        line = input()
        berth_list = [int(c) for c in line.split(sep=" ")]
        id = berth_list[0]
        berth[id].x = berth_list[1]
        berth[id].y = berth_list[2]
        berth[id].transport_time = berth_list[3]
        berth[id].loading_speed = berth_list[4]
    boat_capacity = int(input())
    Boat.capacity = boat_capacity
    okk = input()
    if okk == "OK":
        print("OK")
    sys.stdout.flush()


def Input():
    id, money = map(int, input().split(" "))
    num = int(input())
    for i in range(num):
        x, y, val = map(int, input().split())
        cargo_this = Cargo(x, y, val)
        all_Cargo.append(cargo_this)
        gds[x][y] = val
    for i in range(robot_num):
        robot[i].goods, robot[i].x, robot[i].y, robot[i].status = map(int, input().split())
    for i in range(boat_num):
        boat[i].status, boat[i].pos = map(int, input().split())
    okk = input()
    if okk == "OK":
        print("OK")
    return id


if __name__ == "__main__":
    Init()
    for zhen_id in range(1, 15001):
        id = Input()
        for i in range(robot_num):
            print("move", i, random.randint(0, 3))
            sys.stdout.flush()
        print("OK")
        sys.stdout.flush()
