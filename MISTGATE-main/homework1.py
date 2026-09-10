# 今有物不知其数
# 三三数之剩二，五五数之剩三，七七数之剩二，问几何？

for x in range(1, 1000):
    if x % 3 == 2 and x % 5 == 3 and x % 7 == 2:
        print("这个数是：", x)
        break