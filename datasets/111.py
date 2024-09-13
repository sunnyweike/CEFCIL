import numpy as np
L =[[[]for i in range(50)]for i in range(5)]
for tt in range(10):
    for cc in range(5):
        a = np.random.random(5)
        a /= a.sum()
        if tt > 0:
            index = cc + (tt) * 5
        else:
            index = cc
        for subdata in range(5):
            L[subdata][index]=a[subdata]

print(L[0][0])