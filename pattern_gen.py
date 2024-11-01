n = 12
m = 12
mp = [[-1] * m for i in range(n)]

print(mp)

curx = 0
cury = -1
movex = 0
movey = 1

'''
def valid(x, y):
    return x >= 0 and x < n and y >= 0 and y < m and mp[x][y] == -1


# for _ in range(0, n * m):
for _ in range(n * m - 1, -1, -1):
    while (not valid(curx + movex, cury + movey)) or (cury<m//2 and cur>=m//2):
        movex, movey = movey, -movex

    curx, cury = curx + movex, cury + movey
    mp[curx][cury] = _

for i in range(n):
    print(mp[i])



tmp=[   # 1 side of the 2 sides

        [0, 7, 8, 9, 12, 13],
[1, 6, 5, 10, 11, 14],
[2, 3, 4, 17, 16, 15],
[33, 32, 31, 18, 19, 20],
[34, 29, 30, 25, 24, 21],
[35, 28, 27, 26, 23, 22],
[36, 37, 38, 69, 70, 71],
[43, 42, 39, 68, 65, 64],
[44, 41, 40, 67, 66, 63],
[45, 46, 53, 54, 61, 62],
[48, 47, 52, 55, 60, 59],
[49, 50, 51, 56, 57, 58],


    ]
for i in range(n):
    tmp[i]=tmp[i]+tmp[i][::-1]
    print(tmp[i])
    for j in range(m//2,m):
        tmp[i][j]=n*m-1-tmp[i][j]

for i in range(n):
    print(tmp[i])
'''
cnt = 0
for i in range(0, n, 4):
    for j in range(0, m, 2):
        mp[i][j] = cnt
        cnt = cnt + 1
        mp[i + 1][j] = cnt
        cnt = cnt + 1
        mp[i + 1][j + 1] = cnt
        cnt = cnt + 1
        mp[i][j + 1] = cnt
        cnt = cnt + 1
    for j in range(m - 2, -2, -2):
        mp[i + 2][j + 1] = cnt
        cnt = cnt + 1
        mp[i + 2 + 1][j + 1] = cnt
        cnt = cnt + 1
        mp[i + 2 + 1][j] = cnt
        cnt = cnt + 1
        mp[i + 2][j] = cnt
        cnt = cnt + 1

for _ in mp:
    print(str(_) + ',')
