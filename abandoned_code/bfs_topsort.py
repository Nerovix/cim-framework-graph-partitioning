from collections import deque

def bfs_topsort(g):
    n = len(g)
    vis = [0] * n
    q=deque()
    for i,v in enumerate(g):
        if len(v[1])==0: # 没出度
            vis[i]=1
            q.append(i)

    rk2id=[]
    while q:
        x=q.popleft()
        rk2id.append(x)
        for y in g[x][0]:
            if vis[y]==0:
                vis[y]=1
                q.append(y)

    rk2id.reverse()
    id2rk=[0]*n
    for i,v in enumerate(rk2id):
        id2rk[v]=i

    return rk2id,id2rk
        