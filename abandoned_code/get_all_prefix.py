from prefix_nodeset import prefix_nodeset
import copy

def get_all_prefix(g,rk2id,id2rk):
    s=prefix_nodeset()
    prefix_set=set()
    ind=[0]*len(g)
    for i,v in enumerate(g):
        ind[i]=len(v[0])
    in0set=set([i for i,v in enumerate(ind) if v==0])
    def dfs():
        print(s)
        # print(ind)
        # print()
        prefix_set.add(copy.deepcopy(s))
        head=in0set.copy()
        for i in head:
            s.add(id2rk[i])
            in0set.remove(i)
            if s not in prefix_set:
                for j in g[i][1]:
                    ind[j]-=1
                    if ind[j]==0:
                        in0set.add(j)
                dfs()
                for j in g[i][1]:
                    ind[j]+=1
                    if ind[j]==1:
                        in0set.remove(j)
                s.remove(id2rk[i])
            else:
                s.remove(id2rk[i])
            in0set.add(i)
    
    dfs()
    return prefix_set # 基于拓扑序的