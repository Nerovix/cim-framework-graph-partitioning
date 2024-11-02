from collections import deque
import math
import onnx
from onnx import shape_inference
from onnx import GraphProto, ModelProto, NodeProto


# 返回[[],]表示一个dag，编号跟graph.node一一对应，只单纯反映节点依赖关系
# [[in],[out]]
def build_graph(onnx_graph):

    graph = []
    output_from = {}
    for i, node in enumerate(onnx_graph.node):
        # 输出对到节点上去
        graph.append([[], []])
        for output_name in node.output:
            assert output_from.get(output_name) is None
            output_from[output_name] = i

    for i, node in enumerate(onnx_graph.node):
        for input_name in node.input:
            if output_from.get(input_name) is not None:
                graph[i][0].append(output_from[input_name])
                graph[output_from[input_name]][1].append(i)
    return graph


# 残差的加法节点必须在主链（最长链）上，不然会出问题，所以先小dp一下找主链
# fc放最后，fc之前的都往conv上挂，fc之后的都先保留。fc节点是割点，不会有问题
# nm改来改去alskdjfalaskdjfaksjdlfalsdkjfalksjdflasdf
def get_belong_node(graph, is_conv_node, is_fc_node):
    indeg = [len(g[0]) for g in graph]
    q = deque([i for i, v in enumerate(indeg) if v == 0])
    last = [-1] * len(graph)
    dp = [1] * len(graph)
    while q:
        x = q.popleft()
        for y in graph[x][1]:
            if dp[y] < dp[x] + 1:
                dp[y] = dp[x] + 1
                last[y] = x
            indeg[y] -= 1
            if indeg[y] == 0:
                q.append(y)
    main_chain = []
    x = 0
    for i in range(len(graph)):
        if dp[i] > dp[x]:
            x = i
    while x != -1:
        main_chain.append(x)
        x = last[x]

    main_chain.reverse()
    in_main_chain = [0] * len(graph)
    for i in main_chain:
        in_main_chain[i] = 1

    # 残差的加法一定在主链上。先单独对主链算最近点，挂上去
    # 然后剩下的点就都是残差卷积和残差卷积旁边的小算子，直接也就近挂上去
    belong_node = [-1] * len(graph)
    q = deque([i for i in range(len(graph)) if is_conv_node[i] == 1])
    while q:
        x = q.popleft()
        if belong_node[x] == -1:
            belong_node[x] = x
        for y in graph[x][0] + graph[x][1]:
            if in_main_chain[y] == in_main_chain[x] and \
                    belong_node[y] == -1 and is_fc_node[y] == 0:
                belong_node[y] = belong_node[x]
                q.append(y)

    # 如果这样做一次结束了还有还有剩下的点，那就是存在残差分支没有卷积
    # 正常说残差分支要么没有算子要么一定有至少一个卷积，但是先考虑上
    q = deque([i for i in range(len(graph)) if belong_node[i] != -1])
    while q:
        x = q.popleft()
        if belong_node[x] == -1:
            belong_node[x] = x
        for y in graph[x][0] + graph[x][1]:
            if belong_node[y] == -1 and is_fc_node[y] == 0:
                belong_node[y] = belong_node[x]
                q.append(y)

    # fc自娱自乐
    q = deque([i for i in range(len(graph)) if is_fc_node[i] == 1])
    while q:
        x = q.popleft()
        if belong_node[x] == -1:
            belong_node[x] = x
        for y in graph[x][0] + graph[x][1]:
            if belong_node[y] == -1:
                belong_node[y] = belong_node[x]
                q.append(y)

    assert min(belong_node)>=0
    return belong_node

# find all dependency prefixes for dp
# 依托史
def find_all_prefixes(graph_re_id):
    node_cnt=len(graph_re_id)
    indeg=[0]*node_cnt
    print("ok there are "+str(node_cnt)+" nodes that i need to calc")
    for i in range(node_cnt):
        for j in graph_re_id[i]:
            indeg[j] += 1

    prefixes = dict()
    def dfs(bitmask):
        if prefixes.get(bitmask) is not None:
            return
        prefixes[bitmask] = True
        for i in range(node_cnt):
            if bitmask >> i & 1 == 0 and indeg[i] == 0:
                for j in graph_re_id[i]:
                    indeg[j] -= 1
                dfs(bitmask ^ (1 << i))
                for j in graph_re_id[i]:
                    indeg[j] += 1
    
    dfs(0)
    prefixes_list = list(prefixes.keys())
    prefixes_list.sort()
    return prefixes_list
