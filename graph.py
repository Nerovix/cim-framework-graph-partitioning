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


# 按照最长链分层，每个非conv的节点属于conv节点的层


def getlayer_by_conv_op(g, is_conv):
    outdeg = [0] * len(g)
    for i, v in enumerate(g):
        outdeg[i] = len(v[1])
    q = deque([i for i, v in enumerate(outdeg) if v == 0])
    # 终点只有一个
    assert len(q) == 1

    layer = [0] * len(g)

    while q:
        x = q.popleft()
        if is_conv[x]:
            layer[x] += 1
        for y in g[x][0]:
            outdeg[y] -= 1
            layer[y] = max(layer[y], layer[x])
            if outdeg[y] == 0:
                q.append(y)

    return layer


def build_dependency_bitmask_re_id(graph, is_conv_node, conv_node_re_id):
    indeg = [len(g[0]) for g in graph]
    dependency_bitmask = [0] * len(graph)
    q = deque([i for i, v in enumerate(indeg) if v == 0])
    while q:
        x = q.popleft()
        if is_conv_node[x] == 1:
            dependency_bitmask[x] |= 1 << conv_node_re_id[x]
        for y in graph[x][1]:
            indeg[y] -= 1
            dependency_bitmask[y] |= dependency_bitmask[x]
            if indeg[y] == 0:
                q.append(y)
    dependency_bitmask_re_id = [0] * sum(is_conv_node)
    for i in range(len(graph)):
        if is_conv_node[i]:
            dependency_bitmask_re_id[conv_node_re_id[i]
                                     ] = dependency_bitmask[i]
    return dependency_bitmask_re_id


# find all dependency prefixes for dp
def find_all_prefixes(dependency_bitmask_re_id):

    # input bitmasks are supposed to be pairwise distinct
    for i in range(len(dependency_bitmask_re_id)):
        for j in range(i + 1, len(dependency_bitmask_re_id)):
            assert dependency_bitmask_re_id[i] != dependency_bitmask_re_id[j]

    def remove_extra(bitmask_list):
        if bitmask_list == []:
            return []
        max_bit = max(bitmask_list).bit_length()
        bit_cnt = [0] * max_bit

        for v in bitmask_list:
            for i in range(max_bit):
                if v >> i & 1:
                    bit_cnt[i] += 1

        while True:
            flag = -1
            for i in range(len(bitmask_list)):
                bad = True
                for j in range(max_bit):
                    if bitmask_list[i] >> j & 1 == 1 and bit_cnt[j] == 1:
                        bad = False
                        break
                if bad:
                    flag = i
                    break
            if flag == -1:
                break

            for j in range(max_bit):
                if dependency_bitmask_re_id[flag] >> j & 1 == 1:
                    bit_cnt[j] -= 1

            bitmask_list.pop(flag)

        return bitmask_list

    graph = [[]] * len(dependency_bitmask_re_id)
    print(graph)
    indeg = []
    for i, bmi in enumerate(dependency_bitmask_re_id):
        protential_direct_denpendecies = []
        for j, bmj in enumerate(dependency_bitmask_re_id):
            if bmi & bmj == bmj and i != j:
                # bmj is a subset of bmi
                protential_direct_denpendecies.append(bmj)
        ind = 0
        print()
        for v in remove_extra(protential_direct_denpendecies):
            j = [j for j, bmj in enumerate(
                dependency_bitmask_re_id) if v == bmj][0]
            print(j, i)
            graph[j].append(i)
            ind += 1
        indeg.append(ind)
        print(ind)

    print(graph)

    # now dfs on graph
    prefixes = dict()

    def dfs(bitmask):
        if prefixes.get(bitmask) is not None:
            return
        print("shit:" + bin(bitmask))
        print(indeg)
        prefixes[bitmask] = True
        for i in range(len(dependency_bitmask_re_id)):
            if bitmask >> i & 1 == 0 and indeg[i] == 0:
                # 选i, 删除所有出边
                for j in graph[i]:
                    indeg[j] -= 1
                dfs(bitmask ^ (1 << i))
                for j in graph[i]:
                    indeg[j] += 1

    dfs(0)

    return [prefixes.keys()].sort()
