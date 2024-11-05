from graph import build_graph, find_all_prefixes, get_belong_node
from calc_cost import calc_best_strategy_on_chip
from read_file import get_tensor_shape
import math


def process_alexnet_vgg(onnx_graph):
    graph = build_graph(onnx_graph)

    is_conv_node = []
    is_fc_node = []
    for n in onnx_graph.node:
        if n.op_type == 'Conv' and [
                attr.i for attr in n.attribute if attr.name == 'group'][0] == 1:
            # a pointwise conv
            is_conv_node.append(1)
        else:
            # not pointwise conv, put on simd
            is_conv_node.append(0)
        if n.op_type == 'Gemm' or n.op_type == 'MatMul':
            is_fc_node.append(1)
        else:
            is_fc_node.append(0)

    belong_node = get_belong_node(graph, is_conv_node, is_fc_node)

    print([(i, v)for i, v in enumerate(belong_node)])

    conv_node_re_id = [0] * len(is_conv_node)  # 0-based
    re_id_to_node_id = []
    conv_node_cnt = 0
    for i in range(len(is_conv_node)):
        if is_conv_node[i]:
            conv_node_re_id[i] = conv_node_cnt
            re_id_to_node_id.append(i)
            conv_node_cnt += 1

    re_id_graph = [[] for _ in range(conv_node_cnt)]

    re_id_graph_edgeset = dict()
    for i, g in enumerate(graph):
        for j in g[1]:
            ibel = belong_node[i]
            jbel = belong_node[j]
            if ibel != jbel and is_conv_node[ibel] == 1 and is_conv_node[jbel] == 1:
                re_id_graph[conv_node_re_id[ibel]].append(
                    conv_node_re_id[jbel])
                shape = get_tensor_shape(
                    onnx_graph, onnx_graph.node[re_id_to_node_id[conv_node_re_id[ibel]]].output[0])
                re_id_graph_edgeset[(conv_node_re_id[ibel],
                                     conv_node_re_id[jbel])] = shape
                assert len(shape) == 4, \
                    'shape should be [N * C * H * W], but dimension of the tensor is not 4, wtffff'

    re_id_rev_graph = [[] for _ in range(conv_node_cnt)]
    for i in range(conv_node_cnt):
        for j in re_id_graph[i]:
            re_id_rev_graph[j].append(i)

    # print(onnx_graph.input[0].type.tensor_type.shape)

    prefixes_bitmask_re_id = find_all_prefixes(re_id_graph)
    # print (re_id_to_node_id)
    # for prefix in prefixes_bitmask_re_id:
    #     print([re_id_to_node_id[i] for i in range(conv_node_cnt) if prefix >> i & 1 == 1])
    # print([i for i in range(conv_node_cnt) if prefix >> i & 1 == 1])

    # s=[40,41,42,43,44,45,46,47,48,49,50,51,52]
    # s=[46,47,48,49,50,51,52]
    # cost,alloc=calc_best_strategy_on_chip(
    # s, re_id_graph, re_id_rev_graph, re_id_graph_edgeset, re_id_to_node_id,
    # onnx_graph)

    dp = [math.inf] * len(prefixes_bitmask_re_id)
    dpf = [-1] * len(dp)
    dpalloc = [0] * len(dp)
    # prefixes 排过序了，一定是后面的依赖前面
    for i, iprefix in enumerate(prefixes_bitmask_re_id):
        if iprefix == 0:
            dp[i] = 0
            dpf[i] = -1
            continue
        for j, jprefix in enumerate(prefixes_bitmask_re_id):
            if i > 0 and j < dpf[i - 1]:
                continue
            print(i, iprefix, j, jprefix)
            if i != j and iprefix & jprefix == jprefix:
                if dp[j] == math.inf:
                    continue
                s = iprefix - jprefix
                s = [i for i in range(conv_node_cnt) if s >> i & 1 == 1]
                # s 里面是按照conv重编号的
                cost, alloc = calc_best_strategy_on_chip(
                    s, re_id_graph, re_id_rev_graph, re_id_graph_edgeset, re_id_to_node_id, onnx_graph)
                if dp[j] + cost < dp[i]:
                    dp[i] = dp[j] + cost
                    dpf[i] = j
                    dpalloc[i] = alloc

        print(dpf[i])
    
    u=len(prefixes_bitmask_re_id)-1
    stages=[]
    while prefixes_bitmask_re_id[u]!=0:
        stage=[]
        v=dpf[u]
        stages.append([i for i in range(conv_node_cnt) if (prefixes_bitmask_re_id[u]-prefixes_bitmask_re_id[v]) >> i & 1 == 1])
        u=v
        
    stages.reverse()
    print(stages)

    # 打印一下方案看看
    # askdfja;sldkfjals;kfj
