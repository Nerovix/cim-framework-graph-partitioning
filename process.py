from graph import build_graph, find_all_prefixes, get_belong_node, topsort
from calc_cost import calc_best_strategy_on_chip
from read_file import get_tensor_shape
from partition_result_gen import get_instrctions_for_a_stage
import cimpara as cp
import math


def process(onnx_graph):
    graph = build_graph(onnx_graph)

    is_conv_node = []
    is_fc_node = []
    for n in onnx_graph.node:

        # if n.op_type == 'Conv':
        #     print(n.input[1])
        #     print(n, get_tensor_shape(onnx_graph, n.input[1])[1])

        if n.op_type == 'Conv' and get_tensor_shape(
                onnx_graph, n.input[1])[1] != 1:  # pointwise或者分组卷积
            # a pointwise conv
            is_conv_node.append(1)
        else:
            # not pointwise conv, put on simd
            is_conv_node.append(0)

        if n.op_type == 'Gemm' or n.op_type == 'MatMul':
            is_fc_node.append(1)
        else:
            is_fc_node.append(0)

    # print(is_conv_node)

    belong_node = get_belong_node(graph, is_conv_node, is_fc_node)

    # print(is_conv_node)

    # print([(i, v)for i, v in enumerate(belong_node)])

    conv_node_re_id = [0] * len(is_conv_node)  # 0-based
    re_id_to_node_id = []
    conv_node_cnt = 0
    for i in range(len(is_conv_node)):
        if is_conv_node[i]:
            conv_node_re_id[i] = conv_node_cnt
            re_id_to_node_id.append(i)
            conv_node_cnt += 1

    # print(conv_node_cnt)
    re_id_graph = [[] for _ in range(conv_node_cnt)]

    input_data_conv_node_re_id = dict()
    output_data_conv_node_re_id = dict()

    re_id_graph_edgeset = dict()
    for i, g in enumerate(graph):
        ibel = belong_node[i]
        for j in g[1]:
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
            elif ibel != jbel and is_conv_node[ibel] == 0:
                output_data_conv_node_re_id[conv_node_re_id[ibel]] = get_tensor_shape(
                    onnx_graph, onnx_graph.node[re_id_to_node_id[conv_node_re_id[ibel]]].output[0])

        if onnx_graph.input[0].name in onnx_graph.node[i].input:
            input_data_conv_node_re_id[conv_node_re_id[ibel]] = get_tensor_shape(
                onnx_graph, onnx_graph.input[0].name)

    # print(re_id_graph)

    re_id_rev_graph = [[] for _ in range(conv_node_cnt)]
    for i in range(conv_node_cnt):
        for j in re_id_graph[i]:
            re_id_rev_graph[j].append(i)

    # print(onnx_graph.input[0].type.tensor_type.shape)

    prefixes_bitmask_re_id = find_all_prefixes(re_id_graph)
    # print(prefixes_bitmask_re_id)
    
    # s=[0,1]
    # cost1, alloc1, nodes_re_id1, cores_needed_list1, communicate_on_chip_edgeset,pack1 = calc_best_strategy_on_chip(\
    #         s, re_id_graph, re_id_rev_graph, re_id_graph_edgeset, re_id_to_node_id, input_data_conv_node_re_id, output_data_conv_node_re_id, onnx_graph)
    
    # s=[2,3,4]
    # cost2, alloc2, nodes_re_id2, cores_needed_list2, communicate_on_chip_edgeset,pack2 = calc_best_strategy_on_chip(\
    #         s, re_id_graph, re_id_rev_graph, re_id_graph_edgeset, re_id_to_node_id, input_data_conv_node_re_id, output_data_conv_node_re_id, onnx_graph)
    
    # s=[0,1,2,3,4]
    # cost3, alloc3, nodes_re_id3, cores_needed_list3, communicate_on_chip_edgeset,pack3 = calc_best_strategy_on_chip(\
    #         s, re_id_graph, re_id_rev_graph, re_id_graph_edgeset, re_id_to_node_id, input_data_conv_node_re_id, output_data_conv_node_re_id, onnx_graph)
    # # print(cost1,nodes_re_id1,cores_needed_list1,[len(_) for _ in alloc1],pack1)
    # # print(cost2,nodes_re_id2,cores_needed_list2,[len(_) for _ in alloc2],pack2)
    # print(cost3,nodes_re_id3,cores_needed_list3,[len(_) for _ in alloc3],pack3)
    
    stages=[]

    if cp.partition_mode in [0,3,4,5]:
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
                # if i > 0 and j < dpf[i - 1]:
                # continue
                if i != j and iprefix & jprefix == jprefix:
                    # print("dp", i, iprefix, j, jprefix)
                    if dp[j] == math.inf:
                        continue
                    s = iprefix - jprefix
                    s = [i for i in range(conv_node_cnt) if s >> i & 1 == 1]
                    # s 里面是按照conv重编号的
                    cost, alloc, nodes_re_id, cores_needed_list, communicate_on_chip_edgeset = calc_best_strategy_on_chip(
                        s, re_id_graph, re_id_rev_graph, re_id_graph_edgeset, re_id_to_node_id, input_data_conv_node_re_id, output_data_conv_node_re_id, onnx_graph)
                    if dp[j] + cost < dp[i]:
                        dp[i] = dp[j] + cost
                        dpf[i] = j
                        dpalloc[i] = (
                            alloc,
                            nodes_re_id,
                            cores_needed_list,
                            communicate_on_chip_edgeset)

            # print(dpf[i])

        u = len(prefixes_bitmask_re_id) - 1
        while prefixes_bitmask_re_id[u] != 0:
            stage = []
            v = dpf[u]
            stages.append(dpalloc[u])
            u = v
        stages.reverse()
    else:
        # topsort
        sorted_nodes_re_id=topsort([[re_id_rev_graph[i],re_id_graph[i]] for i in range(len(re_id_graph))])
        u=0
        while u<len(sorted_nodes_re_id):
            l=u
            while u<len(sorted_nodes_re_id):
                cost, alloc, nodes_re_id, cores_needed_list, communicate_on_chip_edgeset = calc_best_strategy_on_chip(
                    [sorted_nodes_re_id[i] for i in range(l,u+1)], re_id_graph, re_id_rev_graph, re_id_graph_edgeset, re_id_to_node_id, input_data_conv_node_re_id, output_data_conv_node_re_id, onnx_graph)
                if cost==math.inf:
                    break
                else:
                    u+=1
            u-=1
            cost, alloc, nodes_re_id, cores_needed_list, communicate_on_chip_edgeset = calc_best_strategy_on_chip(
                [sorted_nodes_re_id[i] for i in range(l,u+1)], re_id_graph, re_id_rev_graph, re_id_graph_edgeset, re_id_to_node_id, input_data_conv_node_re_id, output_data_conv_node_re_id, onnx_graph)
            stages.append((alloc,nodes_re_id,cores_needed_list,communicate_on_chip_edgeset))
            u+=1

                    

    for stage in stages:
        s='['
        for i in range(len(stage[0])):
            s+=str(stage[1][i])
            s+='-'
            s+=str(len(stage[0][i][0]))
            s+='-'
            s+=str(len(stage[0][i]))
            s+=', '
        s+=']'
        print(s)
# '''
    instructions = dict()
    for i in range(cp.P):
        for j in range(cp.Q):
            instructions[f'core_{i}_{j}'] = {
                'stages': {}
            }

    sorted_nodes = topsort(graph)

    for stageid, stage in enumerate(stages):
        alloc, nodes_re_id, cores_needed_list, communicate_on_chip_edgeset = stage
        instruction_cur = get_instrctions_for_a_stage(
            alloc,
            nodes_re_id,
            communicate_on_chip_edgeset,
            re_id_graph,
            re_id_rev_graph,
            re_id_graph_edgeset,
            re_id_to_node_id,
            input_data_conv_node_re_id,
            output_data_conv_node_re_id,
            graph,
            belong_node,
            sorted_nodes,
            onnx_graph
        )

        for i in range(cp.P):
            for j in range(cp.Q):
                core_name = f'core_{i}_{j}'
                instructions[core_name]['stages'][str(stageid)] = {
                    'cluster_id': instruction_cur[core_name]['cluster_id'],
                    'weight_replica_id': instruction_cur[core_name]['weight_replica_id'],
                    'instructions': instruction_cur[core_name]['instructions']
                }

    return instructions
# '''
