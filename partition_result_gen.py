import cimpara as cp
from collections import deque
from read_file import get_tensor_shape

import random
import string


def get_unique_id(length=15):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def get_instrctions_for_a_stage(
        allocation,
        nodes_re_id,
        communicate_on_chip,  # reid为下标的dict，包含所有chip通信关系，不在这上面的要用global通信
        re_id_graph,
        re_id_rev_graph,
        re_id_graph_edgeset,
        re_id_to_node_id,
        input_data_conv_node_re_id,
        output_data_conv_node_re_id,
        graph,  # 若至变量名，命名不动脑子导致的
        # 表示原图（包含全部节点、包括非conv节点的那张依赖关系图）
        belong_node,
        sorted_nodes,
        # 排序后的全部节点
        onnx_graph):

    in_nodes_re_id = [0] * len(graph)
    for i in nodes_re_id:
        in_nodes_re_id[i] = 1

    instructions = dict()
    for i in range(cp.P):
        for j in range(cp.Q):
            instructions[f'core_{i}_{j}'] = {
                'cluster_id': -1,
                'weight_replica_id': -1,
                'instructions': []
            }

    # instruction的dict初始化 + load权重的指令
    for i in range(len(allocation)):
        for j in range(len(allocation[i])):
            filter_shape = get_tensor_shape(
                onnx_graph, onnx_graph.node[re_id_to_node_id[nodes_re_id[i]]].input[1])
            channelcnt = filter_shape[0]
            for core in allocation[i][j]:
                instructions[f'core_{core[0]}_{core[1]}'] = {
                    'cluster_id': onnx_graph.node[re_id_to_node_id[nodes_re_id[i]]].name,
                    'weight_replica_id': j,
                    'instructions': []
                }
                use_channel = min(channelcnt, cp.channels_on_a_core)
                instructions[f'core_{core[0]}_{core[1]}']['instructions'].append({
                    'op': 'read',
                    'attr': {
                        'tensor_type': 'weight',
                        'shape': [use_channel, filter_shape[1], filter_shape[2], filter_shape[3]]
                    }
                })
                channelcnt -= use_channel

    nodes_to_allocation_id = [-1] * len(graph)
    for i in sorted_nodes:
        flag = -1
        for j, re_id in enumerate(nodes_re_id):
            if re_id_to_node_id[re_id] == belong_node[i]:
                flag = j
        if flag != -1:
            nodes_to_allocation_id[i] = flag

    nodecnt = len(nodes_re_id)

    tmp_edgeset = dict()

    for i in range(nodecnt):
        for j in range(nodecnt):
            if (nodes_re_id[i], nodes_re_id[j]) in re_id_graph_edgeset:
                tmp_edgeset[(i, j)] = re_id_graph_edgeset[(
                    nodes_re_id[i], nodes_re_id[j])]
    # 稍微topsort下
    indeg = [0] * nodecnt
    for a, b in tmp_edgeset:
        indeg[b] += 1

    id_topsort = []
    q = deque([i for i in range(nodecnt) if indeg[i] == 0])

    while q:
        x = q.popleft()
        id_topsort.append(x)
        for y in range(nodecnt):
            if (x, y) in tmp_edgeset:
                indeg[y] -= 1
                if indeg[y] == 0:
                    q.append(y)

    def add_instructions(icores, jcores, shape, tensor_name_prefix):
        if jcores is None:  # 写到global
            assert icores is not None, 'wtf global to global wtf is this'
            channelcnt = shape[1]
            for icore in icores:
                use_channel = min(cp.channels_on_a_core, channelcnt)
                instructions[f'core_{icore[0]}_{icore[1]}']['instructions'].append({
                    'op': 'write',
                    'attr': {
                        'tensor_type': 'feature',
                        'shape': [1, use_channel, shape[2], shape[3]]
                    }
                })
                channelcnt -= use_channel
            assert channelcnt == 0
            return

        core_num = len(jcores)
        accumulate_load_channelcnt = [0] * core_num

        if icores is not None:
            p = 0
            channelcnt = shape[1]
            for q in range(len(icores)):
                use_channel = min(channelcnt, cp.channels_on_a_core)
                accumulate_load_channelcnt[p] += use_channel
                frm = icores[q]
                to = jcores[p]

                instructions[f'core_{frm[0]}_{frm[1]}']['instructions'].append({
                    'op': 'send',
                    'attr': {
                        'dist_core_name': f'core_{to[0]}_{to[1]}',
                        'shape': [1, use_channel, shape[2], shape[3]],
                        'name': tensor_name_prefix + f'_btw_clusters_part_{q}'
                    }
                })
                instructions[f'core_{to[0]}_{to[1]}']['instructions'].append({
                    'op': 'receive',
                    'attr': {
                        'src_core_name': f'core_{frm[0]}_{frm[1]}',
                        'shape': [1, use_channel, shape[2], shape[3]],
                        'name': tensor_name_prefix + f'_btw_clusters_part_{q}'
                    }
                })

                p = (p + 1) % core_num
                channelcnt -= use_channel
        else:
            remainder = shape[1] % core_num
            accumulate_load_channelcnt = [0] * core_num
            for l in range(core_num):
                use_channel = shape[1] // core_num + \
                    max(remainder, 1)
                core = jcores[l]
                instructions[f'core_{core[0]}_{core[1]}']['instructions'].append({
                    'op': 'read',
                    'attr': {
                        'tensor_type': 'feature',
                        'shape': [1, use_channel, shape[2], shape[3]]
                    }
                })
                accumulate_load_channelcnt[l] += use_channel
                remainder -= max(remainder, 1)
            # print('read from global',j)

        def add_send_receive(frm, to, src, tensor_name):
            instructions[f'core_{frm[0]}_{frm[1]}']['instructions'].append({
                'op': 'send',
                'attr': {
                    'dist_core_name': f'core_{to[0]}_{to[1]}',
                    'shape': [1, accumulate_load_channelcnt[src], shape[2], shape[3]],
                    'name': tensor_name
                }
            })
            instructions[f'core_{to[0]}_{to[1]}']['instructions'].append({
                'op': 'receive',
                'attr': {
                    'src_core_name': f'core_{frm[0]}_{frm[1]}',
                    'shape': [1, accumulate_load_channelcnt[src], shape[2], shape[3]],
                    'name': tensor_name
                }
            })

        if core_num % 2 == 0:
            for round in range(core_num - 1):
                for bit in range(0, 1):
                    for src in range(bit, core_num, 2):
                        if accumulate_load_channelcnt[src] == 0:
                            continue
                        frm = (src + round) % core_num
                        to = (src + round + 1) % core_num
                        frm = jcores[frm]
                        to = jcores[to]
                        tensor_name = tensor_name_prefix + \
                            f'_in_cluster_part_{src}'
                        add_send_receive(frm, to, src, tensor_name)
        else:
            pos = 0
            for round in range(core_num - 1):
                for _ in range(core_num):
                    frm = pos
                    to = (pos + 1) % core_num
                    pos = (pos + 2) % core_num
                    src = (frm - round + core_num) % core_num
                    if accumulate_load_channelcnt[src] == 0:
                        continue
                    frm = jcores[frm]
                    to = jcores[to]
                    tensor_name = tensor_name_prefix + \
                        f'_in_cluster_part_{src}'
                    add_send_receive(frm, to, src, tensor_name)
    print('nodes_re_id:', nodes_re_id)
    print('id_topsort:', id_topsort)
    print()
    for k in range(cp.batch_size):

        for i_topsort_id in range(nodecnt):
            i = id_topsort[i_topsort_id]
            # 这里i已经按照拓扑序排了。现在加入指令：
            # 先加入读入数据，再加入计算，再加入输出数据

            # ----------读入----------
            # 只考虑从global读，从其他receive放下面
            for j in re_id_rev_graph[nodes_re_id[i]]:
                if in_nodes_re_id[j] == 1:
                    continue  # 不在此stage

                shape = re_id_graph_edgeset[(j, nodes_re_id[i])]
                add_instructions(None,
                                 allocation[i][k % len(allocation[i])],
                                 shape,
                                 get_unique_id())

            # 还要讨论，有可能这是第一个节点
            if nodes_re_id[i] in input_data_conv_node_re_id:
                shape = input_data_conv_node_re_id[nodes_re_id[i]]
                add_instructions(None,
                                 allocation[i][k % len(allocation[i])],
                                 shape,
                                 get_unique_id())

            # ----------计算----------
            in_cluster_nodes = [
                _ for _ in sorted_nodes if belong_node[_] == re_id_to_node_id[nodes_re_id[i]]]
            input_shape = get_tensor_shape(
                # 假定挂在这个conv上的节点的操作都跟这个conv输入的大小一样
                # 这个假定其实不是很科学，但是考虑到相邻的layer的激活值尺寸差别不大，就这么处理了
                onnx_graph,
                onnx_graph.node[re_id_to_node_id[nodes_re_id[i]]].input[0])

            input_shape[0] = 1
            runner = k % len(allocation[i])
            core_num = len(allocation[i][runner])
            for nodeid in in_cluster_nodes:
                channelcnt = 0
                op_type = onnx_graph.node[nodeid].op_type
                if op_type == 'Conv':
                    channelcnt = get_tensor_shape(
                        onnx_graph, onnx_graph.node[nodeid].input[1])[0]
                for coreid in range(core_num):
                    use_channel = min(channelcnt, cp.channels_on_a_core)
                    core = allocation[i][runner][coreid]
                    channelcnt -= use_channel
                    if op_type == 'Conv':
                        group = [
                            attr.i for attr in onnx_graph.node[nodeid].attribute if attr.name == 'group'][0]
                        if group == 1:
                            X_shape = [
                                1, use_channel, input_shape[2], input_shape[3]]
                            W_shape = get_tensor_shape(
                                onnx_graph, onnx_graph.node[nodeid].input[1])
                            assert use_channel != 0
                            W_shape[0] = use_channel
                            padding = [
                                attr.ints for attr in onnx_graph.node[nodeid].attribute if attr.name == 'pads'][0]
                            strides = [
                                attr.ints for attr in onnx_graph.node[nodeid].attribute if attr.name == 'strides'][0]
                            instructions[f'core_{core[0]}_{core[1]}']['instructions'].append({
                                'op': 'conv',
                                'attr': {
                                    'X_shape': X_shape,
                                    'W_shape': W_shape,
                                    'padding': list(padding),
                                    'strides': list(strides)
                                }
                            })
                        else:  # depthwise conv
                            X_shape = [
                                1, use_channel, input_shape[2], input_shape[3]]
                            W_shape = get_tensor_shape(
                                onnx_graph, onnx_graph.node[nodeid].input[1])
                            assert W_shape[1] == 1
                            W_shape[0] = use_channel
                            instructions[f'core_{core[0]}_{core[1]}']['instructions'].append({
                                'op': 'depthwise_conv',
                                'attr': {
                                    'group': group,
                                    'X_shape': X_shape,
                                    'W_shape': W_shape
                                }
                            })
                    elif op_type == 'Add':
                        instructions[f'core_{core[0]}_{core[1]}']['instructions'].append({
                            'op': 'add',
                            'attr': {
                                'shape': [1, use_channel, input_shape[2], input_shape[3]]
                            }
                        })
                    elif op_type == 'Relu':
                        instructions[f'core_{core[0]}_{core[1]}']['instructions'].append({
                            'op': 'relu',
                            'attr': {
                                'shape': [1, use_channel, input_shape[2], input_shape[3]]
                            }
                        })
                    else:
                        # unimportant operators, do nothing
                        pass

            # ----------输出----------
            # 向同stage其他节点输出、同stage其他节点的读入

            for j in range(nodecnt):
                if (nodes_re_id[i], nodes_re_id[j]
                        ) not in re_id_graph_edgeset:
                    continue
                shape = re_id_graph_edgeset[(
                    nodes_re_id[i], nodes_re_id[j])]

                if (nodes_re_id[i],
                        nodes_re_id[j]) in communicate_on_chip:
                    add_instructions(allocation[i][k % len(allocation[i])],
                                     allocation[j][k % len(allocation[j])],
                                     shape,
                                     get_unique_id())
                else:
                    unique_id = get_unique_id()
                    add_instructions(allocation[i][k % len(allocation[i])],
                                     None,
                                     shape,
                                     unique_id)
                    add_instructions(None,
                                     allocation[j][k % len(allocation[j])],
                                     shape,
                                     unique_id)

            # 向global的输出
            flag = False
            shape = None
            for j in re_id_graph[nodes_re_id[i]]:
                if in_nodes_re_id[j] == 0:
                    flag = True
                    shape = re_id_graph_edgeset[(nodes_re_id[i], j)]
                    break
            # 存在某个节点要用这个节点的输入，并且不再当前这个stage里面，说明需要把这个点写出去
            if flag or nodes_re_id[i] in output_data_conv_node_re_id:
                if shape is None:
                    shape = output_data_conv_node_re_id[nodes_re_id[i]]
                add_instructions(allocation[i][k % len(allocation[i])],
                                 None,
                                 shape,
                                 get_unique_id())

    return instructions
