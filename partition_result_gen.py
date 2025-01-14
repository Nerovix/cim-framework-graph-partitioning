import cimpara as cp
from collections import deque
from read_file import get_tensor_shape
from logging_config import logger

import random
import string


def get_unique_id(length=15):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def get_instrctions_for_a_stage(  # see process.py for more details about the parameters
        allocation,
        nodes_reassigned_id,
        communicate_on_chip,
        reassigned_id_graph,
        reassigned_id_rev_graph,
        reassigned_id_graph_edgeset,
        reassigned_id_to_node_id,
        input_data_conv_node_reassigned_id,
        output_data_conv_node_reassigned_id,
        graph,  # Represents the original graph (the dependency graph containing all nodes, including non-conv nodes)
        belong_node,
        sorted_nodes,  # All nodes in the original graph, topsorted
        onnx_graph):

    in_nodes_reassigned_id = [0] * len(graph)
    for i in nodes_reassigned_id:
        in_nodes_reassigned_id[i] = 1

    # Initialize instructions for each core
    instructions = dict()
    for i in range(cp.P):
        for j in range(cp.Q):
            instructions[f'core_{i}_{j}'] = {
                'cluster_id': -1,
                'weight_replica_id': -1,
                'instructions': []
            }

    # Add read instructions for weights
    logger.debug('Adding read instructions for weights...')
    for i in range(len(allocation)):
        for j in range(len(allocation[i])):
            filter_shape = get_tensor_shape(
                onnx_graph,
                onnx_graph.node[reassigned_id_to_node_id[nodes_reassigned_id[i]]].input[1]
            )
            channelcnt = filter_shape[0]
            for core in allocation[i][j]:
                instructions[f'core_{core[0]}_{core[1]}'] = {
                    'cluster_id': onnx_graph.node[
                        reassigned_id_to_node_id[nodes_reassigned_id[i]]].name,
                    'weight_replica_id': j,
                    'instructions': []
                }
                use_channel = min(channelcnt, cp.channels_on_a_core())
                instructions[f'core_{core[0]}_{core[1]}']['instructions'].append({
                    'op': 'read',
                    'attr': {
                        'tensor_type': 'weight',
                        'shape': [use_channel, filter_shape[1],
                                  filter_shape[2], filter_shape[3]]
                    }
                })
                channelcnt -= use_channel

    nodes_to_allocation_id = [-1] * len(graph)
    for i in sorted_nodes:
        flag = -1
        for j, reassigned_id in enumerate(nodes_reassigned_id):
            if reassigned_id_to_node_id[reassigned_id] == belong_node[i]:
                flag = j
        if flag != -1:
            nodes_to_allocation_id[i] = flag

    nodecnt = len(nodes_reassigned_id)

    tmp_edgeset = dict()

    for i in range(nodecnt):
        for j in range(nodecnt):
            if (nodes_reassigned_id[i], nodes_reassigned_id[j]) in reassigned_id_graph_edgeset:
                tmp_edgeset[(i, j)] = reassigned_id_graph_edgeset[(
                    nodes_reassigned_id[i], nodes_reassigned_id[j])]

    # Perform a topsort for nodes_reassigned_id first
    indeg = [0] * nodecnt
    for _, to in tmp_edgeset:
        indeg[to] += 1

    id_topsort = []
    topsort_queue = deque([i for i in range(nodecnt) if indeg[i] == 0])

    while topsort_queue:
        cur_node = topsort_queue.popleft()
        id_topsort.append(cur_node)
        for next_node in range(nodecnt):
            if (cur_node, next_node) in tmp_edgeset:
                indeg[next_node] -= 1
                if indeg[next_node] == 0:
                    topsort_queue.append(next_node)

    def add_communication_instructions(icores, jcores, shape, tensor_name_prefix):
        if jcores is None:  # write to global
            assert icores is not None, 'jcores and icores cannot be None at the same time'
            channelcnt = shape[1]
            for icore in icores:
                use_channel = min(cp.channels_on_a_core(), channelcnt)
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

        if icores is not None:  # send and receive
            p = 0
            channelcnt = shape[1]
            for q in range(len(icores)):
                use_channel = min(channelcnt, cp.channels_on_a_core())
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
        else:  # read from global
            remainder = shape[1] % core_num
            accumulate_load_channelcnt = [0] * core_num
            for l in range(core_num):
                use_channel = shape[1] // core_num + \
                    min(remainder, 1)
                core = jcores[l]
                instructions[f'core_{core[0]}_{core[1]}']['instructions'].append({
                    'op': 'read',
                    'attr': {
                        'tensor_type': 'feature',
                        'shape': [1, use_channel, shape[2], shape[3]]
                    }
                })
                accumulate_load_channelcnt[l] += use_channel
                remainder -= min(remainder, 1)

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

        # send and receive between cores in the same cluster
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

    for k in range(cp.batch_size):  # for every batch
        for i_topsort_id in range(nodecnt):  # for every node in topological order
            i = id_topsort[i_topsort_id]

            logger.debug(f'Processing node reassigned id {i}...')
            # Instructions for node i in topological order:
            # First add input read/receive instructions
            # Then add computation instructions
            # Finally add output write/send instructions

            # Input:
            # Only consider reading from global memory, receiving from others will be handled later
            for j in reassigned_id_rev_graph[nodes_reassigned_id[i]]:
                if in_nodes_reassigned_id[j] == 1:
                    continue  # 不在此stage

                shape = reassigned_id_graph_edgeset[(j, nodes_reassigned_id[i])]
                add_communication_instructions(None,
                                               allocation[i][k % len(allocation[i])],
                                               shape,
                                               get_unique_id())

            # Also check if this is the first node
            if nodes_reassigned_id[i] in input_data_conv_node_reassigned_id:
                shape = input_data_conv_node_reassigned_id[nodes_reassigned_id[i]]
                add_communication_instructions(None,
                                               allocation[i][k % len(allocation[i])],
                                               shape,
                                               get_unique_id())

            # Computation：
            # Get nodes belonging to this conv node
            in_cluster_nodes = [
                _ for _ in sorted_nodes if belong_node[_] == reassigned_id_to_node_id[nodes_reassigned_id[i]]]
            output_shape = get_tensor_shape(
                # Assume other ops attached to this conv node have same input size as the output size of the conv node
                # This assumption is not 100% accurate, but considering activation sizes don't vary much between adjacent layers,
                # We'll handle it this way for now
                # May change how other nodes attach to conv later
                onnx_graph,
                onnx_graph.node[reassigned_id_to_node_id[nodes_reassigned_id[i]]].output[0])

            weight_shape = get_tensor_shape(
                onnx_graph,
                onnx_graph.node[reassigned_id_to_node_id[nodes_reassigned_id[i]]].input[1])
            input_shape = get_tensor_shape(
                onnx_graph,
                onnx_graph.node[reassigned_id_to_node_id[nodes_reassigned_id[i]]].input[0])

            output_shape[0] = 1
            runner = k % len(allocation[i])
            core_num = len(allocation[i][runner])
            for nodeid in in_cluster_nodes:
                channelcnt = 0
                op_type = onnx_graph.node[nodeid].op_type
                channelcnt = weight_shape[0]
                for coreid in range(core_num):
                    use_channel = min(channelcnt, cp.channels_on_a_core())
                    assert use_channel != 0
                    core = allocation[i][runner][coreid]
                    channelcnt -= use_channel
                    if op_type == 'Conv':
                        group = [
                            attr.i for attr in onnx_graph.node[nodeid].attribute if attr.name == 'group'][0]
                        padding = [
                            attr.ints for attr in onnx_graph.node[nodeid].attribute if attr.name == 'pads'][0]
                        strides = [
                            attr.ints for attr in onnx_graph.node[nodeid].attribute if attr.name == 'strides'][0]
                        if group == 1:
                            X_shape = [
                                1, input_shape[1], input_shape[2], input_shape[3]]
                            W_shape = get_tensor_shape(
                                onnx_graph, onnx_graph.node[nodeid].input[1])
                            assert use_channel != 0
                            W_shape[0] = use_channel
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
                                1, input_shape[1], input_shape[2], input_shape[3]]
                            W_shape = get_tensor_shape(
                                onnx_graph, onnx_graph.node[nodeid].input[1])
                            assert W_shape[1] == 1
                            W_shape[0] = use_channel
                            instructions[f'core_{core[0]}_{core[1]}']['instructions'].append({
                                'op': 'depthwise_conv',
                                'attr': {
                                    'group': group,
                                    'X_shape': X_shape,
                                    'W_shape': W_shape,
                                    'padding': list(padding),
                                    'strides': list(strides)
                                }
                            })
                    elif op_type == 'Add' and onnx_graph.node[nodeid].input[1].find('zero_point') == -1:
                        instructions[f'core_{core[0]}_{core[1]}']['instructions'].append({
                            'op': 'add',
                            'attr': {
                                'shape': [1, use_channel, output_shape[2], output_shape[3]]
                            }
                        })
                    else:
                        # unimportant operators, do nothing
                        pass
                assert channelcnt == 0

            # Output：
            # Output to other nodes in same stage and handle their input
            # Since we did a topsort, we can ensure that the output nodes of this node haven't been processed yet
            # So we can directly add receive instructions for the output nodes of node i

            for j in range(nodecnt):
                if (nodes_reassigned_id[i], nodes_reassigned_id[j]
                        ) not in reassigned_id_graph_edgeset:
                    continue
                shape = reassigned_id_graph_edgeset[(
                    nodes_reassigned_id[i], nodes_reassigned_id[j])]

                if (nodes_reassigned_id[i],
                        nodes_reassigned_id[j]) in communicate_on_chip:
                    add_communication_instructions(allocation[i][k % len(allocation[i])],
                                                   allocation[j][k % len(allocation[j])],
                                                   shape,
                                                   get_unique_id())
                else:
                    unique_id = get_unique_id()
                    add_communication_instructions(allocation[i][k % len(allocation[i])],
                                                   None,
                                                   shape,
                                                   unique_id)
                    add_communication_instructions(None,
                                                   allocation[j][k % len(allocation[j])],
                                                   shape,
                                                   unique_id)

            # Output to global memory
            flag = False
            shape = None
            for j in reassigned_id_graph[nodes_reassigned_id[i]]:
                if in_nodes_reassigned_id[j] == 0:
                    flag = True
                    shape = reassigned_id_graph_edgeset[(nodes_reassigned_id[i], j)]
                    break
            # If some node needs this node's output but is not in current stage,
            # We need to write this node's output to global memory
            if flag or nodes_reassigned_id[i] in output_data_conv_node_reassigned_id:
                if shape is None:
                    shape = output_data_conv_node_reassigned_id[nodes_reassigned_id[i]]
                add_communication_instructions(allocation[i][k % len(allocation[i])],
                                               None,
                                               shape,
                                               get_unique_id())

    return instructions
