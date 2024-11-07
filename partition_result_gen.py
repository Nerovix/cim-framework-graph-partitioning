import cimpara as cp
from collections import deque


def get_instrctions_for_a_stage(
        allocation,
        nodes_re_id,
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
        tensor_use_dict,
        # dict套list，表示一个tensor被哪些节点使用，
        onnx_graph):

    in_nodes_re_id=[0]*len(graph)
    for i in nodes_re_id:
        in_nodes_re_id[i]=1

    instructions = dict()
    for i in range(cp.P):
        for j in range(cp.Q):
            instructions[f"core_{i}_{j}"] = {
                "cluster_id": -1,
                "weight_replica_id": -1,
                "instructions": []
            }

    for i in range(allocation):
        for j in range(len(allocation[i])):
            for core in allocation[i][j]:
                instructions[f"core_{core[0]}_{core[1]}"] = {
                    "cluster_id": onnx_graph.node[re_id_to_node_id[nodes_re_id[i]]].name,
                    "weight_replica_id": j,
                    "instructions": []
                }

    nodes = []
    nodes_to_allocation_id = [-1] * len(graph)
    for i in sorted_nodes:
        flag = -1
        for j, re_id in enumerate(nodes_re_id):
            if re_id_to_node_id[re_id] == belong_node[i]:
                flag = j
        if flag != -1:
            nodes.append(i)
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
    for k, v in tmp_edgeset:
        indeg[j] += 1

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
#   todotodotodtodo   
#    !!!先load权重

    for k in range(cp.batch_size):

        for i_topsort_id in range(nodecnt):
            i=id_topsort[i_topsort_id]
            # 这里i已经按照拓扑序排了。现在加入指令：
            # 先加入读入数据，再加入计算，再加入输出数据

            # ----------读入----------
            # 只考虑从global读
            for j in re_id_rev_graph[nodes_re_id[i]]:
                if in_nodes_re_id[j] == 1:
                    continue
                # j->i
                shape = re_id_graph_edgeset[(j, nodes_re_id[i])]

                receiver=k%len(allocation[i])
                

                    for k in range(cp.batch_size):
                        # i是接收方，从global读
                        add_load_receiver(i, k, shape)
                        add_load_global(shape)
                        # print("read from global",i)

            # 还要讨论，有可能这是第一个节点
            if nodes_re_id[i] in input_data_conv_node_re_id:
                shape = input_data_conv_node_re_id[nodes_re_id[i]]
                onnx_graph.input[0].name
                # print(shape)
                for k in range(cp.batch_size):
                    add_load_receiver(i, k, shape)
                    add_load_global(shape)
                    # print("read from global",i)
            
            # ----------输出----------
            # 向同stage其他节点输出、同stage其他节点的读入

            for j in range(nodecnt):
                if (nodes_re_id[i], nodes_re_id[j]
                        ) not in re_id_graph_edgeset:
                    continue
                shape = re_id_graph_edgeset[(
                    nodes_re_id[i], nodes_re_id[j])]
                # 有batch_size个batch，
                # 发送方权重复制了duplicate_times[i]次
                # 接收方权重复制了duplicate_times[j]次
                # 于是第k个batch(0-based)
                # 由发送方的第k%duplicate_times[i]个权重复制到接收方的第k%duplicate_times[j]个权重复制
                # 然后，累加节点压力的时候，发送方要枚举每个权重复制，接收方枚举每个权重复制的每个节点
                for k in range(cp.batch_size):
                    # 发送方
                    add_load_sender(i, k, shape)
                    # 接收方
                    add_load_receiver(j, k, shape)

                    # 如果不是片上通信，不仅要考虑上述一边的发和一边的收，还要考虑global的一读一写
                    if (nodes_re_id[i],
                            nodes_re_id[j]) not in communicate_on_chip:
                        add_load_global(shape)
                        add_load_global(shape)  # 一读一写，两遍
                        # print('global communication',
                        #       onnx_graph.node[re_id_to_node_id[nodes_re_id[i]]].name,
                        #       onnx_graph.node[re_id_to_node_id[nodes_re_id[j]]].name)
                        # print('global communication', i, j)

                    # 如果是片上通信，要统计开销最大的
                    else:
                        sender = k % duplicate_times[i]
                        receiver = k % duplicate_times[j]
                        channelcnt = shape[1]
                        for icore in allocation[i][sender]:
                            use_channel = min(
                                channelcnt, cp.channels_on_a_core)
                            # 挑一个发，然后内部传
                            maxdis = -math.inf
                            for jcore in allocation[j][receiver]:
                                maxdis = max(
                                    maxdis,
                                    (math.fabs(
                                        icore[0] -
                                        jcore[0]) +
                                        math.fabs(
                                        icore[1] -
                                        jcore[1])))

                            most_expensive_request = max(
                                most_expensive_request,
                                use_channel *
                                shape[2] *
                                shape[3] *
                                cp.activation_width //
                                8 *
                                maxdis
                            )
                            channelcnt -= use_channel

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
                for k in range(cp.batch_size):
                    add_load_sender(i, k, shape)
                    add_load_global(shape)
                    # print("write to global",i)

        # print(math.ceil(max([max(_) for _ in chip_node_load]) / cp.B),
        #       math.ceil(global_memory_load / cp.global_memory_bandwidth),
        #       math.ceil(most_expensive_request / cp.B))
        communication_time = max(math.ceil(max([max(_) for _ in chip_node_load]) / cp.B),
                                 math.ceil(global_memory_load / cp.global_memory_bandwidth),
                                 math.ceil(most_expensive_request / cp.B))

        return calc_time_list, communication_time + \
            max(calc_time_list) + load_time_needed_all
