from collections import deque
import math
import onnx
from onnx import shape_inference
from onnx import GraphProto, ModelProto, NodeProto
import cimpara as cp
from read_file import get_tensor_shape
from graph import split_to_chain

'''
重新理清楚

activation_width激活值位宽
weight_width权重位宽

Element:
每个element行m列n
m次(m*activation_width个周期)可进行1个m*(n/weight_width)的mvm

Macro:
每个macro为H*W,
同一行(每W个)element共享输入, H行独立, 然后累加
每m次可进行1个(H*m)*(n*W/weight_width)的mvm


Core:
每个K*T
T个macro groups, 现在存不同权重
同一行(每K个)element共享输入, T行独立, 然后累加
每m次可进行1个相同向量, 不同矩阵(H*m*T)*(n*W/weight_width*K)的mvm
    i.e. 进行(n*W/weight_width*K)个channel的计算, 耗时m*activation_width



'''


def calc_cores_and_time_needed(onnx_graph, node):
    if node.op_type == 'Conv':
        [N, C_in, H, W] = get_tensor_shape(onnx_graph, node.input[0])
        [C_out, C_in, K_h, K_w] = get_tensor_shape(onnx_graph, node.input[1])

        if C_in == 1:
            raise Exception(
                "whhhaaaat??? this seems to be a depthwise conv and shouldn't be considered here!")

        N = 1

        S = [attr.ints for attr in node.attribute if attr.name == 'strides'][0]
        P = [attr.ints for attr in node.attribute if attr.name == 'pads'][0]
        if not S:
            S = [1, 1]
        if not P:
            P = [0, 0, 0, 0]

        # print("conv layer:")
        # print("[N, C_in, H, W]" + str([N, C_in, H, W]))
        # print("[C_out, C_in, K_h, K_w] " + str([C_out, C_in, K_h, K_w]))
        # print("S" + str(S))
        # print("P" + str(P))

        H_out = (H + P[0] + P[2] - K_h) // S[0] + 1
        W_out = (W + P[1] + P[3] - K_w) // S[1] + 1

        # print(H_out)
        # print(W_out)

        # matA = [N * H_out * W_out, C_in * K_h * K_w]
        # matB = [C_in * K_h * K_w, C_out]

        if cp.H * cp.m * cp.T < K_h * K_w * C_in:
            raise Exception('holy the conv op is tooooooo big')

        # 核内允许权重复制的次数：每个channel_out需要K_h * K_w * C_in个权重
        # 每个macro group可以放下H*m个权重
        # 因此放下一个channel_out需要ceil(K_h * K_w * C_in  / (H * m))个macro group
        # 因此能放下floor(T/ceil(...))次权重复制
        duplicate_times = cp.T // math.ceil(K_h * K_w * C_in / (cp.H * cp.m))

        cores_needed = math.ceil(C_out / cp.channels_on_a_core)

        # 需要计算周期：img2col之后有H_out * W_out次输入
        # 每m * activation_width周期可以计算一次
        # 然后除以权重复制次数
        # batchsize留到分配core的时候再考虑
        # 因为涉及到core层面的权重复制

        time_needed = math.ceil(
            H_out *
            W_out *
            cp.m *
            cp.activation_width /
            duplicate_times)

        # 把weight load上去需要的时间
        # /8转化为Byte
        load_time_needed = math.ceil(
            C_out *
            C_in *
            K_h *
            K_w *
            cp.weight_width /
            8 /
            cp.global_memory_bandwidth)  # todo 输入输出带宽要不要对半分？

        # print(cores_needed, C_out, cp.channels_on_a_core)
        return cores_needed, time_needed, load_time_needed

    raise Exception('you must be joking, this is NOT a conv op')


'''
仔细整理一下现在的策略

传进来一个计算图片段。首先，贪心做个计算图片段的链划分
比如
A---B---C---D---E
    |       |
    ----G---H
这种，先划分成ABCDE和GH，然后就按照ABCDEGH这样的顺序往pattern上面放
这样尽可能连续的在一起，然后不连续的，比如BG、DH之间就用global通信。
这样就确定了往pattern上放的顺序

然后要做权重复制。代价是max_所有节点{计算时间*batchsize/此节点权重复制次数}+通信时间(压力最大的路由器)
权重复制越多，通信时间就越高，计算时间就越少
考虑每次取出计算时间最大的节点，然后给他加权重复制的次数，然后重新算代价。这个代价的计算时间应该变少了，通信时间应该增加了
直到新的代价比权重复制之前还要大，说明复制太多了，就停下

我发现这里好像又没有必要让权重复制的次数必须是二的幂次了，所以就先不要求是二的幂次
'''


def calc_best_strategy_on_chip(
        nodes_re_id,
        re_id_graph,
        re_id_rev_graph,
        re_id_graph_edgeset,  # 一个dict，key是一个二元组描述一条边，编号是conv重编号。value是边上数据的shape
        re_id_to_node_id,
        onnx_graph):

    # print("-------------------lets go--------------------")

    allnodescnt = len(re_id_graph)
    in_nodes_re_id = [0] * allnodescnt
    for i in nodes_re_id:
        in_nodes_re_id[i] = 1
    nodecnt = len(nodes_re_id)
    pattern_pos_lists = cp.pattern_pos_lists

    # 先确定往pattern上放的顺序
    chain_list = split_to_chain(nodes_re_id, re_id_graph_edgeset)
    nodes_re_id = []
    communicate_on_chip = dict()
    for chain in chain_list:
        nodes_re_id += chain
        # 顺便确定哪些关系用片上通信，哪些用片外
        for i in range(len(chain) - 1):
            communicate_on_chip[(chain[i], chain[i + 1])] = True

    cores_needed_list = []
    time_needed_list = []
    load_time_needed_all = 0

    for node_re_id in nodes_re_id:
        node = onnx_graph.node[re_id_to_node_id[node_re_id]]
        cores_needed, time_needed, load_time_needed = calc_cores_and_time_needed(
            onnx_graph, node)
        cores_needed_list.append(cores_needed)
        time_needed_list.append(time_needed)
        load_time_needed_all += load_time_needed

    # print("nodes_re_id:" + str(nodes_re_id))
    # print("cores_needed_list:" + str(cores_needed_list))
    # print("time_needed_list:" + str(time_needed_list))
    # print("load_time_needed_all:" + str(load_time_needed_all))
    # print(re_id_graph_edgeset)

    best_time_all_patterns = math.inf
    best_allocation_all_patterns = None
    for pattern_pos_list in pattern_pos_lists:
        duplicate_times = [1] * nodecnt

        # 这个函数简单按照core层的权重复制次数把节点分配到chip上
        def put_nodes_on_chip(duplicate_times):
            assert len(pattern_pos_list) == cp.C

            # 先看节点够不够
            if sum([cores_needed_list[i] * duplicate_times[i]
                   for i in range(nodecnt)]) > cp.C:
                return None

            j = 0
            allocation = []
            for i in range(nodecnt):
                alloc = []
                for _ in range(duplicate_times[i]):
                    alc = []
                    for __ in range(cores_needed_list[i]):
                        alc.append(pattern_pos_list[j])
                        j += 1
                    alloc.append(alc)
                allocation.append(alloc)

            # allocation[i][j][k]=(x,y)表示第i个node，第j个duplicate，第k个core的位置是(x,y)
            return allocation

        def get_cost_for_an_allocation(allocation):
            calc_time_list = [0] * nodecnt
            communication_time = 0
            for i in range(nodecnt):
                calc_time_list[i] = math.ceil(
                    cp.batch_size * time_needed_list[i] / len(allocation[i]))

            most_expensive_request = 0
            chip_node_load = [[0] * cp.Q for _ in range(cp.P)]
            global_memory_load = 0
            # 计算节点间的通信负载

            def add_load_sender(i, k, shape):
                nonlocal chip_node_load
                sender = k % duplicate_times[i]
                channelcnt = shape[1]
                assert len(allocation[i][sender]) == cores_needed_list[i]
                for core in allocation[i][sender]:
                    use_channel = min(
                        channelcnt, cp.channels_on_a_core)
                    chip_node_load[core[0]][core[1]] += use_channel * \
                        shape[2] * shape[3] * cp.activation_width // 8  # 转Byte
                    channelcnt -= use_channel

                assert channelcnt == 0

            def add_load_receiver(j, k, shape):
                nonlocal chip_node_load
                receiver = k % duplicate_times[j]
                for core in allocation[j][receiver]:
                    chip_node_load[core[0]][core[1]] += shape[1] * \
                        shape[2] * shape[3] * cp.activation_width // 8

            def add_load_global(shape):
                nonlocal global_memory_load
                global_memory_load += shape[1] * shape[2] * \
                    shape[3] * cp.activation_width // 8
                # print(shape)

            for i in range(nodecnt):
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
                                for jcore in allocation[j][receiver]:
                                    most_expensive_request = max(
                                        most_expensive_request,
                                        use_channel *
                                        shape[2] *
                                        shape[3] *
                                        cp.activation_width //
                                        8 *
                                        (math.fabs(icore[0] - jcore[0]) + math.fabs(icore[1] - jcore[1]))
                                    )
                                channelcnt -= use_channel

             # 计算节点输入负载
            for i in range(nodecnt):
                for j in re_id_rev_graph[nodes_re_id[i]]:
                    if in_nodes_re_id[j] == 1:
                        continue
                    # j->i
                    shape = re_id_graph_edgeset[(j, nodes_re_id[i])]

                    for k in range(cp.batch_size):
                        # i是接收方，从global读
                        add_load_receiver(i, k, shape)
                        add_load_global(shape)
                        # print("read from global",i)

                # 还要讨论，有可能这是第一个节点
                flag = False
                data_name = onnx_graph.input[0].name
                node = onnx_graph.node[re_id_to_node_id[nodes_re_id[i]]]
                for input_name in node.input:
                    if input_name == data_name:
                        flag = True
                        break
                if flag:
                    shape = get_tensor_shape(onnx_graph, data_name)
                    for k in range(cp.batch_size):
                        add_load_receiver(i, k, shape)
                        add_load_global(shape)
                        # print("read from global",i)

            # 计算节点输出负载
            for i in range(nodecnt):
                flag = False
                shape = None
                for j in re_id_graph[nodes_re_id[i]]:
                    if in_nodes_re_id[j] == 0:
                        flag = True
                        shape = re_id_graph_edgeset[(nodes_re_id[i], j)]
                        break
                # 存在某个节点要用这个节点的输入，并且不再当前这个stage里面，说明需要把这个点写出去
                if flag:
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

        best_time = math.inf
        best_allocation = None
        # print("\ntry a new pattern:")
        while True:
            # print("new loop")
            allocation = put_nodes_on_chip(duplicate_times)
            calc_time_list, cur_time = get_cost_for_an_allocation(allocation)

            if cur_time > best_time:
                break
            best_allocation = allocation
            best_time = cur_time
            calc_time_list_id = [(v, i) for i, v in enumerate(calc_time_list)]
            calc_time_list_id.sort()
            calc_time_list_id.reverse()
            used_core_num = sum([cores_needed_list[i] * duplicate_times[i]
                                 for i in range(nodecnt)])
            j = 0
            assert (len(calc_time_list_id) == nodecnt)
            while j < nodecnt:
                if duplicate_times[calc_time_list_id[j][1]] + 1 <= cp.batch_size and used_core_num + \
                        cores_needed_list[calc_time_list_id[j][1]] <= cp.C:
                    duplicate_times[calc_time_list_id[j][1]] += 1
                    break
                else:
                    j += 1

            if j == nodecnt:
                break
        # print(cores_needed_list)
        # print(duplicate_times)
        # print([cores_needed_list[i] * duplicate_times[i]
        #       for i in range(nodecnt)])
        # print(sum([cores_needed_list[i] * duplicate_times[i]
        #       for i in range(nodecnt)]))
        # print(calc_time_list)
        # print(f"calc time: {max(calc_time_list)}")
        # print(f"communication time: {cur_time - max(calc_time_list)}")
        # print()
        if best_time < best_time_all_patterns:
            best_time_all_patterns = best_time
            best_allocation_all_patterns = best_allocation

    # print([[onnx_graph.node[re_id_to_node_id[i]].name for i in _]
    #       for _ in chain_list])

    return best_time_all_patterns, best_allocation_all_patterns
# 注意！传进来的nodes已经重新排序了！
