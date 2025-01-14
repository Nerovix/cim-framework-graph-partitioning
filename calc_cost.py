import math
from logging_config import logger
import cimpara as cp
from read_file import get_tensor_shape
from graph import split_to_chain

'''
feature_width: bit width of features
weight_width: bit width of weights

Element:
Each element has m rows and n columns
m times (m * feature_width cycles) can perform one m*(n/weight_width) mvm

Macro:
Each macro is H*W,
Elements in the same row (every W elements) share inputs, H rows are independent, then accumulate
m times can perform one (H*m)*(n*W/weight_width) mvm

Core:
Each K*T
T macro groups, currently storing different weights
Elements in the same row (every K elements) share inputs, T rows are independent, then accumulate
m times can perform one same vector, different matrices (H*m*T)*(n*W/weight_width*K) mvm
    i.e., performing (n*W/weight_width*K) channels of computation, taking m*feature_width time
'''


def calc_cores_and_time_needed(onnx_graph, node):
    if node.op_type == 'Conv':
        [N, C_in, H, W] = get_tensor_shape(onnx_graph, node.input[0])
        [C_out, C_in, K_h, K_w] = get_tensor_shape(onnx_graph, node.input[1])

        group = [attr.i for attr in node.attribute if attr.name == 'group'][0]
        assert group == 1, "Depthwise separable convolution is not considered."

        if C_in == 1:
            assert (0), "Depthwise convolution should not be here."

        N = 1

        S = [attr.ints for attr in node.attribute if attr.name == 'strides'][0]
        P = [attr.ints for attr in node.attribute if attr.name == 'pads'][0]
        if not S:
            S = [1, 1]
        if not P:
            P = [0, 0, 0, 0]

        H_out = (H + P[0] + P[2] - K_h) // S[0] + 1
        W_out = (W + P[1] + P[3] - K_w) // S[1] + 1

        # matA = [N * H_out * W_out, C_in * K_h * K_w]
        # matB = [C_in * K_h * K_w, C_out]

        if cp.H * cp.m * cp.K < K_h * K_w * C_in:
            raise Exception('The convolution operation is too large.')

        # replicate times allowed in the core:
        # Each C_out requires K_h * K_w * C_in weights
        # Each macro group can hold H * m weights
        # Therefore, placing a C_out requires ceil(K_h * K_w * C_in / (H * m)) macro groups
        # Therefore, floor(K / ceil(...)) weight copies can be placed

        replicate_times = cp.K // math.ceil(K_h * K_w * C_in / (cp.H * cp.m))

        cores_needed = math.ceil(C_out / cp.channels_on_a_core())

        # Required computation cycles:
        # H_out * W_out inputs after img2col
        # Each computation requires m * feature_width cycles
        # Then divide by the number of weight replications
        # Batch size will be considered when allocating cores
        # as it involves weight replication at the core level

        time_needed = math.ceil(
            H_out *
            W_out *
            cp.m *
            cp.feature_width /
            replicate_times)

        # Time needed to load all weights
        load_time_needed = math.ceil(
            C_out *
            C_in *
            K_h *
            K_w *
            cp.weight_width /
            8 /  # bits to Byte
            cp.global_memory_bandwidth) * replicate_times

        # print(cores_needed, C_out, cp.channels_on_a_core())
        return cores_needed, time_needed, load_time_needed

    raise Exception('This is not a convolution operator.')


def calc_best_strategy_on_chip(
        nodes_reassigned_id,
        reassigned_id_graph,
        reassigned_id_rev_graph,
        reassigned_id_graph_edgeset,  # A dict, the key is a two-tuple describing an edge, the number is the conv renumber. The value is the shape of the edge data.
        reassigned_id_to_node_id,
        input_data_conv_node_reassigned_id,  # see process.py for more details
        output_data_conv_node_reassigned_id,  # see process.py for more details
        onnx_graph):

    logger.debug("Start calc_best_strategy_on_chip, nodes_reassigned_id: " + str(nodes_reassigned_id))

    allnodescnt = len(reassigned_id_graph)
    in_nodes_reassigned_id = [0] * allnodescnt
    for i in nodes_reassigned_id:
        in_nodes_reassigned_id[i] = 1
    nodecnt = len(nodes_reassigned_id)
    pattern_pos_lists = cp.pattern_pos_lists

    # Determine the order to place on the pattern:
    # Split the graph into chains, and then place the chains on the chip
    # By doing so, we maximize on-chip communication and minimize global memory communication
    chain_list = split_to_chain(nodes_reassigned_id, reassigned_id_graph_edgeset)
    nodes_reassigned_id = []
    communicate_on_chip = dict()
    for chain in chain_list:
        nodes_reassigned_id += chain
        # Also determine which dependency edge use on-chip communication, which use global memory
        for i in range(len(chain) - 1):
            communicate_on_chip[(chain[i], chain[i + 1])] = True

    cores_needed_list = []
    time_needed_list = []
    load_time_needed_list = []

    for node_reassigned_id in nodes_reassigned_id:
        node = onnx_graph.node[reassigned_id_to_node_id[node_reassigned_id]]
        cores_needed, time_needed, load_time_needed = calc_cores_and_time_needed(
            onnx_graph, node)
        cores_needed_list.append(cores_needed)
        time_needed_list.append(time_needed)
        load_time_needed_list.append(load_time_needed)

    # Special case: cannot be placed on the chip
    if sum(cores_needed_list) > cp.C:
        logger.debug("Cannot be placed on the chip, no enough cores.")
        return math.inf, None, None, None, None

    logger.debug('nodes_reassigned_id:' + str(nodes_reassigned_id))
    logger.debug('cores_needed_list:' + str(cores_needed_list))
    logger.debug('time_needed_list:' + str(time_needed_list))
    logger.debug('load_time_needed_list:' + str(load_time_needed_list))
    logger.debug('reassigned_id_graph_edgeset:' + str(reassigned_id_graph_edgeset))

    def calc_dis(core0, core1):
        return math.fabs(core0[0] - core1[0]) + math.fabs(core0[1] - core1[1])

    best_time_all_patterns = math.inf
    best_allocation_all_patterns = None
    for pattern_pos_list_idx, pattern_pos_list in enumerate(pattern_pos_lists):
        logger.debug(f'Try pattern {pattern_pos_list_idx}(0-index, {len(pattern_pos_lists)} in total):')
        replicate_times = [1] * nodecnt

        # This function simply allocates nodes to the chip based on the core-level weight replication times
        def put_nodes_on_chip(replicate_times):
            assert len(pattern_pos_list) == cp.C
            # First check if there are enough nodes
            if sum([cores_needed_list[i] * replicate_times[i]
                   for i in range(nodecnt)]) > cp.C:
                return None
            j = 0
            allocation = []
            for i in range(nodecnt):
                alloc = []
                for _ in range(replicate_times[i]):
                    alc = []
                    for __ in range(cores_needed_list[i]):
                        alc.append(pattern_pos_list[j])
                        j += 1
                    alloc.append(alc)
                allocation.append(alloc)

            # allocation[i][j][k]=(x, y) represents the i-th node's j-th replicate's k-th core are allocated to core (x, y) on chip
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
            cluster_internel_communication_cost = [
                [0] * len(allocation[i]) for i in range(len(allocation))]

            # Calculate communication load between nodes

            def add_load_sender(i, k, shape):
                nonlocal chip_node_load
                sender = k % replicate_times[i]
                channelcnt = shape[1]
                assert len(allocation[i][sender]) == cores_needed_list[i]
                for core in allocation[i][sender]:
                    use_channel = min(
                        channelcnt, cp.channels_on_a_core())
                    chip_node_load[core[0]][core[1]] += use_channel * \
                        shape[2] * shape[3] * cp.feature_width // 8  # bit to Byte
                    channelcnt -= use_channel

                assert channelcnt == 0

            def add_load_receiver(j, k, shape, is_from_global):
                nonlocal chip_node_load
                nonlocal cluster_internel_communication_cost
                receiver = k % replicate_times[j]
                # Temporarily accumulate the load received by each receiving core
                accumulate_load = [0] * len(allocation[j][receiver])
                if not is_from_global:
                    # From other nodes, accumulate in a round-robin manner
                    p = 0
                    channelcnt = shape[1]
                    while channelcnt > 0:
                        use_channel = min(
                            channelcnt, cp.channels_on_a_core())
                        accumulate_load[p] += use_channel * \
                            shape[2] * shape[3] * cp.feature_width // 8
                        p = (p + 1) % len(accumulate_load)
                        channelcnt -= use_channel
                else:
                    # From global memory, distribute as evenly as possible
                    remainder = shape[1] % len(accumulate_load)
                    for i in range(len(accumulate_load)):
                        use_channel = shape[1] // len(accumulate_load) + \
                            min(remainder, 1)
                        accumulate_load[i] += use_channel * \
                            shape[2] * shape[3] * cp.feature_width // 8
                        remainder -= min(remainder, 1)

                circle_dis = 0
                for p in range(len(accumulate_load)):
                    core = allocation[j][receiver][p]
                    chip_node_load[core[0]][core[1]] += accumulate_load[p]
                    circle_dis += calc_dis(allocation[j][receiver][p], allocation[j][receiver][(
                        p + 1) % len(accumulate_load)])  # Calculate ring communication distance

                cluster_internel_communication_cost[j][receiver] += math.ceil(
                    circle_dis * max(accumulate_load) / cp.B)

            def update_most_expensive_request(i, j, k, shape):
                nonlocal most_expensive_request
                sender = k % replicate_times[i]
                receiver = k % replicate_times[j]
                channelcnt = shape[1]
                for icoreid, icore in enumerate(allocation[i][sender]):
                    use_channel = min(
                        channelcnt, cp.channels_on_a_core())
                    # choose a receiver and internally transmit
                    jcoreid = icoreid % len(allocation[j][receiver])
                    jcore = allocation[j][receiver][jcoreid]
                    most_expensive_request = max(
                        most_expensive_request,
                        use_channel *
                        shape[2] *
                        shape[3] *
                        cp.feature_width //
                        8 *
                        (math.fabs(icore[0] - jcore[0]) +
                         math.fabs(icore[1] - jcore[1]))
                    )
                    channelcnt -= use_channel
                assert (channelcnt == 0)

            def add_load_global(shape):
                nonlocal global_memory_load
                global_memory_load += shape[1] * shape[2] * \
                    shape[3] * cp.feature_width // 8
                # print(shape)

            for i in range(nodecnt):
                for j in range(nodecnt):
                    if (nodes_reassigned_id[i], nodes_reassigned_id[j]
                            ) not in reassigned_id_graph_edgeset:
                        continue
                    shape = reassigned_id_graph_edgeset[(
                        nodes_reassigned_id[i], nodes_reassigned_id[j])]
                    # There are batch_size batches
                    # The sender has replicate_times[i] replicas
                    # The receiver has replicate_times[j] replicas
                    # The k-th batch(0-based) should be \
                    # sent from the k%replicate_times[i]-th replica to k%replicate_times[j]-th replica
                    for k in range(cp.batch_size):
                        if (nodes_reassigned_id[i],
                                nodes_reassigned_id[j]) not in communicate_on_chip:
                            # If it is not on-chip communication, use global memory
                            add_load_sender(i, k, shape)
                            add_load_global(shape)  # 1 time for write
                            add_load_global(shape)  # 1 time for read
                            add_load_receiver(j, k, shape, True)
                        else:
                            # Use on-chip communication, consider the most expensive request
                            add_load_sender(i, k, shape)
                            add_load_receiver(j, k, shape, False)
                            update_most_expensive_request(i, j, k, shape)

            # Calculate node input load
            for i in range(nodecnt):
                for j in reassigned_id_rev_graph[nodes_reassigned_id[i]]:
                    if in_nodes_reassigned_id[j] == 1:
                        continue
                    # j->nodes_reassigned_id[i]
                    shape = reassigned_id_graph_edgeset[(j, nodes_reassigned_id[i])]

                    for k in range(cp.batch_size):
                        # i is the receiver, reading from global memory
                        add_load_global(shape)
                        add_load_receiver(i, k, shape, True)

                # Also consider if this is the first node in the graph, directly reading input data from global memory
                if nodes_reassigned_id[i] in input_data_conv_node_reassigned_id:
                    shape = input_data_conv_node_reassigned_id[nodes_reassigned_id[i]]
                    onnx_graph.input[0].name
                    # print(shape)
                    for k in range(cp.batch_size):
                        add_load_global(shape)
                        add_load_receiver(i, k, shape, True)
                        # print("read from global",i)

            # Calculate node output load
            for i in range(nodecnt):
                flag = False
                shape = None
                for j in reassigned_id_graph[nodes_reassigned_id[i]]:
                    if in_nodes_reassigned_id[j] == 0:
                        flag = True
                        shape = reassigned_id_graph_edgeset[(nodes_reassigned_id[i], j)]
                        break

                # If some node needs to use the output of this node and it's not in the current stage
                # Or this node's output needs to be written out
                # It indicates that this node's output needs to be written out
                if flag or nodes_reassigned_id[i] in output_data_conv_node_reassigned_id:
                    if shape is None:
                        shape = output_data_conv_node_reassigned_id[nodes_reassigned_id[i]]
                    for k in range(cp.batch_size):
                        add_load_sender(i, k, shape)
                        add_load_global(shape)
                        # print("write to global",i)

            # Calculate communication time
            communication_time = max(math.ceil(max([max(_) for _ in chip_node_load]) / cp.B),
                                     math.ceil(global_memory_load / cp.global_memory_bandwidth),
                                     math.ceil(most_expensive_request / cp.B))
            communication_time += max(max(_)
                                      for _ in cluster_internel_communication_cost)
            communication_time += sum([load_time_needed_list[i]
                                      * replicate_times[i] for i in range(nodecnt)])

            calc_time = max(calc_time_list)

            if cp.partition_mode == 3:
                communication_time *= 2
            elif cp.partition_mode == 4:
                calc_time = sum(calc_time_list)
            elif cp.pattern_map == 5:
                communication_time -= sum([load_time_needed_list[i]
                                           * replicate_times[i] for i in range(nodecnt)])
                communication_time += sum([load_time_needed_list[i] //
                                          2 for i in range(nodecnt)])
            return calc_time_list, calc_time + communication_time  # ,
            # max(calc_time_list), \
            # communication_time, \
            # math.ceil(max([max(_) for _ in chip_node_load]) / cp.B), \
            # math.ceil(global_memory_load / cp.global_memory_bandwidth), \
            # math.ceil(most_expensive_request / cp.B), \
            # max(max(_) for _ in cluster_internel_communication_cost), \
            # sum([load_time_needed_list[i] * replicate_times[i] for i in
            # range(nodecnt)])

        best_time = math.inf
        best_allocation = None
        # best_pack = None

        if cp.partition_mode in [0, 1, 3, 4, 5]:
            logger.debug(f"Iteration begin:")
            while True:
                logger.debug("New loop, replicate_times: {}".format(replicate_times))
                allocation = put_nodes_on_chip(replicate_times)
                pack = get_cost_for_an_allocation(allocation)
                calc_time_list, cur_time = pack[0], pack[1]
                # print(
                #     cur_time,
                #     node_reassigned_id,
                #     cores_needed_list,
                #     replicate_times,
                #     pack)
                if cur_time < best_time:
                    best_allocation = allocation
                    best_time = cur_time
                    # best_pack = pack
                calc_time_list_id = [(v, i)
                                     for i, v in enumerate(calc_time_list)]
                calc_time_list_id.sort()
                calc_time_list_id.reverse()
                used_core_num = sum([cores_needed_list[i] * replicate_times[i]
                                     for i in range(nodecnt)])
                j = 0
                assert (len(calc_time_list_id) == nodecnt)
                while j < nodecnt:
                    if replicate_times[calc_time_list_id[j][1]] + 1 <= cp.batch_size and used_core_num + \
                            cores_needed_list[calc_time_list_id[j][1]] <= cp.C:
                        replicate_times[calc_time_list_id[j][1]] += 1
                        break
                    else:
                        j += 1

                if j == nodecnt:
                    break
            logger.debug('cores_needed_list: ' + str(cores_needed_list))
            logger.debug('replicate_times: ' + str(replicate_times))
            logger.debug('actual cores needed: ' + str([cores_needed_list[i] * replicate_times[i]
                                                        for i in range(nodecnt)]))
            logger.debug(f'cores used in total: {sum([cores_needed_list[i] * replicate_times[i]
                                                      for i in range(nodecnt)])}')
            logger.debug('calc_time_list: ' + str(calc_time_list))
            logger.debug(f'best time: {best_time}')
            logger.debug(f'max calc time: {max(calc_time_list)}')
            logger.debug(f'sum of calc time: {sum(calc_time_list)}')
            logger.debug('Iteration end.')
            # logger.debug(
            #     f'''communication time: {
            #         cur_time -
            #         max(calc_time_list) -
            #         load_time_needed_all}''')
            # logger.debug(f'load time: {load_time_needed_all}')
        else:
            assert cp.partition_mode == 2
            logger.debug(f'Greedy mode, no replicate.')
            allocation = put_nodes_on_chip(replicate_times)
            pack = get_cost_for_an_allocation(allocation)
            calc_time_list, cur_time = pack[0], pack[1]
            best_allocation = allocation
            best_time = cur_time
            # best_pack = pack
        if best_time < best_time_all_patterns:
            best_time_all_patterns = best_time
            best_allocation_all_patterns = best_allocation
            # best_pack_all_patterns = best_pack

    # nodes_reassigned_id have already been reordered, we need to return the new order.
    return best_time_all_patterns, best_allocation_all_patterns, nodes_reassigned_id, cores_needed_list, communicate_on_chip
