from preprocess.graph import build_graph, find_all_prefixes, get_belong_node, topsort
from optimization.cost import calc_best_strategy_on_chip
from preprocess.read_file import get_tensor_shape
from preprocess.model_preprocess import onnx_split_large_conv_pass
from preprocess.read_file import print_graph_nodes
from utils.instructions_gen import get_instrctions_for_a_stage
from utils.visualizer import visualize_computation_graph, visualize_chip_allocation_per_stage
from config.logger_config import logger
import config.cim_config as c_conf
import math


def cg_mapping(model):
    onnx_split_large_conv_pass(model)
    onnx_graph = model.graph

    print_graph_nodes(onnx_graph)

    graph = build_graph(onnx_graph)

    is_conv_node = []
    is_fc_node = []
    for n in onnx_graph.node:
        if n.op_type == 'Conv' and get_tensor_shape(
                onnx_graph, n.input[1])[1] != 1:
            # pointwise
            is_conv_node.append(1)
        else:
            # depthwise
            is_conv_node.append(0)

        if n.op_type == 'Gemm' or n.op_type == 'MatMul':
            is_fc_node.append(1)
        else:
            is_fc_node.append(0)

    belong_node = get_belong_node(graph, is_conv_node, is_fc_node)

    logger.debug(f'belong_node: {[(i, v)for i, v in enumerate(belong_node)]}')
    logger.debug(f'conv_node: {[i for i, v in enumerate(belong_node) if i == v and is_fc_node[i] == 0]}')

    # 0-based, original id -> reassigned id
    conv_node_reassigned_id = [0] * len(is_conv_node)
    reassigned_id_to_node_id = []  # rea
    number_of_conv_nodes = 0
    for i in range(len(is_conv_node)):
        if is_conv_node[i]:
            conv_node_reassigned_id[i] = len(reassigned_id_to_node_id)
            reassigned_id_to_node_id.append(i)

    number_of_conv_nodes = len(reassigned_id_to_node_id)
    logger.info(f'Number of conv nodes: {number_of_conv_nodes}')

    # new graph based on reassigned id
    reassigned_id_graph = [[] for _ in range(number_of_conv_nodes)]

    input_data_conv_node_reassigned_id = dict()  # key: reassigned id, value: shape
    output_data_conv_node_reassigned_id = dict()  # key: reassigned id, value: shape

    # rebuild the dependency graph based on reassigned id and belong_node
    reassigned_id_graph_edgeset = dict()  # shape of every dependency edge
    for i, g in enumerate(graph):
        i_bel = belong_node[i]
        i_bel_reassigned_id = conv_node_reassigned_id[i_bel]
        for j in g[1]:
            j_bel = belong_node[j]
            j_bel_reassigned_id = conv_node_reassigned_id[j_bel]
            # if this edge involves two conv nodes
            if i_bel != j_bel and is_conv_node[i_bel] == 1 and is_conv_node[j_bel] == 1:

                reassigned_id_graph[i_bel_reassigned_id].append(j_bel_reassigned_id)
                shape = get_tensor_shape(
                    onnx_graph,
                    onnx_graph.node[reassigned_id_to_node_id[i_bel_reassigned_id]].output[0]
                )
                reassigned_id_graph_edgeset[(i_bel_reassigned_id, j_bel_reassigned_id)] = shape

                assert len(shape) == 4, \
                    'shape should be [N * C * H * W], but dimension of the tensor is not 4'

            elif i_bel != j_bel and is_conv_node[j_bel] == 0:
                # j_bel is not a conv node, so it is a Gemm node.
                # Gemm nodes always appears last in the DAG, so we won't discuss them at the moment. Just record that i_bel needs to store its result in global memory.
                output_data_conv_node_reassigned_id[i_bel_reassigned_id] = get_tensor_shape(
                    onnx_graph,
                    onnx_graph.node[reassigned_id_to_node_id[i_bel_reassigned_id]].output[0]
                )

        # It's also possible that i_bel is the first node in the graph, it needs to read original input data from global memory.
        if onnx_graph.input[0].name in onnx_graph.node[i].input:
            input_data_conv_node_reassigned_id[i_bel_reassigned_id] = get_tensor_shape(
                onnx_graph, onnx_graph.input[0].name)

    logger.debug(f'reassigned_id_graph_edgeset: {reassigned_id_graph_edgeset}')
    
    logger.debug(f'reassigned_id_graph: {[(i,g) for i,g in enumerate(reassigned_id_graph)]}')
    
    reassigned_id_rev_graph = [[] for _ in range(number_of_conv_nodes)]  # reversed graph
    for i in range(number_of_conv_nodes):
        for j in reassigned_id_graph[i]:
            reassigned_id_rev_graph[j].append(i)

    # Find all 'prefixes'(dependency closures) of the rebuilt DAG.
    prefixes_bitmask_reassigned_id = find_all_prefixes(reassigned_id_graph)
    stages = []

    if c_conf.partition_mode in [0, 3, 4, 5, 6]:
        # DP for strategy
        dp_stages = [math.inf] * len(prefixes_bitmask_reassigned_id)
        dp_stages_from = [-1] * len(dp_stages)
        dp_stages_alloc_info = [0] * len(dp_stages)

        logger.info("DP started")

        # prefixes list is sorted, just enumerate by index
        for i, ith_prefix in enumerate(prefixes_bitmask_reassigned_id):
            if ith_prefix == 0:  # skip empty set
                dp_stages[i] = 0
                dp_stages_from[i] = -1
                continue

            # logger.info(f'DP calculating prefix {i}...')

            for j, jth_prefix in enumerate(prefixes_bitmask_reassigned_id):
                if i != j and ith_prefix & jth_prefix == jth_prefix:  # jth_prefix is a subset of ith_prefix
                    logger.debug(f"Calclulating cost between prefix {j} and prefix {i}")

                    s = ith_prefix - jth_prefix
                    s = [i for i in range(number_of_conv_nodes) if s >> i & 1 == 1]  # s is a list of reassigned id

                    # cost: cycle count
                    # alloc: alloca[i][j][k]=(x, y) represents the i-th node's j-th replicate's k-th core are allocated to core (x, y) on chip
                    # nodes_reassigned_id: A list of reassigned id, consists of the same nodes as s, but reordered in the process of calculating the best strategy. Its order now corresponds to the order of alloc[].
                    # cores_needed_list: A list of list of cores needed for each node in nodes_reassigned_id. cores_needed_list[i] = len(alloc[i])
                    # communicate_on_chip_edgeset: A set of edges that need to communicate on chip. Used for generating instructions. Other edges not in this edgeset use global memory to communicate.
                    cost, alloc, nodes_reassigned_id, cores_needed_list, communicate_on_chip_edgeset = calc_best_strategy_on_chip(
                        s,
                        reassigned_id_graph,
                        reassigned_id_rev_graph,
                        reassigned_id_graph_edgeset,
                        reassigned_id_to_node_id,
                        input_data_conv_node_reassigned_id,
                        output_data_conv_node_reassigned_id,
                        onnx_graph)

                    if dp_stages[j] + cost < dp_stages[i]:
                        dp_stages[i] = dp_stages[j] + cost
                        dp_stages_from[i] = j
                        dp_stages_alloc_info[i] = (
                            alloc,
                            nodes_reassigned_id,
                            cores_needed_list,
                            communicate_on_chip_edgeset)

            logger.info(f'DP state of prefix {i} is transferred from prefix {dp_stages_from[i]}')

        cur_prefix_idx = len(prefixes_bitmask_reassigned_id) - 1  # last prefix, which is the full set
        while prefixes_bitmask_reassigned_id[cur_prefix_idx] != 0:
            next_prefix = dp_stages_from[cur_prefix_idx]
            stages.append(dp_stages_alloc_info[cur_prefix_idx])
            cur_prefix_idx = next_prefix

        stages.reverse()
    else:
        # Naive greedy: do a topsort and try to put as many nodes as possible into one stage until cores are used up
        # Should be worse than DP

        logger.info("Greedy started")

        sorted_nodes_reassigned_id = topsort(
            [[reassigned_id_rev_graph[i], reassigned_id_graph[i]] for i in range(len(reassigned_id_graph))])
        cur_prefix_idx = 0
        while cur_prefix_idx < len(sorted_nodes_reassigned_id):
            last_prefix_idx = cur_prefix_idx
            while cur_prefix_idx < len(sorted_nodes_reassigned_id):
                cost, alloc, nodes_reassigned_id, cores_needed_list, communicate_on_chip_edgeset = calc_best_strategy_on_chip([sorted_nodes_reassigned_id[i] for i in range(
                    last_prefix_idx, cur_prefix_idx + 1)], reassigned_id_graph, reassigned_id_rev_graph, reassigned_id_graph_edgeset, reassigned_id_to_node_id, input_data_conv_node_reassigned_id, output_data_conv_node_reassigned_id, onnx_graph)
                if cost == math.inf:
                    break
                else:
                    cur_prefix_idx += 1
            cur_prefix_idx -= 1
            cost, alloc, nodes_reassigned_id, cores_needed_list, communicate_on_chip_edgeset = calc_best_strategy_on_chip([sorted_nodes_reassigned_id[i] for i in range(
                last_prefix_idx, cur_prefix_idx + 1)], reassigned_id_graph, reassigned_id_rev_graph, reassigned_id_graph_edgeset, reassigned_id_to_node_id, input_data_conv_node_reassigned_id, output_data_conv_node_reassigned_id, onnx_graph)
            stages.append(
                (alloc,
                 nodes_reassigned_id,
                 cores_needed_list,
                 communicate_on_chip_edgeset))
            cur_prefix_idx += 1

    logger.info(f'Stage partitioning completed...')
    logger.info(f'Number of stages: {len(stages)}')
    logger.info(f'Stages: (format: [reassigned id]-[cores needed for every replica]-[replicate times])')
    for stage in stages:
        logger.info('[' +
                    ', '.join([
                        f"{stage[1][i]}-{len(stage[0][i][0])}-{len(stage[0][i])}"
                        for i in range(len(stage[0]))])
                    + ']')

    instructions = {f'core_{i}_{j}': {'stages': {}} for i in range(c_conf.P) for j in range(c_conf.Q)}
    sorted_nodes = topsort(graph)

    if c_conf.visualize_flag:
        logger.info('Visualizing computation graph...')
        if number_of_conv_nodes > 0:  # Check if there's anything to visualize
            visualize_computation_graph(reassigned_id_graph, stages, reassigned_id_to_node_id, output_dir=c_conf.plots_output_path, output_filename=c_conf.plot_output_filename)
        else:
            logger.info("No convolution nodes to visualize in the graph.")

        logger.info(f'Visualizing core allocation for {len(stages)} stages...')
        for stage_idx, stage_content in enumerate(stages):
            alloc, nodes_reassigned_id_in_stage, _, _ = stage_content
            if not alloc:  # Skip if allocation is empty for some reason
                logger.info(f"Skipping visualization for empty Stage {stage_idx}")
                continue

            visualize_chip_allocation_per_stage(
                alloc_info=alloc,
                nodes_in_stage_reassigned_ids=nodes_reassigned_id_in_stage,
                p_cores=c_conf.P,  # Assuming c_conf is imported and P, Q are attributes
                q_cores=c_conf.Q,
                stage_idx=stage_idx,
                reassigned_id_to_node_id=reassigned_id_to_node_id,
                output_dir=c_conf.plots_output_path,
                output_filename_template=c_conf.plot_output_filename_template
            )

    logger.info(f'Generating instructions...')
    for stageid, stage in enumerate(stages):
        alloc, nodes_reassigned_id, cores_needed_list, communicate_on_chip_edgeset = stage
        logger.info(f'Generating instructions for stage {stageid}/{len(stages)}...')
        instruction_cur = get_instrctions_for_a_stage(
            stageid,
            alloc,
            nodes_reassigned_id,
            communicate_on_chip_edgeset,
            reassigned_id_graph,
            reassigned_id_rev_graph,
            reassigned_id_graph_edgeset,
            reassigned_id_to_node_id,
            input_data_conv_node_reassigned_id,
            output_data_conv_node_reassigned_id,
            graph,
            belong_node,
            sorted_nodes,
            onnx_graph
        )

        for i in range(c_conf.P):
            for j in range(c_conf.Q):
                core_name = f'core_{i}_{j}'
                instructions[core_name]['stages'][str(stageid)] = {
                    'cluster_id': instruction_cur[core_name]['cluster_id'],
                    'weight_replica_id': instruction_cur[core_name]['weight_replica_id'],
                    'instructions': instruction_cur[core_name]['instructions']
                }
    logger.info(f'Instructions generated.')
    return instructions
