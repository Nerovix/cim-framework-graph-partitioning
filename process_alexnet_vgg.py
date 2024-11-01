from graph import build_graph, getlayer_by_conv_op, build_dependency_bitmask_re_id, find_all_prefixes
from calc_cost import calc_cores_and_time_needed, calc_best_strategy_on_chip, print_allocation
from read_file import get_tensor_shape


def process_alexnet_vgg(onnx_graph):
    graph = build_graph(onnx_graph)

    is_conv_node = []

    for n in onnx_graph.node:
        if n.op_type == 'Conv' and [
                attr.i for attr in n.attribute if attr.name == 'group'][0] == 1:
            # a pointwise conv
            is_conv_node.append(1)
        else:
            # not pointwise conv, put on simd
            is_conv_node.append(0)

    print([i for i,v in enumerate(is_conv_node) if v==1])

    conv_node_re_id = [0] * len(is_conv_node)  # 0-based
    re_id_to_node_id=[]
    id_cnt = 0
    for i in range(len(is_conv_node)):
        if is_conv_node[i]:
            conv_node_re_id[i] = id_cnt
            re_id_to_node_id.append(i)
            id_cnt += 1

    print(id_cnt)


    dependency_bitmask_re_id = build_dependency_bitmask_re_id(
        graph, is_conv_node, conv_node_re_id)
    
    for i in range(len(dependency_bitmask_re_id)):
        print(str(re_id_to_node_id[i])+' depends on '+str([re_id_to_node_id[j] for j in range(len(dependency_bitmask_re_id)) if dependency_bitmask_re_id[i]>>j&1==1]))
    
    prefixes_bitmask_re_id = find_all_prefixes(dependency_bitmask_re_id)
    
    for i,v in enumerate(prefixes_bitmask_re_id):
        print(v.bitcount())
        print([re_id_to_node_id[i] for i in range(len(dependency_bitmask_re_id)) if v>>i&1==1])
        