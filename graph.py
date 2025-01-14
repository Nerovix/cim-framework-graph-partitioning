from collections import deque
from logging_config import logger

# Returns a [([in_nodes],[out_nodes])] to represent a DAG. 
# The number corresponds to graph.node one by one
# The result DAG simply reflects the node dependency.
def build_graph(onnx_graph):

    graph = []
    output_from = {}
    for i, node in enumerate(onnx_graph.node):
        graph.append([[], []])
        for output_name in node.output:
            assert output_from.get(output_name) is None
            output_from[output_name] = i

    for i, node in enumerate(onnx_graph.node):
        assert len(
            node.output) <= 1, "Every node should have only one output tensor but it seems some nodes do not."
        for input_name in node.input:
            if output_from.get(input_name) is not None:
                graph[i][0].append(output_from[input_name])
                graph[output_from[input_name]][1].append(i)
    return graph


# The add node of the residual must be on the main chain (the longest chain), otherwise there will be problems.
# So first we do a DP to find the main chain.
# Put FC layers last, put the rest of the nodes before the FC layers on a near convolution node.
def get_belong_node(graph, is_conv_node, is_fc_node):
    logger.info('Finding belong node...')
    indeg = [len(g[0]) for g in graph]
    topsort_queue = deque([i for i, v in enumerate(indeg) if v == 0])
    last = [-1] * len(graph)
    dp_longest_chain = [1] * len(graph)
    while topsort_queue:
        cur_node = topsort_queue.popleft()
        for next_node in graph[cur_node][1]:
            if dp_longest_chain[next_node] < dp_longest_chain[cur_node] + 1:
                dp_longest_chain[next_node] = dp_longest_chain[cur_node] + 1
                last[next_node] = cur_node
            indeg[next_node] -= 1
            if indeg[next_node] == 0:
                topsort_queue.append(next_node)
    main_chain = []
    cur_node = 0
    for i in range(len(graph)):
        if dp_longest_chain[i] > dp_longest_chain[cur_node]:
            cur_node = i
    while cur_node != -1:
        main_chain.append(cur_node)
        cur_node = last[cur_node]

    main_chain.reverse()
    in_main_chain = [0] * len(graph)
    for i in main_chain:
        in_main_chain[i] = 1

    # print(main_chain)
    main_chain_edges = dict()
    for i in range(len(main_chain) - 1):
        main_chain_edges[(main_chain[i], main_chain[i + 1])] = True
        main_chain_edges[(main_chain[i + 1], main_chain[i])] = True

    # Add nodes should be on the main chain
    # Calculate the nearest convolution nodes for every node on the main chain, and set its belong_node to the nearest conlolution nodes.
    belong_node = [-1] * len(graph)
    topsort_queue = deque([i for i in range(len(graph)) if is_conv_node[i]
              == 1 and in_main_chain[i] == 1])
    while topsort_queue:
        cur_node = topsort_queue.popleft()
        if belong_node[cur_node] == -1:
            belong_node[cur_node] = cur_node
        for next_node in graph[cur_node][0] + graph[cur_node][1]:
            if belong_node[next_node] != -1:
                continue
            if (cur_node, next_node) in main_chain_edges and is_fc_node[next_node] == 0:
                belong_node[next_node] = belong_node[cur_node]
                topsort_queue.append(next_node)

    # The rest should be the residual convolution nodes and non-convolution nodes near them,
    # Set the belong_node of the non-convolution nodes as their coresponding convolution nodes.
    topsort_queue = deque([i for i in range(len(graph)) if is_conv_node[i]
              == 1 and in_main_chain[i] == 0])
    while topsort_queue:
        cur_node = topsort_queue.popleft()
        if belong_node[cur_node] == -1:
            belong_node[cur_node] = cur_node
        for next_node in graph[cur_node][0] + graph[cur_node][1]:
            if belong_node[next_node] != -1:
                continue
            if (cur_node, next_node) not in main_chain_edges and is_fc_node[next_node] == 0:
                belong_node[next_node] = belong_node[cur_node]
                topsort_queue.append(next_node)


    # If this is done once and there are still nodes left, then there is a residual branch without convolution
    # Normally, a residual branch either has no operator or must have at least one convolution
    # but anyway we put this into consideration as well
    topsort_queue = deque([i for i in range(len(graph)) if belong_node[i] != -1])
    while topsort_queue:
        cur_node = topsort_queue.popleft()
        if belong_node[cur_node] == -1:
            belong_node[cur_node] = cur_node
        for next_node in graph[cur_node][0] + graph[cur_node][1]:
            if belong_node[next_node] == -1 and is_fc_node[next_node] == 0:
                belong_node[next_node] = belong_node[cur_node]
                topsort_queue.append(next_node)

    # Finally, FC layers
    topsort_queue = deque([i for i in range(len(graph)) if is_fc_node[i] == 1])
    while topsort_queue:
        cur_node = topsort_queue.popleft()
        if belong_node[cur_node] == -1:
            belong_node[cur_node] = cur_node
        for next_node in graph[cur_node][0] + graph[cur_node][1]:
            if belong_node[next_node] == -1:
                belong_node[next_node] = belong_node[cur_node]
                topsort_queue.append(next_node)

    assert min(belong_node) >= 0 # no -1 in belong_node, every node belongs to certain convolution node or FC node
    logger.info('Found belong node.')
    return belong_node

# Do a simple DFS to find all dependency closure for dp.
def find_all_prefixes(graph_reassigned_id):
    logger.info('Finding all prefixes...')
    node_cnt = len(graph_reassigned_id)
    indeg = [0] * node_cnt
    for i in range(node_cnt):
        for j in graph_reassigned_id[i]:
            indeg[j] += 1

    prefixes = dict()

    def dfs(bitmask):
        if prefixes.get(bitmask) is not None:
            return
        logger.debug(f'DFS: prefix bitmask = {bitmask}')
        prefixes[bitmask] = True
        for i in range(node_cnt):
            if bitmask >> i & 1 == 0 and indeg[i] == 0:
                for j in graph_reassigned_id[i]:
                    indeg[j] -= 1
                dfs(bitmask ^ (1 << i))
                for j in graph_reassigned_id[i]:
                    indeg[j] += 1

    dfs(0)
    prefixes_list = list(prefixes.keys())
    prefixes_list.sort()
    logger.info(f'Found {len(prefixes_list)} prefixes.')
    return prefixes_list


# split the part of the graph into chains
def split_to_chain(conv_node_reassigned_id, reassigned_id_graph_edgeset):
    nodecnt = len(conv_node_reassigned_id)
    simplified_graph = [[] for _ in range(nodecnt)]
    indeg = [0] * nodecnt
    used = [False] * nodecnt
    usedcnt = 0
    for i in range(nodecnt):
        for j in range(nodecnt):
            if reassigned_id_graph_edgeset.get(
                    (conv_node_reassigned_id[i], conv_node_reassigned_id[j])) is not None:
                simplified_graph[i].append(j)
                indeg[j] += 1
    res = []
    while usedcnt < nodecnt:
        # Just a simple greedy strategy: choose the longest chain every time
        topsort_queue = deque()
        dis = [0] * nodecnt
        last = [-1] * nodecnt
        for i in range(nodecnt):
            if indeg[i] == 0 and not used[i]:
                topsort_queue.append(i)
                dis[i] = 1
                last[i] = -1
        while (topsort_queue):
            cur_node = topsort_queue.popleft()
            for next_node in simplified_graph[cur_node]:
                if used[next_node]:
                    continue
                if dis[next_node] < dis[cur_node] + 1:
                    dis[next_node] = dis[cur_node] + 1
                    last[next_node] = cur_node
                indeg[next_node] -= 1
                if indeg[next_node] == 0:
                    topsort_queue.append(next_node)
        ed = -1
        for i in range(nodecnt):
            if not used[i]:
                if ed == -1 or dis[i] > dis[ed]:
                    ed = i
        m = dis[ed]
        chain = []
        while ed != -1:
            chain.append(conv_node_reassigned_id[ed])
            used[ed] = True
            ed = last[ed]

        assert (len(chain) == m)
        usedcnt += len(chain)
        chain.reverse()
        res.append(chain)
    return res

# simple topsort
def topsort(graph):
    ind = [len(_[0]) for _ in graph]
    queue = deque([i for i, v in enumerate(ind) if v == 0])

    ans = []

    while queue:
        cur_node = queue.popleft()
        ans.append(cur_node)
        for next_node in graph[cur_node][1]:
            ind[next_node] -= 1
            if ind[next_node] == 0:
                queue.append(next_node)

    return ans
