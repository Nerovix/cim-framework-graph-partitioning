import math
import onnx
from onnx import shape_inference
from onnx import GraphProto, ModelProto, NodeProto
import cimpara as cp
from read_file import get_tensor_shape

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


def calc_cores_and_time_needed(graph, node):
    if node.op_type == 'Conv':
        [N, C_in, H, W] = get_tensor_shape(graph, node.input[0])
        [C_out, C_in, K_h, K_w] = get_tensor_shape(graph, node.input[1])

        N = 1

        S = [attr.ints for attr in node.attribute if attr.name == 'strides'][0]
        P = [attr.ints for attr in node.attribute if attr.name == 'pads'][0]
        if not S:
            S = [1, 1]
        if not P:
            P = [0, 0, 0, 0]

        print("conv layer:")
        print("[N, C_in, H, W]" + str([N, C_in, H, W]))
        print("[C_out, C_in, K_h, K_w] " + str([C_out, C_in, K_h, K_w]))
        print("S" + str(S))
        print("P" + str(P))

        H_out = (H + P[0] + P[2] - K_h) // S[0] + 1
        W_out = (W + P[1] + P[3] - K_w) // S[1] + 1

        # print(H_out)
        # print(W_out)

        # matA = [N * H_out * W_out, C_in * K_h * K_w]
        # matB = [C_in * K_h * K_w, C_out]

        if cp.H * cp.m * cp.T < K_h * K_w * C_in:
            raise Exception('holy the conv op is tooooooo big')

        channels_on_a_core = cp.n * cp.W // cp.weight_width * cp.K
        cores_needed = math.ceil(C_out / channels_on_a_core)

        time_needed = N * H_out * W_out * cp.m * cp.activation_width

        return cores_needed, time_needed

    raise Exception('you must be joking, this is NOT a conv op')


# ABANDONED
# def put_nodes_on_chip(cores_needed):
#     # 假定核的个数都是偶数，如果是奇数，给他+1
#     cores_needed = [x + 1 if x % 2 == 1 else x for x in cores_needed]
#     line = 0  # 现在的行组
#     col = 0  # 现在的列（下一个空的）
#     ty = 0  # type

#     P = cp.P
#     Q = cp.Q

#     '''
#     ty == 0, 正常
#     ...###__...
#     ...###__...

#     ty == 1, 刚从上一个行组下来，拐弯消耗了上两格
#     ####...
#     ####...
#     ##__...
#     ____...

#     '''

#     allocation = []
#     for c in cores_needed:
#         c //= 2
#         allo = []
#         if c == 1:
#             # 如果只需要两个核，可能能挤在ty == 1 的碎片中，特殊考虑
#             if ty == 0:
#                 if line % 2 == 0:  # ->
#                     allo.append((line * 2, col))
#                     allo.append((line * 2 + 1, col))
#                     col += 1
#                     if col == Q:
#                         col = Q - 1
#                         line += 1
#                 else:
#                     allo.append((line * 2, col))
#                     allo.append((line * 2 + 1, col))
#                     col -= 1
#                     if col == -1:
#                         col = 0
#                         line += 1
#             else:
#                 if line % 2 == 0:  # ->
#                     allo.append((line * 2 + 1, 0))
#                     allo.append((line * 2 + 1, 1))
#                     col = 2
#                     ty = 0
#                 else:
#                     allo.append((line * 2 + 1, Q - 1))
#                     allo.append((line * 2 + 1, Q - 2))
#                     col = Q - 3
#                     ty = 0
#         else:
#             # 如果需要多个核，先排掉type 1或者type 0且在最后两行形成的碎片
#             if ty == 0:
#                 if line % 2 == 0 and col == Q - 1:
#                     line += 1
#                     col = Q - 1
#                 elif line % 2 == 1 and col == 0:
#                     line += 1
#                     col = 0
#             else:
#                 if line % 2 == 0:
#                     col = 2
#                     ty = 0
#                 else:
#                     col = Q - 3
#                     ty = 0
#             # 然后开始两个两个往上摆
#             while c > 0:
#                 if ty == 0:
#                     if line % 2 == 0:  # ->
#                         if col == 0:
#                             ty = 1
#                             allo.append((line * 2, 0))
#                             allo.append((line * 2, 1))
#                         else:
#                             allo.append((line * 2, col))
#                             allo.append((line * 2 + 1, col))
#                             col += 1
#                             if col == Q:
#                                 col = Q - 1
#                                 line += 1
#                     else:
#                         if col == Q - 1:
#                             ty = 1
#                             allo.append((line * 2, Q - 1))
#                             allo.append((line * 2, Q - 2))
#                         else:
#                             allo.append((line * 2, col))
#                             allo.append((line * 2 + 1, col))
#                             col -= 1
#                             if col == -1:
#                                 col = 0
#                                 line += 1
#                 else:
#                     if line % 2 == 0:  # ->
#                         allo.append((line * 2 + 1, 0))
#                         allo.append((line * 2 + 1, 1))
#                         col = 2
#                         ty = 0
#                     else:
#                         allo.append((line * 2 + 1, Q - 1))
#                         allo.append((line * 2 + 1, Q - 2))
#                         col = Q - 3
#                         ty = 0
#                 c -= 1
#         for (x, y) in allo:
#             if x >= P:
#                 return None
#         allocation.append(allo)

#     return allocation


def print_allocation(allocation):
    chip = [[-1] * cp.Q for _ in range(cp.P)]
    for i in range(len(allocation)):
        for (x, y) in allocation[i]:
            chip[x][y] = i

    for row in chip:
        formatted = [f"{i:2d}" for i in row]
        print(" ".join(formatted))


def calc_best_strategy_on_chip():
    pass