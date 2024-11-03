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
