import heapq
from collections import Counter, namedtuple


# A simple Node for the Huffman tree
class Node(namedtuple("Node", ["freq", "value", "left", "right"])):
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(weights):
    counter = Counter(weights)
    heap = [Node(freq, value, None, None) for value, freq in counter.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        new_node = Node(left.freq + right.freq, None, left, right)
        heapq.heappush(heap, new_node)

    return heap[0]


def assign_codes(node, prefix="", codebook={}):
    if node.value is not None:
        codebook[node.value] = prefix
    else:
        assign_codes(node.left, prefix + "0", codebook)
        assign_codes(node.right, prefix + "1", codebook)
    return codebook


# TODO(frneer): Add tests for the Huffman tree
# test_weights = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
# huffman_tree = build_huffman_tree(test_weights)
# # print(huffman_tree)
# huffman_codes = assign_codes(huffman_tree)
# # print("Huffman Codes:", [len(code) for idx, code in huffman_codes.items()])
