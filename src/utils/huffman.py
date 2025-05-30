from collections import Counter

import numpy as np


def compute_huffman_nominal_complexity(qweights):
    """Compute the nominal complexity of a huffman codification of the
    quantized weights."""
    N = qweights.shape.num_elements()
    counter = Counter(qweights.numpy().flatten())
    total = sum(counter.values())
    probabilities = np.array([freq / total for freq in counter.values()])
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return N * entropy
