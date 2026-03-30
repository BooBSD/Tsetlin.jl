# Source: https://github.com/cair/HierarchicalTsetlinMachine/blob/main/NoisyParityData.py
# License: MIT © 2026 Ole-Christoffer Granmo and the University of Agder


import numpy as np


NOISE = 0.2

# 8-bit XOR

NUMBER_OF_FEATURES = 12
NUMBER_OF_VARIABLES = 4
NUMBER_OF_EXAMPLES = 20_000

# 16-bit XOR

# NUMBER_OF_FEATURES = 24
# NUMBER_OF_VARIABLES = 8
# NUMBER_OF_EXAMPLES = 100_000


def generate_dataset(
    number_of_features: int,
    number_of_variables: int,
    number_of_examples: int,
    noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.random.randint(2, size=(number_of_examples, number_of_features), dtype=np.uint32)
    Y = np.zeros(number_of_examples, dtype=np.uint32)

    for i in range(number_of_examples):
        set_bit_count = 0
        for j in range(number_of_variables):
            start = j * number_of_features // number_of_variables
            set_bit_count += X[i, start : start + 2].sum()
        Y[i] = set_bit_count % 2

    if noise > 0:
        Y = np.where(np.random.rand(number_of_examples) <= noise, 1 - Y, Y) # Adds noise

    return X, Y


X_train, Y_train = generate_dataset(NUMBER_OF_FEATURES, NUMBER_OF_VARIABLES, NUMBER_OF_EXAMPLES, noise=NOISE)
X_test, Y_test = generate_dataset(NUMBER_OF_FEATURES, NUMBER_OF_VARIABLES, NUMBER_OF_EXAMPLES)

np.savetxt("/tmp/NoisyParityTrainingData.txt", np.hstack([X_train, Y_train.reshape(-1, 1)]), fmt='%d')
np.savetxt("/tmp/NoisyParityTestingData.txt", np.hstack([X_test, Y_test.reshape(-1, 1)]), fmt='%d')
