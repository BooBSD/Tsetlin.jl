# Source: https://github.com/cair/HierarchicalTsetlinMachine/blob/main/NoisyParityData.py
# License: MIT © 2026 Ole-Christoffer Granmo and the University of Agder


import numpy as np


noise = 0.2
number_of_features = 12
number_of_variables = 4
number_of_examples = 20000

X_train = np.random.randint(2, size=(number_of_examples, number_of_features), dtype=np.uint32)
Y_train = np.zeros(number_of_examples, dtype=np.uint32)

for i in range(number_of_examples):
	for j in range(number_of_features):
		X_train[i, j] = np.random.randint(2)

	set_bit_count = 0
	for j in range(number_of_variables):
		set_bit_count += X_train[i, j * number_of_features // number_of_variables:j * number_of_features // number_of_variables + 2].sum()
	Y_train[i] = set_bit_count % 2

Y_train = np.where(np.random.rand(number_of_examples) <= noise, 1-Y_train, Y_train) # Adds noise
np.savetxt("/tmp/NoisyParityTrainingData.txt", np.append(X_train, Y_train.reshape((number_of_examples, 1)), axis=1), fmt='%d')

X_test = np.random.randint(2, size=(number_of_examples, number_of_features), dtype=np.uint32)
Y_test = np.zeros(number_of_examples, dtype=np.uint32)
for i in range(number_of_examples):
	for j in range(number_of_features):
		X_test[i, j] = np.random.randint(2)

	set_bit_count = 0
	for j in range(number_of_variables):
		set_bit_count += X_test[i, j * number_of_features // number_of_variables:j * number_of_features // number_of_variables + 2].sum()
	Y_test[i] = set_bit_count % 2

np.savetxt("/tmp/NoisyParityTestingData.txt", np.append(X_test, Y_test.reshape((number_of_examples, 1)), axis=1), fmt='%d')
