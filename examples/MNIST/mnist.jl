include("../../src/Tsetlin.jl")

import Pkg
[Base.find_package(p) === nothing && Pkg.add(p) for p in ["MLDatasets"]]

using MLDatasets: MNIST, FashionMNIST
using .Tsetlin: TMClassifier, InputBatch, train!, save, load, unzip, booleanize, benchmark, compile


DATASET = MNIST
# DATASET = FashionMNIST

MODEL_PATH = joinpath(tempdir(), "tm.tm")

THRESHOLDS = (0.2)                  # 1-bit booleanization
# THRESHOLDS = (0, 0.25, 0.5, 0.75)   # 4-bit booleanization

STATES_NUM = 256
INCLUDE_LIMIT = 128
INDEX = false
SHUFFLE = false
EPOCHS = 1000

CLAUSES = 20
T = 16
S = 800
L = 150
LF = 75

# CLAUSES = 200
# T = 28
# S = 200
# L = 16
# LF = 8

# CLAUSES = 512
# T = 45
# S = 200
# L = 16
# LF = 8

# CLAUSES = 2000
# T = 64
# S = 400
# L = 12
# LF = 4

# CLAUSES = 40
# T = 10
# S = 125
# L = 10
# LF = 5

x_train, y_train = unzip([DATASET(:train)...])
x_test, y_test = unzip([DATASET(:test)...])

input_size = length(first(x_train)) * length(THRESHOLDS)

x_train = InputBatch([booleanize(x, THRESHOLDS...) for x in x_train])
x_test = InputBatch([booleanize(x, THRESHOLDS...) for x in x_test])

# Convert y_train and y_test to the Int8 type to save memory
y_train = Int8.(y_train)
y_test = Int8.(y_test)

# Training the TM model
tm = TMClassifier(input_size, y_train, CLAUSES, T, S, L, LF, states_num=STATES_NUM, include_limit=INCLUDE_LIMIT)
train!(tm, x_train, y_train, x_test, y_test, EPOCHS, index=INDEX, shuffle=SHUFFLE)

# Compiling model
tmc = compile(tm)

save(tmc, MODEL_PATH)
tmc = load(MODEL_PATH)

# Benchmark
benchmark(tmc, x_test, y_test, 1000 * 10, index=INDEX)
