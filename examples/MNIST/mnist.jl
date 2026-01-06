include("../../src/Tsetlin.jl")

import Pkg
[Base.find_package(p) === nothing && Pkg.add(p) for p in ["MLDatasets"]]

using MLDatasets: MNIST, FashionMNIST
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy, save, load, unzip, booleanize, benchmark, compile


x_train, y_train = unzip([MNIST(:train)...])
x_test, y_test = unzip([MNIST(:test)...])
# x_train, y_train = unzip([FashionMNIST(:train)...])
# x_test, y_test = unzip([FashionMNIST(:test)...])

# 4-bit booleanization
x_train = [booleanize(x, 0, 0.25, 0.5, 0.75) for x in x_train]
x_test = [booleanize(x, 0, 0.25, 0.5, 0.75) for x in x_test]
# x_train = [booleanize(x, 0.25) for x in x_train]
# x_test = [booleanize(x, 0.25) for x in x_test]

# Convert y_train and y_test to the Int8 type to save memory
y_train = Int8.(y_train)
y_test = Int8.(y_test)


CLAUSES = 20
T = 20
S = 200
L = 150
LF = 75

# CLAUSES = 200
# T = 20
# S = 200
# L = 16
# LF = 8

# CLAUSES = 512
# T = 32
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

EPOCHS = 1000

# Training the TM model
tm = TMClassifier(x_train[1], y_train, CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=240)
train!(tm, x_train, y_train, x_test, y_test, EPOCHS, index=false)

save(tm, "/tmp/tm.tm")
tm = load("/tmp/tm.tm")

# Compiling model
tmc = compile(tm)

# Benchmark
benchmark(tmc, x_test, y_test, 1000 * 2, warmup=true, index=false)
