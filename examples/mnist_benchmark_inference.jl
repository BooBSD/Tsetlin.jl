include("../src/Tsetlin.jl")

try
    using MLDatasets: MNIST
catch LoadError
    import Pkg
    Pkg.add("MLDatasets")
end

using MLDatasets: MNIST
using .Tsetlin: TMInput, benchmark, load, unzip

x_train, y_train = unzip([m for m in MNIST(split=:train)])
x_test, y_test = unzip([m for m in MNIST(split=:test)])

# Booleanization
x_train = [TMInput(vec([
    [if x > 0 true else false end for x in i];
    [if x > 0.33 true else false end for x in i];
    [if x > 0.66 true else false end for x in i];
])) for i in x_train]
x_test = [TMInput(vec([
    [if x > 0 true else false end for x in i];
    [if x > 0.33 true else false end for x in i];
    [if x > 0.66 true else false end for x in i];
])) for i in x_test]

tm_opt = load("./models/tm_optimized_72.tm")

# Be careful; using swap will drastically decrease benchmark performance!
# Please close all other programs like web browsers and monitor free memory.
# The figure 1600 represents 8.76 GB (calculated as 28 * 28 * 2 * 3 * 8 * 10000 * 1600 / 64 / 1024^3)
# of prepared input data in memory without accounting for overhead from data structures.
# For the MNIST test dataset, 6400 corresponds to 35 GB of prepared input data.
benchmark(tm_opt, x_test, y_test, 1600, batch=true, warmup=true, deep_copy=true)
