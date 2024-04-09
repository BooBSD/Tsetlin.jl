include("../src/Tsetlin.jl")

try
    using MLDatasets: MNIST
catch LoadError
    import Pkg
    Pkg.add("MLDatasets")
end

using MLDatasets: MNIST
using .Tsetlin: TMInput, benchmark, load, unzip

x_test, y_test = unzip([MNIST(split=:test)...])

# Booleanization
x_test = [TMInput(vec([
    [x > 0 ? true : false for x in i];
    [x > 0.33 ? true : false for x in i];
    [x > 0.66 ? true : false for x in i];
])) for i in x_test]

tm_opt = load("./models/tm_optimized_72.tm")

# Be careful; using swap will drastically decrease benchmark performance!
# Please close all other programs such as web browsers and monitor the available memory.
# The figure 1600 represents 8.76 GB (calculated as 28 * 28 * 2 * 3 * 8 * 10000 * 1600 / 64 / 1024^3)
# of prepared input data in memory without accounting for overhead from data structures.
# For the MNIST test dataset, 6400 corresponds to 35 GB of prepared input data.
# You need at least 16GB RAM to run this benchmark.
benchmark(tm_opt, x_test, y_test, 1600, batch=true, warmup=true, deep_copy=true)
