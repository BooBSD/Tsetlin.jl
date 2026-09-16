# include("../../src/TsetlinBitPlanes.jl")
# include("../../src/TsetlinBitPlanes2.jl")
# include("../../src/TsetlinBitPlanes3.jl")
include("../../src/TsetlinBitPlanes4.jl")
# include("../../src/TsetlinBitPlanes5.jl")

import Pkg
[Base.find_package(p) === nothing && Pkg.add(p) for p in ["MLDatasets"]]

using MLDatasets: MNIST, FashionMNIST
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy, save, load, unzip, booleanize, benchmark, compile

x_train, y_train = unzip([MNIST(:train)...])
x_test, y_test = unzip([MNIST(:test)...])
# x_train, y_train = unzip([FashionMNIST(:train)...])
# x_test, y_test = unzip([FashionMNIST(:test)...])

# 1-bit booleanization
x_train = [booleanize(x, 0.2) for x in x_train]
x_test = [booleanize(x, 0.2) for x in x_test]
# 4-bit booleanization
# x_train = [booleanize(x, 0, 0.25, 0.5, 0.75) for x in x_train]
# x_test = [booleanize(x, 0, 0.25, 0.5, 0.75) for x in x_test]

# Convert y_train and y_test to the Int8 type to save memory
y_train = Int8.(y_train)
y_test = Int8.(y_test)

const PLANES = 8

CLAUSES = 20
T = 20
S = 400
L = 150
LF = 75

# CLAUSES = 8
# T = 13
# S = 392 * 2
# L = 196
# LF = 98

# CLAUSES = 16
# T = 17
# S = 392 * 1
# L = 196
# LF = 98

# CLAUSES = 20
# T = 20
# S = 392 * 2
# L = 196
# LF = 98

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

EPOCHS = 1000

MODEL_PATH = joinpath(tempdir(), "tm.tm")

# Training the TM model
tm = TMClassifier(x_train[1], y_train, CLAUSES, T, S, L, LF, planes=PLANES)
tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, exclusive_literals=false, best_tms_compile=false, best_tms_size=0, shuffle=true)

exit()

# save(tms[1][1], MODEL_PATH)
tm = load(MODEL_PATH)
tmc = tm

# Compiling model
# tmc = compile(tm)
#save(tmc, MODEL_PATH)

# accuracy(predict(tmc, x_test), y_test) |> println

# Benchmark
benchmark(tmc, x_test, y_test, 1000 * 32, warmup=true)
# benchmark(tmc, x_test, y_test, 1000 * 2, warmup=true)

exit()


function compress(tm::TMClassifier)::TMClassifier
    tm = deepcopy(tm)
    # for clauses in tm.clauses
    #     clauses.positive_included_literals = reduce((x, y) -> x | y, clauses.positive_included_literals, dims=2, init=typemin(UInt64))
    #     clauses.positive_included_literals_inverted = reduce((x, y) -> x & y, clauses.positive_included_literals_inverted, dims=2, init=typemax(UInt64))
    #     clauses.negative_included_literals = reduce((x, y) -> x | y, clauses.negative_included_literals, dims=2, init=typemin(UInt64))
    #     clauses.negative_included_literals_inverted = reduce((x, y) -> x & y, clauses.negative_included_literals_inverted, dims=2, init=typemax(UInt64))
    # end
    # for clauses in tm.clauses
    #     clauses.positive_included_literals = reduce((x, y) -> x & y, clauses.positive_included_literals, dims=2, init=typemax(UInt64))
    #     clauses.positive_included_literals_inverted = reduce((x, y) -> x | y, clauses.positive_included_literals_inverted, dims=2, init=typemin(UInt64))
    #     clauses.negative_included_literals = reduce((x, y) -> x & y, clauses.negative_included_literals, dims=2, init=typemax(UInt64))
    #     clauses.negative_included_literals_inverted = reduce((x, y) -> x | y, clauses.negative_included_literals_inverted, dims=2, init=typemin(UInt64))
    # end
    for clauses in tm.clauses
        clauses.positive_included_literals = reduce((x, y) -> x | y, clauses.positive_included_literals, dims=2, init=typemin(UInt64))
        clauses.positive_included_literals_inverted = reduce((x, y) -> x | y, clauses.positive_included_literals_inverted, dims=2, init=typemin(UInt64))
        clauses.negative_included_literals = reduce((x, y) -> x | y, clauses.negative_included_literals, dims=2, init=typemin(UInt64))
        clauses.negative_included_literals_inverted = reduce((x, y) -> x | y, clauses.negative_included_literals_inverted, dims=2, init=typemin(UInt64))
    end
    # for clauses in tm.clauses
    #     clauses.positive_included_literals = reduce((x, y) -> x & y, clauses.positive_included_literals, dims=2, init=typemax(UInt64))
    #     clauses.positive_included_literals_inverted = reduce((x, y) -> x & y, clauses.positive_included_literals_inverted, dims=2, init=typemax(UInt64))
    #     clauses.negative_included_literals = reduce((x, y) -> x & y, clauses.negative_included_literals, dims=2, init=typemax(UInt64))
    #     clauses.negative_included_literals_inverted = reduce((x, y) -> x & y, clauses.negative_included_literals_inverted, dims=2, init=typemax(UInt64))
    # end
    # for ta in tm.clauses
    #     a = reduce((x, y) -> x | y, clauses.positive_included_literals, dims=2, init=typemin(UInt64))
    #     b = reduce((x, y) -> x & y, clauses.positive_included_literals, dims=2, init=typemax(UInt64))
    #     clauses.positive_included_literals = [a b]
    #     a = reduce((x, y) -> x | y, clauses.positive_included_literals_inverted, dims=2, init=typemin(UInt64))
    #     b = reduce((x, y) -> x & y, clauses.positive_included_literals_inverted, dims=2, init=typemax(UInt64))
    #     clauses.positive_included_literals_inverted = [a b]
    #     a = reduce((x, y) -> x | y, clauses.negative_included_literals, dims=2, init=typemin(UInt64))
    #     b = reduce((x, y) -> x & y, clauses.negative_included_literals, dims=2, init=typemax(UInt64))
    #     clauses.negative_included_literals = [a b]
    #     a = reduce((x, y) -> x | y, clauses.negative_included_literals_inverted, dims=2, init=typemin(UInt64))
    #     b = reduce((x, y) -> x & y, clauses.negative_included_literals_inverted, dims=2, init=typemax(UInt64))
    #     clauses.negative_included_literals_inverted = [a b]
    # end
    return tm
end


# accuracy(predict(tmc, x_test), y_test) |> println
# tmcc = compress(tmc)
# accuracy(predict(tmcc, x_test), y_test) |> println

# save(tm, "/tmp/tm.tm")
# save(tmc, "/tmp/tmc.tm")
# save(tmcc, "/tmp/tmcc.tm")

# for LF in 1:1000
#     tmcc.LF = LF
#     (LF, accuracy(predict(tmcc, x_test), y_test)) |> println
# end
