include("../../src/Tsetlin.jl")
include("../TEXT/HDC.jl")


import Pkg
[Base.find_package(p) === nothing && Pkg.add(p) for p in ["MLDatasets"]]

using Random
using Serialization
using Base.Threads
using MLDatasets: MNIST, FashionMNIST
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy, save, load, unzip, benchmark, compile


# x_train, y_train = unzip([MNIST(:train, Tx=Float32)...])
# x_test, y_test = unzip([MNIST(:test, Tx=Float32)...])
x_train, y_train = unzip([FashionMNIST(:train, Tx=Float32)...])
x_test, y_test = unzip([FashionMNIST(:test, Tx=Float32)...])

HV_PATH = "/tmp/hvectors_mnist"
HV_DIMENSIONS = 1024 * 64
BUNDLE_ACC_TYPE = Int16
BINARIZE_THRESHOLD = 0.05


if isfile(HV_PATH)
    hvectors, hvectors_x, hvectors_y, color = deserialize(HV_PATH)
else
    hvectors::Dict{Int, BitVector} = Dict()
    for hv in 1:length(x_train[1])
        hvectors[hv] = bitrand(HV_DIMENSIONS)
    end
    hvectors_x::Dict{Int, BitVector} = Dict()
    for hv in 1:size(x_train[1])[1]
        hvectors_x[hv] = bitrand(HV_DIMENSIONS)
    end
    hvectors_y::Dict{Int, BitVector} = Dict()
    for hv in 1:size(x_train[1])[2]
        hvectors_y[hv] = bitrand(HV_DIMENSIONS)
    end
    color = bitrand(HV_DIMENSIONS)
    serialize(HV_PATH, (hvectors, hvectors_x, hvectors_y, color))
end


# function booleanize(x::Matrix{Float32}, acc::Vector{T}, scratch::BitVector)::TMInput where T <: Integer
#     fill!(acc, zero(T))
#     @inbounds for i in eachindex(x)
#         hv = hvectors[i]
#         shift = ceil(Int, x[i] / BINARIZE_THRESHOLD)
#         circshift!(scratch, hv, shift)
#         bundle_add!(acc, scratch)
#     end
#     hv = binarize_bundle(acc)
#     return TMInput(hv.chunks, hv.len)
# end

# function booleanize(x::Matrix{Float32}, acc::Vector{T}, scratch::BitVector)::TMInput where T <: Integer
#     fill!(acc, zero(T))
#     @inbounds for i in eachindex(x)
#         hv = hvectors[i]
#         shift = ceil(Int, x[i] / BINARIZE_THRESHOLD)
#         @inbounds for s in 1:shift
#             circshift!(scratch, hv, s)
#             bundle_add!(acc, scratch)
#         end
#     end
#     hv = binarize_bundle(acc)
#     return TMInput(hv.chunks, hv.len)
# end

function booleanize(x::Matrix{Float32}, acc::Vector{T}, scratch::BitVector)::TMInput where T <: Integer
    fill!(acc, zero(T))
    # acc2 = copy(acc)
    @inbounds for (i, c) in enumerate(eachcol(x))
        X = hvectors_x[i]
        @inbounds for (j, p) in enumerate(c)
            Y = hvectors_y[j]
            hv = copy(X)
            bind!(hv, Y)
            shift = ceil(Int, p / BINARIZE_THRESHOLD)
            @inbounds for s in 1:shift
                circshift!(scratch, hv, s)
                bundle_add!(acc, scratch)
            end
        end
    end
    # @inbounds for (i, c) in enumerate(eachcol(x))
    #     X = hvectors_x[i]
    #     @inbounds for (j, p) in enumerate(c)
    #         Y = hvectors_y[j]
    #         fill!(acc2, zero(T))
    #         bundle_add!(acc2, X)
    #         bundle_add!(acc2, Y)
    #         shift = ceil(Int, p / BINARIZE_THRESHOLD)
    #         @inbounds for s in 1:shift
    #             circshift!(scratch, color, s)
    #             bundle_add!(acc2, scratch)
    #         end
    #         hv = binarize_bundle(acc2)
    #         bundle_add!(acc, hv)
    #     end
    # end
    # @inbounds for (i, r) in enumerate(eachrow(x))
    #     hv = hvectors_y[i]
    #     @inbounds for p in r
    #         shift = ceil(Int, p / BINARIZE_THRESHOLD)
    #         circshift!(scratch, hv, shift)
    #         bundle_add!(acc, scratch)
    #     end
    # end
    # @inbounds for i in eachindex(x)
    #     hv = hvectors[i]
    #     shift = ceil(Int, x[i] / BINARIZE_THRESHOLD)
    #     @inbounds for s in 1:shift
    #         circshift!(scratch, hv, s)
    #         bundle_add!(acc, scratch)
    #     end
    # end
    hv = binarize_bundle(acc)
    return TMInput(hv.chunks, hv.len)
end


function prepare_dataset(X::Vector{Matrix{Float32}})::Vector{TMInput}
    acc = Vector{Vector{BUNDLE_ACC_TYPE}}(undef, nthreads(:default))
    scratch = Vector{BitVector}(undef, nthreads(:default))
    for tid in 1:Threads.nthreads(:default)
        acc[tid] = zeros(BUNDLE_ACC_TYPE, HV_DIMENSIONS)
        scratch[tid] = BitVector(undef, HV_DIMENSIONS)
    end
    dataset = Vector{TMInput}(undef, length(X))
    @threads for i in eachindex(X)
        tid::Int64 = Threads.threadid() - Threads.nthreads(:interactive)
        dataset[i] = booleanize(X[i], acc[tid], scratch[tid])
    end
    return dataset
end


if isfile("/tmp/train")
    x_train, y_train = deserialize("/tmp/train")
    x_test, y_test = deserialize("/tmp/test")
else
    "Preparing HDC dataset... " |> print
    x_train, y_train = prepare_dataset(x_train), Int8.(y_train)
    x_test, y_test = prepare_dataset(x_test), Int8.(y_test)
    serialize("/tmp/train", (x_train, y_train))
    serialize("/tmp/test", (x_test, y_test))
    "Done." |> println
end


# CLAUSES = 20
# T = 286
# S = 1024 * 16
# L = 8192
# LF = 8192

# CLAUSES = 20
# T = 572
# S = 65536
# L = 32768
# LF = 32768

# CLAUSES = 256
# T = 1024
# S = 1024 * 16
# L = 8192
# LF = 8192

CLAUSES = 256
T = 2048
S = 65536
L = 32768
LF = 32768


EPOCHS = 1000

# Training the TM model
tm = TMClassifier(x_train[1], y_train, CLAUSES, T, S, L, LF, states_num=256, include_limit=240)
train!(tm, x_train, y_train, x_test, y_test, EPOCHS, index=false)
