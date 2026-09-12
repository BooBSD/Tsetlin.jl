include("../../src/Tsetlin.jl")

import Pkg
[Base.find_package(p) === nothing && Pkg.add(p) for p in ["MLDatasets"]]

using Dates
using Random
using Serialization
using Base.Threads
using MLDatasets: MNIST, FashionMNIST
using .Tsetlin: TMInput, TMClassifier, train!, unzip


const HV_PATH = joinpath(tempdir(), "hvectors_fmnist")
const DATASET_PATH = joinpath(tempdir(), "dataset_fmnist")
const DATASET_CACHING = false
const HV_DIMENSIONS = 1024 * 32
const BUNDLE_ACC_TYPE = Float32

const HV_NUMBERS = 28 + 28

const CLAUSES = 128
const T = 1024
const S = 1024 * 32
const L = 1024 * 16
const LF = 1024 * 16

# const CLAUSES = 512
# const T = 2048
# const S = 1024 * 32
# const L = 1024 * 16
# const LF = 1024 * 16

const STATES_NUM = 64000
const INCLUDE_LIMIT = 32000
const EPOCHS = 1000


function bundle!(
    acc::Vector{BUNDLE_ACC_TYPE},
    scratch::BitVector,
    image::Matrix{Float32},
    hvectors::Memory{BitVector},
)::BitVector
    fill!(acc, zero(BUNDLE_ACC_TYPE))
    @inbounds for x in 1:28
        hv_x = hvectors[x]
        for y in 1:28
            hv_y = hvectors[28 + y]
            @simd for i in eachindex(scratch.chunks)
                scratch.chunks[i] = hv_x.chunks[i] ⊻ hv_y.chunks[i]
            end
            weight = image[x, y]
            w_pos = weight
            w_neg = -weight
            @simd for j in eachindex(scratch)
                acc[j] += ifelse(scratch[j], w_pos, w_neg)
            end
        end
    end
    zero_val = zero(BUNDLE_ACC_TYPE)
    @inbounds @simd for i in eachindex(acc)
        val = acc[i]
        scratch[i] = ifelse(val != zero_val, val > zero_val, iseven(i))
    end
    return scratch
end


if DATASET_CACHING
    print("\nLoading cached dataset... ")
    hvectors = deserialize(HV_PATH)
    X_train, y_train, X_test, y_test = deserialize(DATASET_PATH)
    println("Done.")
else
    print("\nPreparing dataset... ")
    # x_train, y_train = unzip([MNIST(:train, Tx=Float32)...])
    # x_test, y_test = unzip([MNIST(:test, Tx=Float32)...])
    x_train, y_train = unzip([FashionMNIST(:train, Tx=Float32)...])
    x_test, y_test = unzip([FashionMNIST(:test, Tx=Float32)...])
    n_default = Threads.nthreads(:default)
    n_interact = Threads.nthreads(:interactive)
    prepare_time = @elapsed begin
        hvectors = Memory{BitVector}(undef, HV_NUMBERS)
        for i in 1:HV_NUMBERS
            hvectors[i] = bitrand(HV_DIMENSIONS)
        end
        serialize(HV_PATH, hvectors)

        X_train = Vector{TMInput}(undef, length(x_train))
        X_test = Vector{TMInput}(undef, length(x_test))
        accs = [zeros(BUNDLE_ACC_TYPE, HV_DIMENSIONS) for _ in 1:n_default]
        scratchs = [BitVector(undef, HV_DIMENSIONS) for _ in 1:n_default]
        @threads for i in eachindex(x_train)
            tid = Threads.threadid() - n_interact
            hv = bundle!(accs[tid], scratchs[tid], x_train[i], hvectors)
            X_train[i] = TMInput(hv.chunks, hv.len)
        end
        @threads for i in eachindex(x_test)
            tid = Threads.threadid() - n_interact
            hv = bundle!(accs[tid], scratchs[tid], x_test[i], hvectors)
            X_test[i] = TMInput(hv.chunks, hv.len)
        end
    end
    # Convert y_train and y_test to the Int8 type to save memory
    y_train = Int8.(y_train)
    y_test = Int8.(y_test)
    serialize(DATASET_PATH, (X_train, y_train, X_test, y_test))
    println("Done. Elapsed in $(Time(0) + Second(floor(Int, prepare_time))).")
end


# Training the TM model
tm = TMClassifier(X_train[1], y_train, CLAUSES, T, S, L, LF, states_num=STATES_NUM, include_limit=INCLUDE_LIMIT)
train!(tm, X_train, y_train, X_test, y_test, EPOCHS, exclusive_literals=false)
