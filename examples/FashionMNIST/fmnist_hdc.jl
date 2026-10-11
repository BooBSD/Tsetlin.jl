include("../../src/Tsetlin.jl")

import Pkg
[Base.find_package(p) === nothing && Pkg.add(p) for p in ["MLDatasets"]]

using Dates
using Random
using Serialization
using Base.Threads
using MLDatasets: MNIST, FashionMNIST
using .Tsetlin: TMClassifier, InputVector, InputBatch, train!, unzip


# const DATASET = MNIST
const DATASET = FashionMNIST
const HV_PATH = joinpath(tempdir(), "hvectors_fmnist")
const DATASET_PATH = joinpath(tempdir(), "dataset_fmnist")
const BUNDLE_ACC_TYPE = Float32
const HV_DIMENSIONS = 1024 * 32
const DATASET_CACHING = false

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
    chunks = scratch.chunks
    @inbounds for x in 1:28
        hv_x = hvectors[x]
        for y in 1:28
            hv_y = hvectors[28 + y]
            weight = image[x, y]
            w_pos = weight
            w_neg = -weight
            hv_x_chunks = hv_x.chunks
            hv_y_chunks = hv_y.chunks
            for c in eachindex(chunks)
                chunk = hv_x_chunks[c] ⊻ hv_y_chunks[c]
                base = c * 64 - 63
                @simd for j in 0:63
                    a = base + j
                    pos = ((chunk >> j) & 1) == 1
                    acc[a] += ifelse(pos, w_pos, w_neg)
                end
            end
        end
    end
    zero_val = zero(BUNDLE_ACC_TYPE)
    @inbounds for c in eachindex(chunks)
        chunk = zero(UInt64)
        base = c * 64 - 63
        @simd for j in 0:63
            a = base + j
            val = acc[a]
            bit = ifelse(val != zero_val, val > zero_val, iseven(a))
            chunk |= (UInt64(bit) << j)
        end
        chunks[c] = chunk
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
    x_train, y_train = unzip([DATASET(:train, Tx=Float32)...])
    x_test, y_test = unzip([DATASET(:test, Tx=Float32)...])
    n_default = Threads.nthreads(:default)
    n_interact = Threads.nthreads(:interactive)
    prepare_time = @elapsed begin
        hvectors = Memory{BitVector}(undef, HV_NUMBERS)
        for i in 1:HV_NUMBERS
            hvectors[i] = bitrand(HV_DIMENSIONS)
        end
        serialize(HV_PATH, hvectors)

        X_train = InputBatch(HV_DIMENSIONS, length(x_train))
        X_test = InputBatch(HV_DIMENSIONS, length(x_test))
        accs = [zeros(BUNDLE_ACC_TYPE, HV_DIMENSIONS) for _ in 1:n_default]
        scratchs = [BitVector(undef, HV_DIMENSIONS) for _ in 1:n_default]
        @threads for i in eachindex(x_train)
            tid = Threads.threadid() - n_interact
            hv = bundle!(accs[tid], scratchs[tid], x_train[i], hvectors)
            copyto!(view(X_train, :, i), InputVector(hv, copy=false))
        end
        @threads for i in eachindex(x_test)
            tid = Threads.threadid() - n_interact
            hv = bundle!(accs[tid], scratchs[tid], x_test[i], hvectors)
            copyto!(view(X_test, :, i), InputVector(hv, copy=false))
        end
    end
    # Convert y_train and y_test to the Int8 type to save memory
    y_train = Int8.(y_train)
    y_test = Int8.(y_test)
    serialize(DATASET_PATH, (X_train, y_train, X_test, y_test))
    println("Done. Elapsed in $(Time(0) + Second(round(Int, prepare_time))).")
end

# Training the TM model
tm = TMClassifier(HV_DIMENSIONS, y_train, CLAUSES, T, S, L, LF, states_num=STATES_NUM, include_limit=INCLUDE_LIMIT)
train!(tm, X_train, y_train, X_test, y_test, EPOCHS)
