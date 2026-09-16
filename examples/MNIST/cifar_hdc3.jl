include("../../src/Tsetlin.jl")
include("../TEXT/HDC.jl")


import Pkg
[Base.find_package(p) === nothing && Pkg.add(p) for p in ["MLDatasets"]]

using Dates
using Random
using Serialization
using Base.Threads
using MLDatasets: CIFAR10
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy, save, load, unzip, benchmark, compile

x_train, y_train = unzip([CIFAR10(:train, Tx=Float32)...])
x_test, y_test = unzip([CIFAR10(:test, Tx=Float32)...])

# const MEAN = reshape(Float32[0.4914, 0.4822, 0.4465], 1, 1, 3)
# const STD  = reshape(Float32[0.2023, 0.1994, 0.2010], 1, 1, 3)

# # normalize_cifar(x) = (x .- MEAN) ./ STD
# normalize_cifar(x) = x .* 2.0f0 .- 1.0f0

# x_train = normalize_cifar.(x_train)
# x_test = normalize_cifar.(x_test)

const HV_PATH = "/tmp/hvectors_cifar"
const DATASET_PATH = "/tmp/dataset_cifar"
const DATASET_CACHING = false
# const HV_DIMENSIONS = 1024 * 4
# const HV_DIMENSIONS = 1024 * 16
const HV_DIMENSIONS = 1024 * 32
# const HV_DIMENSIONS = 1024 * 64
const BUNDLE_ACC_TYPE = Float32

# HV_NUMBERS = 3 + 16 + 16 + 2 + 2
HV_NUMBERS = 3 + 8 + 8 + 4 + 4

HV_NUMBERS_2 = 3 + 16 + 16 + 2 + 2
HV_NUMBERS_3 = 3 + 32 + 32

HV_NUMBERS_4 = 3 + 4 + 4 + 8 + 8

# const CLAUSES = 16
# const T = 128
# const S = 1024 * 4
# const L = 1024 * 2
# const LF = 1024 * 2

# const CLAUSES = 64
# const T = 256
# const S = 1024 * 4
# const L = 1024 * 2
# const LF = 1024 * 2

# const CLAUSES = 64
# const T = 512
# const S = 1024 * 16
# const L = 1024 * 8
# const LF = 1024 * 8

# const CLAUSES = 64
# const T = 1024
# const S = 1024 * 64
# const L = 1024 * 32
# const LF = 1024 * 32

const CLAUSES = 128
const T = 1024
const S = 1024 * 32
const L = 1024 * 16
const LF = 1024 * 16

# const CLAUSES = 256
# const T = 512
# const S = 1024 * 4
# const L = 1024 * 2
# const LF = 1024 * 2

# const CLAUSES = 256
# const T = 1024
# const S = 1024 * 16
# const L = 1024 * 8
# const LF = 1024 * 8

# const CLAUSES = 256
# const T = 2048
# const S = 1024 * 32 # 64
# const L = 1024 * 32
# const LF = 1024 * 32

# const CLAUSES = 512
# const T = 2048
# const S = 1024 * 32
# const L = 1024 * 16
# const LF = 1024 * 16

# const CLAUSES = 1024
# const T = 2048
# const S = 1024 * 16
# const L = 1024 * 8
# const LF = 1024 * 8

# const CLAUSES = 2048
# const T = 1024 * 4
# const S = 1024 * 32
# const L = 1024 * 16
# const LF = 1024 * 16

# const CLAUSES = 1024 * 4
# const T = 1024 * 8
# const S = 1024 * 32 # 64
# const L = 1024 * 32
# const LF = 1024 * 32

EPOCHS = 1000

# Sparse
@inline function random_hv(dim::Int, k::Int)::BitVector
    hv = falses(dim)
    indices = randperm(dim)[1:k]
    hv[indices] .= true
    return hv
end

function bundle!(
    acc::Vector{BUNDLE_ACC_TYPE},
    scratch::BitVector,
    image::Array{Float32, 3},
    hvectors::Memory{BitVector},
    hvectors2::Memory{BitVector},
    hvectors3::Memory{BitVector},
    hvectors4::Memory{BitVector},
)::BitVector
    fill!(acc, zero(BUNDLE_ACC_TYPE))
    # @inbounds for layer in 1:3
    #     hv_layer = hvectors[layer]
    #     for X in 1:8
    #         hv_X = hvectors[3 + X]
    #         for Y in 1:8
    #             hv_Y = hvectors[3 + 8 + Y]
    #             weight = zero(BUNDLE_ACC_TYPE)
    #             fill!(scratch.chunks, zero(UInt64))
    #             for x in 1:4
    #                 hv_x = hvectors[3 + 8 + 8 + x]
    #                 for y in 1:4
    #                     hv_y = hvectors[3 + 8 + 8 + 4 + y]
    #                     @simd for i in eachindex(scratch.chunks)
    #                         scratch.chunks[i] = scratch.chunks[i] ⊻ hv_x.chunks[i] ⊻ hv_y.chunks[i]
    #                     end
    #                     weight += image[X * 4 - 4 + x, Y * 4 - 4 + y, layer]
    #                 end
    #             end
    #             @simd for i in eachindex(scratch.chunks)
    #                 scratch.chunks[i] = hv_layer.chunks[i] ⊻ hv_X.chunks[i] ⊻ hv_Y.chunks[i] ⊻ scratch.chunks[i]
    #             end
    #             w_pos = weight
    #             w_neg = -weight
    #             @simd for j in eachindex(scratch)
    #                 acc[j] += ifelse(scratch[j], w_pos, w_neg)
    #             end
    #         end
    #     end
    # end

    # @inbounds for layer in 1:3
    #     hv_layer = hvectors2[layer]
    #     for X in 1:16
    #         hv_X = hvectors2[3 + X]
    #         for Y in 1:16
    #             hv_Y = hvectors2[3 + 16 + Y]
    #             weight = zero(BUNDLE_ACC_TYPE)
    #             fill!(scratch.chunks, zero(UInt64))
    #             for x in 1:2
    #                 hv_x = hvectors2[3 + 16 + 16 + x]
    #                 for y in 1:2
    #                     hv_y = hvectors2[3 + 16 + 16 + 2 + y]
    #                     @simd for i in eachindex(scratch.chunks)
    #                         scratch.chunks[i] = scratch.chunks[i] ⊻ hv_x.chunks[i] ⊻ hv_y.chunks[i]
    #                     end
    #                     weight += image[X * 2 - 2 + x, Y * 2 - 2 + y, layer]
    #                 end
    #             end
    #             @simd for i in eachindex(scratch.chunks)
    #                 scratch.chunks[i] = hv_layer.chunks[i] ⊻ hv_X.chunks[i] ⊻ hv_Y.chunks[i] ⊻ scratch.chunks[i]
    #             end
    #             w_pos = weight
    #             w_neg = -weight
    #             @simd for j in eachindex(scratch)
    #                 acc[j] += ifelse(scratch[j], w_pos, w_neg)
    #             end
    #         end
    #     end
    # end

    @inbounds for layer in 1:3
        hv_layer = hvectors3[layer]
        for x in 1:32
            hv_x = hvectors3[3 + x]
            for y in 1:32
                hv_y = hvectors3[3 + 32 + y]
                @simd for i in eachindex(scratch.chunks)
                    scratch.chunks[i] = hv_layer.chunks[i] ⊻ hv_x.chunks[i] ⊻ hv_y.chunks[i]
                end
                weight = image[x, y, layer]
                w_pos = weight
                w_neg = -weight
                @simd for j in eachindex(scratch)
                    acc[j] += ifelse(scratch[j], w_pos, w_neg)
                end
            end
        end
    end

    # hv_channel_R = hvectors3[1]
    # hv_channel_G = hvectors3[2]
    # hv_channel_B = hvectors3[3]
    # @simd for i in eachindex(scratch.chunks)
    #     scratch.chunks[i] = hv_channel_R.chunks[i] ⊻ hv_channel_G.chunks[i] ⊻ hv_channel_B.chunks[i]
    # end
    # @inbounds for x in 1:32
    #     hv_x = hvectors3[3 + x]
    #     for y in 1:32
    #         hv_y = hvectors3[3 + 32 + y]
    #         weight = image[x, y, 1] + image[x, y, 2] + image[x, y, 3]
    #         # weight = max(image[x, y, 1], image[x, y, 2], image[x, y, 3])
    #         # weight = zero(BUNDLE_ACC_TYPE)
    #         # n = 0
    #         # for i in 1:3
    #         #     w = image[x, y, i]
    #         #     pred = w > weight
    #         #     weight = ifelse(pred, w, weight)
    #         #     n = ifelse(pred, i, n)
    #         # end
    #         @simd for i in eachindex(scratch.chunks)
    #             scratch.chunks[i] = scratch.chunks[i] ⊻ hv_x.chunks[i] ⊻ hv_y.chunks[i]
    #             # scratch.chunks[i] = hvectors3[n].chunks[i] ⊻ hv_x.chunks[i] ⊻ hv_y.chunks[i]
    #         end
    #         w_pos = weight
    #         w_neg = -weight
    #         @simd for j in eachindex(scratch)
    #             acc[j] += ifelse(scratch[j], w_pos, w_neg)
    #         end
    #     end
    # end

    # @inbounds for layer in 1:3
    #     hv_layer = hvectors4[layer]
    #     for X in 1:4
    #         hv_X = hvectors4[3 + X]
    #         for Y in 1:4
    #             hv_Y = hvectors4[3 + 4 + Y]
    #             weight = zero(BUNDLE_ACC_TYPE)
    #             fill!(scratch.chunks, zero(UInt64))
    #             for x in 1:8
    #                 hv_x = hvectors4[3 + 4 + 4 + x]
    #                 for y in 1:8
    #                     hv_y = hvectors4[3 + 4 + 4 + 8 + y]
    #                     @simd for i in eachindex(scratch.chunks)
    #                         scratch.chunks[i] = scratch.chunks[i] ⊻ hv_x.chunks[i] ⊻ hv_y.chunks[i]
    #                     end
    #                     weight += image[X * 8 - 8 + x, Y * 8 - 8 + y, layer]
    #                 end
    #             end
    #             @simd for i in eachindex(scratch.chunks)
    #                 scratch.chunks[i] = hv_layer.chunks[i] ⊻ hv_X.chunks[i] ⊻ hv_Y.chunks[i] ⊻ scratch.chunks[i]
    #             end
    #             w_pos = weight
    #             w_neg = -weight
    #             @simd for j in eachindex(scratch)
    #                 acc[j] += ifelse(scratch[j], w_pos, w_neg)
    #             end
    #         end
    #     end
    # end

    # return acc .> 0
    zero_val = zero(BUNDLE_ACC_TYPE)
    @inbounds @simd for i in eachindex(acc)
        val = acc[i]
        # if val != zero_val
        #     scratch[i] = val > zero_val
        # else
        #     scratch[i] = rand(Bool)
        #     # "!!!!!!!!11111!!!!!" |> println
        # end
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
    n_default = Threads.nthreads(:default)
    n_interact = Threads.nthreads(:interactive)
    prepare_time = @elapsed begin
        # bits_per_hv = round(Int, HV_DIMENSIONS * (1 - 0.5 ^ (1 / 3)))
        # bits_per_hv = round(Int, HV_DIMENSIONS * (1 - 0.5 ^ (1 / 2)))
        # bits_per_hv = round(Int, HV_DIMENSIONS / 2)
        hvectors = Memory{BitVector}(undef, HV_NUMBERS)
        for i in 1:HV_NUMBERS
            # hvectors[i] = random_hv(HV_DIMENSIONS, bits_per_hv)
            hvectors[i] = bitrand(HV_DIMENSIONS)
        end

        hvectors2 = Memory{BitVector}(undef, HV_NUMBERS_2)
        for i in 1:HV_NUMBERS_2
            hvectors2[i] = bitrand(HV_DIMENSIONS)
        end
        hvectors3 = Memory{BitVector}(undef, HV_NUMBERS_3)
        for i in 1:HV_NUMBERS_3
            hvectors3[i] = bitrand(HV_DIMENSIONS)
        end
        hvectors4 = Memory{BitVector}(undef, HV_NUMBERS_4)
        for i in 1:HV_NUMBERS_4
            hvectors4[i] = bitrand(HV_DIMENSIONS)
        end

        # hvectors::Memory{BitVector} = generate_orthogonal_pool(HV_DIMENSIONS, HV_NUMBERS, 1)
        serialize(HV_PATH, hvectors)

        X_train = Vector{TMInput}(undef, length(x_train))
        X_test = Vector{TMInput}(undef, length(x_test))
        accs = [zeros(BUNDLE_ACC_TYPE, HV_DIMENSIONS) for _ in 1:n_default]
        scratchs = [BitVector(undef, HV_DIMENSIONS) for _ in 1:n_default]
        @threads for i in eachindex(x_train)
            tid = Threads.threadid() - n_interact
            hv = bundle!(accs[tid], scratchs[tid], x_train[i], hvectors, hvectors2, hvectors3, hvectors4)
            X_train[i] = TMInput(hv.chunks, hv.len)
        end
        @threads for i in eachindex(x_test)
            tid = Threads.threadid() - n_interact
            hv = bundle!(accs[tid], scratchs[tid], x_test[i], hvectors, hvectors2, hvectors3, hvectors4)
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
# tm = TMClassifier(X_train[1], y_train, CLAUSES, T, S, L, LF, states_num=256, include_limit=128)
tm = TMClassifier(X_train[1], y_train, CLAUSES, T, S, L, LF, states_num=64000, include_limit=32000)
tms = train!(tm, X_train, y_train, X_test, y_test, EPOCHS, index=false, exclusive_literals=false, best_tms_size=0)
