include("../../src/Tsetlin.jl")

import Pkg
[Base.find_package(p) === nothing && Pkg.add(p) for p in ["MLDatasets"]]

using Dates
using Random
using Serialization
using Base.Threads
using Printf: @printf
using MLDatasets: CIFAR100
using .Tsetlin: TMInput, TMClassifier, train!, unzip, vote, accuracy, literals_sum


const HV_PATH = joinpath(tempdir(), "hvectors_cifar")
const DATASET_PATH = joinpath(tempdir(), "dataset_cifar")
const DATASET_CACHING = false
const HV_DIMENSIONS = 1024 * 64
const BUNDLE_ACC_TYPE = Float32
const RANDOM_NEGATIVE_SAMPLE = false

const HV_NUMBERS = 3 + 32 + 32

const CLAUSES = 128
const T = 1024 * 2
const S = 1024 * 64
const L = 1024 * 32
const LF = 1024 * 32

# const CLAUSES = 512
# const T = 1024 * 4
# const S = 1024 * 64
# const L = 1024 * 32
# const LF = 1024 * 32

const STATES_NUM = 64000
const INCLUDE_LIMIT = 32000
const EPOCHS = 1000

const R_mask = bitrand(HV_DIMENSIONS)


function bundle(
    acc::Vector{BUNDLE_ACC_TYPE},
    image::Array{Float32, 3},
    hvectors::Memory{BitVector},
)::BitVector
    scratch = BitVector(undef, HV_DIMENSIONS)
    fill!(acc, zero(BUNDLE_ACC_TYPE))
    @inbounds for channel in 1:3
        hv_channel = hvectors[channel]
        for x in 1:32
            hv_x = hvectors[3 + x]
            for y in 1:32
                hv_y = hvectors[3 + 32 + y]
                @simd for i in eachindex(scratch.chunks)
                    scratch.chunks[i] = hv_channel.chunks[i] ⊻ hv_x.chunks[i] ⊻ hv_y.chunks[i]
                end
                weight = image[x, y, channel]
                w_pos = weight
                w_neg = -weight
                @simd for j in eachindex(scratch)
                    acc[j] += ifelse(scratch[j], w_pos, w_neg)
                end
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


@inline function mix_target!(result::TMInput, x::BitVector, target::BitVector)
    @inbounds @simd for i in 1:length(x.chunks)
        # XOR algo
        result.chunks[i] = x.chunks[i] ⊻ target.chunks[i]
        # Mix algo
        # result.chunks[i] = ((x.chunks[i] & ~R_mask.chunks[i]) | (target.chunks[i] & R_mask.chunks[i]))
    end
end


if DATASET_CACHING
    print("\nLoading cached dataset... ")
    hv_targets = deserialize(HV_PATH)
    X_train, y_train, X_test, y_test = deserialize(DATASET_PATH)
    println("Done.")
else
    print("\nPreparing dataset... ")
    x_train, y_train = unzip([CIFAR100(:train, Tx=Float32)...])
    x_test, y_test = unzip([CIFAR100(:test, Tx=Float32)...])
    n_default = Threads.nthreads(:default)
    n_interact = Threads.nthreads(:interactive)
    prepare_time = @elapsed begin
        # Convert y_train and y_test to the Int8 type to save memory
        y_train = [Int8(y[:fine]) for y in y_train]
        y_test = [Int8(y[:fine]) for y in y_test]

        targets_length = length(unique(y_train))
        hv_targets = Dict{Int8, BitVector}()
        for y in unique(y_train)
            hv_targets[y] = bitrand(HV_DIMENSIONS)
        end
        serialize(HV_PATH, hv_targets)

        hvectors = Memory{BitVector}(undef, HV_NUMBERS)
        for i in 1:HV_NUMBERS
            hvectors[i] = bitrand(HV_DIMENSIONS)
        end

        X_train = Vector{BitVector}(undef, length(x_train))
        X_test = Vector{BitVector}(undef, length(x_test))
        accs = [zeros(BUNDLE_ACC_TYPE, HV_DIMENSIONS) for _ in 1:n_default]
        scratchs = [BitVector(undef, HV_DIMENSIONS) for _ in 1:n_default]
        @threads for i in eachindex(x_train)
            tid = Threads.threadid() - n_interact
            X_train[i] = bundle(accs[tid], x_train[i], hvectors)
        end
        @threads for i in eachindex(x_test)
            tid = Threads.threadid() - n_interact
            X_test[i] = bundle(accs[tid], x_test[i], hvectors)
        end
    end
    serialize(DATASET_PATH, (X_train, y_train, X_test, y_test))
    println("Done. Elapsed in $(Time(0) + Second(floor(Int, prepare_time))).")
end


@inline function get_negative_idx(ys::AbstractVector{Int8}, target::Int8)
    n = length(ys)
    while true
        idx = rand(1:n)
        @inbounds if ys[idx] != target
            return idx
        end
    end
end


function main()
    x_sample = TMInput(HV_DIMENSIONS)
    mix_target!(x_sample, X_train[1], first(values(hv_targets)))
    tm = TMClassifier(x_sample, [true, false], CLAUSES, T, S, L, LF, states_num=STATES_NUM, include_limit=INCLUDE_LIMIT)
    density = round(sum(x_sample) / length(x_sample) * 100, digits=2)
    println("\nClasses: $(tm.classes_num), clauses: $(tm.clauses_num), T: $(tm.T), S: $(tm.S) (s: $(tm.s)), L: $(tm.L), LF: $(tm.LF), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit).")
    println("Input vector size: $(length(x_sample)) bits, density: $(density)%, training dataset size: $(length(X_train)).")
    println("Expected average clause literal density: $(round(tm.L / length(x_sample) * 100, digits=2))%. Using literals index: false.")
    println("Running in $(nthreads()) threads. Training over $(EPOCHS) epochs:\n")

    n_default = Threads.nthreads(:default)
    n_interact = Threads.nthreads(:interactive)
    results = [TMInput(undef, HV_DIMENSIONS) for _ in 1:n_default]
    best_acc = 0.0
    all_time = @elapsed begin
        @inbounds for e in 1:EPOCHS
            training_time = @elapsed begin
                @threads for i in randperm(length(y_train))
                    X = X_train[i]
                    y = y_train[i]
                    tid = Threads.threadid() - n_interact
                    res = results[tid]
                    mix_target!(res, X, hv_targets[y])
                    train!(tm, res, true, exclusive_literals=false)
                    if RANDOM_NEGATIVE_SAMPLE
                        neg_idx = get_negative_idx(y_train, y)
                        mix_target!(res, X, hv_targets[y_train[neg_idx]])
                        train!(tm, res, false, exclusive_literals=false)
                    else
                        @inbounds for (cls, hv_target) in hv_targets
                            if cls != y
                                mix_target!(res, X, hv_targets[cls])
                                train!(tm, res, false, exclusive_literals=false)
                            end
                        end
                    end
                end
            end
            testing_time = @elapsed begin
                len = length(X_test)
                predicted::Vector{Int8} = Vector{Int8}(undef, len)
                @threads for i in 1:len
                    x = X_test[i]
                    tid = Threads.threadid() - n_interact
                    res = results[tid]
                    best_vote = typemin(Int64)
                    best_cls = typemin(Int8)
                    @inbounds for (cls, hv_target) in hv_targets
                        mix_target!(res, x, hv_target)
                        pos, neg = vote(tm, tm.clauses, res)
                        v = pos - neg
                        is_better = v > best_vote
                        best_cls = ifelse(is_better, cls, best_cls)
                        best_vote = ifelse(is_better, v, best_vote)
                    end
                    predicted[i] = best_cls
                end
                acc = accuracy(y_test, predicted)
                best_acc = ifelse(acc > best_acc, acc, best_acc)
            end
            @printf("#%s  Accuracy: %.2f%%  Best: %.2f%%  Training: %.3fs  Testing: %.3fs\n", e, acc * 100, best_acc * 100, training_time, testing_time)
        end
    end
    elapsed = Time(0) + Second(floor(Int, all_time))
    average_clause_density = round((literals_sum(tm) / (tm.classes_num * tm.clauses_num * 2)) / length(x_sample) * 100, digits=2)
    println("\n$(EPOCHS) epochs done in $(elapsed).")
    println("Classes: $(tm.classes_num), clauses: $(tm.clauses_num), T: $(tm.T), S: $(tm.S) (s: $(tm.s)), L: $(tm.L), LF: $(tm.LF), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit).")
    println("Input vector size: $(length(x_sample)) bits, density: $(density)%, training dataset size: $(length(X_train)).")
    println("Average clause literal density: $(average_clause_density)%. Using literals index: false.\n")
end


main()
