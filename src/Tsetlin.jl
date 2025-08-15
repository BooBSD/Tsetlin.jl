module Tsetlin

export TMInput, TMClassifier, train!, predict, accuracy, save, load

using Dates
using Random
using Base.Threads
using Serialization
using Statistics: mean, median
using Printf: @printf, @sprintf


Base.exit_on_sigint(false)

unzip(a) = (getfield.(a, x) for x in fieldnames(eltype(a)))

isbitset(x::UInt64, n::UInt64)::Bool = (x >> n) & one(UInt64)

abstract type AbstractTATeam end
abstract type AbstractTMClassifier end

class_type(tm::AbstractTMClassifier)::DataType = typeof(tm).parameters[1]

mutable struct TATeam <: AbstractTATeam
    const include_limit::UInt8
    const state_min::UInt8
    const state_max::UInt8
    positive_clauses::Matrix{UInt8}
    negative_clauses::Matrix{UInt8}
    positive_clauses_inverted::Matrix{UInt8}
    negative_clauses_inverted::Matrix{UInt8}
    positive_included_literals::Vector{Vector{UInt16}}
    negative_included_literals::Vector{Vector{UInt16}}
    positive_included_literals_inverted::Vector{Vector{UInt16}}
    negative_included_literals_inverted::Vector{Vector{UInt16}}
    const clause_size::Int64

    function TATeam(clause_size::Int64, clauses_num::Int64, include_limit::Int64, state_min::Int64, state_max::Int64)
        positive_clauses = fill(UInt8(include_limit - 1), clause_size, floor(Int, clauses_num / 2))
        negative_clauses = fill(UInt8(include_limit - 1), clause_size, floor(Int, clauses_num / 2))
        positive_clauses_inverted = fill(UInt8(include_limit - 1), clause_size, floor(Int, clauses_num / 2))
        negative_clauses_inverted = fill(UInt8(include_limit - 1), clause_size, floor(Int, clauses_num / 2))
        positive_included_literals = fill([], floor(Int, clauses_num / 2))
        negative_included_literals = fill([], floor(Int, clauses_num / 2))
        positive_included_literals_inverted = fill([], floor(Int, clauses_num / 2))
        negative_included_literals_inverted = fill([], floor(Int, clauses_num / 2))
        return new(include_limit, state_min, state_max, positive_clauses, negative_clauses, positive_clauses_inverted, negative_clauses_inverted, positive_included_literals, negative_included_literals, positive_included_literals_inverted, negative_included_literals_inverted, clause_size)
    end
end


mutable struct TMClassifier{ClassType} <: AbstractTMClassifier
    clauses_num::Int64
    T::Int64
    S::Int64
    s::Int64
    L::Int64
    const include_limit::Int64
    const state_min::Int64
    const state_max::Int64
    const clauses::Dict{ClassType, TATeam}

    function TMClassifier{ClassType}(clauses_num::Int64, T::Int64, S::Int64; states_num::Int64=256, include_limit::Int64=128, L::Int64=16) where ClassType
        return new{ClassType}(clauses_num, T, S, 0, L, include_limit, typemin(UInt8), states_num - 1, Dict())
    end
end


mutable struct TATeamCompiled <: AbstractTATeam
    positive_included_literals::Vector{Vector{UInt16}}
    negative_included_literals::Vector{Vector{UInt16}}
    positive_included_literals_inverted::Vector{Vector{UInt16}}
    negative_included_literals_inverted::Vector{Vector{UInt16}}

    function TATeamCompiled(clauses_num::Int64)
        positive_included_literals = fill([], floor(Int, clauses_num / 2))
        negative_included_literals = fill([], floor(Int, clauses_num / 2))
        positive_included_literals_inverted = fill([], floor(Int, clauses_num / 2))
        negative_included_literals_inverted = fill([], floor(Int, clauses_num / 2))
        return new(positive_included_literals, negative_included_literals, positive_included_literals_inverted, negative_included_literals_inverted)
    end
end


struct TMClassifierCompiled{ClassType} <: AbstractTMClassifier
    clauses_num::Int64
    T::Int64
    S::Int64
    s::Int64
    L::Int64
    clauses::Dict{ClassType, TATeamCompiled}

    function TMClassifierCompiled{ClassType}(clauses_num::Int64, T::Int64, S::Int64, s::Int64, L::Int64) where ClassType
        return new{ClassType}(clauses_num, T, S, s, L, Dict())
    end
end


abstract type AbstractTMInput <: AbstractVector{Bool} end


struct TMInput <: AbstractTMInput
    x::BitVector

    function TMInput(x::AbstractArray{Bool})
        return new(BitVector(vec(x)))
    end
end

Base.IndexStyle(::Type{<:TMInput}) = IndexLinear()
Base.size(x::TMInput)::Tuple{Int64} = size(x.x)
Base.getindex(x::TMInput, i::Int)::Bool = x.x[i]


function booleanize(x::AbstractArray{Float32}, thresholds::Number...)::TMInput
    return TMInput(vcat((x .> t for t in thresholds)...))
end


function bits_2_word(X::Vector{TMInput}, j::Int64, size::Int64)::UInt64
    ex::UInt64 = zero(UInt64)
    @inbounds for i in 1:size
        ex *= 2
        ex += X[i].x[j]
    end
    if size < 64
        @inbounds for i in size+1:64
            ex *= 2
            ex += true
        end
    end
    return bitreverse(ex)
end


struct TMInputBatch <: AbstractTMInput
    x::Vector{UInt64}
    batch_size::Int64

    function TMInputBatch(X::Vector{TMInput})
        @assert 1 <= length(X) <= 64 "Size of input Vector must be in 1..64."
        if length(X) == 64
            return new([bits_2_word(X, j, 64) for j in 1:length(X[1])], 64)
        else
            return new([bits_2_word(X, j, length(X) < 64 ? length(X) : 64) for j in 1:length(X[1])], length(X))
        end
    end
end

Base.IndexStyle(::Type{<:TMInputBatch}) = IndexLinear()
Base.length(x::TMInputBatch)::Int64 = length(x.x)
Base.getindex(x::TMInputBatch, i::Int)::UInt64 = x.x[i]


function batches(X::Vector{TMInput})::Vector{TMInputBatch}
    _d, _r = divrem(length(X), 64)
    _X::Vector{TMInputBatch} = Vector{TMInputBatch}(undef, _r == 0 ? _d : _d + 1)  # Predefine vector for @threads access
    @threads for (j, i) in collect(enumerate(1:64:length(X)))
        _X[j] = (j <= _d) ? TMInputBatch(X[i:i+63]) : TMInputBatch(X[i:i+_r - 1])
    end
    return _X
end


function initialize!(tm::TMClassifier, X::Vector{TMInput}, Y::Vector)
    tm.s = round(Int, length(first(X)) / tm.S)
    for cls in collect(Set(Y))
        tm.clauses[cls] = TATeam(length(first(X)), tm.clauses_num, tm.include_limit, tm.state_min, tm.state_max)
    end
end


function check_clause(x::TMInput, literals::Vector{UInt16}, literals_inverted::Vector{UInt16})::Bool
    @inbounds for i in eachindex(literals)
        if !x.x[literals[i]]
            return false
        end
    end
    @inbounds for i in eachindex(literals_inverted)
        if x.x[literals_inverted[i]]
            return false
        end
    end
    return true
end


function check_clause(x::TMInputBatch, literals::Vector{UInt16}, literals_inverted::Vector{UInt16})::UInt64
    b::UInt64 = typemin(UInt64)
    @inbounds for i in eachindex(literals)
        b |= ~x.x[literals[i]]
    end
    @inbounds for i in eachindex(literals_inverted)
        b |= x.x[literals_inverted[i]]
    end
    return b
end


function vote(ta::AbstractTATeam, x::TMInput)::Tuple{Int64, Int64}
    pos = sum(check_clause(x, ta.positive_included_literals[i], ta.positive_included_literals_inverted[i]) for i in eachindex(ta.positive_included_literals))
    neg = sum(check_clause(x, ta.negative_included_literals[i], ta.negative_included_literals_inverted[i]) for i in eachindex(ta.negative_included_literals))
    return pos, neg
end


function vote(ta::AbstractTATeam, x::TMInputBatch, votes::Vector{Int64})::Vector{Int64}
    @inbounds for (pil, piil, nil, niil) in zip(ta.positive_included_literals, ta.positive_included_literals_inverted, ta.negative_included_literals, ta.negative_included_literals_inverted)
        p::UInt64 = check_clause(x, pil, piil)
        n::UInt64 = check_clause(x, nil, niil)
        @inbounds @simd for i in 1:64
            votes[i] += (isbitset(n, UInt64(i - 1)) - isbitset(p, UInt64(i - 1)))
        end
    end
    return votes
end


function feedback!(tm::TMClassifier, ta::TATeam, x::TMInput, clauses1::Matrix{UInt8}, clauses_inverted1::Matrix{UInt8}, clauses2::Matrix{UInt8}, clauses_inverted2::Matrix{UInt8}, literals1::Vector{Vector{UInt16}}, literals_inverted1::Vector{Vector{UInt16}}, literals2::Vector{Vector{UInt16}}, literals_inverted2::Vector{Vector{UInt16}}, positive::Bool)
    v::Int64 = clamp(-(vote(ta, x)...), -tm.T, tm.T)
    update::Float64 = (positive ? (tm.T - v) : (tm.T + v)) / (tm.T * 2)

    # Feedback 1
    @inbounds for (j, (c, ci)) in enumerate(zip(eachcol(clauses1), eachcol(clauses_inverted1)))
        if (rand() < update)
            if check_clause(x, literals1[j], literals_inverted1[j])
                if (length(literals1[j]) + length(literals_inverted1[j])) <= tm.L
                    @inbounds for i = 1:ta.clause_size
                        if (x.x[i] == true) && (c[i] < ta.state_max)
                            c[i] += one(UInt8)
                        end
                        if (x.x[i] == false) && (ci[i] < ta.state_max)
                            ci[i] += one(UInt8)
                        end
                    end
                end
                @inbounds for i = 1:ta.clause_size
                    if (x.x[i] == false) && (c[i] < ta.include_limit) && (c[i] > ta.state_min)
                        c[i] -= one(UInt8)
                    end
                    if (x.x[i] == true) && (ci[i] < ta.include_limit) && (ci[i] > ta.state_min)
                        ci[i] -= one(UInt8)
                    end
                end
            else
                @inbounds for _ in 1:tm.s
                    i = rand(1:ta.clause_size)  # Here's one random only.
                    if c[i] > ta.state_min
                        c[i] -= one(UInt8)
                    end
                    i = rand(1:ta.clause_size)  # Here's one random only.
                    if ci[i] > ta.state_min
                        ci[i] -= one(UInt8)
                    end
                end
            end
            literals1[j] = [@inbounds i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
            literals_inverted1[j] = [@inbounds i for i = 1:ta.clause_size if ci[i] >= ta.include_limit]
        end
    end
    # Feedback 2
    @inbounds for (j, (c, ci)) in enumerate(zip(eachcol(clauses2), eachcol(clauses_inverted2)))
        if (rand() < update)
            if check_clause(x, literals2[j], literals_inverted2[j])
                @inbounds for i = 1:ta.clause_size
                    if (x.x[i] == false) && (c[i] < ta.include_limit)
                        c[i] += one(UInt8)
                    end
                    if (x.x[i] == true) && (ci[i] < ta.include_limit)
                        ci[i] += one(UInt8)
                    end
                end
                literals2[j] = [@inbounds i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
                literals_inverted2[j] = [@inbounds i for i = 1:ta.clause_size if ci[i] >= ta.include_limit]
            end
        end
    end
end


function predict(tm::AbstractTMClassifier, x::AbstractTMInput)::Any
    best_vote::Int64 = typemin(Int64)
    best_cls::Any = nothing
    @inbounds for (cls, ta) in tm.clauses
        v::Int64 = -(vote(ta, x)...)
        if v > best_vote
            best_vote = v
            best_cls = cls
        end
    end
    return best_cls
end


# function predefine_batch_arrays(tm::AbstractTMClassifier)::Tuple{Vector{Int64}, Vector{class_type(tm)}, Vector{Int64}}
#     best_vote::Vector{Int64} = fill(typemin(Int64), 64)
#     best_cls::Vector = Vector{class_type(tm)}(undef, 64)
#     votes::Vector{Int64} = Vector{Int64}(undef, 64)
#     return best_vote, best_cls, votes
# end
function predefine_batch_arrays(tm::AbstractTMClassifier)::Tuple{Vector{Int64}, Vector{Int64}}
    best_vote::Vector{Int64} = fill(typemin(Int64), 64)
    votes::Vector{Int64} = Vector{Int64}(undef, 64)
    return best_vote, votes
end


#function predict(tm::AbstractTMClassifier, x::TMInputBatch, best_vote::Vector{Int64}, best_cls::Vector, votes::Vector{Int64})
function predict(tm::AbstractTMClassifier, x::TMInputBatch, best_vote::Vector{Int64}, votes::Vector{Int64})
    fill!(best_vote, typemin(Int64))
    best_cls = Vector{class_type(tm)}(undef, x.batch_size)
    @inbounds for (cls, ta) in tm.clauses
        fill!(votes, 0)
        votes = vote(ta, x, votes)
        @inbounds for i in 1:x.batch_size
            if votes[i] > best_vote[i]
                best_vote[i] = votes[i]
                best_cls[i] = cls
            end
        end
    end
    # Yes, we need to allocate a new array here using collect()
#    return best_cls[1:x.batch_size]
    return best_cls
end

function predict(tm::AbstractTMClassifier, x::TMInputBatch)::Vector
    predict(tm, x, predefine_batch_arrays(tm)...)
end


function predict(tm::AbstractTMClassifier, X::Vector{TMInput})::Vector
    predicted::Vector = Vector{class_type(tm)}(undef, length(X))  # Predefine vector for @threads access
    @threads for i in eachindex(X)
        predicted[i] = predict(tm, X[i])
    end
    return predicted
end


function predict(tm::AbstractTMClassifier, X::Vector{TMInputBatch})::Vector
    # Predefine vectors for @threads access
    predicted = Vector{Vector{class_type(tm)}}(undef, length(X))
    thread_best_vote = Vector{Vector{Int64}}(undef, Threads.nthreads())
#    thread_best_cls = Vector{Vector{class_type(tm)}}(undef, Threads.nthreads())
    thread_votes = Vector{Vector{Int64}}(undef, Threads.nthreads())
    for tid in 1:Threads.nthreads()
#        thread_best_vote[tid], thread_best_cls[tid], thread_votes[tid] = predefine_batch_arrays(tm)
        thread_best_vote[tid], thread_votes[tid] = predefine_batch_arrays(tm)
    end
    @threads for i in eachindex(X)
        tid::Int64 = Threads.threadid()
#        predicted[i] = predict(tm, X[i], thread_best_vote[tid], thread_best_cls[tid], thread_votes[tid])
        predicted[i] = predict(tm, X[i], thread_best_vote[tid], thread_votes[tid])
    end
    return predicted
end


function accuracy(predicted::Vector{T}, Y::Vector{T})::Float64 where T
    @assert length(predicted) == length(Y)
    return sum(@inbounds 1 for (p, y) in zip(predicted, Y) if p == y; init=0) / length(Y)
end


function accuracy(predicted::Vector{Vector{T}}, Y::Vector{T})::Float64 where T
    c::Int64 = 0
    a::Int64 = 0
    @inbounds for batch in predicted
        @inbounds @simd for pred in batch
            c += 1
            a += (pred == Y[c])
        end
    end
    @assert c == length(Y)
    return a / c
end


function train!(tm::TMClassifier, x::TMInput, y::Any; shuffle::Bool=true)
    if shuffle
        classes = Random.shuffle(collect(keys(tm.clauses)))
    else
        classes = keys(tm.clauses)
    end
#    feedback!(tm, tm.clauses[y], x, tm.clauses[y].positive_clauses, tm.clauses[y].positive_clauses_inverted, tm.clauses[y].negative_clauses, tm.clauses[y].negative_clauses_inverted, tm.clauses[y].positive_included_literals, tm.clauses[y].positive_included_literals_inverted, tm.clauses[y].negative_included_literals, tm.clauses[y].negative_included_literals_inverted, true)
    for cls in classes
        if cls != y
            feedback!(tm, tm.clauses[y], x, tm.clauses[y].positive_clauses, tm.clauses[y].positive_clauses_inverted, tm.clauses[y].negative_clauses, tm.clauses[y].negative_clauses_inverted, tm.clauses[y].positive_included_literals, tm.clauses[y].positive_included_literals_inverted, tm.clauses[y].negative_included_literals, tm.clauses[y].negative_included_literals_inverted, true)
            feedback!(tm, tm.clauses[cls], x, tm.clauses[cls].negative_clauses, tm.clauses[cls].negative_clauses_inverted, tm.clauses[cls].positive_clauses, tm.clauses[cls].positive_clauses_inverted, tm.clauses[cls].negative_included_literals, tm.clauses[cls].negative_included_literals_inverted, tm.clauses[cls].positive_included_literals, tm.clauses[cls].positive_included_literals_inverted, false)
        end
    end
end


function train!(tm::TMClassifier, X::Vector{TMInput}, Y::Vector; shuffle::Bool=true)
    # If not initialized yet
    if length(tm.clauses) == 0
        initialize!(tm, X, Y)
    end
    if shuffle
        X, Y = unzip(Random.shuffle(collect(zip(X, Y))))
    end
    @threads for i in eachindex(Y)
        train!(tm, X[i], Y[i], shuffle=shuffle)
    end
end


function train!(tm::TMClassifier, x_train::Vector, y_train::Vector, x_test::Vector, y_test::Vector, epochs::Int64; batch::Bool=true, shuffle::Bool=true, verbose::Int=1, best_tms_size::Int64=16, best_tms_compile::Bool=true)::Vector{Tuple{AbstractTMClassifier, Float64}}
    @assert best_tms_size in 1:2000
    if batch
        x_test = batches(x_test)
    end
    if length(tm.clauses) == 0
        initialize!(tm, x_train, y_train)
    end
    if verbose > 0
        println("\nRunning in $(nthreads()) threads.")
        println("Clauses: $(tm.clauses_num), T: $(tm.T), S: $(tm.S) (s: $(tm.s)), L: $(tm.L), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit).")
        println("Classes: $(length(tm.clauses)). Input vector size: $(length(x_train[1])) bits. Training dataset size: $(length(y_train)). Testing dataset size: $(length(y_test)).")
        println("Accuracy over $(epochs) epochs:\n")
    end
    best_tms = Tuple{AbstractTMClassifier, Float64}[]
    all_time = @elapsed begin
        for i in 1:epochs
            training_time = @elapsed train!(tm, x_train, y_train, shuffle=shuffle)
            testing_time = @elapsed begin
                acc = accuracy(predict(tm, x_test), y_test)
            end
            push!(best_tms, (best_tms_compile ? compile(tm, verbose=verbose - 1) : deepcopy(tm), acc))
            sort!(best_tms, by=last, rev=true)
            best_tms = best_tms[1:clamp(length(best_tms), length(best_tms), best_tms_size)]
            if verbose > 0
                @printf("#%s  Accuracy: %.2f%%  Best: %.2f%%  Training: %.3fs  Testing: %.3fs\n", i, acc * 100, best_tms[1][2] * 100, training_time, testing_time)
            end
        end
    end
    if verbose > 0
        elapsed = Time(0) + Second(floor(Int, all_time))
        @printf("\n%s epochs done in %s. Best accuracy: %.2f%%.\n", epochs, elapsed, best_tms[1][2] * 100)
        println("Clauses: $(tm.clauses_num), T: $(tm.T), S: $(tm.S) (s: $(tm.s)), L: $(tm.L), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit).")
        println("Classes: $(length(tm.clauses)). Input vector size: $(length(x_train[1])) bits. Training dataset size: $(length(y_train)). Testing dataset size: $(length(y_test)).\n")
    end
    return best_tms
end


function compile(tm::TMClassifier; verbose::Int=0)::TMClassifierCompiled
    if verbose > 0
        print("Compiling model... ")
        pos = []
        neg = []
        pos_inv = []
        neg_inv = []
    end
        all_time = @elapsed begin
        tmc = TMClassifierCompiled{class_type(tm)}(tm.clauses_num, tm.T, tm.S, tm.s, tm.L)
        for (cls, ta) in tm.clauses
            tmc.clauses[cls] = TATeamCompiled(tm.clauses_num)
            for (j, (c, ci)) in enumerate(zip(eachcol(ta.positive_clauses), eachcol(ta.positive_clauses_inverted)))
                tmc.clauses[cls].positive_included_literals[j] = [i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
                tmc.clauses[cls].positive_included_literals_inverted[j] = [i for i = 1:ta.clause_size if ci[i] >= ta.include_limit]
                if verbose > 0
                    append!(pos, length(tmc.clauses[cls].positive_included_literals[j]))
                    append!(pos_inv, length(tmc.clauses[cls].positive_included_literals_inverted[j]))
                end
            end
            for (j, (c, ci)) in enumerate(zip(eachcol(ta.negative_clauses), eachcol(ta.negative_clauses_inverted)))
                tmc.clauses[cls].negative_included_literals[j] = [i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
                tmc.clauses[cls].negative_included_literals_inverted[j] = [i for i = 1:ta.clause_size if ci[i] >= ta.include_limit]
                if verbose > 0
                    append!(neg, length(tmc.clauses[cls].negative_included_literals[j]))
                    append!(neg_inv, length(tmc.clauses[cls].negative_included_literals_inverted[j]))
                end
            end
        end
    end
    if verbose > 0
        @printf("Done. Time elapsed: %.3fs\n", all_time)
        pos_sum = sum(pos)
        neg_sum = sum(neg)
        pos_inv_sum = sum(pos_inv)
        neg_inv_sum = sum(neg_inv)
        total = (pos_sum + neg_sum + pos_inv_sum + neg_inv_sum)
        println("Included literals:")
        @printf("  Positive: %s, Negative: %s, Positive Inverted: %s, Negative Inverted: %s, Total: %s, Per clause: %.2f\n", pos_sum, neg_sum, pos_inv_sum, neg_inv_sum, total, total / (length(tm.clauses) * tm.clauses_num * 2))
        @printf("  Positive min: %s, max: %s, mean: %.2f, median: %.2f\n", minimum(pos), maximum(pos), mean(pos), median(pos))
        @printf("  Negative min: %s, max: %s, mean: %.2f, median: %.2f\n", minimum(neg), maximum(neg), mean(neg), median(neg))
        @printf("  Positive Inverted min: %s, max: %s, mean: %.2f, median: %.2f\n", minimum(pos_inv), maximum(pos_inv), mean(pos_inv), median(pos_inv))
        @printf("  Negative Inverted min: %s, max: %s, mean: %.2f, median: %.2f\n", minimum(neg_inv), maximum(neg_inv), mean(neg_inv), median(neg_inv))
    end
    return tmc
end


function compile(tms::Vector{Tuple{AbstractTMClassifier, Float64}})::Vector{Tuple{TMClassifierCompiled, Float64}}
    return [(compile(tm, verbose=0), acc) for (tm, acc) in tms]
end


function optimize!(tm::AbstractTMClassifier, X::Vector{TMInput}; verbose::Int=0)
    if verbose > 0
        print("Optimizing model... ")
    end
    all_time = @elapsed begin

        for (cls, ta) in tm.clauses
            for (check, included_literals) in ((false, ta.positive_included_literals), (false, ta.negative_included_literals), (true, ta.positive_included_literals_inverted), (true, ta.negative_included_literals_inverted))
                @threads for c in included_literals
                    d = Dict(k => 0 for k in c)
                    for x in X
                        for k in keys(d)
                            if x.x[k] == check
                                d[k] += 1
                            end
                        end
                    end
                    sort!(c, lt=(a, b) -> d[a] > d[b])
                end
            end
        end
    end
    if verbose > 0
        @printf("Done. Time elapsed: %.3fs\n", all_time)
    end
end


function save(tm::Union{AbstractTMClassifier, Tuple{AbstractTMClassifier, Float64}, Vector{Tuple{AbstractTMClassifier, Float64}}}, filepath::AbstractString)
    if !endswith(filepath, ".tm")
        filepath = string(filepath, ".tm")
    end
    print("Saving model to $(filepath)... ")
    Serialization.serialize(filepath, tm)
    println("Done.\n")
end


function load(filepath::AbstractString)
    if !endswith(filepath, ".tm")
        filepath = string(filepath, ".tm")
    end
    print("Loading model from $(filepath)... ")
    println("Done.\n")
    return Serialization.deserialize(filepath)
end


function mix(clauses_vec::Matrix{UInt8}...)::Matrix{UInt8}
    return [max(ms...) for ms in zip(clauses_vec...)]
end


function merge!(new_tm::TMClassifier, tms::TMClassifier...; algo::Symbol=:merge)
    @assert algo in (:merge, :join)
    @threads for cls in collect(keys(new_tm.clauses))
        if algo == :merge
            new_tm.clauses[cls].positive_clauses = mix((tm.clauses[cls].positive_clauses for tm in tms)...)
            new_tm.clauses[cls].negative_clauses = mix((tm.clauses[cls].negative_clauses for tm in tms)...)
            new_tm.clauses[cls].positive_clauses_inverted = mix((tm.clauses[cls].positive_clauses_inverted for tm in tms)...)
            new_tm.clauses[cls].negative_clauses_inverted = mix((tm.clauses[cls].negative_clauses_inverted for tm in tms)...)
        elseif algo == :join
            new_tm.clauses[cls].positive_clauses = hcat((tm.clauses[cls].positive_clauses for tm in tms)...)
            new_tm.clauses[cls].negative_clauses = hcat((tm.clauses[cls].negative_clauses for tm in tms)...)
            new_tm.clauses[cls].positive_clauses_inverted = hcat((tm.clauses[cls].positive_clauses_inverted for tm in tms)...)
            new_tm.clauses[cls].negative_clauses_inverted = hcat((tm.clauses[cls].negative_clauses_inverted for tm in tms)...)

            clauses_num_half = size(new_tm.clauses[cls].positive_clauses, 2)
            new_tm.clauses_num = clauses_num_half * 2

            new_tm.clauses[cls].positive_included_literals = fill([], floor(Int, clauses_num_half))
            new_tm.clauses[cls].negative_included_literals = fill([], floor(Int, clauses_num_half))
            new_tm.clauses[cls].positive_included_literals_inverted = fill([], floor(Int, clauses_num_half))
            new_tm.clauses[cls].negative_included_literals_inverted = fill([], floor(Int, clauses_num_half))
        end
        @inbounds for (j, c) in enumerate(eachcol(new_tm.clauses[cls].positive_clauses))
            new_tm.clauses[cls].positive_included_literals[j] = [@inbounds i for i = 1:new_tm.clauses[cls].clause_size if c[i] >= new_tm.clauses[cls].include_limit]
        end
        @inbounds for (j, c) in enumerate(eachcol(new_tm.clauses[cls].negative_clauses))
            new_tm.clauses[cls].negative_included_literals[j] = [@inbounds i for i = 1:new_tm.clauses[cls].clause_size if c[i] >= new_tm.clauses[cls].include_limit]
        end
        @inbounds for (j, c) in enumerate(eachcol(new_tm.clauses[cls].positive_clauses_inverted))
            new_tm.clauses[cls].positive_included_literals_inverted[j] = [@inbounds i for i = 1:new_tm.clauses[cls].clause_size if c[i] >= new_tm.clauses[cls].include_limit]
        end
        @inbounds for (j, c) in enumerate(eachcol(new_tm.clauses[cls].negative_clauses_inverted))
            new_tm.clauses[cls].negative_included_literals_inverted[j] = [@inbounds i for i = 1:new_tm.clauses[cls].clause_size if c[i] >= new_tm.clauses[cls].include_limit]
        end
    end
end


function merge!(new_tm::TMClassifierCompiled, tms::TMClassifierCompiled...; algo::Symbol=:merge)
    @assert algo in (:merge, :join)
    @threads for cls in collect(keys(new_tm.clauses))
        if algo == :merge
            new_tm.clauses[cls].positive_included_literals = [sort(collect(Set(vcat(ls...)))) for ls in zip((tm.clauses[cls].positive_included_literals for tm in tms)...)]
            new_tm.clauses[cls].negative_included_literals = [sort(collect(Set(vcat(ls...)))) for ls in zip((tm.clauses[cls].negative_included_literals for tm in tms)...)]
            new_tm.clauses[cls].positive_included_literals_inverted = [sort(collect(Set(vcat(ls...)))) for ls in zip((tm.clauses[cls].positive_included_literals_inverted for tm in tms)...)]
            new_tm.clauses[cls].negative_included_literals_inverted = [sort(collect(Set(vcat(ls...)))) for ls in zip((tm.clauses[cls].negative_included_literals_inverted for tm in tms)...)]
        elseif algo == :join
            new_tm.clauses[cls].positive_included_literals = sort(collect(Set(vcat((tm.clauses[cls].positive_included_literals for tm in tms)...))))
            new_tm.clauses[cls].negative_included_literals = sort(collect(Set(vcat((tm.clauses[cls].negative_included_literals for tm in tms)...))))
            new_tm.clauses[cls].positive_included_literals_inverted = sort(collect(Set(vcat((tm.clauses[cls].positive_included_literals_inverted for tm in tms)...))))
            new_tm.clauses[cls].negative_included_literals_inverted = sort(collect(Set(vcat((tm.clauses[cls].negative_included_literals_inverted for tm in tms)...))))
        end
    end
end


function merge(tms::AbstractTMClassifier...; algo::Symbol=:merge)::AbstractTMClassifier
    @assert algo in (:merge, :join)
    new_tm = deepcopy(first(tms))
    merge!(new_tm, tms...; algo=algo)
    return new_tm
end


function combine(tms, k::Int64, x_test::Vector, y_test::Vector; algo::Symbol=:merge, batch::Bool=true)::Tuple{AbstractTMClassifier, Float64}
    @assert algo in (:merge, :join)
    if batch
        x_test = batches(x_test)
    end
    tm = deepcopy(tms[1][1])
    best = (nothing, 0.0)
    combinations = Set(s for s in (Set(c) for c in Iterators.product((1:length(tms) for _ in 1:k)...)) if length(s) == k)
    println("Trying to find best combine accuracy among $(length(combinations)) of $(k) combined models (algo: $(algo), batch size: $(length(tms)))...")
    all_time = @elapsed begin
        for (i, c) in enumerate(combinations)
            merging_time = @elapsed begin
                merge!(tm, (tms[t][1] for t in c)...; algo=algo)
            end
            testing_time = @elapsed begin
                acc = accuracy(predict(tm, x_test), y_test)
            end
            if acc >= last(best)
                best = (deepcopy(tm), acc)
            end
            @printf("#%s  Accuracy: %s = %.2f%%  Best: %.2f%%  Merging: %.3fs  Testing: %.3fs\n", i, join([@sprintf("%.2f%%", tms[t][2] * 100) for t in c], " + "), acc * 100, last(best) * 100, merging_time, testing_time)
        end
    end
    elapsed = Time(0) + Second(floor(Int, all_time))
    @printf("Time elapsed: %s. Best %s combined models accuracy (algo: %s, batch size: %s): %.2f%%.\n\n", elapsed, k, algo, length(tms), last(best) * 100)
    return best
end


function benchmark(tm::AbstractTMClassifier, X::Vector{TMInput}, Y::Vector, loops::Int64; batch::Bool=true, warmup::Bool=true)
    @printf("CPU: %s\n", Sys.cpu_info()[1].model)
    @printf("Running in %s threads.\n", nthreads())
    println("Input vector size: $(length(X[1])) bits.")
    print("Preparing input data for benchmark... ")
    GC.gc()
    prepare_time = @elapsed begin
        # Permutate in random order
        len = length(Y)
        perm = Vector{Int32}(undef, len * loops)
        i::Int64 = 0
        @inbounds @fastmath for _ in 1:loops
            @inbounds @fastmath for r in Random.shuffle(UnitRange{Int32}(1:len))
                i += 1
                perm[i] = r
            end
        end
        # Multiply X and Y by loops times
        if batch
            X = batches(X[perm])
        else
            _X::Vector{TMInput} = Vector{TMInput}(undef, length(perm))
            @threads for i in eachindex(_X)
                # This is 4.5x faster than deepcopy()
                _X[i] = TMInput(X[perm[i]].x)
            end
            X = _X
        end
        Y = Y[perm]
    end
    @printf("Done. Elapsed %.3f seconds.\n", prepare_time)
    GC.gc()
    X_size = Base.summarysize(X[1]) * length(X)
    if warmup
        print("Warm-up started... ")
        warmup_time = @elapsed begin
            predict(tm, X)
        end
        @printf("Done. Elapsed %.3f seconds.\n", warmup_time)
    end
    model_type = last(split(string(typeof(tm)), '.'))
    if batch
        @printf("Benchmark for %s model in batch mode (batch size = %s) started... ", model_type, ndigits(typemax(typeof(X[1][1])), base=2))
    else
        @printf("Benchmark for %s model started... ", model_type)
    end
    GC.gc()
    GC.enable(false)
    bench_time = @elapsed begin
        predicted = predict(tm, X)
    end
    println("Done.")
    GC.enable(true)
    if batch
        predicted = vcat(predicted...)
    end
    @printf("%s predictions processed in %.3f seconds.\n", length(predicted), bench_time)
    @printf("Performance: %s predictions per second.\n", floor(Int, length(predicted) / bench_time))
    @printf("Throughput: %.3f GB/s.\n", X_size / 1024^3 / bench_time)
    @printf("Input data size: %.3f GB.\n", X_size / 1024^3)
    @printf("Parameters during training: %s.\n", tm.clauses_num * length(keys(tm.clauses)) * length(X[1]) * 2)
    @printf("Parameters after training and compilation: %s.\n", diff_count(tm)[1])
    @printf("Accuracy: %.2f%%.\n\n", accuracy(predicted, Y) * 100)
end


function diff_count(tm::AbstractTMClassifier)::Tuple{Int64, Int64, Int64, Int64, Int64}
    pos = []
    pos_inv = []
    neg = []
    neg_inv = []
    for (k, clauses) in tm.clauses
        for c in clauses.positive_included_literals
            append!(pos, [c])
        end
        for c in clauses.negative_included_literals
            append!(neg, [c])
        end
        for c in clauses.positive_included_literals_inverted
            append!(pos_inv, [c])
        end
        for c in clauses.negative_included_literals_inverted
            append!(neg_inv, [c])
        end
    end
    pos = vcat(pos...)
    neg = vcat(neg...)
    pos_inv = vcat(pos_inv...)
    neg_inv = vcat(neg_inv...)
    count = length(pos) + length(neg) + length(pos_inv) + length(neg_inv)
    # FIXME: irrelevant info
    return count, length(union(pos, neg)), length(intersect(pos, neg)), length(pos) - length(Set(pos)), length(neg) - length(Set(neg))
end


function transpose(tm::TMClassifierCompiled; algo::Symbol=:join)::TMClassifierCompiled
    @assert algo == :join
    @assert iseven(tm.clauses_num)
    tmc = TMClassifierCompiled(length(tm.clauses) * 2, tm.T, tm.R, tm.L)
    for cls in 1:Int(tm.clauses_num / 2)
        tmc.clauses[cls] = TATeamCompiled(length(tm.clauses) * 2)
        tmc.clauses[cls].positive_included_literals = [ta.positive_included_literals[cls] for (_, ta) in tm.clauses]
        tmc.clauses[cls].negative_included_literals = [ta.negative_included_literals[cls] for (_, ta) in tm.clauses]
    end
    return tmc
end

end # module
