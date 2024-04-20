module Tsetlin

export TMInput, TMClassifier, train!, predict, accuracy, save, load

using Dates
using Random
using Base.Threads
using Serialization
using Statistics: mean, median
using Printf: @printf, @sprintf


unzip(a) = (getfield.(a, x) for x in fieldnames(eltype(a)))

isbitset(x::UInt64, n::UInt64)::Bool = (x >> n) & one(UInt64)

function poly_2_eval(p)::UInt64
    ex::UInt64 = zero(UInt64)
    @inbounds for e in p
        ex *= 2
        ex += e
    end
    return bitreverse(ex % UInt64)
end


abstract type AbstractTATeam end
abstract type AbstractTMClassifier end


mutable struct TATeam <: AbstractTATeam
    const include_limit::UInt8
    const state_min::UInt8
    const state_max::UInt8
    positive_clauses::Matrix{UInt8}
    negative_clauses::Matrix{UInt8}
    positive_included_literals::Vector{Vector{UInt16}}
    negative_included_literals::Vector{Vector{UInt16}}
    const clause_size::Int64

    function TATeam(clause_size::Int64, clauses_num::Int64, include_limit::Int64, state_min::Int64, state_max::Int64)
        positive_clauses = fill(UInt8(include_limit - 1), clause_size, floor(Int, clauses_num / 2))
        negative_clauses = fill(UInt8(include_limit - 1), clause_size, floor(Int, clauses_num / 2))
        positive_included_literals = fill([], floor(Int, clauses_num / 2))
        negative_included_literals = fill([], floor(Int, clauses_num / 2))
        return new(include_limit, state_min, state_max, positive_clauses, negative_clauses, positive_included_literals, negative_included_literals, clause_size)
    end
end


mutable struct TMClassifier <: AbstractTMClassifier
    clauses_num::Int64
    T::Int64
    R::Float64
    L::Int64
    const include_limit::Int64
    const state_min::Int64
    const state_max::Int64
    const clauses::Dict{Any, TATeam}

    function TMClassifier(clauses_num::Int64, T::Int64, R::Float64; states_num::Int64=256, include_limit::Int64=128, L::Int64=16)
        return new(clauses_num, T, R, L, include_limit, typemin(UInt8), states_num - 1, Dict())
    end
end


mutable struct TATeamCompiled <: AbstractTATeam
    positive_included_literals::Vector{Vector{UInt16}}
    negative_included_literals::Vector{Vector{UInt16}}

    function TATeamCompiled(clauses_num::Int64)
        positive_included_literals = fill([], floor(Int, clauses_num / 2))
        negative_included_literals = fill([], floor(Int, clauses_num / 2))
        return new(positive_included_literals, negative_included_literals)
    end
end


struct TMClassifierCompiled <: AbstractTMClassifier
    clauses_num::Int64
    T::Int64
    R::Float64
    L::Int64
    clauses::Dict{Any, TATeamCompiled}

    function TMClassifierCompiled(clauses_num::Int64, T::Int64, R::Float64, L::Int64)
        return new(clauses_num, T, R, L, Dict())
    end
end


abstract type AbstractTMInput <: AbstractVector{Bool} end

struct TMInput <: AbstractTMInput
    x::Vector{Bool}

    function TMInput(x::Vector{Bool}; negate::Bool=true)
        return negate ? new([x; [!_x for _x in x]]) : new(x)
    end
end

Base.IndexStyle(::Type{<:TMInput}) = IndexLinear()
Base.size(x::TMInput)::Tuple{Int64} = size(x.x)
Base.getindex(x::TMInput, i::Int)::Bool = x.x[i]


struct TMInputBatch <: AbstractTMInput
    x::Vector{UInt64}
    batch_size::Int64

    function TMInputBatch(X::Vector{TMInput})
        @assert 1 <= length(X) <= 64 "Size of input Vector must be in 1..64."
        if length(X) == 64
            return new([poly_2_eval(!X[i][j] for i in 1:64) for j in 1:length(X[1])], 64)
        else
            return new([poly_2_eval(i <= length(X) ? !X[i][j] : true for i in 1:64) for j in 1:length(X[1])], length(X))
        end
    end
end

Base.IndexStyle(::Type{<:TMInputBatch}) = IndexLinear()
Base.size(x::TMInputBatch)::Tuple{Int64} = size(x.x)
Base.getindex(x::TMInputBatch, i::Int)::UInt64 = x.x[i]


function initialize!(tm::TMClassifier, X::Vector{TMInput}, Y::Vector)
    for cls in collect(Set(Y))
        tm.clauses[cls] = TATeam(length(first(X)), tm.clauses_num, tm.include_limit, tm.state_min, tm.state_max)
    end
end


function check_clause(x::TMInput, literals::Vector{UInt16})::Bool
    @inbounds for i in eachindex(literals)
        if !x[literals[i]]
            return false
        end
    end
    return true
end


function check_clause(x::TMInputBatch, literals::Vector{UInt16})::UInt64
    b::UInt64 = typemin(UInt64)
    @inbounds for i in eachindex(literals)
        b |= x[literals[i]]
    end
    return b
end


function vote(ta::AbstractTATeam, x::TMInput)::Tuple{Int64, Int64}
    pos = sum(check_clause(x, ta.positive_included_literals[i]) for i in eachindex(ta.positive_included_literals))
    neg = sum(check_clause(x, ta.negative_included_literals[i]) for i in eachindex(ta.negative_included_literals))
    return pos, neg
end


function vote(ta::AbstractTATeam, x::TMInputBatch)::Tuple{Vector{Int64}, Vector{Int64}}
    pos_sum::Vector{Int64} = fill(0, 64)
    neg_sum::Vector{Int64} = fill(0, 64)
    @inbounds for p in (check_clause(x, ta.positive_included_literals[j]) for j in eachindex(ta.positive_included_literals))
        @inbounds @simd for i in 1:64
            pos_sum[i] -= isbitset(p, UInt64(i - 1))
        end
    end
    @inbounds for n in (check_clause(x, ta.negative_included_literals[j]) for j in eachindex(ta.negative_included_literals))
        @inbounds @simd for i in 1:64
            neg_sum[i] -= isbitset(n, UInt64(i - 1))
        end
    end
    return pos_sum, neg_sum
end


function feedback!(tm::TMClassifier, ta::TATeam, x::TMInput, clauses1::Matrix{UInt8}, clauses2::Matrix{UInt8}, literals1::Vector{Vector{UInt16}}, literals2::Vector{Vector{UInt16}}, positive::Bool)
    v::Int64 = clamp(-(vote(ta, x)...), -tm.T, tm.T)
    update::Float64 = (positive ? (tm.T - v) : (tm.T + v)) / (tm.T * 2)

    # Feedback 1
    @inbounds for (j, c) in enumerate(eachcol(clauses1))
        if (rand() < update)
            if check_clause(x, literals1[j])
                if (length(literals1[j]) <= tm.L)
                    @inbounds for i = 1:ta.clause_size
                        if (x[i] == true) && (c[i] < ta.state_max)
                            c[i] += one(UInt8)
                        end
                    end
                end
                @inbounds for i = 1:ta.clause_size
                    if (rand() > tm.R) && (x[i] == false) && (c[i] < ta.include_limit) && (c[i] > ta.state_min)
                        c[i] -= one(UInt8)
                    end
                end
            else
                @inbounds for i = 1:ta.clause_size
                    if (rand() > tm.R) && (c[i] > ta.state_min)
                        c[i] -= one(UInt8)
                    end
                end
            end
            literals1[j] = [@inbounds i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
        end
    end
    # Feedback 2
    @inbounds for (j, c) in enumerate(eachcol(clauses2))
        if (rand() < update)
            if check_clause(x, literals2[j])
                @inbounds for i = 1:ta.clause_size
                    if (rand() <= tm.R) && (x[i] == false) && (c[i] < ta.include_limit)
                        c[i] += one(UInt8)
                    end
                end
                literals2[j] = [@inbounds i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
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


function predict(tm::AbstractTMClassifier, x::TMInputBatch)::Vector{Any}
    best_vote::Vector{Int64} = fill(typemin(Int64), x.batch_size)
    best_cls::Vector{Any} = fill(nothing, x.batch_size)
    @inbounds for (cls, ta) in tm.clauses
        pos_sum, neg_sum = vote(ta, x)
        @inbounds for i in 1:x.batch_size
            v::Int64 = pos_sum[i] - neg_sum[i]
            if v > best_vote[i]
                best_vote[i] = v
                best_cls[i] = cls
            end
        end
    end
    return best_cls
end


function predict(tm::AbstractTMClassifier, X::Vector{TMInput})::Vector
    predicted::Vector = Vector{eltype(first(keys(tm.clauses)))}(undef, length(X))  # Predefine vector for @threads access
    @threads for i in eachindex(X)
        predicted[i] = predict(tm, X[i])
    end
    return predicted
end


function predict(tm::AbstractTMClassifier, X::Vector{TMInputBatch})::Vector
    predicted::Vector = Vector{Vector{eltype(first(keys(tm.clauses)))}}(undef, length(X))  # Predefine vector for @threads access
    @threads for i in eachindex(X)
        predicted[i] = predict(tm, X[i])
    end
    return vcat(predicted...)
end


function accuracy(predicted::Vector, Y::Vector)::Float64
    @assert eltype(predicted) == eltype(Y)
    @assert length(predicted) == length(Y)
    return sum(@inbounds 1 for (p, y) in zip(predicted, Y) if p == y; init=0) / length(Y)
end


function train!(tm::TMClassifier, x::TMInput, y::Any; shuffle::Bool=true)
    if shuffle
        classes = Random.shuffle(collect(keys(tm.clauses)))
    else
        classes = keys(tm.clauses)
    end
    for cls in classes
        if cls != y
            feedback!(tm, tm.clauses[y], x, tm.clauses[y].positive_clauses, tm.clauses[y].negative_clauses, tm.clauses[y].positive_included_literals, tm.clauses[y].negative_included_literals, true)
            feedback!(tm, tm.clauses[cls], x, tm.clauses[cls].negative_clauses, tm.clauses[cls].positive_clauses, tm.clauses[cls].negative_included_literals, tm.clauses[cls].positive_included_literals, false)
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


function train!(tm::TMClassifier, x_train::Vector, y_train::Vector, x_test::Vector, y_test::Vector, epochs::Int64; batch::Bool=true, shuffle::Bool=true, verbose::Int=1, best_tms_size::Int64=16, best_tms_compile::Bool=true)::Tuple{TMClassifier, Vector{Tuple{Float64, AbstractTMClassifier}}}
    @assert best_tms_size in 1:2000
    if batch
        x_test = batches(x_test)
    end
    if verbose > 0
        println("\nRunning in $(nthreads()) threads.")
        println("Accuracy over $(epochs) epochs (Clauses: $(tm.clauses_num), T: $(tm.T), R: $(tm.R), L: $(tm.L), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit)):\n")
    end
    best_tm = (0.0, nothing)
    best_tms = Tuple{Float64, AbstractTMClassifier}[]
    all_time = @elapsed begin
        for i in 1:epochs
            training_time = @elapsed train!(tm, x_train, y_train, shuffle=shuffle)
            testing_time = @elapsed begin
                acc = accuracy(predict(tm, x_test), y_test)
            end
            if acc >= first(best_tm)
                best_tm = (acc, deepcopy(tm))
            end
            push!(best_tms, (acc, best_tms_compile ? compile(tm, verbose=verbose - 1) : deepcopy(tm)))
            sort!(best_tms, by=first, rev=true)
            best_tms = best_tms[1:clamp(length(best_tms), length(best_tms), best_tms_size)]
            if verbose > 0
                @printf("#%s  Accuracy: %.2f%%  Best: %.2f%%  Training: %.3fs  Testing: %.3fs\n", i, acc * 100, best_tm[1] * 100, training_time, testing_time)
            end
        end
    end
    if verbose > 0
        println("\nDone. $(epochs) epochs (Clauses: $(tm.clauses_num), T: $(tm.T), R: $(tm.R), L: $(tm.L), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit)).")
        elapsed = Time(0) + Second(floor(Int, all_time))
        @printf("Time elapsed: %s. Best accuracy was: %.2f%%.\n\n", elapsed, best_tm[1] * 100)
    end
    return best_tm[2], best_tms
end


function compile(tm::TMClassifier; verbose::Int=0)::TMClassifierCompiled
    if verbose > 0
        print("Compiling model... ")
        pos = []
        neg = []
    end
        all_time = @elapsed begin
        tmc = TMClassifierCompiled(tm.clauses_num, tm.T, tm.R, tm.L)
        for (cls, ta) in tm.clauses
            tmc.clauses[cls] = TATeamCompiled(tm.clauses_num)
            for (j, c) in enumerate(eachcol(ta.positive_clauses))
                tmc.clauses[cls].positive_included_literals[j] = [i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
                if verbose > 0
                    append!(pos, length(tmc.clauses[cls].positive_included_literals[j]))
                end
            end
            for (j, c) in enumerate(eachcol(ta.negative_clauses))
                tmc.clauses[cls].negative_included_literals[j] = [i for i = 1:ta.clause_size if c[i] >= ta.include_limit]
                if verbose > 0
                    append!(neg, length(tmc.clauses[cls].negative_included_literals[j]))
                end
            end
        end
    end
    if verbose > 0
        @printf("Done. Time elapsed: %.3fs\n", all_time)
        pos_sum = sum(pos)
        neg_sum = sum(neg)
        println("Included literals:")
        @printf("  Positive: %s, Negative: %s, Total: %s, Per clause: %.2f\n", pos_sum, neg_sum, (pos_sum + neg_sum), (pos_sum + neg_sum) / (length(tm.clauses) * tm.clauses_num))
        @printf("  Positive min: %s, max: %s, mean: %.2f, median: %.2f\n", minimum(pos), maximum(pos), mean(pos), median(pos))
        @printf("  Negative min: %s, max: %s, mean: %.2f, median: %.2f\n", minimum(neg), maximum(neg), mean(neg), median(neg))
    end
    return tmc
end


function compile(tms::Vector{Tuple{Float64, AbstractTMClassifier}})::Vector{Tuple{Float64, TMClassifierCompiled}}
    return [(acc, compile(tm, verbose=0)) for (acc, tm) in tms]
end


function optimize!(tm::AbstractTMClassifier, X::Vector{TMInput}; verbose::Int=0)
    if verbose > 0
        print("Optimizing model... ")
    end
    all_time = @elapsed begin
        for (cls, ta) in tm.clauses
            for included_literals in (ta.positive_included_literals, ta.negative_included_literals)
                @threads for c in included_literals
                    d = Dict(k => 0 for k in c)
                    for x in X
                        for k in keys(d)
                            if x[k] == false
                                d[k] += 1
                            end
                        end
                    end
                    sort!(c, lt=(a, b) -> d[a] > d[b])
                end
                sort!(included_literals)
            end
        end
    end
    if verbose > 0
        @printf("Done. Time elapsed: %.3fs\n", all_time)
    end
end


function save(tm::Union{AbstractTMClassifier, Tuple{Float64, AbstractTMClassifier}, Vector{Tuple{Float64, AbstractTMClassifier}}}, filepath::AbstractString)
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
        elseif algo == :join
            new_tm.clauses[cls].positive_clauses = hcat((tm.clauses[cls].positive_clauses for tm in tms)...)
            new_tm.clauses[cls].negative_clauses = hcat((tm.clauses[cls].negative_clauses for tm in tms)...)

            clauses_num_half = size(new_tm.clauses[cls].positive_clauses, 2)
            new_tm.clauses_num = clauses_num_half * 2

            new_tm.clauses[cls].positive_included_literals = fill([], floor(Int, clauses_num_half))
            new_tm.clauses[cls].negative_included_literals = fill([], floor(Int, clauses_num_half))
        end
        @inbounds for (j, c) in enumerate(eachcol(new_tm.clauses[cls].positive_clauses))
            new_tm.clauses[cls].positive_included_literals[j] = [@inbounds i for i = 1:new_tm.clauses[cls].clause_size if c[i] >= new_tm.clauses[cls].include_limit]
        end
        @inbounds for (j, c) in enumerate(eachcol(new_tm.clauses[cls].negative_clauses))
            new_tm.clauses[cls].negative_included_literals[j] = [@inbounds i for i = 1:new_tm.clauses[cls].clause_size if c[i] >= new_tm.clauses[cls].include_limit]
        end
    end
end


function merge!(new_tm::TMClassifierCompiled, tms::TMClassifierCompiled...; algo::Symbol=:merge)
    @assert algo in (:merge, :join)
    @threads for cls in collect(keys(new_tm.clauses))
        if algo == :merge
            new_tm.clauses[cls].positive_included_literals = [sort(collect(Set(vcat(ls...)))) for ls in zip((tm.clauses[cls].positive_included_literals for tm in tms)...)]
            new_tm.clauses[cls].negative_included_literals = [sort(collect(Set(vcat(ls...)))) for ls in zip((tm.clauses[cls].negative_included_literals for tm in tms)...)]
        elseif algo == :join
            new_tm.clauses[cls].positive_included_literals = sort(collect(Set(vcat((tm.clauses[cls].positive_included_literals for tm in tms)...))))
            new_tm.clauses[cls].negative_included_literals = sort(collect(Set(vcat((tm.clauses[cls].negative_included_literals for tm in tms)...))))
        end
    end
end


function merge(tms::AbstractTMClassifier...; algo::Symbol=:merge)::AbstractTMClassifier
    @assert algo in (:merge, :join)
    new_tm = deepcopy(first(tms))
    merge!(new_tm, tms...; algo=algo)
    return new_tm
end


function combine(tms, k::Int64, x_test::Vector, y_test::Vector; algo::Symbol=:merge, batch::Bool=true)::Tuple{Float64, AbstractTMClassifier}
    @assert algo in (:merge, :join)
    if batch
        x_test = batches(x_test)
    end
    tm = deepcopy(tms[1][2])
    best = (0.0, nothing)
    combinations = Set(s for s in (Set(c) for c in Iterators.product((1:length(tms) for _ in 1:k)...)) if length(s) == k)
    println("Trying to find best combine accuracy among $(length(combinations)) of $(k) combined models (algo: $(algo), batch size: $(length(tms)))...")
    all_time = @elapsed begin
        for (i, c) in enumerate(combinations)
            merging_time = @elapsed begin
                merge!(tm, (tms[t][2] for t in c)...; algo=algo)
            end
            testing_time = @elapsed begin
                acc = accuracy(predict(tm, x_test), y_test)
            end
            if acc >= first(best)
                best = (acc, deepcopy(tm))
            end
            @printf("#%s  Accuracy: %s = %.2f%%  Best: %.2f%%  Merging: %.3fs  Testing: %.3fs\n", i, join([@sprintf("%.2f%%", tms[t][1] * 100) for t in c], " + "), acc * 100, first(best) * 100, merging_time, testing_time)
        end
    end
    elapsed = Time(0) + Second(floor(Int, all_time))
    @printf("Time elapsed: %s. Best %s combined models accuracy (algo: %s, batch size: %s): %.2f%%.\n\n", elapsed, k, algo, length(tms), first(best) * 100)
    return best
end


function batches(X::Vector{TMInput})::Vector{TMInputBatch}
    _d, _r = divrem(length(X), 64)
    _X::Vector{TMInputBatch} = Vector{TMInputBatch}(undef, _r == 0 ? _d : _d + 1)  # Predefine vector for @threads access
    @threads for (j, i) in collect(enumerate(1:64:length(X)))
        _X[j] = (j <= _d) ? TMInputBatch(X[i:i+63]) : TMInputBatch(X[i:i+_r - 1])
    end
    return _X
end


function benchmark(tm::AbstractTMClassifier, X::Vector{TMInput}, Y::Vector, loops::Int64; batch::Bool=false, deep_copy::Bool=false, warmup::Bool=true)
    @printf("CPU: %s\n", Sys.cpu_info()[1].model)
    print("Preparing input data for benchmark... ")
    GC.gc()
    prepare_time = @elapsed begin
        # Permutate in random order
        perm = Random.shuffle(1:length(Y) * loops)
        # Multiply X and Y by loops times
        if deep_copy && !batch
            X = vcat((deepcopy(X) for _ in 1:loops)...)[perm]
            Y = vcat((deepcopy(Y) for _ in 1:loops)...)[perm]
        else
            X = vcat((X for _ in 1:loops)...)[perm]
            Y = vcat((Y for _ in 1:loops)...)[perm]
        end
        if batch
            X = batches(X)
        end
    end
    @printf("Done. Elapsed %.3f seconds.\n", prepare_time)
    GC.gc()
    if warmup
        @printf("Warm-up started in %s threads... ", nthreads())
        warmup_time = @elapsed begin
            predict(tm, X)
        end
        @printf("Done. Elapsed %.3f seconds.\n", warmup_time)
    end
    model_type = last(split(string(typeof(tm)), '.'))
    if batch
        @printf("Benchmark for %s model in batch mode (batch size = %s) started in %s threads... ", model_type, ndigits(typemax(typeof(X[1][1])), base=2), nthreads())
    else
        @printf("Benchmark for %s model started in %s threads... ", model_type, nthreads())
    end
    GC.gc()
    GC.enable(false)
    bench_time = @elapsed begin
        predicted = predict(tm, X)
    end
    X_size = Base.summarysize(X)
    GC.enable(true)
    println("Done.")
    @printf("%s predictions processed in %.3f seconds.\n", length(predicted), bench_time)
    @printf("Performance: %s predictions per second.\n", floor(Int, length(predicted) / bench_time))
    @printf("Throughput: %.3f GB/s.\n", X_size / 1024^3 / bench_time)
    @printf("Input data size: %.3f GB.\n", X_size / 1024^3)
    @printf("Parameters during training: %s.\n", tm.clauses_num * length(keys(tm.clauses)) * length(X[1]))
    @printf("Parameters after training and compilation: %s.\n", diff_count(tm)[3])
    @printf("Accuracy: %.2f%%.\n\n", accuracy(predicted, Y) * 100)
end


function diff_count(tm::AbstractTMClassifier)::Tuple{Int64, Int64, Int64}
    literals::Int64 = 0
    pos = []
    neg = []
    for (k, clauses) in tm.clauses
        for c in clauses.positive_included_literals
            append!(pos, [c])
            literals += length(c)
        end
        for c in clauses.negative_included_literals
            append!(neg, [c])
            literals += length(c)
        end
    end
    return length(pos) - length(Set(pos)), length(neg) - length(Set(neg)), literals
end

end # module
