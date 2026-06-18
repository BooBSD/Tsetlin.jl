module Tsetlin

export TMInput, TMClassifier, train!, predict, accuracy, save, load

using Dates
using Random
using Base.Threads
using Serialization
using Printf: @printf


Base.exit_on_sigint(false)

unzip(a) = (getfield.(a, x) for x in fieldnames(eltype(a)))


abstract type AbstractTMInput <: AbstractVector{Bool} end

# Mutable struct is up to 10% faster
mutable struct TMInput <: AbstractTMInput
    const chunks::Memory{UInt64}
    const len::UInt32

    function TMInput(x::AbstractArray{Bool})
        len = length(x)
        chunks = Memory{UInt64}(undef, cld(len, 64))
        idx = firstindex(x)
        @inbounds for n in eachindex(chunks)
            chunk = zero(UInt64)
            for i in 0:63
                idx > len && break
                chunk |= UInt64(x[idx]) << i
                idx += 1
            end
            chunks[n] = chunk
        end
        return new(chunks, len)
    end

    function TMInput(len::Integer)
        num_chunks = cld(len, 64)
        chunks = Memory{UInt64}(undef, num_chunks)
        fill!(chunks, zero(UInt64))
        return new(chunks, len)
    end

    function TMInput(::UndefInitializer, len::Integer)
        chunks = Memory{UInt64}(undef, cld(len, 64))
        return new(chunks, len)
    end

    function TMInput(chunks::AbstractArray{UInt64}, len::Integer)
        return new(chunks, len)
    end
end

Base.IndexStyle(::Type{<:TMInput}) = IndexLinear()
Base.size(x::TMInput)::Tuple{Int64} = (x.len,)
Base.sum(x::TMInput)::Int = sum(count_ones, x.chunks)
@inline function Base.getindex(x::TMInput, i::Int)
    @boundscheck checkbounds(x, i)
    chunk_idx = ((i - 1) >>> 6) + 1
    bit_idx = (i - 1) & 63
    @inbounds return ((x.chunks[chunk_idx] >> bit_idx) & 1) == 1
end
@inline function Base.setindex!(x::TMInput, v::Bool, i::Int)
    @boundscheck checkbounds(x, i)
    chunk_idx = ((i - 1) >>> 6) + 1
    bit_idx = (i - 1) & 63
    mask = UInt64(1) << bit_idx
    @inbounds begin
        c = x.chunks[chunk_idx]
        x.chunks[chunk_idx] = ifelse(v, c | mask, c & ~mask)
    end
    return x
end
@inline Base.setindex!(x::TMInput, v, i::Int) = setindex!(x, convert(Bool, v), i)


booleanize(x, ts...) = TMInput(vec(vec(x) .> reshape([ts...], 1, :)))


const STATE_TYPES = (UInt8, UInt16)


mutable struct TMClauses{StateType}
    const positive_included_literals::Matrix{UInt64}
    const positive_included_literals_inverted::Matrix{UInt64}
    const negative_included_literals::Matrix{UInt64}
    const negative_included_literals_inverted::Matrix{UInt64}
    const positive_included_literals_idx::Matrix{UInt64}
    const negative_included_literals_idx::Matrix{UInt64}
    positive_clauses::Union{Matrix{StateType}, Nothing}
    positive_clauses_inverted::Union{Matrix{StateType}, Nothing}
    negative_clauses::Union{Matrix{StateType}, Nothing}
    negative_clauses_inverted::Union{Matrix{StateType}, Nothing}

    function TMClauses{StateType}(clause_size::Int64, ta_clauses_num::Int64, include_limit::Int64) where StateType
        chunks_size = cld(clause_size, 64)
        chunks_idx_size = cld(chunks_size, 64)
        positive_clauses = fill(StateType(include_limit - 1), clause_size, ta_clauses_num)
        negative_clauses = fill(StateType(include_limit - 1), clause_size, ta_clauses_num)
        positive_clauses_inverted = fill(StateType(include_limit - 1), clause_size, ta_clauses_num)
        negative_clauses_inverted = fill(StateType(include_limit - 1), clause_size, ta_clauses_num)
        positive_included_literals_idx = fill(zero(UInt64), chunks_idx_size, ta_clauses_num)
        negative_included_literals_idx = fill(zero(UInt64), chunks_idx_size, ta_clauses_num)
        positive_included_literals = fill(zero(UInt64), chunks_size, ta_clauses_num)
        negative_included_literals = fill(zero(UInt64), chunks_size, ta_clauses_num)
        positive_included_literals_inverted = fill(zero(UInt64), chunks_size, ta_clauses_num)
        negative_included_literals_inverted = fill(zero(UInt64), chunks_size, ta_clauses_num)
        return new{StateType}(positive_included_literals, positive_included_literals_inverted, negative_included_literals, negative_included_literals_inverted, positive_included_literals_idx, negative_included_literals_idx, positive_clauses, positive_clauses_inverted, negative_clauses, negative_clauses_inverted)
    end
end


mutable struct TMClassifier{ClassType, N, TMType, C}
    classes_num::Int64
    clauses_num::Int64
    T::Int64
    S::Int64
    s::Int64
    L::Int64
    LF::Int64
    const clause_size::UInt32
    const include_limit::UInt16
    const state_min::UInt16
    const state_max::UInt16
    const clauses::TMType
    const classes::Memory{ClassType}

    function TMClassifier(x::TMInput, Y::Vector, clauses_num::Int64, T::Int64, S::Int64, L::Int64, LF::Int64; states_num::Int64=256, include_limit::Int64=128)
        state_max_available = maximum(typemax.(STATE_TYPES))
        state_max = states_num - 1
        @assert 2 <= states_num <= state_max_available + 1 "states_num must be between 2 to $(state_max_available + 1)."
        @assert 1 <= include_limit <= state_max "include_limit must be between 1 to $(state_max)."
        ClassType = typeof(first(Y))
        clause_size = length(x)
        N = length(x.chunks)
        s = round(Int, length(x) / S)
        StateType = STATE_TYPES[findfirst(T -> state_max <= typemax(T), STATE_TYPES)]
        if ClassType == Bool
            TMType = TMClauses{StateType}
            clauses = TMClauses{StateType}(clause_size, clauses_num, include_limit)
            classes_num = 2
            ta_clauses_num = clauses_num
            classes = Memory{Bool}([true, false])
        else
            ta_clauses_num = floor(Int, clauses_num / 2)
            TMType = Memory{TMClauses{StateType}}
            ys = sort(unique(Y))
            classes_num = length(ys)
            clauses::TMType = TMType(undef, classes_num)
            classes = Memory{ClassType}(ys)
            for i in eachindex(ys)
                clauses[i] = TMClauses{StateType}(clause_size, ta_clauses_num, include_limit)
            end
        end
        return new{ClassType, N, TMType, ta_clauses_num}(classes_num, clauses_num, T, S, s, L, LF, clause_size, include_limit, 0, state_max, clauses, classes)
    end
end


@inline function check_clause(tm::TMClassifier{<:Any, N}, x::TMInput, literals::SubArray{UInt64}, literals_inverted::SubArray{UInt64}, literals_idx::SubArray{UInt64})::Int64 where N
    LF = tm.LF
    c = 0
    chunks = x.chunks
    nidx = cld(N, 64)
    @inbounds for i in 1:nidx
        (c >= LF) && return 0  # helps for huge inputs
        idx = literals_idx[i]
        (idx == zero(UInt64)) && continue
        base = i * 64 - 63
        max_n = 63 - leading_zeros(idx)
        # min_n = trailing_zeros(idx)
        # @simd for n in min_n:max_n  # Potentially faster on very sparse inputs
        @simd for n in 0:max_n  # Faster on a MNIST
            id = base + n
            chunk = chunks[id]
            # val = (~chunk & literals[id]) | (chunk & literals_inverted[id])
            val = (((literals[id] ⊻ literals_inverted[id]) & chunk) ⊻ literals[id])
            c += count_ones(val)
        end
    end
    return max(0, LF - c)
end


@inline function check_clause(tm::TMClassifier{<:Any, N}, x::TMInput, literals::SubArray{UInt64}, literals_inverted::SubArray{UInt64})::Int64 where N
    c = 0
    chunks = x.chunks
    @inbounds @simd for n in 1:N
        chunk = chunks[n]
        lit = literals[n]
        lit_inv = literals_inverted[n]
        # val = (~chunk & lit) | (chunk & lit_inv)
        val = (((lit ⊻ lit_inv) & chunk) ⊻ lit)
        c += count_ones(val)
    end
    return max(0, tm.LF - c)
end


@inline function vote(tm::TMClassifier{<:Any, <:Any, <:Any, C}, clauses::TMClauses, x::TMInput; index::Bool=false)::Tuple{Int64, Int64} where C
    pos = 0
    neg = 0
    if !index
        @inbounds for i in 1:C
            pos += check_clause(tm, x, @view(clauses.positive_included_literals[:, i]), @view(clauses.positive_included_literals_inverted[:, i]))
        end
        @inbounds for i in 1:C
            neg += check_clause(tm, x, @view(clauses.negative_included_literals[:, i]), @view(clauses.negative_included_literals_inverted[:, i]))
        end
    else
        @inbounds for i in 1:C
            pos += check_clause(tm, x, @view(clauses.positive_included_literals[:, i]), @view(clauses.positive_included_literals_inverted[:, i]), @view(clauses.positive_included_literals_idx[:, i]))
        end
        @inbounds for i in 1:C
            neg += check_clause(tm, x, @view(clauses.negative_included_literals[:, i]), @view(clauses.negative_included_literals_inverted[:, i]), @view(clauses.negative_included_literals_idx[:, i]))
        end
    end
    return pos, neg
end


@inline function update_index(tm::TMClassifier{<:Any, N}, literals::SubArray{UInt64}, literals_inverted::SubArray{UInt64}, literals_idx::SubArray{UInt64}) where N
    @inbounds for n in 0:((N - 1) >> 6)
        base = n << 6
        idx_mask = zero(UInt64)
        limit = min(63, N - 1 - base)
        @simd for n in 0:limit
            i = base + n + 1
            combined = literals[i] | literals_inverted[i]
            is_active = (combined != 0)
            idx_mask |= (UInt64(is_active) << n)
        end
        literals_idx[n + 1] = idx_mask
    end
end


@inline function include_literals_sum(a::SubArray{UInt64}, b::SubArray{UInt64}, N::Int64)::Int64
    c = 0
    @inbounds @simd for n in 1:N
        c += count_ones(a[n] | b[n])
    end
    return c
end


@inline function get_rands()::Tuple{Float32, Float32}
    rnd = rand(UInt64)
    r1 = rnd % UInt32
    r2 = UInt32(rnd >> 32)
    f1 = Float32(r1 >>> 8) * Float32(0x1p-24)
    f2 = Float32(r2 >>> 8) * Float32(0x1p-24)
    return f1, f2
end


function feedback!(tm::TMClassifier{<:Any, N, <:Any, C}, clauses::TMClauses{StateType}, x::TMInput, clauses1::Matrix{StateType}, clauses_inverted1::Matrix{StateType}, clauses2::Matrix{StateType}, clauses_inverted2::Matrix{StateType}, literals1::Matrix{UInt64}, literals_inverted1::Matrix{UInt64}, literals2::Matrix{UInt64}, literals_inverted2::Matrix{UInt64}, literals1_idx::Matrix{UInt64}, literals2_idx::Matrix{UInt64}, positive::Bool, index::Bool, exclusive_literals::Bool=false) where {N, StateType, C}
    T = tm.T
    pos, neg = vote(tm, clauses, x, index=index)
    v = clamp(pos - neg, -T, T)
    # update = ifelse(positive, T - v, T + v) / (T * 2)
    update = 0.5f0 + ifelse(positive, -v, v) / Float32(T * 2)
    include_limit = StateType(tm.include_limit)
    state_max = StateType(tm.state_max)
    state_min = StateType(tm.state_min)
    clause_size = tm.clause_size
    last_bit = 63 - ((N << 6) - clause_size)
    chunks = x.chunks

    # Feedback 1
    @inbounds for j in 1:C
        rndup1, rndup2 = get_rands()

        # if rand() < update
        if rndup1 < update
            c = @view(clauses1[:, j])
            ci = @view(clauses_inverted1[:, j])
            l = @view(literals1[:, j])
            li = @view(literals_inverted1[:, j])
            l_idx = @view(literals1_idx[:, j])
            if (!index ? check_clause(tm, x, l, li) : check_clause(tm, x, l, li, l_idx)) > 0
                if include_literals_sum(l, li, N) < tm.L
                    # @inbounds for i = 1:tm.clause_size
                    #     if (x.x[i] == true) && (c[i] < state_max)
                    #         c[i] += one(StateType)
                    #     end
                    #     if (x.x[i] == false) && (ci[i] < state_max)
                    #         ci[i] += one(StateType)
                    #     end
                    # end
                    # Two loops are a bit faster than one.
                    @inbounds for n in 1:N
                        std_mask = chunks[n]
                        (std_mask == zero(UInt64)) && continue
                        base = n * 64 - 63
                        stop_bit = ifelse(n == N, last_bit, 63)
                        l_mask = zero(UInt64)
                        @simd for i in 0:stop_bit
                            ii = base + i
                            c[ii] += StateType((c[ii] < state_max) & (std_mask >> i))
                            l_mask |= UInt64(c[ii] >= include_limit) << i
                        end
                        l[n] = ifelse(exclusive_literals, l_mask & ~li[n], l_mask)  # contradiction fix
                    end
                    @inbounds for n in 1:N
                        inv_mask = ~chunks[n]
                        (inv_mask == zero(UInt64)) && continue
                        base = n * 64 - 63
                        stop_bit = ifelse(n == N, last_bit, 63)
                        li_mask = zero(UInt64)
                        @simd for i in 0:stop_bit
                            ii = base + i
                            ci[ii] += StateType((ci[ii] < state_max) & (inv_mask >> i))
                            li_mask |= UInt64(ci[ii] >= include_limit) << i
                        end
                        li[n] = ifelse(exclusive_literals, li_mask & ~l[n], li_mask)  # contradiction fix
                    end
                end
                # @inbounds for i = 1:tm.clause_size
                #     # No random
                #     if (x.x[i] == false) && (c[i] < include_limit) && (c[i] > state_min)
                #         c[i] -= one(StateType)
                #     end
                #     # No random
                #     if (x.x[i] == true) && (ci[i] < include_limit) && (ci[i] > state_min)
                #         ci[i] -= one(StateType)
                #     end
                # end
                # Two loops are a bit faster than one.
                @inbounds for n in 1:N
                    std_mask = ~chunks[n] & ~l[n]
                    (std_mask == zero(UInt64)) && continue
                    base = n * 64 - 63
                    stop_bit = ifelse(n == N, last_bit, 63)
                    l_mask = zero(UInt64)
                    @simd for i in 0:stop_bit
                        ii = base + i
                        c[ii] -= StateType((c[ii] > state_min) & (std_mask >> i))
                        l_mask |= UInt64(c[ii] >= include_limit) << i
                    end
                    l[n] = ifelse(exclusive_literals, l_mask & ~li[n], l_mask)  # contradiction fix
                end
                @inbounds for n in 1:N
                    inv_mask = chunks[n] & ~li[n]
                    (inv_mask == zero(UInt64)) && continue
                    base = n * 64 - 63
                    stop_bit = ifelse(n == N, last_bit, 63)
                    li_mask = zero(UInt64)
                    @simd for i in 0:stop_bit
                        ii = base + i
                        ci[ii] -= StateType((ci[ii] > state_min) & (inv_mask >> i))
                        li_mask |= UInt64(ci[ii] >= include_limit) << i
                    end
                    li[n] = ifelse(exclusive_literals, li_mask & ~l[n], li_mask)  # contradiction fix
                end
            else
                @inbounds for _ in 1:tm.s
                    # Extracting two random UInt32 values from a single UInt64
                    rnd = rand(UInt64)
                    rnd1 = rnd % UInt32
                    rnd2 = UInt32(rnd >> 32)

                    i = (rnd1 % clause_size) + one(UInt32)
                    c[i] -= StateType(c[i] > state_min)
                    d = (i + 63) >> 6
                    r = (i - 1) & 63
                    l_mask = l[d] & ~(one(UInt64) << r) | UInt64(c[i] >= include_limit) << r
                    l[d] = ifelse(exclusive_literals, l_mask & ~li[d], l_mask)  # contradiction fix

                    i = (rnd2 % clause_size) + one(UInt32)
                    ci[i] -= StateType(ci[i] > state_min)
                    d = (i + 63) >> 6
                    r = (i - 1) & 63
                    li_mask = li[d] & ~(one(UInt64) << r) | UInt64(ci[i] >= include_limit) << r
                    li[d] = ifelse(exclusive_literals, li_mask & ~l[d], li_mask)  # contradiction fix
                end
            end
            index && update_index(tm, l, li, l_idx)
        end
    # end
    # Feedback 2
    # @inbounds for j in 1:C
        # if rand() < update
        if rndup2 < update
            c = @view(clauses2[:, j])
            ci = @view(clauses_inverted2[:, j])
            l = @view(literals2[:, j])
            li = @view(literals_inverted2[:, j])
            l_idx = @view(literals2_idx[:, j])
            (!index ? check_clause(tm, x, l, li) : check_clause(tm, x, l, li, l_idx)) > 0 || continue
            # @inbounds for i = 1:tm.clause_size
            #     if (x.x[i] == false) && (c[i] < include_limit)
            #         c[i] += one(StateType)
            #     end
            #     if (x.x[i] == true) && (ci[i] < include_limit)
            #         ci[i] += one(StateType)
            #     end
            # end
            # Two loops are a bit faster than one.
            @inbounds for n in 1:N
                std_mask = ~chunks[n] & ~l[n]
                (std_mask == zero(UInt64)) && continue
                base = n * 64 - 63
                stop_bit = ifelse(n == N, last_bit, 63)
                l_mask = zero(UInt64)
                @simd for i in 0:stop_bit
                    ii = base + i
                    c[ii] += StateType((std_mask >> i) & one(UInt64))
                    l_mask |= UInt64(c[ii] >= include_limit) << i
                end
                l[n] = ifelse(exclusive_literals, l_mask & ~li[n], l_mask)  # contradiction fix
            end
            @inbounds for n in 1:N
                inv_mask = chunks[n] & ~li[n]
                (inv_mask == zero(UInt64)) && continue
                base = n * 64 - 63
                stop_bit = ifelse(n == N, last_bit, 63)
                li_mask = zero(UInt64)
                @simd for i in 0:stop_bit
                    ii = base + i
                    ci[ii] += StateType((inv_mask >> i) & one(UInt64))
                    li_mask |= UInt64(ci[ii] >= include_limit) << i
                end
                li[n] = ifelse(exclusive_literals, li_mask & ~l[n], li_mask)  # contradiction fix
            end
            index && update_index(tm, l, li, l_idx)
        end
    end
end


function predict(tm::TMClassifier{ClassType}, x::TMInput; index::Bool=false)::ClassType where ClassType <: Bool
    pos, neg = vote(tm, tm.clauses, x, index=index)
    return pos > neg
end


function predict(tm::TMClassifier{ClassType}, x::TMInput; index::Bool=false)::ClassType where ClassType
    best_vote = typemin(Int64)
    best_cls = typemin(ClassType)
    classes = tm.classes
    tm_clauses = tm.clauses
    @inbounds for i in eachindex(classes)
        cls = classes[i]
        clauses = tm_clauses[i]
        pos, neg = vote(tm, clauses, x, index=index)
        v = pos - neg
        is_better = v > best_vote
        best_cls = ifelse(is_better, cls, best_cls)
        best_vote = ifelse(is_better, v, best_vote)
    end
    return best_cls
end


function predict(tm::TMClassifier{ClassType}, X::Vector{TMInput}; index::Bool=false)::Vector{ClassType} where ClassType
    predicted::Vector{ClassType} = Vector{ClassType}(undef, length(X))  # Predefine vector for @threads access
    @threads for i in eachindex(X)
        predicted[i] = predict(tm, X[i], index=index)
    end
    return predicted
end


function accuracy(predicted::Vector{T}, Y::Vector{T})::Float64 where T
    @assert length(predicted) === length(Y)
    return sum(@inbounds p === y for (p, y) in zip(predicted, Y); init=0) / length(Y)
end


function train!(tm::TMClassifier{ClassType}, x::TMInput, y::ClassType; index::Bool=false, exclusive_literals::Bool=false) where ClassType <: Bool
    clauses = tm.clauses
    if y == true
        feedback!(tm, clauses, x, clauses.positive_clauses, clauses.positive_clauses_inverted, clauses.negative_clauses, clauses.negative_clauses_inverted, clauses.positive_included_literals, clauses.positive_included_literals_inverted, clauses.negative_included_literals, clauses.negative_included_literals_inverted, clauses.positive_included_literals_idx, clauses.negative_included_literals_idx, true, index, exclusive_literals)
    else
        feedback!(tm, clauses, x, clauses.negative_clauses, clauses.negative_clauses_inverted, clauses.positive_clauses, clauses.positive_clauses_inverted, clauses.negative_included_literals, clauses.negative_included_literals_inverted, clauses.positive_included_literals, clauses.positive_included_literals_inverted, clauses.negative_included_literals_idx, clauses.positive_included_literals_idx, false, index, exclusive_literals)
    end
end


function train!(tm::TMClassifier{ClassType}, x::TMInput, y::ClassType; index::Bool=false, exclusive_literals::Bool=false) where ClassType
    classes = tm.classes
    tm_clauses = tm.clauses
    @inbounds for i in eachindex(tm_clauses)
        clauses = tm_clauses[i]
        if classes[i] == y
            feedback!(tm, clauses, x, clauses.positive_clauses, clauses.positive_clauses_inverted, clauses.negative_clauses, clauses.negative_clauses_inverted, clauses.positive_included_literals, clauses.positive_included_literals_inverted, clauses.negative_included_literals, clauses.negative_included_literals_inverted, clauses.positive_included_literals_idx, clauses.negative_included_literals_idx, true, index, exclusive_literals)
        else
            feedback!(tm, clauses, x, clauses.negative_clauses, clauses.negative_clauses_inverted, clauses.positive_clauses, clauses.positive_clauses_inverted, clauses.negative_included_literals, clauses.negative_included_literals_inverted, clauses.positive_included_literals, clauses.positive_included_literals_inverted, clauses.negative_included_literals_idx, clauses.positive_included_literals_idx, false, index, exclusive_literals)
        end
    end
end


function train!(tm::TMClassifier{ClassType}, X::Vector{TMInput}, Y::Vector{ClassType}; shuffle::Bool=true, index::Bool=false, exclusive_literals::Bool=false) where ClassType
    @threads for i in ifelse(shuffle, randperm(length(Y)), eachindex(Y))
        train!(tm, X[i], Y[i], index=index, exclusive_literals=exclusive_literals)
    end
end


function train!(tm::TMClassifier{ClassType}, x_train::Vector{TMInput}, y_train::Vector{ClassType}, x_test::Vector{TMInput}, y_test::Vector{ClassType}, epochs::Int64; shuffle::Bool=true, index::Bool=false, verbose::Int=1, best_tms_size::Int64=0, best_tms_compile::Bool=true, exclusive_literals::Bool=false)::Vector{Tuple{TMClassifier, Float64}} where ClassType
    @assert best_tms_size in 0:2000
    if verbose > 0
        density = round(sum(sum(x) for x in x_train) / (length(x_train[1]) * length(x_train)) * 100, digits=2)
        println("\nClasses: $(tm.classes_num), clauses: $(tm.clauses_num), T: $(tm.T), S: $(tm.S) (s: $(tm.s)), L: $(tm.L), LF: $(tm.LF), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit).")
        println("Input vector size: $(length(x_train[1])) bits, density: $(density)%, training dataset size: $(length(y_train)), testing dataset size: $(length(y_test)).")
        println("Expected average clause literal density: $(round(tm.L / length(x_train[1]) * 100, digits=2))%. Using literals index: $(index).")
        println("Running in $(nthreads()) threads. Accuracy over $(epochs) epochs:\n")
    end
    best_acc::Float64 = 0.0
    best_tms = Tuple{TMClassifier, Float64}[]
    all_time = @elapsed begin
        @inbounds for i in 1:epochs
            training_time = @elapsed train!(tm, x_train, y_train, shuffle=shuffle, index=index, exclusive_literals=exclusive_literals)
            testing_time = @elapsed predicted = predict(tm, x_test, index=index)
            acc = accuracy(predicted, y_test)
            best_acc = ifelse(acc > best_acc, acc, best_acc)
            if best_tms_size > 0
                push!(best_tms, (best_tms_compile ? compile(tm) : deepcopy(tm), acc))
                sort!(best_tms, by=last, rev=true)
                best_tms = best_tms[1:clamp(length(best_tms), length(best_tms), best_tms_size)]
            end
            if verbose > 0
                @printf("#%s  Accuracy: %.2f%%  Best: %.2f%%  Training: %.3fs  Testing: %.3fs\n", i, acc * 100, best_acc * 100, training_time, testing_time)
            end
        end
    end
    if verbose > 0
        elapsed = Time(0) + Second(floor(Int, all_time))
        multiplier = ifelse(ClassType == Bool, 1, 2)
        average_clause_density = round((literals_sum(tm) / (tm.classes_num * tm.clauses_num * multiplier)) / length(x_train[1]) * 100, digits=2)
        @printf("\n%s epochs done in %s. Best accuracy: %.2f%%.\n", epochs, elapsed, best_acc * 100)
        println("Classes: $(tm.classes_num), clauses: $(tm.clauses_num), T: $(tm.T), S: $(tm.S) (s: $(tm.s)), L: $(tm.L), LF: $(tm.LF), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit).")
        println("Input vector size: $(length(x_train[1])) bits, density: $(density)%, training dataset size: $(length(y_train)), testing dataset size: $(length(y_test)).")
        println("Average clause literal density: $(average_clause_density)%. Using literals index: $(index).\n")
    end
    return best_tms
end


@inline compile!(clauses::TMClauses) = clauses.positive_clauses = clauses.negative_clauses = clauses.positive_clauses_inverted = clauses.negative_clauses_inverted = nothing

function compile(tm::TMClassifier{<:Bool})
    tmc = deepcopy(tm)
    compile!(tmc.clauses)
    return tmc
end

function compile(tm::TMClassifier)
    tmc = deepcopy(tm)
    foreach(compile!, tmc.clauses)
    return tmc
end


function save(tm::Union{TMClassifier, Tuple{TMClassifier, Float64}, Vector{Tuple{TMClassifier, Float64}}}, filepath::AbstractString)
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


function benchmark(tm::TMClassifier{ClassType}, X::Vector{TMInput}, Y::Vector{ClassType}, loops::Int64; warmup::Bool=true, index::Bool=false) where ClassType
    density = round(sum(sum(x) for x in X) / (length(X[1]) * length(X)) * 100, digits=2)
    multiplier = ifelse(ClassType == Bool, 1, 2)
    average_clause_density = round((literals_sum(tm) / (tm.classes_num * tm.clauses_num * multiplier)) / length(X[1]) * 100, digits=2)
    @printf("CPU: %s\n", Sys.cpu_info()[1].model)
    @printf("Running in %s threads.\n", nthreads())
    println("Input vector size: $(length(X[1])) bits. Density: $(density)%")
    println("Average clause literal density: $(average_clause_density)%. Using literals index: $(index).")
    print("Preparing input data for benchmark... ")
    GC.gc()
    prepare_time = @elapsed begin
        # Permutate in random order
        len = length(Y)
        x_len = X[1].len
        perm = Vector{Int32}(undef, len * loops)
        i::Int64 = 0
        @inbounds @fastmath for _ in 1:loops
            @inbounds @fastmath for r in Random.shuffle(UnitRange{Int32}(1:len))
                i += 1
                perm[i] = r
            end
        end
        # Multiply X and Y by loops times
        _X::Vector{TMInput} = Vector{TMInput}(undef, length(perm))
        @threads for i in eachindex(_X)
            # This is 3.5x faster than deepcopy()
            _X[i] = TMInput(X[perm[i]].chunks, x_len)
            # _X[i] = deepcopy(X[perm[i]])
        end
        X = _X
        Y = Y[perm]
    end
    @printf("Done. Elapsed %.3f seconds.\n", prepare_time)
    GC.gc()
    X_size = Base.summarysize(X[1]) * length(X)
    if warmup
        print("Warm-up started... ")
        warmup_time = @elapsed begin
            predict(tm, X, index=index)
        end
        @printf("Done. Elapsed %.3f seconds.\n", warmup_time)
    end
    print("Benchmark for TMClassifier model started... ")
    GC.gc()
    GC.enable(false)
    bench_time = @elapsed begin
        predicted = predict(tm, X, index=index)
    end
    println("Done.")
    GC.enable(true)
    @printf("%s predictions processed in %.3f seconds.\n", length(predicted), bench_time)
    @printf("Performance: %s predictions per second.\n", floor(Int, length(predicted) / bench_time))
    @printf("Throughput: %.3f GB/s.\n", X_size / 1024^3 / bench_time)
    @printf("Input data size: %.3f GB.\n", X_size / 1024^3)
    multiplier = (ClassType == Bool) ? 2 : length(tm.clauses)
    @printf("Parameters during training: %s.\n", tm.clauses_num * multiplier * length(X[1]) * 2)
    @printf("Parameters after training and compilation: %s.\n", literals_sum(tm))
    @printf("Accuracy: %.2f%%.\n\n", accuracy(predicted, Y) * 100)
end


function literals_count(clauses::TMClauses)
    return (
        sum(count_ones, clauses.positive_included_literals),
        sum(count_ones, clauses.negative_included_literals),
        sum(count_ones, clauses.positive_included_literals_inverted),
        sum(count_ones, clauses.negative_included_literals_inverted)
    )
end

literals_sum(tm::TMClassifier{<:Bool}) = sum(literals_count(tm.clauses))
literals_sum(tm::TMClassifier) = sum(sum(literals_count(clauses)) for clauses in tm.clauses)

end # module
