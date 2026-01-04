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


abstract type AbstractTMInput <: AbstractVector{Bool} end

# Mutable struct is up to 10% faster
mutable struct TMInput <: AbstractTMInput
    const chunks::Memory{UInt64}
    const len::Int64

    function TMInput(x::AbstractArray{Bool})
        len = length(x)
        chunks::Memory{UInt64} = Memory{UInt64}(undef, ceil(Int, len / 64))
        @inbounds for i in eachindex(chunks)
            chunk::UInt64 = zero(UInt64)
            @inbounds @simd for ii in 0:63
                n = i * 64 - 63 + ii
                chunk |= x[n] << ii
            end
            chunks[i] = chunk
        end
        return new(chunks, len)
    end
    
    function TMInput(x::AbstractArray{UInt64}, len::Int64)
        return new(copy(x), len)
    end
end

Base.IndexStyle(::Type{<:TMInput}) = IndexLinear()
Base.size(x::TMInput)::Tuple{Int64} = (x.len,)
Base.getindex(x::TMInput, i::Int)::Bool = x.chunks[i]
Base.sum(x::TMInput)::Int = sum(count_ones, x.chunks)
@inline function Base.getindex(x::TMInput, i::Int)::Bool
    @boundscheck checkbounds(x, i)
    chunk_idx = (i - 1) >>> 6 + 1
    bit_idx = (i - 1) & 63
    @inbounds return (x.chunks[chunk_idx] >> bit_idx) & 1 == 1
end


booleanize(x, ts...) = TMInput(vec(vec(x) .> reshape([ts...], 1, :)))


mutable struct TATeam{StateType}
    positive_clauses::Union{Matrix{StateType}, Nothing}
    negative_clauses::Union{Matrix{StateType}, Nothing}
    positive_clauses_inverted::Union{Matrix{StateType}, Nothing}
    negative_clauses_inverted::Union{Matrix{StateType}, Nothing}
    const positive_included_literals_idx::Matrix{UInt64}
    const negative_included_literals_idx::Matrix{UInt64}
    const positive_included_literals::Matrix{UInt64}
    const negative_included_literals::Matrix{UInt64}
    const positive_included_literals_inverted::Matrix{UInt64}
    const negative_included_literals_inverted::Matrix{UInt64}
    const positive_included_literals_sum::Memory{Int64}
    const negative_included_literals_sum::Memory{Int64}
    const positive_included_literals_sum_clamp::Memory{Int64}
    const negative_included_literals_sum_clamp::Memory{Int64}
    const clause_size::Int64
    const include_limit::StateType
    const state_min::StateType
    const state_max::StateType

    function TATeam{StateType}(clause_size::Int64, ta_clauses_num::Int64, include_limit::Int64, state_min::Int64, state_max::Int64) where StateType
        chunks_size = ceil(Int, clause_size / 64)
        chunks_idx_size = ceil(Int, chunks_size / 64)
        chunks_bits_size = ceil(Int, clause_size / 64) * 64
        positive_clauses = fill(StateType(include_limit - 1), chunks_bits_size, ta_clauses_num)
        negative_clauses = fill(StateType(include_limit - 1), chunks_bits_size, ta_clauses_num)
        positive_clauses_inverted = fill(StateType(include_limit - 1), chunks_bits_size, ta_clauses_num)
        negative_clauses_inverted = fill(StateType(include_limit - 1), chunks_bits_size, ta_clauses_num)
        positive_included_literals_idx = fill(zero(UInt64), chunks_idx_size, ta_clauses_num)
        negative_included_literals_idx = fill(zero(UInt64), chunks_idx_size, ta_clauses_num)
        positive_included_literals = fill(zero(UInt64), chunks_size, ta_clauses_num)
        negative_included_literals = fill(zero(UInt64), chunks_size, ta_clauses_num)
        positive_included_literals_inverted = fill(zero(UInt64), chunks_size, ta_clauses_num)
        negative_included_literals_inverted = fill(zero(UInt64), chunks_size, ta_clauses_num)
        positive_included_literals_sum = zeros(Int64, ta_clauses_num)
        negative_included_literals_sum = zeros(Int64, ta_clauses_num)
        positive_included_literals_sum_clamp = zeros(Int64, ta_clauses_num)
        negative_included_literals_sum_clamp = zeros(Int64, ta_clauses_num)
        return new{StateType}(positive_clauses, negative_clauses, positive_clauses_inverted, negative_clauses_inverted, positive_included_literals_idx, negative_included_literals_idx, positive_included_literals, negative_included_literals, positive_included_literals_inverted, negative_included_literals_inverted, positive_included_literals_sum, negative_included_literals_sum, positive_included_literals_sum_clamp, negative_included_literals_sum_clamp, clause_size, include_limit, state_min, state_max)
    end
end


mutable struct TMClassifier{ClassType, N, I, TMType}
    classes_num::Int64
    clauses_num::Int64
    T::Int64
    S::Int64
    s::Int64
    L::Int64
    LF::Int64
    const include_limit::Int64
    const state_min::Int64
    const state_max::Int64
    const clauses::TMType

    function TMClassifier(x::TMInput, Y::Vector, clauses_num::Int64, T::Int64, S::Int64; states_num::Int64=256, include_limit::Int64=128, L::Int64=16, LF::Int64=4)
        ClassType = typeof(first(Y))
        N = length(x.chunks)
        I = ceil(Int, N / 64)
        s = round(Int, length(x) / S)
        state_max = states_num - 1
        StateType = ifelse(state_max <= typemax(UInt8), UInt8, UInt16)
        if ClassType == Bool
            TMType = TATeam{StateType}
            clauses = TATeam{StateType}(length(x), clauses_num, include_limit, 0, state_max)
            classes_num = 2
        else
            ta_clauses_num = floor(Int, clauses_num / 2)
            TMType = Dict{ClassType, TATeam{StateType}}
            clauses::TMType = Dict()
            for cls in unique(Y)
                clauses[cls] = TATeam{StateType}(length(x), ta_clauses_num, include_limit, 0, state_max)
            end
            classes_num = length(unique(Y))
        end
        return new{ClassType, N, I, TMType}(classes_num, clauses_num, T, S, s, L, LF, include_limit, 0, states_num - 1, clauses)
    end
end


@inline function check_clause(tm::TMClassifier{<:Any, <:Any, I}, x::TMInput, literals::SubArray{UInt64}, literals_inverted::SubArray{UInt64}, literals_sum_clamp::Int64, literals_idx::SubArray{UInt64})::Int64 where I
    c::Int64 = 0
    p_idx = pointer(literals_idx, 1)
    p_chunks = pointer(x.chunks, 1)
    p_lits = pointer(literals, 1)
    p_lits_inv = pointer(literals_inverted, 1)
    @inbounds for i in 1:I
        (c >= literals_sum_clamp) && return 0  # helps for huge inputs
        idx = unsafe_load(p_idx, i)
        base::Int64 = i * 64 - 63
        # @inbounds while idx != 0
        #     n::Int64 = trailing_zeros(idx)
        #     chunk_idx::Int64 = base + n
        #     x_chunk = unsafe_load(p_chunks, chunk_idx)
        #     lit = unsafe_load(p_lits, chunk_idx)
        #     lit_inv = unsafe_load(p_lits_inv, chunk_idx)
        #     val = (~x_chunk & lit) | (x_chunk & lit_inv)
        #     c += count_ones(val)
        #     idx &= idx - 1
        # end
        @inbounds for n in 0:63  # Faster on a big sparse inputs
            (idx & (1 << n)) == zero(UInt64) && continue
            chunk_idx::Int64 = base + n
            x_chunk = unsafe_load(p_chunks, chunk_idx)
            lit = unsafe_load(p_lits, chunk_idx)
            lit_inv = unsafe_load(p_lits_inv, chunk_idx)
            val = (~x_chunk & lit) | (x_chunk & lit_inv)
            c += count_ones(val)
        end
    end
    return max(literals_sum_clamp - c, 0)
end


@inline function check_clause(tm::TMClassifier{<:Any, N}, x::TMInput, literals::SubArray{UInt64}, literals_inverted::SubArray{UInt64}, literals_sum_clamp::Int64)::Int64 where N
    c::Int64 = 0
    @inbounds for i in 1:N
        val = (~x.chunks[i] & literals[i]) | (x.chunks[i] & literals_inverted[i])
        c += count_ones(val)
    end
    return max(literals_sum_clamp - c, 0)
end


function vote(tm::TMClassifier, ta::TATeam, x::TMInput; index::Bool=false)::Tuple{Int64, Int64}
    pos::Int64 = 0
    neg::Int64 = 0
    if index
        @inbounds @simd for i in eachindex(ta.positive_included_literals_sum_clamp)
            pos += check_clause(tm, x, @view(ta.positive_included_literals[:, i]), @view(ta.positive_included_literals_inverted[:, i]), ta.positive_included_literals_sum_clamp[i], @view(ta.positive_included_literals_idx[:, i]))
            neg += check_clause(tm, x, @view(ta.negative_included_literals[:, i]), @view(ta.negative_included_literals_inverted[:, i]), ta.negative_included_literals_sum_clamp[i], @view(ta.negative_included_literals_idx[:, i]))
        end
    else
        @inbounds @simd for i in eachindex(ta.positive_included_literals_sum_clamp)
            pos += check_clause(tm, x, @view(ta.positive_included_literals[:, i]), @view(ta.positive_included_literals_inverted[:, i]), ta.positive_included_literals_sum_clamp[i])
            neg += check_clause(tm, x, @view(ta.negative_included_literals[:, i]), @view(ta.negative_included_literals_inverted[:, i]), ta.negative_included_literals_sum_clamp[i])
        end
    end
    return pos, neg
end


@inline function aux_update(tm::TMClassifier{<:Any, N}, ta::TATeam{StateType}, j::Int64, c::SubArray{StateType}, ci::SubArray{StateType}, literals::SubArray{UInt64}, literals_inverted::SubArray{UInt64}, literals_sum::Memory{Int64}, literals_sum_clamp::Memory{Int64}, literals_idx::SubArray{UInt64}) where {N, StateType}
    limit, LF, lsum = ta.include_limit, tm.LF, 0
    last_N = 63 - ((N << 6) - ta.clause_size)
    idx_mask = zero(UInt64)
    @inbounds for i in 1:N
        a, b = zero(UInt64), zero(UInt64)
        base = (i - 1) << 6
        stop_bit = (i == N) ? last_N : 63
        @simd for ii in 0:stop_bit
            bit = one(UInt64) << ii
            a |= (c[base + ii + 1] >= limit) ? bit : zero(UInt64)
            b |= (ci[base + ii + 1] >= limit) ? bit : zero(UInt64)
        end
        literals[i], literals_inverted[i] = a, b
        lsum += count_ones(a) + count_ones(b)

        # Update index
        if (a | b) != 0
            idx_mask |= (one(UInt64) << ((i - 1) & 63))
        end
        if (i & 63) == 0 || i == N
            literals_idx[(i - 1) >> 6 + 1] = idx_mask
            idx_mask = zero(UInt64)
        end
    end
    literals_sum[j] = lsum
    literals_sum_clamp[j] = ifelse(0 < lsum < LF, lsum, LF)
end


function feedback!(tm::TMClassifier{<:Any, N}, ta::TATeam{StateType}, x::TMInput, clauses1::Matrix{StateType}, clauses_inverted1::Matrix{StateType}, clauses2::Matrix{StateType}, clauses_inverted2::Matrix{StateType}, literals1::Matrix{UInt64}, literals_inverted1::Matrix{UInt64}, literals2::Matrix{UInt64}, literals_inverted2::Matrix{UInt64}, literals1_sum::Memory{Int64}, literals1_sum_clamp::Memory{Int64}, literals2_sum::Memory{Int64}, literals2_sum_clamp::Memory{Int64}, literals1_idx::Matrix{UInt64}, literals2_idx::Matrix{UInt64}, positive::Bool, index::Bool) where {N, StateType}
    v::Int64 = clamp(-(vote(tm, ta, x, index=index)...), -tm.T, tm.T)
    update::Float64 = ifelse(positive, tm.T - v, tm.T + v) / (tm.T * 2)

    # Feedback 1
    @inbounds for (j, (c, ci, l1, li1, l1_idx)) in enumerate(zip(eachcol(clauses1), eachcol(clauses_inverted1), eachcol(literals1), eachcol(literals_inverted1), eachcol(literals1_idx)))
        if rand() < update
            if (index ? check_clause(tm, x, l1, li1, literals1_sum_clamp[j], l1_idx) : check_clause(tm, x, l1, li1, literals1_sum_clamp[j])) > 0
                if literals1_sum[j] < tm.L
                    # @inbounds for i = 1:ta.clause_size
                    #     if (x.x[i] == true) && (c[i] < ta.state_max)
                    #         c[i] += one(StateType)
                    #     end
                    #     if (x.x[i] == false) && (ci[i] < ta.state_max)
                    #         ci[i] += one(StateType)
                    #     end
                    # end
                    @inbounds @simd for i in 1:N
                        pos::UInt64 = x.chunks[i]
                        neg::UInt64 = ~x.chunks[i]
                        base::Int64 = i * 64 - 63
                        # Fast iterate over ones
                        @inbounds while pos != zero(UInt64)
                            ii::Int64 = trailing_zeros(pos)
                            iii::Int64 = base + ii
                            c[iii] += (c[iii] < ta.state_max)
                            # l1[i] &= ~(1 << ii)
                            # l1[i] |= (c[iii] >= ta.include_limit) << ii
                            pos &= pos - one(UInt64)
                        end
                        @inbounds while neg != zero(UInt64)
                            ii::Int64 = trailing_zeros(neg)
                            iii::Int64 = base + ii
                            ci[iii] += (ci[iii] < ta.state_max)
                            # li1[i] &= ~(1 << ii)
                            # li1[i] |= (ci[iii] >= ta.include_limit) << ii
                            neg &= neg - one(UInt64)
                        end
                    end
                end
                # @inbounds for i = 1:ta.clause_size
                #     # No random
                #     if (x.x[i] == false) && (c[i] < ta.include_limit) && (c[i] > ta.state_min)
                #         c[i] -= one(StateType)
                #     end
                #     # No random
                #     if (x.x[i] == true) && (ci[i] < ta.include_limit) && (ci[i] > ta.state_min)
                #         ci[i] -= one(StateType)
                #     end
                # end
                @inbounds @simd for i in 1:N
                    pos::UInt64 = ~x.chunks[i] & ~l1[i]
                    neg::UInt64 = x.chunks[i] & ~li1[i]
                    base::Int64 = i * 64 - 63
                    # Fast iterate over ones
                    @inbounds while pos != zero(UInt64)
                        ii::Int64 = trailing_zeros(pos)
                        iii::Int64 = base + ii
                        c[iii] -= (c[iii] > ta.state_min)
                        # l1[i] &= ~(1 << ii)
                        # l1[i] |= (c[iii] >= ta.include_limit) << ii
                        pos &= pos - one(UInt64)
                    end
                    @inbounds while neg != zero(UInt64)
                        ii::Int64 = trailing_zeros(neg)
                        iii::Int64 = base + ii
                        ci[iii] -= (ci[iii] > ta.state_min)
                        # li1[i] &= ~(1 << ii)
                        # li1[i] |= (ci[iii] >= ta.include_limit) << ii
                        neg &= neg - one(UInt64)
                    end
                end
            else
                @inbounds for _ in 1:tm.s
                    i = rand(1:ta.clause_size)  # Here's one random only.
                    c[i] -= (c[i] > ta.state_min)
                    # d, r = divrem(i + 63, 64)
                    # l1[d] &= ~(1 << r)
                    # l1[d] |= (c[i] >= ta.include_limit) << r
                    i = rand(1:ta.clause_size)  # And here's another.
                    ci[i] -= (ci[i] > ta.state_min)
                    # d, r = divrem(i + 63, 64)
                    # li1[d] &= ~(1 << r)
                    # li1[d] |= (ci[i] >= ta.include_limit) << r
                end
            end
            aux_update(tm, ta, j, c, ci, l1, li1, literals1_sum, literals1_sum_clamp, l1_idx)
        end
    end
    # Feedback 2
    @inbounds for (j, (c, ci, l2, li2, l2_idx)) in enumerate(zip(eachcol(clauses2), eachcol(clauses_inverted2), eachcol(literals2), eachcol(literals_inverted2), eachcol(literals2_idx)))
        if rand() < update
            if (index ? check_clause(tm, x, l2, li2, literals2_sum_clamp[j], l2_idx) : check_clause(tm, x, l2, li2, literals2_sum_clamp[j])) > 0
                # @inbounds for i = 1:ta.clause_size
                #     if (x.x[i] == false) && (c[i] < ta.include_limit)
                #         c[i] += one(StateType)
                #     end
                #     if (x.x[i] == true) && (ci[i] < ta.include_limit)
                #         ci[i] += one(StateType)
                #     end
                # end
                @inbounds @simd for i in 1:N
                    pos::UInt64 = ~x.chunks[i] & ~l2[i]
                    neg::UInt64 = x.chunks[i] & ~li2[i]
                    base::Int64 = i * 64 - 63
                    # Fast iterate over ones
                    @inbounds while pos != zero(UInt64)
                        ii::Int64 = trailing_zeros(pos)
                        iii::Int64 = base + ii
                        c[iii] += one(StateType)
                        # l2[i] &= ~(1 << ii)
                        # l2[i] |= (c[iii] >= ta.include_limit) << ii
                        pos &= pos - one(UInt64)
                    end
                    @inbounds while neg != zero(UInt64)
                        ii::Int64 = trailing_zeros(neg)
                        iii::Int64 = base + ii
                        ci[iii] += one(StateType)
                        # li2[i] &= ~(1 << ii)
                        # li2[i] |= (ci[iii] >= ta.include_limit) << ii
                        neg &= neg - one(UInt64)
                    end
                end
                aux_update(tm, ta, j, c, ci, l2, li2, literals2_sum, literals2_sum_clamp, l2_idx)
            end
        end
    end
end


function predict(tm::TMClassifier{ClassType}, x::TMInput; index::Bool=false)::ClassType where ClassType <: Bool
    pos, neg = vote(tm, tm.clauses, x, index=index)
    return pos > neg
end


function predict(tm::TMClassifier{ClassType}, x::TMInput; index::Bool=false)::ClassType where ClassType
    best_vote::Int64 = typemin(Int64)
    best_cls::ClassType = typemin(ClassType)
    @inbounds for (cls, ta) in tm.clauses
        v::Int64 = -(vote(tm, ta, x, index=index)...)
        if v > best_vote
            best_vote = v
            best_cls = cls
        end
    end
    return best_cls
end

# function predict(tm::TMClassifier{ClassType}, x::TMInput; index::Bool=false)::ClassType where ClassType
#     best_vote::Int64 = typemin(Int64)
#     best_cls::ClassType = typemin(ClassType)
#     @inbounds for (cls, ta) in tm.clauses
#         v::Int64 = -(vote(tm, ta, x, index=index)...)
#         best_cls = ifelse(v > best_vote, cls, best_cls)
#         best_vote = ifelse(v > best_vote, v, best_vote)
#     end
#     return best_cls
# end


function predict(tm::TMClassifier{ClassType}, X::Vector{TMInput}; index::Bool=true)::Vector{ClassType} where ClassType
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


function train!(tm::TMClassifier{ClassType}, x::TMInput, y::ClassType; index::Bool=false) where ClassType <: Bool
    ta = tm.clauses
    if y == true
        feedback!(tm, ta, x, ta.positive_clauses, ta.positive_clauses_inverted, ta.negative_clauses, ta.negative_clauses_inverted, ta.positive_included_literals, ta.positive_included_literals_inverted, ta.negative_included_literals, ta.negative_included_literals_inverted, ta.positive_included_literals_sum, ta.positive_included_literals_sum_clamp, ta.negative_included_literals_sum, ta.negative_included_literals_sum_clamp, ta.positive_included_literals_idx, ta.negative_included_literals_idx, true, index)
    else
        feedback!(tm, ta, x, ta.negative_clauses, ta.negative_clauses_inverted, ta.positive_clauses, ta.positive_clauses_inverted, ta.negative_included_literals, ta.negative_included_literals_inverted, ta.positive_included_literals, ta.positive_included_literals_inverted, ta.negative_included_literals_sum, ta.negative_included_literals_sum_clamp, ta.positive_included_literals_sum, ta.positive_included_literals_sum_clamp, ta.negative_included_literals_idx, ta.positive_included_literals_idx, false, index)
    end
end


function train!(tm::TMClassifier{ClassType}, x::TMInput, y::ClassType; index::Bool=false) where ClassType
    ta1::TATeam = tm.clauses[y]
    feedback!(tm, ta1, x, ta1.positive_clauses, ta1.positive_clauses_inverted, ta1.negative_clauses, ta1.negative_clauses_inverted, ta1.positive_included_literals, ta1.positive_included_literals_inverted, ta1.negative_included_literals, ta1.negative_included_literals_inverted, ta1.positive_included_literals_sum, ta1.positive_included_literals_sum_clamp, ta1.negative_included_literals_sum, ta1.negative_included_literals_sum_clamp, ta1.positive_included_literals_idx, ta1.negative_included_literals_idx, true, index)
    @inbounds for (cls, ta) in tm.clauses
        if cls !== y
            feedback!(tm, ta, x, ta.negative_clauses, ta.negative_clauses_inverted, ta.positive_clauses, ta.positive_clauses_inverted, ta.negative_included_literals, ta.negative_included_literals_inverted, ta.positive_included_literals, ta.positive_included_literals_inverted, ta.negative_included_literals_sum, ta.negative_included_literals_sum_clamp, ta.positive_included_literals_sum, ta.positive_included_literals_sum_clamp, ta.negative_included_literals_idx, ta.positive_included_literals_idx, false, index)
        end
    end
end


# function train!(tm::TMClassifier{ClassType}, x::TMInput, y::ClassType; index::Bool=false) where ClassType
#     c::Int64 = typemin(Int64)
#     cls::ClassType = typemin(ClassType)
#     @inbounds for (class, ta) in tm.clauses
#         if class !== y
#             v::Int64 = vote(tm, ta, x, index=index)
#             if v > c
#                 c = v
#                 cls = class
#             end
#         end
#     end
#     feedback!(tm, tm.clauses[y], x, tm.clauses[y].positive_clauses, tm.clauses[y].positive_clauses_inverted, tm.clauses[y].negative_clauses, tm.clauses[y].negative_clauses_inverted, tm.clauses[y].positive_included_literals, tm.clauses[y].positive_included_literals_inverted, tm.clauses[y].negative_included_literals, tm.clauses[y].negative_included_literals_inverted, tm.clauses[y].positive_included_literals_sum, tm.clauses[y].positive_included_literals_sum_clamp, tm.clauses[y].negative_included_literals_sum, tm.clauses[y].negative_included_literals_sum_clamp, tm.clauses[y].positive_included_literals_idx, tm.clauses[y].negative_included_literals_idx, true, index)
#     feedback!(tm, tm.clauses[cls], x, tm.clauses[cls].negative_clauses, tm.clauses[cls].negative_clauses_inverted, tm.clauses[cls].positive_clauses, tm.clauses[cls].positive_clauses_inverted, tm.clauses[cls].negative_included_literals, tm.clauses[cls].negative_included_literals_inverted, tm.clauses[cls].positive_included_literals, tm.clauses[cls].positive_included_literals_inverted, tm.clauses[cls].negative_included_literals_sum, tm.clauses[cls].negative_included_literals_sum_clamp, tm.clauses[cls].positive_included_literals_sum, tm.clauses[cls].positive_included_literals_sum_clamp, tm.clauses[cls].negative_included_literals_idx, tm.clauses[cls].positive_included_literals_idx, false, index)
# end


function train!(tm::TMClassifier{ClassType}, X::Vector{TMInput}, Y::Vector{ClassType}; shuffle::Bool=true, index::Bool=false) where ClassType
    @threads for i in ifelse(shuffle, randperm(length(Y)), eachindex(Y))
        train!(tm, X[i], Y[i], index=index)
    end
end


function train!(tm::TMClassifier{ClassType}, x_train::Vector{TMInput}, y_train::Vector{ClassType}, x_test::Vector{TMInput}, y_test::Vector{ClassType}, epochs::Int64; shuffle::Bool=true, index::Bool=false, verbose::Int=1, best_tms_size::Int64=0, best_tms_compile::Bool=true)::Vector{Tuple{TMClassifier, Float64}} where ClassType
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
            training_time = @elapsed train!(tm, x_train, y_train, shuffle=shuffle, index=index)
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


@inline compile!(ta::TATeam) = ta.positive_clauses = ta.negative_clauses = ta.positive_clauses_inverted = ta.negative_clauses_inverted = nothing

function compile(tm::TMClassifier{<:Bool})
    tmc = deepcopy(tm)
    compile!(tmc.clauses)
    return tmc
end

function compile(tm::TMClassifier)
    tmc = deepcopy(tm)
    foreach(compile!, values(tmc.clauses))
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
            _X[i] = TMInput(X[perm[i]].chunks, X[perm[i]].len)
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
    multiplier = (ClassType == Bool) ? 2 : length(keys(tm.clauses))
    @printf("Parameters during training: %s.\n", tm.clauses_num * multiplier * length(X[1]) * 2)
    @printf("Parameters after training and compilation: %s.\n", literals_sum(tm))
    @printf("Accuracy: %.2f%%.\n\n", accuracy(predicted, Y) * 100)
end


function literals_count(ta::TATeam)
    return (
        sum(count_ones, ta.positive_included_literals),
        sum(count_ones, ta.negative_included_literals),
        sum(count_ones, ta.positive_included_literals_inverted),
        sum(count_ones, ta.negative_included_literals_inverted)
    )
end

literals_sum(tm::TMClassifier{<:Bool}) = sum(literals_count(tm.clauses))
literals_sum(tm::TMClassifier) = sum(sum(literals_count(c)) for (_, c) in tm.clauses)

end # module
