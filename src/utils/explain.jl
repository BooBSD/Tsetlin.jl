include("../Tsetlin.jl")

export explain

using Base.Threads
using .Tsetlin: TMInput, TMClassifier, TATeam


struct ExplainedLiteralSum
    positive_included_literals::Vector{Int16}
    positive_included_literals_inverted::Vector{Int16}
    negative_included_literals::Vector{Int16}
    negative_included_literals_inverted::Vector{Int16}
end


struct ExplainedClause
    matched_literals::BitVector
    matched_literals_inverted::BitVector
    failed_literals::BitVector
    failed_literals_inverted::BitVector
    vote::Int64
end


struct ExplainedClauses
    clauses::Vector{ExplainedClause}
    votes::Int64
end


@inline function explain(literals::Matrix{UInt64}, clause_size::UInt32)::Vector{Int16}
    res = zeros(Int16, clause_size)
    bv = BitVector(undef, clause_size)
    @inbounds for lits in eachcol(literals)
        copyto!(bv.chunks, lits)
        @simd for i in 1:clause_size
            res[i] += bv[i]
        end
    end
    return res
end


@inline function explain(tm::TMClassifier, ta::TATeam)::ExplainedLiteralSum
    return ExplainedLiteralSum(
        explain(ta.positive_included_literals, tm.clause_size),
        explain(ta.positive_included_literals_inverted, tm.clause_size),
        explain(ta.negative_included_literals, tm.clause_size),
        explain(ta.negative_included_literals_inverted, tm.clause_size),
    )
end


function explain(tm::TMClassifier{ClassType})::ExplainedLiteralSum where ClassType <: Bool
    return explain(tm, tm.clauses)
end


function explain(tm::TMClassifier{ClassType})::Dict{ClassType, ExplainedLiteralSum} where ClassType
    res::Dict{ClassType, ExplainedLiteralSum} = Dict()
    @inbounds for (cls, ta) in zip(tm.classes, tm.clauses)
        res[cls] = explain(tm, ta)
    end
    return res
end


function explain(tm::TMClassifier{<:Any, N}, x::TMInput, literals::SubArray{UInt64}, literals_inverted::SubArray{UInt64})::ExplainedClause where N
    matched_literals = BitVector(undef, x.len)
    matched_literals_inverted = BitVector(undef, x.len)
    failed_literals = BitVector(undef, x.len)
    failed_literals_inverted = BitVector(undef, x.len)
    c = 0
    @inbounds for i in 1:N
        matched_literals.chunks[i] = x.chunks[i] & literals[i]
        matched_literals_inverted.chunks[i] = ~x.chunks[i] & literals_inverted[i]
        failed_literals.chunks[i] = ~x.chunks[i] & literals[i]
        failed_literals_inverted.chunks[i] = x.chunks[i] & literals_inverted[i]
        c += count_ones(failed_literals.chunks[i] | failed_literals_inverted.chunks[i])
    end
    vote = max(0, tm.LF - c)
    return ExplainedClause(
        matched_literals,
        matched_literals_inverted,
        failed_literals,
        failed_literals_inverted,
        vote,
    )
end


function explain(tm::TMClassifier{<:Any, <:Any, <:Any, <:Any, C}, ta::TATeam, x::TMInput)::Tuple{ExplainedClauses, ExplainedClauses} where C
    pos = Vector{ExplainedClause}(undef, C)
    neg = Vector{ExplainedClause}(undef, C)
    @inbounds for i in 1:C
        pos[i] = explain(tm, x, @view(ta.positive_included_literals[:, i]), @view(ta.positive_included_literals_inverted[:, i]))
        neg[i] = explain(tm, x, @view(ta.negative_included_literals[:, i]), @view(ta.negative_included_literals_inverted[:, i]))
    end
    return (
        ExplainedClauses(pos, sum(c.vote for c in pos)),
        ExplainedClauses(neg, sum(c.vote for c in neg)),
    )
end


function explain(tm::TMClassifier{ClassType}, x::TMInput)::Dict{ClassType, ExplainedClauses} where ClassType <: Bool
    pos, neg = explain(tm, tm.clauses, x)
    return Dict(
        true => pos,
        false => neg,
    )
end


function explain(tm::TMClassifier{ClassType}, x::TMInput)::Dict{ClassType, Dict{Bool, ExplainedClauses}} where ClassType
    res::Dict{ClassType, Dict{Bool, ExplainedClauses}} = Dict()
    @inbounds for (cls, ta) in zip(tm.classes, tm.clauses)
        pos, neg = explain(tm, ta, x)
        res[cls] = Dict(
            true => pos,
            false => neg,
        )
    end
    return res
end


function explain(tm::TMClassifier{ClassType}, X::Vector{TMInput})::Vector{Dict{ClassType, Dict{Bool, ExplainedClauses}}} where ClassType
    res = Vector{Dict{ClassType, Dict{Bool, ExplainedClauses}}}(undef, length(X))
    @threads for i in eachindex(X)
        res[i] = explain(tm, X[i])
    end
    return res
end
