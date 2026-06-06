include("../Tsetlin.jl")

export explain

using .Tsetlin: TMInput, TMClassifier, TATeam


struct LiteralsSum
    positive_included_literals::Vector{Int16}
    positive_included_literals_inverted::Vector{Int16}
    negative_included_literals::Vector{Int16}
    negative_included_literals_inverted::Vector{Int16}
end


@inline function explain(literals::Matrix{UInt64}, clause_size::Int64)::Vector{Int16}
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


@inline function explain(ta::TATeam)::LiteralsSum
    return LiteralsSum(
        explain(ta.positive_included_literals, ta.clause_size),
        explain(ta.positive_included_literals_inverted, ta.clause_size),
        explain(ta.negative_included_literals, ta.clause_size),
        explain(ta.negative_included_literals_inverted, ta.clause_size),
    )
end


function explain(tm::TMClassifier{ClassType})::LiteralsSum where ClassType <: Bool
    return explain(tm.clauses)
end


function explain(tm::TMClassifier{ClassType})::Dict{ClassType, LiteralsSum} where ClassType
    res::Dict{ClassType, LiteralsSum} = Dict()
    @inbounds for (cls, ta) in zip(tm.classes, tm.clauses)
        res[cls] = explain(ta)
    end
    return res
end
