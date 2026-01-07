export HDC


function bind(A::BitVector, B::BitVector)::BitVector
    if length(A) != length(B)
        error("The dimensions of the vectors must match")
    end
    return A .⊻ B
end


function bundle(vectors::Vector{BitVector})::BitVector
    isempty(vectors) && error("The vector list is empty")
    dim = length(vectors[1])
    n = length(vectors)
    counts = zeros(Int, dim)
    @inbounds for v in vectors
        length(v) == dim || error("All vectors must be of the same dimension")
        counts .+= v
    end
    threshold = n ÷ 2
    
    if isodd(n)
        return counts .> threshold
    else
        result = BitVector(undef, dim)
        @inbounds for i in 1:dim
            if counts[i] == threshold
                result[i] = rand(Bool)
            else
                result[i] = counts[i] > threshold
            end
        end
        return result
    end
end


# Monte Carlo sparse context subsampling

function weighted_subsample(tokens::SubArray{T}; lambda::Float64 = 0.05, min_p::Float64 = 0.05)::Tuple{Vector{T}, Vector{Int}, Vector{T}} where T
    n = length(tokens)
    selected::Vector{T} = []
    indexes::Vector{Int} = []
    selected_prev::Vector{T} = []
    @inbounds for i in 1:n
        dist_from_end = n - i
        p = exp(-lambda * dist_from_end)        
        p = max(p, min_p)
        if rand() < p
            push!(selected, tokens[i])
            push!(indexes, dist_from_end + 1)
            if i != 1
                push!(selected_prev, tokens[i - 1])
            else
                push!(selected_prev, tokens[i])
            end
        end
    end
    return selected, indexes, selected_prev
end


function gen_context_hvector(context_window::SubArray{UInt8})::BitVector
    context, indexes, context_prev = weighted_subsample(context_window, lambda=LAMBDA, min_p=MIN_P)
    hvs = Vector{BitVector}(undef, length(context))
    @inbounds for i in eachindex(context)
        token_current = circshift(hvectors[context[i]], indexes[i])
        token_previous = hvectors[context_prev[i]]
        if i != 1
            hvs[i] = bind(token_current, token_previous)
        else
            hvs[i] = token_current
        end
    end
    return bundle(hvs)
end
