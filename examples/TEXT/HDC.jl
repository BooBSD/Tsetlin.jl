export HDC


bind(A::BitVector, B::BitVector)::BitVector = A .⊻ B

bind!(target::BitVector, source::BitVector) = (target .⊻= source)

bundle(vectors::Vector{<:Integer}...) = sum(vectors)


function bundle_add!(acc::AbstractVector{T}, hv::BitVector) where T <: Integer
    @inbounds @simd for i in eachindex(acc, hv)
        acc[i] += ifelse(hv[i], one(T), -one(T))
    end
    return acc
end


function binarize_bundle(acc::AbstractVector{T})::BitVector where T <: Integer
    res = BitVector(undef, length(acc))
    zero_val = zero(T)
    @inbounds for i in eachindex(acc)
        val = acc[i]
        if val == zero_val
            res[i] = rand(Bool)
        else
            res[i] = val > zero_val
        end
    end
    return res
end


function bundle(vectors::Vector{BitVector}; acc_type::Type{T} = BUNDLE_ACC_TYPE)::BitVector where T <: Integer
    isempty(vectors) && error("The vector list is empty")
    dim = length(vectors[1])
    acc = zeros(acc_type, dim)
    @inbounds for v in vectors
        length(v) == dim || error("All vectors must be of the same dimension")
        bundle_add!(acc, v)
    end
    return binarize_bundle(acc)
end


# Monte Carlo sparse context subsampling

function gen_context_hvector!(
    acc::Vector{T},
    scratch::BitVector,
    context_window::SubArray{UInt8},
    hvectors::Dict{UInt8, BitVector};
    lambda::Float64 = 0.05,
    min_p::Float64 = 0.05,
    D::Int64 = HV_DIMENSIONS
) where T <: Integer
    fill!(acc, zero(T))
    n = 0
    len = length(context_window)
    @inbounds for i in 1:len
        dist_from_end = len - i
        p = max(exp(-lambda * dist_from_end), min_p)
        if rand() < p
            curr_val = context_window[i]
            circshift!(scratch, hvectors[curr_val], dist_from_end + 1)
            if n > 0
                prev_val = context_window[i - 1]
                bind!(scratch, hvectors[prev_val])
            end
            bundle_add!(acc, scratch)
            n += 1
        end
    end
end
