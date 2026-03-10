export HDC

using Random


# Sparse
function random_hv(dim::Int, k::Int)::BitVector
    hv = falses(dim)
    indices = randperm(dim)[1:k]
    hv[indices] .= true
    return hv
end

# Dense
bind!(target::BitVector, source::BitVector) = (target .⊻= source)

# Sparse
@inline function bundle_add!(acc::BitVector, hv::BitVector)::BitVector
    acc .|= hv
    return hv
end

# Zipf
function compute_weights(n::Int, alpha::T)::Vector{T} where T<:AbstractFloat
    normalization_sum = 0.0
    weights = Vector{T}(undef, n)
    @inbounds for k in 1:n
        val = 1.0 / (k ^ alpha)
        normalization_sum += val
        weights[k] = val 
    end
    inv_norm = 1.0 / normalization_sum
    @inbounds @simd for k in 1:n
        weights[k] *= inv_norm
    end
    return weights
end

# Exponential
function compute_weights_exp(n::Int, lambda::T)::Vector{T} where T<:AbstractFloat
    normalization_sum = zero(T)
    weights = Vector{T}(undef, n)
    decay_factor = exp(-lambda)
    val = decay_factor 
    @inbounds for k in 1:n
        normalization_sum += val
        weights[k] = val
        val *= decay_factor
    end
    inv_norm = one(T) / normalization_sum
    @inbounds @simd for k in 1:n
        weights[k] *= inv_norm
    end
    return weights
end


@inline function gen_ngram(hvectors::Dict{UInt8, BitVector}, context::AbstractVector{UInt8}, scratch::BitVector, scratch2::BitVector)::BitVector
    len = length(context)
    fill!(scratch2.chunks, zero(UInt64))
    @inbounds for i in eachindex(context)
        dist_from_end = len + 1 - i
        circshift!(scratch, hvectors[context[i]], dist_from_end)
        bundle_add!(scratch2, scratch)
    end
    return scratch2
end


function gen_context_hvector!(
    acc::Vector{T},
    scratch::BitVector,
    scratch2::BitVector,
    context_window::AbstractVector{UInt8},
    hvectors::Dict{UInt8, BitVector};
    alpha::T=ALPHA_CONTEXT,
    noise::T=zero(T)
)::BitVector where T<:AbstractFloat
    len = length(context_window)
    fill!(acc, zero(T))
    weights = compute_weights(len, alpha)
    # weights = compute_weights_exp(len, alpha)
    n = 1
    @inbounds for i in 1:len
        if n > NGRAM - 1
            token = context_window[i]
            dist_from_end = len - i + 1
            weight = weights[dist_from_end]
            w_pos = weight
            w_neg = -weight
            hv = gen_ngram(hvectors, @view(context_window[i-NGRAM+1:i]), scratch, scratch2)
            circshift!(scratch, hv, dist_from_end)
            @simd for j in eachindex(scratch)
                acc[j] += ifelse(scratch[j], w_pos, w_neg)
            end
        end
        n += 1
    end
    if noise != zero(T)
        @inbounds @simd for j in 1:HV_DIMENSIONS
            acc[j] += rand((noise, -noise))
        end
    end
    # return acc .> 0
    zero_val = zero(T)
    @inbounds for i in eachindex(acc)
        val = acc[i]
        if val != zero_val
            scratch[i] = val > zero_val
        else
            scratch[i] = rand(Bool)
            # "!!!!!!!!11111!!!!!" |> println
        end
    end
    return scratch
end
