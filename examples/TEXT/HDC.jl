export HDC


bind!(target::BitVector, source::BitVector) = (target .⊻= source)


# Context subsampling

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


function gen_context_hvector!(
    acc::Vector{T},
    scratch::BitVector,
    context_window::AbstractVector{UInt8},
    hvectors::Dict{UInt8, BitVector};
    alpha::T=ALPHA_CONTEXT
)::BitVector where T<:AbstractFloat
    len = length(context_window)
    fill!(acc, zero(T))
    weights = compute_weights(len, alpha)
    # weights = compute_weights_exp(len, alpha)
    n = 1
    @inbounds for i in 1:len
        if n > 1
            token = context_window[i]
            dist_from_end = len - i + 1
            weight = weights[dist_from_end]
            circshift!(scratch, hvectors[token], dist_from_end)
            bind!(scratch, hvectors[context_window[i - 1]])
            w_pos = weight
            w_neg = -weight
            @simd for j in eachindex(scratch)
                acc[j] += ifelse(scratch[j], w_pos, w_neg)
            end
        end
        n += 1
    end
    # @inbounds @simd for j in 1:HV_DIMENSIONS
    #     acc[j] += rand((0.02, -0.02))
    # end
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
