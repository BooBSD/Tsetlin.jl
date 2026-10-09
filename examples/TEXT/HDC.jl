export HDC

using Random


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


@inline function gen_ngram!(hvectors::Dict{UInt8, BitVector}, context::AbstractVector{UInt8}, scratch::BitVector, scratch2::BitVector)::BitVector
    len = length(context)
    fill!(scratch.chunks, zero(UInt64))
    scratch_chunks = scratch.chunks
    @inbounds for i in eachindex(context)
        dist_from_end = len + 1 - i
        circshift!(scratch2, hvectors[context[i]], dist_from_end)
        hvectors_chunks = scratch2.chunks
        @simd for n in eachindex(scratch_chunks)
            scratch_chunks[n] = scratch_chunks[n] ⊻ hvectors_chunks[n]
        end
    end
    return scratch
end


function gen_context_hvector!(
    acc::Vector{T},
    scratch::BitVector,
    scratch2::BitVector,
    context_window::AbstractVector{UInt8},
    hvectors::Dict{UInt8, BitVector},
    shift_hvectors::Memory{BitVector};
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
            gen_ngram!(hvectors, @view(context_window[i-NGRAM+1:i]), scratch, scratch2)
            chunks = scratch.chunks
            shift_chunks = shift_hvectors[dist_from_end].chunks
            for c in eachindex(chunks)
                chunk = chunks[c] ⊻ shift_chunks[c]
                base = c * 64 - 63
                @simd for j in 0:63
                    a = base + j
                    pos = ((chunk >> j) & 1) == 1
                    acc[a] += ifelse(pos, w_pos, w_neg)
                end
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
