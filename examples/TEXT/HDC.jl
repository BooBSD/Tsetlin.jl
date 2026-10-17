export HDC


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
    alpha::T=ALPHA_CONTEXT
)::BitVector where T<:AbstractFloat
    len = length(context_window)
    fill!(acc, zero(T))
    chunks = scratch.chunks
    n = 1
    @inbounds for i in 1:len
        if n > NGRAM - 1
            dist_from_end = len - i + 1
            weight = one(T) / (T(dist_from_end) ^ alpha)
            w_pos = weight
            w_neg = -weight
            gen_ngram!(hvectors, @view(context_window[i-NGRAM+1:i]), scratch, scratch2)
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
    zero_val = zero(T)
    @inbounds for c in eachindex(chunks)
        chunk = zero(UInt64)
        base = c * 64 - 63
        @simd for j in 0:63
            a = base + j
            val = acc[a]
            bit = ifelse(val != zero_val, val > zero_val, iseven(a))
            # bit = val != zero_val ? val > zero_val : rand(Bool)
            chunk |= (UInt64(bit) << j)
        end
        chunks[c] = chunk
    end
    return scratch
end
