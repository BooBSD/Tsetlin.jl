# Source: https://github.com/aamini/FastConv.jl/blob/master/src/utils.jl
# License: MIT © 2016 Alexander Amini

export fastconv

using Base.Cartesian


@generated function fastconv(E::Array{T, N}, k::Array{T, N}) where {T, N}
    quote
        retsize = [size(E)...] + [size(k)...] .- 1
        retsize = tuple(retsize...)
        ret = zeros(T, retsize)
        convn!(ret, E, k)
        return ret
    end
end


@generated function convn!(out::Array{T}, E::Array{T, N}, k::Array{T, N}) where {T, N}
    quote
        @inbounds begin
            @nloops $N x E begin
                @nloops $N i k begin
                    (@nref $N out d -> (x_d + i_d - 1)) += (@nref $N E x) * (@nref $N k i)
                end
            end
        end
        return out
    end
end
