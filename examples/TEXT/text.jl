include("../../src/Tsetlin.jl")

using Random
using Base.Threads
using Serialization
using .Tsetlin: TMInput, TMClassifier, train!, predict, accuracy, save, load, compile


CORPUS_URL = "https://raw.githubusercontent.com/BooBSD/char-rnn/refs/heads/patch-1/data/tinyshakespeare/input.txt"
CORPUS_PATH = "/tmp/input.txt"
TM_PATH = "/tmp/tm_text.tm"
HV_PATH = "/tmp/hvectors"
HV_DIMENSIONS = 64 * 256  # 64 * 256
CONTEXT_SIZE = 256  # 256
EPOCHS = 1000
SAMPLES_PER_EPOCH = 1_000_000
LAMBDA = 0.05  # 0.05
MIN_P = 0.05  # 0.05, 0.1
ALPHA_NORM = 1.0  # 1.0, 2.25


CLAUSES = 64  # Not bad results!
T = 512 * 1
S = 16000
L = 8192
LF = 8192

# CLAUSES = 256  # Not bad, but slow.
# T = 1024 * 1
# S = 16000
# L = 8192
# LF = 8192


isfile(CORPUS_PATH) || download(CORPUS_URL, CORPUS_PATH)
CORPUS = read(CORPUS_PATH)
CORPUS_LENGTH = length(CORPUS)

tokens = CORPUS |> unique |> sort # Sorted !!!!!!
println("Characters: $(join(Char.(tokens)))\n")

function get_stochastic_updates(weight::Float64)::Int
    base_count = floor(Int, weight)
    probability = weight - base_count
    extra = rand() < probability ? 1 : 0    
    return base_count + extra
end

tokens_count::Dict{UInt8, Int} = Dict()
for t in tokens
    tokens_count[t] = count(==(t), CORPUS)
end
max_freq = maximum(values(tokens_count))
tokens_probs::Dict{UInt8, Float64} = Dict()
for (char, count) in tokens_count
    raw_ratio = max_freq / count
    tokens_probs[char] = 1.0 + ALPHA_NORM * log(raw_ratio)
end

sort!(tokens, by=t -> tokens_probs[t], rev=true)

for t in tokens
    println("$(Char(t)): $(get_stochastic_updates(tokens_probs[t]))")
end

if isfile(HV_PATH)
    hvectors = deserialize(HV_PATH)
else
    hvectors::Dict{UInt8, BitVector} = Dict()
    for hv in tokens
        hvectors[hv] = bitrand(HV_DIMENSIONS)
    end
    serialize(HV_PATH, hvectors)
end

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


# Training the TM model
function train()
    hv_sample = gen_context_hvector(@view(CORPUS[1:CONTEXT_SIZE]))
    x_sample = TMInput(hv_sample.chunks, hv_sample.len)
    y_samples = collect(keys(hvectors))
    tm = TMClassifier(x_sample, y_samples, CLAUSES, T, S, L=L, LF=LF, states_num=65536, include_limit=65000)
    save(compile(tm), TM_PATH)  # Save empty model for sample()
    for epoch in 1:EPOCHS
        elapsed = @elapsed begin
            cnt = 0
            @threads for n in 1:SAMPLES_PER_EPOCH
                start = rand(1:CORPUS_LENGTH - CONTEXT_SIZE - 1)
                finish = rand(start:start + CONTEXT_SIZE - 1)
                y = CORPUS[finish + 1]
                # Balancing classes
                for i in 1:get_stochastic_updates(tokens_probs[y])
                    context = @view(CORPUS[start:finish])
                    hv = gen_context_hvector(context)
                    x = TMInput(hv.chunks, hv.len)
                    train!(tm, x, y)
                    cnt += 1
                end
            end
            cnt |> println
        end
        println("Epoch #$(epoch) elapsed in $(elapsed)")
        save(compile(tm), TM_PATH)
    end
end


# Dirty hack to force text generation starting from "ROLE:"
PROMPT = "--\n\n"

function sample()
    tm = load(TM_PATH)
    SUBSAMPLES = 21
    TOKENS_GENERATE = 10000
    prompt = PROMPT[max(end - CONTEXT_SIZE + 1, 1):end]
    prompt_vector = [UInt8(t) for t in prompt]
    for n in 1:TOKENS_GENERATE
        con = @view(prompt_vector[max(end - CONTEXT_SIZE + 1, 1):end])

        hvs = Vector{BitVector}(undef, SUBSAMPLES)
        for i in 1:SUBSAMPLES
            context = @view(con[rand(max(end - CONTEXT_SIZE + 1, 1):end):end])
            # context = con
            hvs[i] = gen_context_hvector(context)
        end
        hv = bundle(hvs)
        push!(prompt_vector, predict(tm, TMInput(hv.chunks, hv.len)))

        print(Char(prompt_vector[n + length(prompt)]))
    end
    println("\n")
end


if !isfile(TM_PATH)
    train()
else
    sample()
end
