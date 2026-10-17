include("../../src/Tsetlin.jl")
include("config.jl")
include("HDC.jl")

using Serialization
using .Tsetlin: TMClassifier, InputVector, load, vote


tm = load(TM_PATH)
hvectors, shift_hvectors = deserialize(HV_PATH)


# Predict with sampling
function predict(variants::Vector{ClassType}, tm::TMClassifier{ClassType}, x::AbstractVector{UInt64}; show_samples::Bool=false)::ClassType where ClassType
    best_vote = typemin(Int64)
    best_cls = typemin(ClassType)
    classes = tm.classes
    tm_clauses = tm.clauses
    empty!(variants)
    @inbounds for i in eachindex(classes)
        cls = classes[i]
        clauses = tm_clauses[i]
        pos, neg = vote(tm, clauses, x, index=false)
        v = pos - neg
        is_better = v > best_vote
        best_cls = ifelse(is_better, cls, best_cls)
        best_vote = ifelse(is_better, v, best_vote)
        if v > 0
            push!(variants, cls)
        end
    end
    if show_samples && length(variants) >= 2
        Char.(variants) |> print
    end
    return length(variants) < 2 ? best_cls : rand(variants)
end


function generate(tm::TMClassifier{ClassType}, hvectors::Dict{UInt8, BitVector}, shift_hvectors::Memory{BitVector}; show_samples::Bool=false) where ClassType
    prompt = PROMPT[max(end - CONTEXT_SIZE + 1, 1):end]
    prompt_vector = [UInt8(t) for t in prompt]
    variants = sizehint!(Vector{ClassType}(), length(tm.classes))
    for n in 1:TOKENS_GENERATE
        acc = zeros(BUNDLE_ACC_TYPE, HV_DIMENSIONS)
        local_scratch = BitVector(undef, HV_DIMENSIONS)
        local_scratch2 = BitVector(undef, HV_DIMENSIONS)

        context = @view(prompt_vector[max(end - CONTEXT_SIZE + 1, 1):end])
        if RANDOMLY_REDUCE_CONTEXT_SIZE
            context = @view(context[rand(max(end - CONTEXT_SIZE + 1, 1):end):end])
        end
        hv = gen_context_hvector!(acc, local_scratch, local_scratch2, context, hvectors, shift_hvectors)
        push!(prompt_vector, predict(variants, tm, InputVector(hv, copy=false), show_samples=show_samples))

        print(Char(prompt_vector[n + length(prompt)]))
    end
    println("\n")
end


generate(tm, hvectors, shift_hvectors, show_samples=false)
