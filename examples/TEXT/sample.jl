include("../../src/Tsetlin.jl")
include("config.jl")
include("HDC.jl")

using Serialization
using .Tsetlin: TMInput, TMClassifier, predict, load


hvectors = deserialize(HV_PATH)
tm = load(TM_PATH)

prompt = PROMPT[max(end - CONTEXT_SIZE + 1, 1):end]
prompt_vector = [UInt8(t) for t in prompt]

for n in 1:TOKENS_GENERATE
    acc = zeros(BUNDLE_ACC_TYPE, HV_DIMENSIONS)
    local_scratch = BitVector(undef, HV_DIMENSIONS)
    local_scratch2 = BitVector(undef, HV_DIMENSIONS)

    context = @view(prompt_vector[max(end - CONTEXT_SIZE + 1, 1):end])
    if RANDOMLY_REDUCE_CONTEXT_SIZE
        context = @view(context[rand(max(end - CONTEXT_SIZE + 1, 1):end):end])
    end
    hv = gen_context_hvector!(acc, local_scratch, local_scratch2, context, hvectors; noise=TEMPERATURE_NOISE)
    push!(prompt_vector, predict(tm, TMInput(hv.chunks, hv.len)))

    print(Char(prompt_vector[n + length(prompt)]))
end
println("\n")
