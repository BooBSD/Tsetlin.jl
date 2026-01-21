include("../../src/Tsetlin.jl")
include("HDC.jl")
include("config.jl")


using Serialization
using .Tsetlin: TMInput, TMClassifier, predict, load


hvectors = deserialize(HV_PATH)
tm = load(TM_PATH)

prompt = PROMPT[max(end - CONTEXT_SIZE + 1, 1):end]
prompt_vector = [UInt8(t) for t in prompt]

for n in 1:TOKENS_GENERATE
    con = @view(prompt_vector[max(end - CONTEXT_SIZE + 1, 1):end])

    local_acc = zeros(BUNDLE_ACC_TYPE, HV_DIMENSIONS)
    local_scratch = BitVector(undef, HV_DIMENSIONS)
    acc = zeros(BUNDLE_ACC_TYPE, HV_DIMENSIONS)

    for i in 1:SUBSAMPLES
        context = @view(con[rand(max(end - CONTEXT_SIZE + 1, 1):end):end])
        # context = con
        gen_context_hvector!(local_acc, local_scratch, context, hvectors)
        # bundle_add!(acc, binarize_bundle(local_acc))
        bundle!(acc, local_acc)
    end
    hv = binarize_bundle(acc)
    # push!(prompt_vector, predict(tm, TMInput(hv.chunks, hv.len), diff=200))
    push!(prompt_vector, predict(tm, TMInput(hv.chunks, hv.len)))

    print(Char(prompt_vector[n + length(prompt)]))
end
println("\n")
