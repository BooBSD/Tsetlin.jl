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
