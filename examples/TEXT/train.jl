include("../../src/Tsetlin.jl")
include("HDC.jl")
include("config.jl")


using Dates
using Base.Threads
using Serialization
using .Tsetlin: TMInput, TMClassifier, train!, save, compile, literals_sum


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

# Training the TM model
hv_sample = gen_context_hvector(@view(CORPUS[1:CONTEXT_SIZE]))
x_sample = TMInput(hv_sample.chunks, hv_sample.len)
y_samples = collect(keys(hvectors))
tm = TMClassifier(x_sample, y_samples, CLAUSES, T, S, L=L, LF=LF, states_num=65536, include_limit=65000)
save(compile(tm), TM_PATH)  # Save empty model for sample()

density = round(sum(x_sample) / length(x_sample) * 100, digits=2)
println("\nClasses: $(tm.classes_num), clauses: $(tm.clauses_num), T: $(tm.T), S: $(tm.S) (s: $(tm.s)), L: $(tm.L), LF: $(tm.LF), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit).")
println("Input vector size: $(length(x_sample)) bits, density: $(density)%, training dataset size: $(CORPUS_LENGTH).")
println("Expected average clause literal density: $(round(tm.L / length(x_sample) * 100, digits=2))%. Using literals index: false.")
println("Running in $(nthreads()) threads. Training over $(EPOCHS) epochs:\n")
all_time = @elapsed begin
    for epoch in 1:EPOCHS
        epoch_time = @elapsed begin
            counter = Atomic{Int}(0)
            @threads for _ in 1:SAMPLES_PER_EPOCH
                if counter[] > SAMPLES_PER_EPOCH
                    break
                end
                start = rand(1:CORPUS_LENGTH - CONTEXT_SIZE - 1)
                finish = rand(start:start + CONTEXT_SIZE - 1)
                y = CORPUS[finish + 1]
                # Balancing classes
                @inbounds for i in 1:get_stochastic_updates(tokens_probs[y])
                    context = @view(CORPUS[start:finish])
                    hv = gen_context_hvector(context)
                    x = TMInput(hv.chunks, hv.len)
                    train!(tm, x, y)
                    atomic_add!(counter, 1)
                end
            end
        end
        epoch_time = Time(0) + Second(floor(Int, epoch_time))
        println("Epoch #$(epoch) elapsed in $(epoch_time)")
        save(compile(tm), TM_PATH)
    end
end
elapsed = Time(0) + Second(floor(Int, all_time))
average_clause_density = round((literals_sum(tm) / (tm.classes_num * tm.clauses_num * 2)) / length(x_sample) * 100, digits=2)
println("\n$(EPOCHS) epochs done in $(elapsed).")
println("Classes: $(tm.classes_num), clauses: $(tm.clauses_num), T: $(tm.T), S: $(tm.S) (s: $(tm.s)), L: $(tm.L), LF: $(tm.LF), states_num: $(tm.state_max + 1), include_limit: $(tm.include_limit).")
println("Input vector size: $(length(x_sample)) bits, density: $(density)%, training dataset size: $(CORPUS_LENGTH).")
println("Average clause literal density: $(average_clause_density)%. Using literals index: false.\n")
