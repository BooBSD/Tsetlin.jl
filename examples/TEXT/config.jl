export config


const CORPUS_URL = "https://raw.githubusercontent.com/BooBSD/char-rnn/refs/heads/patch-1/data/tinyshakespeare/input.txt"
const CORPUS_PATH = joinpath(tempdir(), "input.txt")
const TM_PATH = joinpath(tempdir(), "tm_text.tm")
const HV_PATH = joinpath(tempdir(), "hvectors")
const HV_DIMENSIONS = 1024 * 16
const BUNDLE_ACC_TYPE = Float32
const NGRAM = 2
const CONTEXT_SIZE = 256
const RANDOMLY_REDUCE_CONTEXT_SIZE = false
const TRAIN_TEST_SPLIT_RATIO = 0.9
const EPOCHS = 1_000
const SAVE_MODEL_EVERY_EPOCH = 1
const SAMPLES_PER_EPOCH = 1_000_000
const ALPHA_NORM = 0.6
const ALPHA_CONTEXT::BUNDLE_ACC_TYPE = 0.7
const TEMPERATURE_NOISE::BUNDLE_ACC_TYPE = 0.02
const TOKENS_GENERATE = 10_000

const STATES_NUM = 65536
const INCLUDE_LIMIT = 65000
const SPARSE_INDEX = false

const CLAUSES = 64  # Not bad results!
const T = 512 * 1
const S = 1024 * 16
const L = 1024 * 8
const LF = 1024 * 8

# const CLAUSES = 256  # Not bad, but slow.
# const T = 1024 * 1
# const S = 1024 * 16
# const L = 1024 * 8
# const LF = 1024 * 8

# const CLAUSES = 4  # Small 1 MB model.
# const T = 128 * 1
# const S = 1024 * 16
# const L = 1024 * 8
# const LF = 1024 * 8

isfile(CORPUS_PATH) || download(CORPUS_URL, CORPUS_PATH)
corpus = read(CORPUS_PATH)
const TRAIN_SIZE = round(Int, TRAIN_TEST_SPLIT_RATIO * length(corpus))

# const PROMPT = corpus[1:CONTEXT_SIZE]
# const PROMPT = corpus[TRAIN_SIZE:TRAIN_SIZE+CONTEXT_SIZE-1]

# Simple hack to force text generation starting from "ROLE:"
const PROMPT = "--\n\n"
