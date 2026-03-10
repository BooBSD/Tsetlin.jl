export config


CORPUS_URL = "https://raw.githubusercontent.com/BooBSD/char-rnn/refs/heads/patch-1/data/tinyshakespeare/input.txt"
CORPUS_PATH = "/tmp/input.txt"
TM_PATH = "/tmp/tm_text.tm"
HV_PATH = "/tmp/hvectors"
HV_DIMENSIONS = 1024 * 16
BUNDLE_ACC_TYPE = Float32
NGRAM = 2
CONTEXT_SIZE = 256
TRAIN_SIZE = 900_000
EPOCHS = 1_000
SAVE_MODEL_EVERY_EPOCH = 1
SAMPLES_PER_EPOCH = 1_000_000
ALPHA_NORM = 0.6
ALPHA_CONTEXT::BUNDLE_ACC_TYPE = 0.9
TEMPERATURE_NOISE::BUNDLE_ACC_TYPE = 0.02
TOKENS_GENERATE = 10_000

STATES_NUM = 65536
INCLUDE_LIMIT = 65000

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

# CLAUSES = 4  # Small 1 MB model.
# T = 128 * 1
# S = 16000
# L = 8192
# LF = 8192

# isfile(CORPUS_PATH) || download(CORPUS_URL, CORPUS_PATH)
# PROMPT = read(CORPUS_PATH)[1:CONTEXT_SIZE]
# PROMPT = read(CORPUS_PATH)[TRAIN_SIZE:TRAIN_SIZE+CONTEXT_SIZE-1]

# Simple hack to force text generation starting from "ROLE:"
PROMPT = "--\n\n"
