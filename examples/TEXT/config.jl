export config


CORPUS_URL = "https://raw.githubusercontent.com/BooBSD/char-rnn/refs/heads/patch-1/data/tinyshakespeare/input.txt"
CORPUS_PATH = "/tmp/input.txt"
TM_PATH = "/tmp/tm_text.tm"
HV_PATH = "/tmp/hvectors"
HV_DIMENSIONS = 1024 * 16
BUNDLE_ACC_TYPE = Int16
CONTEXT_SIZE = 256
EPOCHS = 1000
SAMPLES_PER_EPOCH = 1_000_000
LAMBDA = 0.05  # 0.05, 0.1
MIN_P = 0.05  # 0.05, 0.1
ALPHA_NORM = 1.0  # 1.0, 2.25
SUBSAMPLES = 10
TOKENS_GENERATE = 10_000
# Simple hack to force text generation starting from "ROLE:"
PROMPT = "--\n\n"

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
