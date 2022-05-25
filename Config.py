import go

ARENA_GAME_COUNT = 15

MCTS_SIMULATIONS = 100
C_PUCT = 1
DIRICHLET_ALPHA = 0.03
DIRICHLET_EPSILON = 0.25

NETWORK_FILTERS = 64
NETWORK_RESIDUAL_NUM = 9
NETWORK_INPUT_SHAPE = (go.N, go.N, 9)
NETWORK_OUTPUT_SIZE = (go.N * go.N) + 1

TRAINING_EPOCHS = 15
TRAINING_LOG_PATH = 'training_log.csv'
TRAINING_ITERATIONS = 20
SAVE_THRESHOLD = 0.55

SELFPLAY_GAMES = 10
SELFPLAY_TEMPERATURE_THRESHOLD = 15
SELFPLAY_TEMPERATURE_EARLY = 1
SELFPLAY_TEMPERATURE_TERMINAL = 0
