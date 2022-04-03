# ============ DonkeyCar Config ================== #
# Raw camera input

CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160

MARGIN_TOP = CAMERA_HEIGHT // 3
# MARGIN_TOP = 0

# ============ End of DonkeyCar Config ============ #

# Camera max FPS
FPS = 40


# Region Of Interest
# r = [margin_left, margin_top, width, height]
ROI = [0, MARGIN_TOP, CAMERA_WIDTH, CAMERA_HEIGHT - MARGIN_TOP]

# Fixed input dimension for the autoencoder
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 80
N_CHANNELS = 3
RAW_IMAGE_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH, N_CHANNELS)
INPUT_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)

# Arrow keys, used by opencv when displaying a window
UP_KEY = 82
DOWN_KEY = 84
RIGHT_KEY = 83
LEFT_KEY = 81
ENTER_KEY = 10
SPACE_KEY = 32
EXIT_KEYS = [113, 27]  # Escape and q
S_KEY = 115  # S key
