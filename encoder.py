import numpy as np

LAYER_DICT = {
    "conv": 1,
    "maxpool": 2,
    "dropout": 3,
    "linear": 4,
    "batchnorm": 5,
    "relu": 6,
    "globalaveragepooling": 7,
}
MAX_LAYERS = 15
FEATURES_PER_LAYER = 3


def encode_architecture(arch_json):
    vector = []
    layers = arch_json.get("layers", [])

    for layer in layers:
        l_type = LAYER_DICT.get(layer["type"].lower(), 0)

        if l_type == 1:  # Conv
            vector.extend([l_type, layer.get("filters", 0), layer.get("kernel", 0)])
        elif l_type == 3:  # Dropout
            vector.extend([l_type, layer.get("p", 0), 0])
        elif l_type == 4:  # Linear
            vector.extend([l_type, layer.get("units", 0), 0])
        else:
            vector.extend([l_type, 0, 0])

    # Padding
    current_len = len(vector) // FEATURES_PER_LAYER
    padding_needed = MAX_LAYERS - current_len

    if padding_needed > 0:
        vector.extend([0, 0, 0] * padding_needed)

    return np.array(vector[: MAX_LAYERS * FEATURES_PER_LAYER])
