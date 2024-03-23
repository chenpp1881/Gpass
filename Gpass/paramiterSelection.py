import numpy as np


def PSelection(loss_values, Parameter_sets_path, weight=[0.5, 0.5], Top_P=10):
    slopes = np.abs(np.diff(loss_values))

    normalized_loss = (loss_values - np.min(loss_values)) / (np.max(loss_values) - np.min(loss_values))
    normalized_slope = (slopes - np.min(slopes)) / (np.max(slopes) - np.min(slopes))

    w_loss, w_slope = weight

    scores = w_loss * normalized_loss + w_slope * normalized_slope

    selected_indices = np.argsort(scores)[-Top_P:]

    for index in selected_indices:
        print("Parameter sets:", Parameter_sets_path[index])
        print("Score:", scores[index])
        print("Loss value:", loss_values[index])
        print("Loss slopes:", slopes[index])
        print("---------------------")