import torch
import numpy as np
import cv2

# Tiles array of numpy arrays
def imtile(arr, nrow, ncol):
    assert arr.shape[0] == nrow * ncol
    # Combine columns
    cols_merged = [cv2.vconcat([arr[ncol*row + col] for row in range(nrow)])
            for col in range(ncol)]
    rows_merged = cv2.hconcat([col for col in cols_merged])
    rows_merged = (rows_merged*255).astype(np.uint8)
    return cv2.cvtColor(rows_merged, cv2.COLOR_RGB2BGR)
