import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    p_arr=np.asarray(p)
    y_arr=np.asarray(y)
    p_y=np.sum(p_arr*y_arr)
    p_s=np.sum(p_arr)
    y_s=np.sum(y_arr)

    D=(2.0*p_y + eps)/(p_s+y_s+eps)
    loss=1.0-D
    return float(loss)