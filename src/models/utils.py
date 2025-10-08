from pyts.image import RecurrencePlot
import numpy as np
import torch

def time_series_2_recurrence_plot(x):
    # normalize input to numpy array (handle torch tensors if provided)
    try:
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)
    except Exception:
        arr = np.asarray(x)

    # Supported shapes:
    # (n_samples, length), (length,), (n_samples, n_channels, length)
    if arr.ndim == 1:
        arr2 = arr[None, :]
        imgs = RecurrencePlot().fit_transform(arr2)  # (1, L, L)
        return imgs.astype(np.float32)
    elif arr.ndim == 2:
        arr2 = arr
        imgs = RecurrencePlot().fit_transform(arr2)  # (n_samples, L, L)
        return imgs.astype(np.float32)
    elif arr.ndim == 3:
        n_samples, n_channels, length = arr.shape
        # If single channel, behave as before and return (n_samples, length, length)
        if n_channels == 1:
            arr2 = arr[:, 0, :]
            imgs = RecurrencePlot().fit_transform(arr2)  # (n_samples, L, L)
            return imgs.astype(np.float32)
        # For multichannel, compute one RP per channel per sample -> (n_samples, n_channels, L, L)
        arr2 = arr.reshape(n_samples * n_channels, length)
        imgs = RecurrencePlot().fit_transform(arr2)  # (n_samples*n_channels, L, L)
        imgs = imgs.reshape(n_samples, n_channels, length, length)
        return imgs.astype(np.float32)
    else:
        raise ValueError(f"Unsupported input shape: {arr.shape}")

if __name__ == "__main__":
    x = torch.randn(4,5,32)
    out = time_series_2_recurrence_plot(x)
    print(out.shape)
    import matplotlib.pyplot as plt

    plt.imshow(out[0][:1].swapaxes(0,2), cmap="gray")