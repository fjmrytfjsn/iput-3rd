import time
import numpy as np
import cupy as cp
import torch
import matplotlib.pyplot as plt
import gc
from scipy.signal import convolve2d
from cupyx.scipy.ndimage import convolve as cp_convolve


def generate_matrix(size):
    return [[(i + j) % 2 for j in range(size)] for i in range(size)]


def generate_matrix_np(size):
    return np.array(
        [[(i + j) % 2 for j in range(size)] for i in range(size)], dtype=np.float32
    )


def generate_matrix_cp(size):
    return cp.array(
        [[(i + j) % 2 for j in range(size)] for i in range(size)], dtype=cp.float32
    )


def generate_matrix_torch(size, device):
    return (
        torch.tensor(
            [[(i + j) % 2 for j in range(size)] for i in range(size)],
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )


def convolution_python(img, kernel):
    h, w = len(img), len(img[0])
    kh, kw = len(kernel), len(kernel[0])
    output = [[0.0 for _ in range(w - kw + 1)] for _ in range(h - kh + 1)]

    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            for ki in range(kh):
                for kj in range(kw):
                    output[i][j] += img[i + ki][j + kj] * kernel[ki][kj]
    return output


def convolution_numpy(img, kernel):
    return convolve2d(img, kernel, mode="valid")


def convolution_cupy(img, kernel):
    cp.cuda.Stream.null.synchronize()
    result = cp_convolve(img, kernel, mode="constant", cval=0.0)
    cp.cuda.Stream.null.synchronize()
    return result


def convolution_torch(img, kernel):
    result = torch.nn.functional.conv2d(img, kernel, padding=0)
    torch.cuda.synchronize()
    return result


def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step = 1000
    time_limit = 30
    max_size = 25000

    kernel_py = [[1 / 9.0] * 3 for _ in range(3)]
    kernel_np = np.ones((3, 3)) / 9.0
    kernel_cp = cp.ones((3, 3), dtype=cp.float32) / 9.0
    kernel_torch = torch.tensor(kernel_np).unsqueeze(0).unsqueeze(0).to(device).float()

    python_times = []
    numpy_times = []
    cupy_times = []
    torch_times = []
    conv_sizes = []

    size = 0
    flag_python = True
    flag_numpy = True
    flag_cupy = True
    flag_torch = True

    print("Starting convolution benchmark...")

    with open("conv-result.txt", "w") as f:
        while True:
            if not (flag_python or flag_numpy or flag_cupy or flag_torch):
                print("All methods exceeded the threshold.", file=f)
                break
            if size >= max_size:
                print("Maximum size reached.", file=f)
                break

            size += step
            conv_sizes.append(size)
            print(f"Size: {size}", end=",\t", file=f)

            # Python
            if flag_python:
                u_py = generate_matrix(size)
                start = time.time()
                convolution_python(u_py, kernel_py)
                python_times.append(time.time() - start)
                print(f"Python: {python_times[-1]:.4f}s", end=", \t", file=f)
                del u_py
                if python_times[-1] > time_limit:
                    flag_python = False
            else:
                python_times.append(python_times[-1])
                print("Python: --------", end=", \t", file=f)

            # NumPy
            if flag_numpy:
                u_np = generate_matrix_np(size)
                start = time.time()
                convolution_numpy(u_np, kernel_np)
                numpy_times.append(time.time() - start)
                print(f"NumPy: {numpy_times[-1]:.4f}s", end=", \t", file=f)
                del u_np
                if numpy_times[-1] > time_limit:
                    flag_numpy = False
            else:
                numpy_times.append(numpy_times[-1])
                print("NumPy: --------", end=", \t", file=f)

            # CuPy
            if flag_cupy:
                try:
                    u_cp = generate_matrix_cp(size)
                    start = time.time()
                    convolution_cupy(u_cp, kernel_cp)
                    cupy_times.append(time.time() - start)
                    print(f"CuPy: {cupy_times[-1]:.4f}s", end=", \t", file=f)
                    del u_cp
                    if cupy_times[-1] > time_limit:
                        flag_cupy = False
                except Exception as e:
                    print("CuPy: Error", file=f)
                    cupy_times.append(cupy_times[-1] if cupy_times else 0)
                    flag_cupy = False
            else:
                cupy_times.append(cupy_times[-1])
                print("CuPy: --------", end=", \t", file=f)

            # PyTorch
            if flag_torch:
                try:
                    u_th = generate_matrix_torch(size, device)
                    start = time.time()
                    convolution_torch(u_th, kernel_torch)
                    torch_times.append(time.time() - start)
                    print(f"PyTorch: {torch_times[-1]:.4f}s", file=f)
                    if torch_times[-1] > time_limit:
                        flag_torch = False
                except Exception as e:
                    print("PyTorch: Error", file=f)
                    torch_times.append(torch_times[-1] if torch_times else 0)
                    flag_torch = False
                finally:
                    if "u_th" in locals():
                        del u_th
                    torch.cuda.empty_cache()
            else:
                torch_times.append(torch_times[-1])
                print("PyTorch: --------", file=f)

            gc.collect()

    # === Plotting ===
    plt.figure(figsize=(12, 5))
    plt.plot(conv_sizes, python_times[: len(conv_sizes)], label="Python")
    plt.plot(conv_sizes, numpy_times[: len(conv_sizes)], label="NumPy")
    plt.plot(conv_sizes, cupy_times[: len(conv_sizes)], label="CuPy")
    plt.plot(conv_sizes, torch_times[: len(conv_sizes)], label="PyTorch")
    plt.title("Matrix Convolution Time (3x3 kernel)")
    plt.xlabel("Matrix Size (NxN)")
    plt.ylabel("Time (s)")
    plt.ylim(0, time_limit)
    plt.legend()
    plt.grid()
    plt.savefig("matrix_convolution_time.png")
    plt.show()


if __name__ == "__main__":
    benchmark()
