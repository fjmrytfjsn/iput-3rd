import time
import numpy as np
import cupy as cp
import torch
import matplotlib.pyplot as plt
import gc


def generate_matrix_python(size):
    return [[(i + j) for j in range(size)] for i in range(size)]


def generate_matrix_np(size):
    return np.array(
        [[(i + j) for j in range(size)] for i in range(size)], dtype=np.float32
    )


def generate_matrix_cp(size):
    return cp.array(
        [[(i + j) for j in range(size)] for i in range(size)], dtype=cp.float32
    )


def generate_matrix_torch(size, device):
    return torch.tensor(
        [[(i + j) for j in range(size)] for i in range(size)],
        dtype=torch.float32,
        device=device,
    )


def matrix_multiplication_python(u, v):
    size = len(u)
    result = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += u[i][k] * v[k][j]
    return result


def matrix_multiplication_numpy(u, v):
    return np.matmul(u, v)


def matrix_multiplication_cupy(u, v):
    cp.cuda.Stream.null.synchronize()
    result = cp.matmul(u, v)
    cp.cuda.Stream.null.synchronize()
    return result


def matrix_multiplication_torch(u, v):
    torch.cuda.synchronize()
    result = torch.matmul(u, v)
    torch.cuda.synchronize()
    return result


def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step = 200
    time_limit = 40

    python_times = []
    numpy_times = []
    cupy_times = []
    torch_times = []
    multiple_sizes = []

    i = 0
    size = 0
    flag_python = True
    flag_numpy = True
    flag_cupy = True
    flag_torch = True
    print("Starting benchmark...")
    with open("multiply-result.txt", "w") as f:
        while True:
            print(
                f"flag_python: {flag_python}, flag_numpy: {flag_numpy}, flag_cupy: {flag_cupy}, flag_torch: {flag_torch}"
            )
            if not (flag_python or flag_numpy or flag_cupy or flag_torch):
                print("All methods exceeded the threshold.")
                break

            size += step if flag_python else step * 5
            print(f"Size: {size}")

            print(f"Size: {size}", end=",  \t", file=f)

            # Python
            if flag_python:
                u_py = generate_matrix_python(size)
                v_py = generate_matrix_python(size)
                start = time.time()
                matrix_multiplication_python(u_py, v_py)
                python_times.append(time.time() - start)
                print(f"Python: {python_times[-1]:.4f}s", end=",\t", file=f)
                del u_py, v_py
                if python_times[-1] > time_limit:
                    flag_python = False
            else:
                python_times.append(python_times[-1])
                print("Python: --------", end=",\t", file=f)

            # NumPy
            if flag_numpy:
                u_np = generate_matrix_np(size)
                v_np = generate_matrix_np(size)
                start = time.time()
                matrix_multiplication_numpy(u_np, v_np)
                numpy_times.append(time.time() - start)
                print(f"NumPy: {numpy_times[-1]:.4f}s", end=", \t", file=f)
                del u_np, v_np
                if numpy_times[-1] > time_limit:
                    flag_numpy = False
            else:
                numpy_times.append(numpy_times[-1])
                print("NumPy: --------", end=", \t", file=f)

            # CuPy
            if flag_cupy:
                u_cp = generate_matrix_cp(size)
                v_cp = generate_matrix_cp(size)
                start = time.time()
                matrix_multiplication_cupy(u_cp, v_cp)
                cupy_times.append(time.time() - start)
                print(f"Cupy: {cupy_times[-1]:.4f}s", end=", \t", file=f)
                del u_cp, v_cp
                if cupy_times[-1] > time_limit:
                    flag_cupy = False
            else:
                cupy_times.append(cupy_times[-1])
                print("CuPy: --------", end=", \t", file=f)

            # PyTorch
            if flag_torch:
                try:
                    u_th = generate_matrix_torch(size, device)
                    v_th = generate_matrix_torch(size, device)
                    start = time.time()
                    matrix_multiplication_torch(u_th, v_th)
                    torch_times.append(time.time() - start)
                    print(f"PyTorch: {torch_times[-1]:.4f}s", file=f)
                    if torch_times[-1] > time_limit:
                        flag_torch = False
                except Exception as e:
                    if "out of memory" in str(e):
                        print("handling OOM error")
                        print("PyTorch: OOM", file=f)
                        u_th = None
                        v_th = None
                        torch_times.append(torch_times[-1])
                        flag_torch = False
                    else:
                        raise e
                finally:
                    if u_th is not None and v_th is not None:
                        del u_th, v_th
                    torch.cuda.empty_cache()

            else:
                torch_times.append(torch_times[-1])
                print("PyTorch: --------", file=f)

            multiple_sizes.append(size)
            gc.collect()
            i += 1

    # === Plotting ===
    plt.figure(figsize=(12, 5))
    plt.plot(multiple_sizes, python_times[: len(multiple_sizes)], label="Python")
    plt.plot(multiple_sizes, numpy_times[: len(multiple_sizes)], label="NumPy")
    plt.plot(multiple_sizes, cupy_times[: len(multiple_sizes)], label="Cupy")
    plt.plot(multiple_sizes, torch_times[: len(multiple_sizes)], label="PyTorch")
    plt.title("Matrix Multiplication Time")
    plt.xlabel("Matrix Size (NxN)")
    plt.ylabel("Time (s)")
    plt.ylim(0, time_limit)
    plt.legend()
    plt.grid()
    plt.savefig("matrix_multiplication_time.png")
    plt.show()


if __name__ == "__main__":
    benchmark()
