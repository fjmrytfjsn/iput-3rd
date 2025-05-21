import time
import matplotlib.pyplot as plt


def generate_matrix(rows, cols):
    u = [[(i + j) % 2 for j in range(cols)] for i in range(rows)]
    v = [[(i + j) % 2 for j in range(cols)] for i in range(rows)]
    return u, v


def matrix_multiplication(u, v):
    rows = len(u)
    cols = len(v[0])
    result = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            for k in range(len(v)):
                result[i][j] += u[i][k] * v[k][j]

    return result


def main():
    max_size = 1000
    step = 100

    times = []
    sizes = []
    for size in range(step, max_size + 1, step):
        u, v = generate_matrix(size, size)
        start_time = time.time()
        matrix_multiplication(u, v)
        end_time = time.time()

        times.append(end_time - start_time)
        sizes.append(size)
        print(f"Size: {size}, Time: {end_time - start_time:.4f} seconds")

    plt.plot(sizes, times)
    plt.xlabel("Matrix Size (rows and cols)")
    plt.ylabel("Time (seconds)")
    plt.title("Matrix Multiplication Time Complexity")
    plt.grid()
    plt.savefig("matrix_multiplication_time_complexity.png")
    plt.show()


if __name__ == "__main__":
    main()
