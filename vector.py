import numpy as np
import timeit


class MultiDimensionalArray:
    def __init__(self, dim, start, stop):
        self.dim = dim
        self.start = start
        self.stop = stop
        self.array = np.random.rand(*dim)

    def direct_access(self, indices):
        return self.array[indices]

    def diagonal_access(self, indices):
        diag = np.diag(self.array, k=indices[0] - indices[1])
        return diag[::-1] if indices[0] > indices[1] else diag

    def fft_access(self, indices):
        shape = self.array.shape
        indices = np.fft.fftfreq(shape[0]) * shape[0]
        indices = indices.astype(int)
        indices = np.fft.fftshift(indices) + shape[1] // 2
        return self.array[indices, indices]


if __name__ == '__main__':
    dim = (100, 100)
    start = (-50, -50)
    stop = (50, 50)

    array = MultiDimensionalArray(dim, start, stop)

    indices = (75, 25)

    t_direct = timeit.timeit(lambda: array.direct_access(indices), number=1000)
    t_diagonal = timeit.timeit(
        lambda: array.diagonal_access(indices), number=1000)
    t_fft = timeit.timeit(lambda: array.fft_access(indices), number=1000)

    print("Direct Access: {:.6f} seconds".format(t_direct))
    print("Diagonal Access: {:.6f} seconds".format(t_diagonal))
    print("FFT Access: {:.6f} seconds".format(t_fft))

    print("Memory Usage:")
    print("Direct Access: {:.2f} MB".format(array.array.nbytes / 1024 / 1024))
    print("Diagonal Access: {:.2f} MB".format(
        array.array.diagonal().nbytes / 1024 / 1024))
    print("FFT Access: {:.2f} MB".format(
        array.fft_access(indices).nbytes / 1024 / 1024))
