from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

# Define bounds and parameters
xlo, ylo = -2.5, -1.5
xhi, yhi = 0.75, 1.5
nx, ny = 2048, 1536
dx, dy = (xhi - xlo) / nx, (yhi - ylo) / ny
iter_limit = 200
set_threshold = 2

def mandelbrot_test(x, y):
    z = 0
    c = x + y * 1j
    for i in range(iter_limit):
        z = z ** 2 + c
        if abs(z) > set_threshold:
            return i
    return i

def calculate_chunk(start, end):
    """
    Calculates a chunk of the Mandelbrot set rows from 'start' to 'end'.
    """
    local_result = np.zeros([end - start, nx])
    for i, row in enumerate(range(start, end)):
        y = row * dy + ylo
        for j in range(nx):
            x = j * dx + xlo
            local_result[i, j] = mandelbrot_test(x, y)
    return local_result

def calculate_serial():
    """
    Serial implementation of Mandelbrot set calculation for comparison.
    """
    result = np.zeros([ny, nx])
    for i in range(ny):
        y = i * dy + ylo
        for j in range(nx):
            x = j * dx + xlo
            result[i, j] = mandelbrot_test(x, y)
    return result

if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Determine workload per process
    rows_per_process = ny // size
    start_row = rank * rows_per_process
    end_row = (rank + 1) * rows_per_process if rank != size - 1 else ny

    # Start timing
    start_time = time.perf_counter()

    # Each process calculates its chunk
    local_result = calculate_chunk(start_row, end_row)

    # Gather all chunks to the root process
    if rank == 0:
        mandelbrot_set = np.zeros([ny, nx])
    else:
        mandelbrot_set = None
    
    comm.Gather(local_result, mandelbrot_set, root=0)

    # End timing
    stop_time = time.perf_counter()

    # Perform correctness and performance comparison
    if rank == 0:
        print(f"Parallel calculation took {stop_time - start_time:.2f} seconds with {size} processes")

        # Serial calculation for comparison
        serial_start_time = time.perf_counter()
        serial_result = calculate_serial()
        serial_stop_time = time.perf_counter()
        print(f"Serial calculation took {serial_stop_time - serial_start_time:.2f} seconds")

        # Check exact correctness
        if np.array_equal(mandelbrot_set, serial_result):
            print("Parallel and serial results match")
        else:
            print("Mismatch between parallel and serial results.")

        # Plot the results
        plt.imshow(mandelbrot_seat, interpolation="nearest", cmap="Greys")
        plt.gca().set_aspect("equal")
        plt.axis("off")
        plt.show()
