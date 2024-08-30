import multiprocessing

def parallel_map(func, data, num_processes=None):
    with multiprocessing.Pool(processes=num_processes) as pool:
        result = pool.map(func, data)
    return result

def distribute_matrices(matrices, num_workers):
    chunk_size = len(matrices) // num_workers
    return [matrices[i:i + chunk_size] for i in range(0, len(matrices), chunk_size)]
