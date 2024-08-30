from threading import Thread

def worker(data_chunk):
    print(f"Processing {data_chunk}")

def parallel_process(data, chunk_size):
    threads = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        thread = Thread(target=worker, args=(chunk,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

