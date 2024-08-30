import multiprocessing
import logging

def parallel_map(func, data, num_processes=None):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    num_processes = num_processes or multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            result = pool.map(func, data)
        except Exception as e:
            logger.error("An error occurred during parallel processing", exc_info=True)
            pool.terminate()  # ensure all processes are terminated in case of an error
            raise e
        else:
            pool.close()  # no more tasks will be sent to the pool
            pool.join()   # wait for the worker processes to exit
    return result
