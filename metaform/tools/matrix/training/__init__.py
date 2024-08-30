# Import relevant classes and functions from each module
from .ipc import IPC
from .memory_management import MemoryManager
from .fixed_precision import FixedPrecisionHandler  # Assuming you have a FixedPrecisionHandler class
from .parallelization import Parallelizer  # Assuming you have a Parallelizer class

__all__ = [
    'IPC',
    'MemoryManager',
    'FixedPrecisionHandler',
    'Optimizer',
    'Parallelizer'
]
