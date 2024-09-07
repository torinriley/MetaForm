from setuptools import setup, Extension # type: ignore
from Cython.Build import cythonize # type: ignore

extensions = [
    Extension(
        name="memory_management.memory_pool",
        sources=["memory_pool.pyx", "memory_pool.cpp"],
        language="c++", 
    )
]

setup(
    name="memory_management",
    ext_modules=cythonize(extensions),
)