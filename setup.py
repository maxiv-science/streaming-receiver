from setuptools import Extension, setup
from Cython.Build import cythonize

sources = [
    "streaming_receiver/receiver/processing.pyx",
    "streaming_receiver/receiver/cbf.c",
    "streaming_receiver/receiver/mono12p.c",
    "streaming_receiver/receiver/downsample.c",
]
extentions = [
    Extension(
        "streaming_receiver.receiver.processing", sources, extra_compile_args=["-mavx2"]
    )
]

setup(
    # name="streaming-receiver",
    ext_modules=cythonize(extentions)
)
