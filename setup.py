from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize

sources = ['receiver/processing.pyx', 
           'receiver/cbf.c', 
           'receiver/mono12p.c',
           'receiver/downsample.c']
extentions = [Extension('receiver.processing', 
                        sources,
                        extra_compile_args=['-mavx2'])]

setup(
    name='streaming-receiver',
    packages=find_packages(),
    install_requires=["numpy", "h5py", "pyyaml", "pyzmq", "uvicorn", "fastapi", "cbor2"],
    ext_modules=cythonize(extentions),
    #include_dirs=[np.get_include()],
    extras_require={
        'tests': ["pytest", "pytest-asyncio", "aiohttp"]
    },
    entry_points = {
        'console_scripts': ['streaming-receiver = app.main:main',]
    }
)
 
