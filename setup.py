import setuptools

setuptools.setup(
    name='aicfit',
    version='0.0.1',
    description='Perform parameter search using AIC for the selected distribution.',
    url='https://github.com/nijigen-plot/aicfit',
    author='nijigen-plot',
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=[
        'numpy>=1.21.1',
        'pandas>=1.3.0'
    ],
    python_requires='>=3.7',
)