from setuptools import setup

with open("README.md", 'r') as file:
    long_description = file.read()

setup(
    name='Integrated uncertainty gradients',
    version='0.1',
    description='Demonstration of integrated artificial uncertainty '
                + 'gradients for neural networks using Tensorflow.',
    long_description=long_description,
    url='',
    author='David Drakard',
    author_email='research@ddrakard.com',
    license='MIT',
    package_dir = {'': 'src'},
    install_requires=[
        'tensorflow',
        'keras',
        'tensorflow-probability',
        'numpy',
        'matplotlib',
        'pandas',
        'tensorflow-datasets',
        'pylint'
    ],
    zip_safe=False
)
