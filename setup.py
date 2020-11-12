from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/DoubleML/doubleml-for-py/issues',
    'Documentation': 'http://doubleml.org',
    'Source Code': 'https://github.com/DoubleML/doubleml-for-py'
}

setup(
    name='DoubleML',
    version='0.1.dev0',
    author='Bach, P., Chernozhukov, V., Kurz, M. S., and Spindler, M.',
    maintainer='Malte S. Kurz',
    maintainer_email='malte.simon.kurz@uni-hamburg.de',
    description='Double Machine Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://doubleml.org',
    packages=find_packages(),
    install_requires=[
        'joblib',
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'statsmodels',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
