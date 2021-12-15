from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

PROJECT_URLS = {
    'Documentation': 'https://docs.doubleml.org',
    'Source Code': 'https://github.com/DoubleML/doubleml-for-py',
    'Bug Tracker': 'https://github.com/DoubleML/doubleml-for-py/issues',
}

setup(
    name='DoubleML',
    version='0.4.1',
    author='Bach, P., Chernozhukov, V., Kurz, M. S., and Spindler, M.',
    maintainer='Malte S. Kurz',
    maintainer_email='malte.simon.kurz@uni-hamburg.de',
    description='Double Machine Learning in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://docs.doubleml.org',
    project_urls=PROJECT_URLS,
    packages=find_packages(),
    install_requires=[
        'joblib',
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'statsmodels',
    ],
    python_requires=">=3.6",
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
