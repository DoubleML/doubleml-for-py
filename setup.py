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
    version='0.10.dev0',
    author='Bach, P., Chernozhukov, V., Klaassen, S., Kurz, M. S., and Spindler, M.',
    maintainer='Sven Klaassen',
    maintainer_email='sven.klaassen@uni-hamburg.de',
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
        'scikit-learn<1.6.0',
        'statsmodels',
        'plotly',
    ],
    python_requires=">=3.9",
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)
