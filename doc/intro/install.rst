:parenttoc: True

Installing DoubleML
===================

How to :ref:`install the R package DoubleML <install_r>` is described below,
the :ref:`installation of the Python package DoubleML <install_python>` is described in the following.

.. _install_python:

Python: Installing DoubleML
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are three different ways to install the python package :ref:`DoubleML <doubleml_package>`:

- Install the latest official release via pip or conda. This is the recommended approach for most users.
- Install a released version of DoubleML from a .whl file.
- Building the package from source. This is recommended if you want to work with the latest development version of the package. This also the best way if you wish to contribute to DoubleML.

Python: Installing the latest release from pip or conda
-------------------------------------------------------

.. tabbed:: Linux

    .. dropdown:: pip with virtual environment
        :open:

        Install ``python3`` and ``python3`` via the package manager of your distribution, e.g., with

        .. code-block:: Bash

            $ sudo apt-get install python3 python3-pip

        For Python releases see also `<https://www.python.org/downloads/source/>`_.

        To avoid potential conflicts with other packages it is recommended to use a virtual environment.
        The package virtualenv can be installed with `python3 -m pip install virtualenv`.
        We setup a virtualenv named ``dml-venv`` and activate it

        .. code-block:: Bash

            $ virtualenv -p python3 dml-venv
            $ source dml-venv/bin/activate

        To install :ref:`DoubleML <doubleml_package>` run

        .. code-block:: Bash

            $ pip install -U DoubleML

        To check your installation of :ref:`DoubleML <doubleml_package>` use

        .. code-block:: Bash

            $ python -m pip show DoubleML # to see which version and where DoubleML is installed
            $ python -m pip freeze # to see all packages installed in the active virtualenv
            $ python -c "import doubleml as dml; print(dml.__version__)"

    .. dropdown:: pip without virtual environment

        Install ``python3`` and ``python3`` via the package manager of your distribution, e.g., with

        .. code-block:: Bash

            $ sudo apt-get install python3 python3-pip

        For Python releases see also `<https://www.python.org/downloads/source/>`_.

        To install :ref:`DoubleML <doubleml_package>` run

        .. code-block:: Bash

            $ pip3 install -U DoubleML

        To check your installation of :ref:`DoubleML <doubleml_package>` use

        .. code-block:: Bash

            $ python3 -m pip show DoubleML # to see which version and where DoubleML is installed
            $ python3 -m pip freeze # to see all installed packages
            $ python3 -c "import doubleml as dml; print(dml.__version__)"

    .. dropdown:: conda with environment

        Install ``conda`` as described `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

        To avoid potential conflicts with other packages it is recommended to use a conda environment.

        We setup a conda environment named ``dml-venv`` and activate it

        .. code-block:: Bash

            $ conda create -n dml-venv
            $ conda activate dml-venv

        To install :ref:`DoubleML <doubleml_package>` run

        .. code-block:: Bash

            $ conda install -c conda-forge doubleml

        To check your installation of :ref:`DoubleML <doubleml_package>` use

        .. code-block:: Bash

            $ conda list DoubleML # to see which version and where DoubleML is installed
            $ conda list # to see all packages installed in the active conda environment
            $ python -c "import doubleml as dml; print(dml.__version__)"

    .. dropdown:: conda without environment

        Install ``conda`` as described `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

        To install :ref:`DoubleML <doubleml_package>` run

        .. code-block:: Bash

            $ conda install -c conda-forge doubleml

        To check your installation of :ref:`DoubleML <doubleml_package>` use

        .. code-block:: Bash

            $ conda list DoubleML # to see which version and where DoubleML is installed
            $ conda list # to see all installed packages
            $ python -c "import doubleml as dml; print(dml.__version__)"

.. tabbed:: macOS

    .. dropdown:: pip with virtual environment
        :open:

        Install Python 3 using ``brew install python`` or from `<https://www.python.org/downloads/mac-osx/>`_.

        To avoid potential conflicts with other packages it is recommended to use a virtual environment.
        We setup a virtual environment named ``dml-venv`` and activate it

        .. code-block:: Bash

            $ python -m venv dml-venv
            $ source dml-venv/bin/activate

        To install :ref:`DoubleML <doubleml_package>` run

        .. code-block:: Bash

            $ pip install -U DoubleML

        To check your installation of :ref:`DoubleML <doubleml_package>` use

        .. code-block:: Bash

            $ python -m pip show DoubleML # to see which version and where DoubleML is installed
            $ python -m pip freeze # to see all packages installed in the active virtualenv
            $ python -c "import doubleml as dml; print(dml.__version__)"

    .. dropdown:: pip without virtual environment

        Install Python 3 using ``brew install python`` or from `<https://www.python.org/downloads/mac-osx/>`_.

        To install :ref:`DoubleML <doubleml_package>` run

        .. code-block:: Bash

            $ pip install -U DoubleML

        To check your installation of :ref:`DoubleML <doubleml_package>` use

        .. code-block:: Bash

            $ python -m pip show DoubleML # to see which version and where DoubleML is installed
            $ python -m pip freeze # to see all packages installed in the active virtualenv
            $ python -c "import doubleml as dml; print(dml.__version__)"

    .. dropdown:: conda with environment

        Install ``conda`` as described `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

        To avoid potential conflicts with other packages it is recommended to use a conda environment.

        We setup a conda environment named ``dml-venv`` and activate it

        .. code-block:: Bash

            $ conda create -n dml-venv
            $ conda activate dml-env

        To install :ref:`DoubleML <doubleml_package>` run

        .. code-block:: Bash

            $ conda install -c conda-forge doubleml

        To check your installation of :ref:`DoubleML <doubleml_package>` use

        .. code-block:: Bash

            $ conda list DoubleML # to see which version and where DoubleML is installed
            $ conda list # to see all packages installed in the active conda environment
            $ python -c "import doubleml as dml; print(dml.__version__)"

    .. dropdown:: conda without environment

        Install ``conda`` as described `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

        To install :ref:`DoubleML <doubleml_package>` run

        .. code-block:: Bash

            $ conda install -c conda-forge doubleml

        To check your installation of :ref:`DoubleML <doubleml_package>` use

        .. code-block:: Bash

            $ conda list DoubleML # to see which version and where DoubleML is installed
            $ conda list # to see all installed packages
            $ python -c "import doubleml as dml; print(dml.__version__)"

.. tabbed:: Windows

    .. dropdown:: pip with virtualenv
        :open:

        Install Python 3. Releases are available here `<https://www.python.org/downloads/windows/>`_.

        To avoid potential conflicts with other packages it is recommended to use a virtual environment.
        We setup a virtual environment named ``dml-venv`` and activate it

        .. code-block:: Bash

            $ python -m venv dml-venv
            $ dml-venv\Scripts\activate

        To install :ref:`DoubleML <doubleml_package>` run

        .. code-block:: Bash

            $ pip install -U DoubleML

        To check your installation of :ref:`DoubleML <doubleml_package>` use

        .. code-block:: Bash

            $ python -m pip show DoubleML # to see which version and where DoubleML is installed
            $ python -m pip freeze # to see all packages installed in the active virtualenv
            $ python -c "import doubleml as dml; print(dml.__version__)"

    .. dropdown:: pip without virtual environment

        Install Python 3. Releases are available here `<https://www.python.org/downloads/windows/>`_.

        To install :ref:`DoubleML <doubleml_package>` run

        .. code-block:: Bash

            $ pip install -U DoubleML

        To check your installation of :ref:`DoubleML <doubleml_package>` use

        .. code-block:: Bash

            $ python -m pip show DoubleML # to see which version and where DoubleML is installed
            $ python -m pip freeze # to see all packages installed in the active virtualenv
            $ python -c "import doubleml as dml; print(dml.__version__)"

    .. dropdown:: conda with environment

        Install ``conda`` as described `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

        To avoid potential conflicts with other packages it is recommended to use a conda environment.

        We setup a conda environment named ``dml-venv`` and activate it

        .. code-block:: Bash

            $ conda create -n dml-venv
            $ conda activate dml-env

        To install :ref:`DoubleML <doubleml_package>` run

        .. code-block:: Bash

            $ conda install -c conda-forge doubleml

        To check your installation of :ref:`DoubleML <doubleml_package>` use

        .. code-block:: Bash

            $ conda list DoubleML # to see which version and where DoubleML is installed
            $ conda list # to see all packages installed in the active conda environment
            $ python -c "import doubleml as dml; print(dml.__version__)"

    .. dropdown:: conda without environment

        Install ``conda`` as described `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

        To install :ref:`DoubleML <doubleml_package>` run

        .. code-block:: Bash

            $ conda install -c conda-forge doubleml

        To check your installation of :ref:`DoubleML <doubleml_package>` use

        .. code-block:: Bash

            $ conda list DoubleML # to see which version and where DoubleML is installed
            $ conda list # to see all installed packages
            $ python -c "import doubleml as dml; print(dml.__version__)"



Python: Installing a released version from a .whl file
------------------------------------------------------

Released versions of the DoubleML package in form of .whl files can be obtained from
`GitHub Releases <https://github.com/DoubleML/doubleml-for-py/releases>`_.
After setting up python and pip as described above use

.. code-block:: Bash

    $ pip install -U DoubleML-0.1.1-py3-none-any.whl

Python: Building the package from source
----------------------------------------

This is recommended if you want to work with the latest development version of the package or wish to contribute to DoubleML.

First download the latest source code from GitHub via

.. code-block:: Bash

    $ git clone git@github.com:DoubleML/doubleml-for-py.git
    $ cd doubleml-for-py

Then build the package from source using pip in the editable mode.
The advantage of building the package with the flag ``--editable`` is that changes of the source code will immediately be
re-interpreted when the python interpreter restarts without having to re-build the package
:ref:`DoubleML <doubleml_package>`.

.. code-block:: Bash

    $ pip install --editable .

An alternative to pip with the ``--editable`` flag is the ``develope`` mode of setuptools. To use it call

.. code-block:: Bash

    $ python setup.py develop


.. _install_r:

R: Installing DoubleML
^^^^^^^^^^^^^^^^^^^^^^

R: Installing the latest release from CRAN
------------------------------------------

Install the last release from CRAN

.. code-block:: R

    install.packages("DoubleML")

The package can be loaded after completed installation

.. code-block:: R

    library(DoubleML)

R: Installing the development version from GitHub
--------------------------------------------------

The :ref:`DoubleML <doubleml_package>` package for R can be downloaded using the command (previous installation of the
`remotes package <https://remotes.r-lib.org/index.html>`_  is required).

.. code-block:: R

    remotes::install_github("DoubleML/doubleml-for-r")

Load the package after completed installation.

.. code-block:: R

    library(DoubleML)

