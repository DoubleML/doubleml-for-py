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
- Install the package from source. This is recommended if you want to work with the latest development version of the package. This also the best way if you wish to contribute to DoubleML.


.. tabbed:: Linux

    .. tabbed:: pip

        Install ``python3`` and ``python3`` via the package manager of your distribution, e.g., with

        .. code-block:: Bash

            $ sudo apt-get install python3 python3-pip

        To avoid potential conflicts with other packages it is recommended to use a virtual environment.

        .. tabbed:: with virtualenv

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

        .. tabbed:: without virtualenv

            To install :ref:`DoubleML <doubleml_package>` run

            .. code-block:: Bash

                $ pip3 install -U DoubleML

            To check your installation of :ref:`DoubleML <doubleml_package>` use

            .. code-block:: Bash

                $ python3 -m pip show DoubleML # to see which version and where DoubleML is installed
                $ python3 -m pip freeze # to see all packages installed in the active virtualenv
                $ python3 -c "import doubleml as dml; print(dml.__version__)"

    .. tabbed:: conda




.. _install_r:

R: Installing DoubleML
^^^^^^^^^^^^^^^^^^^^^^


