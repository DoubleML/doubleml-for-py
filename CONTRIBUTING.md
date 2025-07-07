# DoubleML - Contributing Guidelines <a href="https://docs.doubleml.org"><img src="https://raw.githubusercontent.com/DoubleML/doubleml-for-py/main/doc/logo.png" align="right" width = "120" /></a>

DoubleML is a community effort.
Everyone is welcome to contribute.
All contributors should adhere to this contributing guidelines
and our [code of conduct](https://github.com/DoubleML/doubleml-for-py/blob/main/CODE_OF_CONDUCT.md).
The contributing guidelines are particularly helpful to get started for your first contribution.

## Submit a Bug Report :bug:
To submit a **bug report**, you can use our
[issue template for bug reports](https://github.com/DoubleML/doubleml-for-py/issues/new/choose).

- A good bug report contains a **minimum reproducible code snippet**, like for example

```python
import numpy as np
import doubleml as dml
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
np.random.seed(3141)
ml_g = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
ml_m = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
obj_dml_data = make_plr_CCDDHNR2018(alpha=0.5, n_obs=500, dim_x=20)
dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
dml_plr_obj.fit().summary
```

- State the **result you would have expected** and the **result you actually got**.
In case of an exception the full traceback is appreciated.

- State the **versions of your code** by running the following lines and copy-paste the result.

```python
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import doubleml; print("DoubleML", sklearn.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
```

## Submit a Feature Request :bulb:
We welcome **feature requests and suggestions** towards improving and/or extending the DoubleML package.
For feature requests you can use the corresponding
[issue template](https://github.com/DoubleML/doubleml-for-py/issues/new/choose).

## Submit a Question or Start a Discussion
We use **[GitHub Discussions](https://github.com/DoubleML/doubleml-for-py/discussions)** to give the community a platform
for asking questions about the DoubleML package and for discussions on topics related to the package.

## Contribute Code :computer:
Everyone is welcome to contribute to the DoubleML code base.
The following guidelines and hints help you to get started.

### Development Workflow
In the following, the recommended way to contribute to DoubleML is described in detail.
The most important steps are: To **fork** the repo, then **add your changes** and finally submit a **pull-request**.
1. **Fork** the [DoubleML repo](https://github.com/DoubleML/doubleml-for-py)
by clicking on the Fork button (this requires a GitHub account).

2. **Clone** your fork to your local machine via
```bash
$ git clone git@github.com:YourGitHubAccount/doubleml-for-py.git
$ cd doubleml-for-py
```

3. Create a **feature branch** via
```bash
$ git checkout -b my_feature_branch
```

4. (Optionally) you can add the `upstream` remote.
```bash
$ git remote add upstream https://github.com/DoubleML/doubleml-for-py.git
```
This allows you to easily keep your repository in synch via
```bash
$ git fetch upstream
$ git merge upstream/main
```

5. **Install DoubleML in editable mode** (more details can be found
[here](https://docs.doubleml.org/stable/intro/install.html#python-building-the-package-from-source))
via
```bash
$ pip install --editable .[dev, rdd]
```

6. **Develop** your code changes. The changes can be added and pushed via
```bash
$ git add your_new_file your_modified_file
$ git commit -m "A commit message which briefly summarizes the changes made"
$ git push origin my_feature_branch
```

7. Generate a **pull request** from your fork.
Please follow our guidelines for pull requests.
When opening the PR you will be guided with a checklist.

### Checklist for Pull Requests (PR)
- [x] The **title** of the pull request summarizes the changes made.

- [x] The PR contains a **detailed description** of all changes and additions
(you may want to comment on the diff in GitHub).

- [x] **References** to related issues or PRs are added.

- [x] The code passes **all (unit) tests** (see
[below](https://github.com/DoubleML/doubleml-for-py/blob/main/CONTRIBUTING.md#unit-test-and-test-coverage)
for details).
To check, please run
```bash
$ pytest .
```

- [x] If you add an **enhancements** or **new feature**, **unit tests**
(with a certain level of coverage) are **mandatory** for getting the PR merged.

- [x] Check whether your changes adhere to the **PEP8 standards**.
For the check you can use the following code
```bash
$ git diff upstream/main -u -- "*.py" | ruff check --diff
```

- [x] Check wether the code formatting adheres to the **Black code style**
by running
```bash
$ black . --check --diff
```

If your PR is still **work in progress**, please consider marking it a **draft PR**
(see also [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)).

### (Optional) Set up pre-commit Hooks

To ensure code quality and consistency before committing your changes, we recommend using [pre-commit hooks](https://pre-commit.com/). Pre-commit hooks will automatically run checks like code formatting and linting on your staged files.

1. **Install hooks**:
   If you haven't already, install the required hooks by running:
   ```bash
   $ pre-commit install
   ```

2. **Run pre-commit manually**:
    To run the pre-commit checks manually, use:
   ```bash
   $ pre-commit run --all-files
   ```

### Unit Tests and Test Coverage
We use the package **pytest for unit testing**.
Unit testing is considered to be a fundamental part of the development workflow.
The tests are located in the `tests` subfolder.
The test coverage is determined with the `pytest-cov` package.
Coverage reports for the package, PRs, branches etc. are available from
[codecov](https://app.codecov.io/gh/DoubleML/doubleml-for-py).
It is mandatory to equip new features with an appropriate level of unit test coverage.
To **run all unit tests** (for further option see the [pytest docu](https://docs.pytest.org)) call
```bash
$ pytest --cov .
```
If `pytest` is called with the `--cov` flag, a unit test coverage report is being generated.

### Contribute a New Model Class
The **DoubleML package** is particularly designed in a flexible way to make it **easily extendable** with regard to
**new model classes**.
**Contributions** in this direction **are very much welcome**, and we are happy to help authors to integrate their models in the
DoubleML OOP structure.
If you need assistance, just open an issue or contact one of the maintainers
[@SvenKlaassen](https://github.com/SvenKlaassen) or [@PhilippBach](https://github.com/PhilippBach).

The **abstract base class `DoubleML` implements all core functionalities** based on a linear Neyman orthogonal score
function.
To contribute a new model class, you only need to **specify all nuisance functions** that need to be estimated for the
new model class (e.g. regressions or classifications).
Furthermore, the **score components for the Neyman orthogonal score function need to be implemented**.
All other functionality is automatically available via inheritance from the abstract base class.
A **template for new model classes** is available
[here](https://github.com/DoubleML/doubleml-docs/blob/main/model_templates/double_ml_model_template.py).

## Contribute Documentation :books:
The **documentation** of DoubleML is generated with **sphinx** and hosted at
[https://docs.doubleml.org](https://docs.doubleml.org).
The Python API documentation is generated from docstrings in the source code.
The source code for the website, user guide, example gallery, etc. is available in a separate repository
[https://github.com/DoubleML/doubleml-docs](https://github.com/DoubleML/doubleml-docs).

### Contribute to the API Documentation
The **API documentation** is generated from **docstrings** in the source code.
It can be generated locally (dev requirements sphinx and pydata-sphinx-theme need to be installed) via
```bash
$ cd doc/
$ make html
```

### Contribute to the User Guide and Documentation
The **documentation of DoubleML** is hosted at [https://docs.doubleml.org](https://docs.doubleml.org).
The **source code** for the website, user guide, example gallery, etc. is available in a **separate repository
[doubleml-docs](https://github.com/DoubleML/doubleml-docs)**.
Changes, issues and PRs for the documentation (except the API documentation) should be discussed in the
[doubleml-docs](https://github.com/DoubleML/doubleml-docs) repo.
We welcome contributions to the user guide, especially case studies for the
[example gallery](https://docs.doubleml.org/stable/examples/index.html).
A step-by-step guide for contributions to the example gallery is available
[here](https://github.com/DoubleML/doubleml-docs/wiki/Contribute-to-our-Website-and-Example-Gallery).
