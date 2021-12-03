# DoubleML - Contributing Guidelines <a href="https://docs.doubleml.org"><img src="https://raw.githubusercontent.com/DoubleML/doubleml-for-py/master/doc/logo.png" align="right" width = "120" /></a>

DoubleML is a community effort.
Everyone is welcome to contribute.
This contributing guidelines help you to get started for your first contribution,
and we recommend to also read our
[code of conduct](https://github.com/DoubleML/doubleml-for-py/blob/master/CODE_OF_CONDUCT.md).

## Submit a Bug Report :bug:
To submit a bug report, you can use our
[issue template for bug reports](https://github.com/DoubleML/doubleml-for-py/issues/new/choose).

- A good bug reports contains a **minimum reproducible code snippet, like for example

```python
import numpy as np
import doubleml as dml
from doubleml.datasets import make_plr_CCDDHNR2018
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
np.random.seed(3141)
learner = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
ml_g = learner
ml_m = learner
obj_dml_data = make_plr_CCDDHNR2018(alpha=0.5, n_obs=500, dim_x=20)
dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
dml_plr_obj.fit().summary
```

- State the result you would have expected and the result you actually got.
In case of an exception the full traceback is appreciated.
- State the versions of your code by running the following lines and copy-paste the result.

```python
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import doubleml; print("DoubleML", sklearn.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
```

## Submit a Feature Request :bulb:
We welcome feature request and suggestions towards improving and/or extending the DoubleML package.
For feature requests you can use the corresponding
[issue template](https://github.com/DoubleML/doubleml-for-py/issues/new/choose).

## Submit a Question or Start a Discussion
We use [GitHub Discussions](https://github.com/DoubleML/doubleml-for-py/discussions) to give the community a platform
for asking questions about the DoubleML package and for discussions on topics related to the package.

## Contribute Code :computer:
Everyone is welcome to contribute to the DoubleML code base.
The following guidelines and hints help you to get started.

### Development Workflow
The recommended way to contribute to DoubleML is described in detail in following.
The most important steps are: To fork the repo, then add your changes and finally submit a pull-request.
1. Fork the [DoubleML repo](https://github.com/DoubleML/doubleml-for-py)
by clicking on the Fork button (this requires a GitHub account).
2. Clone your fork to your local machine, e.g., via
```bash
$ git clone git@github.com:YourGitHubAccount/doubleml-for-py.git
$ cd doubleml-for-py
```
3. Create a feature branch
```bash
$ git checkout -b my_feature_branch
```
4. (Optionally) you can add the `upstream` remote.
This allows you to easily keep your repository in synch as demonstrated in the following.
```bash
$ git remote add upstream https://github.com/DoubleML/doubleml-for-py.git
```
```bash
$ # Update your repo from upstream via
$ git fetch upstream
$ git merge upstream/master
```
5. Install the development dependencies via
```bash
$ pip install -r requirements.txt
$ pip install -r requirements-dev.txt
```
6. Install DoubleML in editable mode via (more details can be found
[here](https://docs.doubleml.org/stable/intro/install.html#python-building-the-package-from-source))
```bash
$ pip install --editable .
```
7. Develop your code changes. The changes can be added and pushed via
```bash
$ git add your_new_file your_modified_file
$ git commit -m "A commit message which briefly summarizes the changes made"
$ git push origin my_feature_branch
```
8. Generate a pull request from your fork.
Please follow our guidelines for pull requests.
When opening the PR you will be guided via a checklist.

### Checklist for Pull Requests (PR)
If your PR is still work in progress please consider marking it a draft PR
(see also [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)).
-[x] The title of the pull request summarizes the changes made.
-[x] The PR contains a detailed description of all changes and additions.
-[x] References to related issues or PRs are added.
(you may want to comment on the diff in GitHub).
-[x] The code passes all (unit) test (see below for details).
To check, please run
```bash
$ pytest .
```
-[x] If you added an enhancements or new feature unit tests
(with a certain level of coverage) are mandatory for getting the PR merged.
-[x] Check whether your changes adhere to the PEP8 standards.
For the check you can use the following code
```bash
$ git diff upstream/master -u -- "*.py" | flake8 --diff --max-line-length=127
```

### Unit Test and Test Coverage

### Contribute a New Model Class

## Contribute Documentation :books:

### Contribute to the API Documentation

### Contribute to the User Guide and Documentation

### Contribute a Case Study for the Example Gallery
