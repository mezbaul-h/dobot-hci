# hypy

This is a template to kickstart Python-based projects. It includes essential features to help you initialize a project easily.

### Features

1. Linting.
2. Code formatting.
3. Code security issues diagnosing.
4. Testing & code coverage.
5. Multi-package structure.
6. Publishing packages on PyPI.
7. GitHub workflow setup with linting, testing, and publishing actions.


## USAGE

You can create your new repository based on this template. Just click `Use this template > Create a new repository` from the top right-hand side of the screen.

### Initial Setup

After creating your repo with this template, take care of these things:

1. Remove `sample_package` and `sample_package2`.
2. Update line 15 in `scripts/gh_release.py` to import `__version__` from the correct package.
3. Update `pyproject.toml` with information appropriate for your project.
4. Change `LICENSE`, `CHANGELOG.md`, and `README.md`.
5. Create these action secrets for the workflow: `CODECOV_TOKEN`, `GH_ACCESS_TOKEN`, `PYPI_API_TOKEN`. If you don't create them, no issues will arise, but actions dependent on these won't run.

### Linting

You can run the linters to check source code styling issues:

```shell
make check
```

Or, if you don't have `make` installed:
```shell
bash scripts/check.sh
```

### Auto Source Code Formatting

You can use the code formatters to automatically format your code using:

```shell
make fmt
```

Or, if you don't have `make` installed:

```shell
bash scripts/fmt.sh
```

### Testing

You can run the test suite and generate a coverage report with:

```shell
make test
```

Or, if you don't have `make` installed:

```shell
bash scripts/test.sh
```

### Multi-Package Setup

This project already includes `sample_package` and `sample_package2`. Whenever you release, both packages will be built and distributed when you run build commands. You can add more packages or even just one, depending on your needs.

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
