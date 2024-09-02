# Dobot HCI

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


```shell
QT_QPA_PLATFORM=wayland python -m dobot_hci
```
