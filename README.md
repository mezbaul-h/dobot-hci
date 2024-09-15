# Multimodal Dobot HCI

This repository contains the source code implementing the system presented in our paper: *Advancing Human-Robot Interaction: A Multimodal Approach Combining Video & Speech Language Models with Fuzzy Logic*.

## Environment

The project was developed and tested under the following environment specifications:

- **CPU**: Intel Core i9-14900HX
- **GPU**: NVIDIA 4070 8GB
- **CUDA**: 12.5
- **Camera**: Intel RealSense D435i (connected via USB 2.0)
- **Robot**: DOBOT Magician (connected via USB 2.0)
- **Microphone**: Samsung Buds 2 (connected via Bluetooth)
- **Operating System**: Ubuntu 22.04 LTS
- **Python**: 3.10

## Installation

Before proceeding, ensure the following native packages are installed:

```bash
sudo apt install build-essential ffmpeg libxcb-xinerama0 portaudio19-dev python3-dev
```

Then, install the Python dependencies with:

```bash
make deps
```
To operate the DOBOT Magician, you need to add the current user to the `dialout` group for USB permissions:

```bash
sudo usermod -a -G dialout $USER
```

> [!IMPORTANT]
> You will need to restart your machine for these changes to take effect.

> [!NOTE]
> The installation process is optimized for the environment specified above (or similar Debian/Ubuntu-based systems). If using a different setup, modifications might be necessary to ensure compatibility.


## Execution
To start the project, simply run:

```bash
python -m dobot_hci -rs
```

## License
This project is licensed under the MIT License.
