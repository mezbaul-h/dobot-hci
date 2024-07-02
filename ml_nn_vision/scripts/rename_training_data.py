from pathlib import Path

TRAINING_DATA_ROOT = "wisdom/training_data"


for recorder in Path(TRAINING_DATA_ROOT).glob(pattern="*"):
    if recorder.is_dir():
        for gesture in recorder.glob(pattern="*"):
            if gesture.is_dir():
                for data_file in gesture.glob(pattern="*.npy"):
                    if data_file.is_file():
                        file_ts = data_file.name.split("__")[-1].split(".")[0]
                        data_file.rename(data_file.parent / f"{recorder.name}__{gesture.name}__{file_ts}.npy")
