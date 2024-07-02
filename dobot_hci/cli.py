from .recorders import ObjectRecorder


def main():
    with ObjectRecorder() as recorder:
        recorder.run()
