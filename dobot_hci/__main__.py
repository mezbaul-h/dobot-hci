import torch.multiprocessing as mp

mp.set_start_method("spawn")

from dobot_hci.cli import main

if __name__ == "__main__":
    main()
