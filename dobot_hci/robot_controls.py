from typing import List

import torch.multiprocessing as mp


class RobotAction:
    def __init__(self, **kwargs):
        self.log_queue = kwargs["log_queue"]

    def move_object_near(self, source_object_name, target_object_name):
        ...

    def move_object_on_top_of(self, source_object_name, target_object_name):
        ...

    def move_object_to_down(self, object_name):
        ...

    def move_object_to_down_of(self, source_object_name, target_object_name):
        ...

    def move_object_to_left(self, object_name):
        ...

    def move_object_to_left_of(self, source_object_name, target_object_name):
        ...

    def move_object_to_right(self, object_name):
        ...

    def move_object_to_right_of(self, source_object_name, target_object_name):
        ...

    def move_object_to_up(self, object_name):
        ...

    def move_object_to_up_of(self, source_object_name, target_object_name):
        ...


def execute_robot_action(method: str, arguments: List[str], log_queue: mp.Queue) -> None:
    try:
        robot_action = RobotAction(log_queue=log_queue)

        getattr(robot_action, method)(*arguments)
    except AttributeError:
        raise AttributeError(f'Method "{method}" not found in RobotAction')
