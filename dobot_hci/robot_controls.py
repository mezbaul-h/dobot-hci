from typing import List

import torch.multiprocessing as mp


class RobotAction:
    def __init__(self, object_positions, **kwargs):
        self.log_queue = kwargs["log_queue"]
        self.object_positions = object_positions

    @property
    def end_effector_position(self):
        for k in self.object_positions:
            if k.startswith("end_effector"):
                return self.object_positions[k]

        raise ValueError("End effector position not found")

    @staticmethod
    def get_position_center(position):
        return (position[0] + position[2]) / 2, (position[1] + position[3]) / 2

    def move_object_near(self, source_object_name, target_object_name):
        end_effector_position = self.get_position_center(self.end_effector_position)
        source_object_position = self.get_position_center(self.object_positions[source_object_name])
        target_object_position = self.get_position_center(self.object_positions[target_object_name])

    def move_object_on_top_of(self, source_object_name, target_object_name):
        end_effector_position = self.get_position_center(self.end_effector_position)
        source_object_position = self.get_position_center(self.object_positions[source_object_name])
        target_object_position = self.get_position_center(self.object_positions[target_object_name])

    def move_object_to_down(self, object_name):
        end_effector_position = self.get_position_center(self.end_effector_position)
        object_position = self.get_position_center(self.object_positions[object_name])

    def move_object_to_down_of(self, source_object_name, target_object_name):
        end_effector_position = self.get_position_center(self.end_effector_position)
        source_object_position = self.get_position_center(self.object_positions[source_object_name])
        target_object_position = self.get_position_center(self.object_positions[target_object_name])

    def move_object_to_left(self, object_name):
        end_effector_position = self.get_position_center(self.end_effector_position)
        object_position = self.get_position_center(self.object_positions[object_name])

    def move_object_to_left_of(self, source_object_name, target_object_name):
        end_effector_position = self.get_position_center(self.end_effector_position)
        source_object_position = self.get_position_center(self.object_positions[source_object_name])
        target_object_position = self.get_position_center(self.object_positions[target_object_name])

    def move_object_to_right(self, object_name):
        end_effector_position = self.get_position_center(self.end_effector_position)
        object_position = self.get_position_center(self.object_positions[object_name])

    def move_object_to_right_of(self, source_object_name, target_object_name):
        end_effector_position = self.get_position_center(self.end_effector_position)
        source_object_position = self.get_position_center(self.object_positions[source_object_name])
        target_object_position = self.get_position_center(self.object_positions[target_object_name])

    def move_object_to_up(self, object_name):
        end_effector_position = self.get_position_center(self.end_effector_position)
        object_position = self.get_position_center(self.object_positions[object_name])

    def move_object_to_up_of(self, source_object_name, target_object_name):
        end_effector_position = self.get_position_center(self.end_effector_position)
        source_object_position = self.get_position_center(self.object_positions[source_object_name])
        target_object_position = self.get_position_center(self.object_positions[target_object_name])


def execute_robot_action(method: str, arguments: List[str], object_positions, log_queue: mp.Queue) -> None:
    try:
        robot_action = RobotAction(object_positions, log_queue=log_queue)

        getattr(robot_action, method)(*arguments)
    except AttributeError:
        raise
    except KeyError:
        raise
    except ValueError:
        raise