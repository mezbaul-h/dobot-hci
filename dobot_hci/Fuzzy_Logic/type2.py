import numpy as np
from pyit2fls import IT2FLS, IT2FS_Gaussian_UncertStd, crisp, max_s_norm, min_t_norm


class RobotArmFuzzyController:
    def __init__(self, dead_zone=0.5):
        # Define the domain for inputs and outputs
        self.X = np.linspace(-100, 100, 200)
        self.dead_zone = dead_zone

        # Create Type-2 Fuzzy Sets
        self.NegativeVeryLarge = IT2FS_Gaussian_UncertStd(self.X, [-100, 20, 0.1, 1.0])
        self.NegativeLarge = IT2FS_Gaussian_UncertStd(self.X, [-75, 20, 0.1, 1.0])
        self.NegativeMedium = IT2FS_Gaussian_UncertStd(self.X, [-50, 15, 0.1, 1.0])
        self.NegativeSmall = IT2FS_Gaussian_UncertStd(self.X, [-25, 15, 0.1, 1.0])
        self.NegativeVerySmall = IT2FS_Gaussian_UncertStd(self.X, [-10, 5, 0.1, 1.0])  # New set
        self.Zero = IT2FS_Gaussian_UncertStd(self.X, [0, 2, 0.1, 1.0])  # Narrower zero
        self.PositiveVerySmall = IT2FS_Gaussian_UncertStd(self.X, [10, 5, 0.1, 1.0])  # New set
        self.PositiveSmall = IT2FS_Gaussian_UncertStd(self.X, [25, 15, 0.1, 1.0])
        self.PositiveMedium = IT2FS_Gaussian_UncertStd(self.X, [50, 15, 0.1, 1.0])
        self.PositiveLarge = IT2FS_Gaussian_UncertStd(self.X, [75, 20, 0.1, 1.0])
        self.PositiveVeryLarge = IT2FS_Gaussian_UncertStd(self.X, [100, 20, 0.1, 1.0])

        # Create a Type-2 Fuzzy Logic System
        self.it2fls = IT2FLS()

        # Add input and output variables
        self.it2fls.add_input_variable("X")
        self.it2fls.add_input_variable("Y")
        self.it2fls.add_output_variable("Xout")
        self.it2fls.add_output_variable("Yout")

        # Add rules
        self._add_rules()

    def _add_rules(self):
        for input_var in ["X", "Y"]:
            output_var = input_var + "out"
            self.it2fls.add_rule([(input_var, self.NegativeVeryLarge)], [(output_var, self.PositiveVeryLarge)])
            self.it2fls.add_rule([(input_var, self.NegativeLarge)], [(output_var, self.PositiveLarge)])
            self.it2fls.add_rule([(input_var, self.NegativeMedium)], [(output_var, self.PositiveMedium)])
            self.it2fls.add_rule([(input_var, self.NegativeSmall)], [(output_var, self.PositiveSmall)])
            self.it2fls.add_rule(
                [(input_var, self.NegativeVerySmall)], [(output_var, self.PositiveVerySmall)]
            )  # New rule
            self.it2fls.add_rule(
                [(input_var, self.PositiveVerySmall)], [(output_var, self.NegativeVerySmall)]
            )  # New rule
            self.it2fls.add_rule([(input_var, self.PositiveSmall)], [(output_var, self.NegativeSmall)])
            self.it2fls.add_rule([(input_var, self.PositiveMedium)], [(output_var, self.NegativeMedium)])
            self.it2fls.add_rule([(input_var, self.PositiveLarge)], [(output_var, self.NegativeLarge)])
            self.it2fls.add_rule([(input_var, self.PositiveVeryLarge)], [(output_var, self.NegativeVeryLarge)])
            self.it2fls.add_rule([(input_var, self.Zero)], [(output_var, self.Zero)])

    def evaluate(self, x_diff, y_diff):
        """
        Evaluate the fuzzy system to get the next move.

        :param current_x: Current X position of the arm
        :param current_y: Current Y position of the arm
        :param target_x: Target X position
        :param target_y: Target Y position
        :return: Tuple of (move_x, move_y) representing the next move
        """
        # x_diff = target_x - current_x
        # y_diff = target_y - current_y

        # Check if within dead zone
        if abs(x_diff) < self.dead_zone and abs(y_diff) < self.dead_zone:
            return 0, 0

        it2out, tr = self.it2fls.evaluate({"X": x_diff, "Y": y_diff}, min_t_norm, max_s_norm, self.X)
        x_out = -crisp(tr["Xout"])  # Reverse direction
        y_out = -crisp(tr["Yout"])  # Reverse direction

        # Implement variable step size
        distance_to_target = np.sqrt(x_diff**2 + y_diff**2)
        step_size = self._calculate_step_size(distance_to_target)

        return x_out * step_size, y_out * step_size

    def _calculate_step_size(self, distance):
        """
        Calculate a variable step size based on distance to target.
        """
        max_step = 1.0
        min_step = 0.1
        distance_threshold = 50  # Adjust based on your workspace

        if distance > distance_threshold:
            return max_step
        else:
            return max(min_step, (distance / distance_threshold) * max_step)


# Example usage:
if __name__ == "__main__":
    controller = RobotArmFuzzyController(dead_zone=0.5)

    # Example positions (replace these with your camera input)
    current_x, current_y = -20, 30
    target_x, target_y = 75, -60

    # Get the fuzzy system output
    move_x, move_y = controller.evaluate(current_x, current_y, target_x, target_y)

    print(f"Current Position: ({current_x}, {current_y})")
    print(f"Target Position: ({target_x}, {target_y})")
    print(f"Fuzzy Output - Move: ({move_x:.2f}, {move_y:.2f})")

    # In your actual implementation, you would use this output to move your robot arm
    # move_robot_arm(move_x, move_y)
