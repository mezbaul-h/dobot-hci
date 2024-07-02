import click

from .common import CameraFeedProcessor, RealSenseFeedProcessor
from .experiments import RealTimeYoloRecorder
from .inferrers import RealTimeLandmarkInference
from .recorders import RealTimeBodyPoseRecorder, RealTimeHandGestureRecorder
from .settings import LABELS


@click.group()
def command_group():
    ...


camera_feed_processor_class_map = {
    "camera": CameraFeedProcessor,
    "realsense": RealSenseFeedProcessor,
}


@command_group.command()
@click.option(
    "-cf",
    "--camera-feed",
    default=list(camera_feed_processor_class_map.keys())[0],
    help="Camera feed to use.",
    required=True,
    type=click.Choice(list(camera_feed_processor_class_map.keys()), case_sensitive=True),
)
@click.option(
    "-cp",
    "--checkpoint",
    help="Checkpoint file to use.",
    required=True,
    type=str,
)
def infer(camera_feed, checkpoint):
    with RealTimeLandmarkInference(
        camera_feed_processor_class=camera_feed_processor_class_map[camera_feed],
        network_checkpoint_filename=checkpoint,
    ) as inferrer:
        inferrer.run()


RECORDERS = {
    "body": RealTimeBodyPoseRecorder,
    "hand": RealTimeHandGestureRecorder,
    "yolo": RealTimeYoloRecorder,
}


@command_group.command()
@click.option(
    "-l",
    "--label",
    help="Label to use.",
    required=True,
    type=click.Choice(LABELS, case_sensitive=True),
)
@click.option(
    "-cf",
    "--camera-feed",
    default=list(camera_feed_processor_class_map.keys())[0],
    help="Camera feed to use.",
    required=True,
    type=click.Choice(list(camera_feed_processor_class_map.keys()), case_sensitive=True),
)
@click.option(
    "-r",
    "--recorder",
    help="Recorder to use.",
    required=True,
    type=click.Choice(list(RECORDERS.keys()), case_sensitive=True),
)
def record(label, camera_feed, recorder):
    recorder_cls = RECORDERS[recorder]

    with recorder_cls(
        camera_feed_processor_class=camera_feed_processor_class_map[camera_feed],
        label=label,
    ) as recorder:
        recorder.run()
