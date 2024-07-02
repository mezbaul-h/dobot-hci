from PIL import Image

from .camera_feeds import RealSenseFeedProcessor
from .models.vision import Florence2FT


class ObjectRecorder:
    def __enter__(self):
        # self.run()
        self._lazy_initialization()

        return self

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Release the VideoCapture and destroy OpenCV windows
        self.camera_feed_proc.close()

    def __init__(self, **kwargs):
        self.camera_feed_proc = RealSenseFeedProcessor()
        self.vision_model = Florence2FT()

    def _lazy_initialization(self):
        ...

    def run(self):
        def key_event_hook(e):
            ...

        def process_frame_hook(**kwargs):
            frame = kwargs["frame"]
            frame_rgb = kwargs["frame_rgb"]
            frame_shape = frame.shape
            h, w, c = frame_shape

            # Process the image with MediaPipe
            # results = self.pose.process(frame_rgb)
            pil_image = Image.fromarray(frame_rgb)
            print(self.vision_model.infer(pil_image))

        self.camera_feed_proc.run(process_frame_hook=process_frame_hook)
