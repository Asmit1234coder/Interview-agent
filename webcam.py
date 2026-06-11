import io
import cv2
from PIL import Image
from src.config import JPEG_QUALITY,MAX_FRAME_SIZE

class Camera:
    """one webcam,opened once,read one JPEG frame at a time."""
    def __init__(self,device_index:int=0)->None:
        """"open webcame.Raises error if no camera is found."""
        self._capture=cv2.VideoCapture(device_index)
        if not self._capture.isOpened:
            raise RuntimeError(
                f"No camera found at index {device_index}"
            )
        
    def read_jpeg_frame(self)->bytes | None:
        """Grab one frame amd return it as jPEG bytes or None on failure.
        """
        ok, frame=self._capture.read()
        if not ok:
            return None
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image=Image.fromarray(rgb_frame)
        image.thumbnail(MAX_FRAME_SIZE)

        buffer=io.BytesIO()
        image.save(buffer,format="JPEG",quality=JPEG_QUALITY)
        return buffer.getvalue()
    
    def close(self)->None:
        """Return webcam so other apps can use it"""
        self._capture.release()