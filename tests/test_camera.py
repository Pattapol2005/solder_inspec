import unittest
from unittest.mock import patch

from camera import Camera


class CameraTests(unittest.TestCase):
    def test_named_start_uses_directshow_index_and_skips_other_webcams(self):
        from types import SimpleNamespace
        import cv2
        with patch('cv2_enumerate_cameras.enumerate_cameras') as enum, patch('camera.cv2.VideoCapture') as capture:
            enum.return_value=[SimpleNamespace(index=0,name='Iriun Webcam'),
                               SimpleNamespace(index=2,name='USB 2.0 Camera')]
            capture.return_value.isOpened.return_value=False
            Camera(None,640,480,'DirectShow')._run()
            capture.assert_called_once_with(2,cv2.CAP_DSHOW)
            capture.reset_mock()
            enum.return_value=enum.return_value[:1]
            camera=Camera(None,640,480,'DirectShow')
            camera._run()
            capture.assert_not_called()
            self.assertIn('USB 2.0 Camera',camera.error)

    def test_open_failure_releases_device(self):
        with patch('camera.cv2.VideoCapture') as factory:
            device = factory.return_value
            device.isOpened.return_value = False
            camera = Camera(99, 640, 480)
            camera.start()
            camera.thread.join(timeout=2)
            self.assertFalse(camera.thread.is_alive())
            self.assertIsNotNone(camera.error)
            self.assertIsNone(camera.latest())
            device.release.assert_called_once()

    def test_disconnect_does_not_leave_previous_frame_active(self):
        with patch('camera.cv2.VideoCapture') as factory, patch('camera.time.sleep'):
            device = factory.return_value
            device.isOpened.return_value = True
            device.read.return_value = (False, None)
            camera = Camera(0, 640, 480)
            camera.start()
            camera.thread.join(timeout=2)
            self.assertFalse(camera.thread.is_alive())
            self.assertIsNotNone(camera.error)
            device.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
