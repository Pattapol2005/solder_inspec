import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from runtime import ResourceMonitor, probe_system


class RuntimeTests(unittest.TestCase):
    def test_cpu_only_pytorch_explains_cuda_unavailability(self):
        torch = SimpleNamespace(__version__='2.8.0+cpu',
                                version=SimpleNamespace(cuda=None))
        events = []
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict('sys.modules', {'torch': torch}), patch('runtime.platform.system', return_value='Linux'):
                result = probe_system(folder, lambda *args: events.append(args))
        self.assertEqual(result['device'], 'cpu')
        cuda_result = [(state, detail) for name, state, detail in events
                       if name == 'CUDA / PyTorch' and state != 'checking'][0]
        self.assertEqual(cuda_result[0], 'warning')
        self.assertIn('เป็นรุ่น CPU', cuda_result[1])
        self.assertIn('install_cuda.bat', cuda_result[1])

    def test_failed_cuda_execution_falls_back_and_checks_continue(self):
        torch = SimpleNamespace(__version__='test',
                                version=SimpleNamespace(cuda='12.8'),
                                cuda=SimpleNamespace(is_available=lambda: True),
                                ones=Mock(side_effect=RuntimeError('CUDA execution failed')))
        events = []
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict('sys.modules', {'torch': torch}), patch('runtime.platform.system', return_value='Linux'):
                result = probe_system(folder, lambda *args: events.append(args))
        self.assertEqual(result['device'], 'cpu')
        self.assertIsNone(result['model'])
        completed = {name: (state, detail) for name, state, detail in events if state != 'checking'}
        self.assertEqual(len(completed), 8)
        self.assertEqual(completed['CUDA / PyTorch'][0], 'warning')
        self.assertEqual(completed['best_inspec.pt'][0], 'warning')
        self.assertEqual(completed['best_locator.pt'][0], 'warning')
        self.assertEqual(completed['Storage'][0], 'ok')

    def test_resource_queue_keeps_only_latest_sample(self):
        monitor = ResourceMonitor()
        monitor.publish({'cpu': 10, 'gpu': None})
        monitor.publish({'cpu': 20, 'gpu': None})
        self.assertEqual(monitor.samples.qsize(), 1)
        self.assertEqual(monitor.samples.get_nowait(), {'cpu': 20, 'gpu': None})


if __name__ == '__main__':
    unittest.main()
