import time
import tkinter as tk
import unittest
from unittest.mock import patch

from startup import Startup


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.entered = []
        # Inject diagnostics through the same queue used by the worker.
        self.worker = patch.object(Startup, 'work', return_value=None)
        self.worker.start()
        self.timing = patch.multiple('startup', STEP_HOLD_MS=1, RESULT_HOLD_MS=1, FINAL_HOLD_MS=25)
        self.timing.start()
        self.startup = Startup(self.root, self.entered.append)
        self.payload = dict(device='cpu', model=None, signature=None)

    def tearDown(self):
        if not self.startup.closed:
            self.startup.close()
        else:
            self.root.destroy()
        self.timing.stop()
        self.worker.stop()

    def pump_until(self, condition, timeout=2):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.005)
        self.assertTrue(condition())

    def deliver(self, state='ok'):
        for name in self.startup.rows:
            self.startup.events.put(('row', (name, 'checking', 'checking')))
            self.startup.events.put(('row', (name, state if name == 'best_inspec.pt' else 'ok', 'result')))
        self.startup.events.put(('done', self.payload))
        self.pump_until(lambda: self.startup.ready)

    def assert_manual_entry(self):
        deadline = time.monotonic() + .15
        while time.monotonic() < deadline:
            self.root.update()
            time.sleep(.005)
        self.assertEqual(self.entered, [])
        self.assertIsNone(self.startup.auto_timer)
        self.assertEqual(self.startup.enter['state'], 'normal')
        self.startup.enter.invoke()
        self.assertEqual(self.entered, [self.startup.payload])

    def test_all_checks_pass_enters_automatically_once(self):
        self.deliver()
        self.pump_until(lambda: bool(self.entered))
        self.assertEqual(self.entered, [self.payload])
        self.startup.finish()
        self.assertEqual(len(self.entered), 1)

    def test_warning_waits_for_user(self):
        self.deliver('warning')
        self.assert_manual_entry()

    def test_error_waits_for_user(self):
        self.deliver('error')
        self.assert_manual_entry()

    def test_fatal_diagnostic_failure_still_allows_manual_entry(self):
        self.startup.events.put(('fatal', 'diagnostics unavailable'))
        self.pump_until(lambda: self.startup.ready)
        self.assertIn('diagnostics unavailable', self.startup.note.get())
        self.assert_manual_entry()

    def test_incomplete_checks_cannot_enter_automatically(self):
        self.startup.events.put(('done', self.payload))
        self.pump_until(lambda: self.startup.ready)
        self.assert_manual_entry()


if __name__ == '__main__':
    unittest.main()
