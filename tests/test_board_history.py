import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from board_history import BoardHistory,STATUS_COLOURS,recycle_board


class BoardHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
        self.folder=self.base/'completed_boards'
        for key,state in (('board_pass','PASS'),('board_fail','FAIL')):
            folder=self.folder/key
            folder.mkdir(parents=True)
            (folder/'board.json').write_text(json.dumps(dict(board_number=key,saved_at='2026-11-18',
                                                           overall_status=state,results={})))
        self.root=tk.Tk()
        self.root.withdraw()
        self.app=SimpleNamespace(report_root=self.base,info=tk.StringVar(master=self.root),show_evidence=Mock())
        self.history=BoardHistory(self.root,self.app)
        self.history.pack(fill='both',expand=True)
        self.history.reload()
        self.pump(lambda:not self.history.loading)

    def tearDown(self):
        self.root.destroy()
        self.temp.cleanup()

    def pump(self,condition):
        deadline=time.monotonic()+4
        while not condition() and time.monotonic()<deadline:
            self.root.update()
            time.sleep(.005)
        self.assertTrue(condition())

    def choose(self,key):
        self.history.boards.selection_set(key)
        self.history.select_board()

    def test_result_colours_apply_to_selected_and_unselected_rows(self):
        for key,state in (('board_pass','PASS'),('board_fail','FAIL')):
            self.choose(key)
            self.assertIn(state,self.history.boards.item(key,'tags'))
            self.assertEqual(str(self.history.boards.tag_configure(state,'foreground')),STATUS_COLOURS[state])
            self.assertEqual(str(self.history.style.lookup('BoardHistory.Treeview','foreground',('selected',))),
                             STATUS_COLOURS[state])

    def test_cancel_delete_preserves_board_and_evidence(self):
        self.choose('board_fail')
        with patch('board_history.messagebox.askyesno',return_value=False),patch('board_history.send2trash') as recycle:
            self.history.delete_selected()
        recycle.assert_not_called()
        self.assertTrue((self.folder/'board_fail'/'board.json').is_file())

    def test_delete_recycles_only_selected_archive_and_refreshes_history(self):
        self.choose('board_fail')
        def move_to_test_recycle(path):
            Path(path).rename(self.base/'recycled')
        with patch('board_history.messagebox.askyesno',return_value=True),patch('board_history.send2trash',side_effect=move_to_test_recycle) as recycle:
            self.history.delete_selected()
            self.pump(lambda:not self.history.deleting and not self.history.loading)
        recycle.assert_called_once_with(str((self.folder/'board_fail').resolve()))
        self.assertNotIn('board_fail',self.history.entries)
        self.assertIn('board_pass',self.history.entries)
        self.assertTrue((self.folder/'board_pass'/'board.json').is_file())
        self.assertTrue((self.base/'recycled'/'board.json').is_file())

    def test_recycle_failure_keeps_board_visible(self):
        self.choose('board_fail')
        with patch('board_history.messagebox.askyesno',return_value=True),patch('board_history.send2trash',side_effect=OSError('busy')),patch('board_history.messagebox.showerror') as error:
            self.history.delete_selected()
            self.pump(lambda:not self.history.deleting and not self.history.loading)
        error.assert_called_once()
        self.assertIn('board_fail',self.history.entries)
        self.assertTrue((self.folder/'board_fail'/'board.json').is_file())

    def test_invalid_delete_target_never_reaches_recycle_bin(self):
        with patch('board_history.send2trash') as recycle:
            for key in ('..','../board_pass','board_fail/../board_pass','board_unknown'):
                with self.assertRaises(ValueError):recycle_board(self.folder,key)
        recycle.assert_not_called()


if __name__=='__main__':unittest.main()
