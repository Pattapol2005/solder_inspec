import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from board import BoardChecklist, COMPONENTS, crop_input, restore_box, parse_joints


class BoardTests(unittest.TestCase):
    def test_crop_and_inverse_boxes_for_all_rotations(self):
        frame=np.arange(60*80*3,dtype=np.uint8).reshape(60,80,3)
        roi=(10,15,60,45)
        # Original local box x=5..20, y=7..17 inside a 50x30 crop.
        rotated={0:(5,7,20,17),90:(13,5,23,20),180:(30,13,45,23),270:(7,30,17,45)}
        for angle,box in rotated.items():
            crop,actual=crop_input(frame,roi,angle)
            self.assertEqual(actual,roi)
            self.assertEqual(crop.shape[:2],(30,50) if angle in (0,180) else (50,30))
            self.assertEqual(restore_box(box,roi,angle),(15.,22.,30.,32.))
        self.assertTrue(np.array_equal(crop_input(frame,roi,90)[0],cv2.rotate(frame[15:45,10:60],cv2.ROTATE_90_CLOCKWISE)))
        for invalid in [(0,0,0,5),(-1,0,4,5),(0,0,81,60)]:
            with self.assertRaises(ValueError):crop_input(frame,invalid)

    def test_joint_ranges_are_bounded(self):
        self.assertEqual(parse_joints('1-4, 8,4',16),[1,2,3,4,8])
        for invalid in ['', '0', '17', '8-1', '1-999999999', '1-', '1,']:
            with self.assertRaises(ValueError):parse_joints(invalid,16)

    def test_partial_ic_coverage_history_and_reinspection(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=BoardChecklist(Path(folder)/'boards')
            report=Path(folder)/'evidence';report.mkdir();(report/'result.json').write_text('{}')
            def record(state,joints):
                ledger.record('U1',state,'test',report,len(joints),(0,0,100,100),0,joints)
            record('PASS',list(range(1,9)))
            self.assertEqual(ledger.session['results']['U1']['status'],'REVIEW')
            self.assertEqual(ledger.completed(),0)
            record('FAIL',[9])
            record('PASS',list(range(10,15)))
            item=ledger.session['results']['U1']
            self.assertEqual(item['status'],'FAIL')
            self.assertEqual(item['covered'],14)
            self.assertEqual(ledger.completed(),1)
            record('PASS',[9])
            self.assertEqual(ledger.session['results']['U1']['status'],'PASS')
            self.assertEqual(len(ledger.session['results']['U1']['history']),3)
            record('REVIEW',[1])
            self.assertEqual(ledger.completed(),0)
            restored=BoardChecklist(ledger.folder)
            self.assertEqual(restored.session,ledger.session)
            old_id=ledger.session['board_id']
            ledger.new_board()
            self.assertNotEqual(ledger.session['board_id'],old_id)
            self.assertTrue((ledger.folder/(old_id+'.json')).exists())
            self.assertEqual(ledger.session['results'],{})
            ledger.export_csv(Path(folder)/'check.csv')
            with (Path(folder)/'check.csv').open(encoding='utf-8-sig',newline='') as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),62)
            self.assertEqual(next(r for r in rows if r['component']=='U1')['total'],'14')

    def test_no_completion_without_evidence_or_full_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=BoardChecklist(Path(folder)/'boards')
            with self.assertRaises(ValueError):ledger.record('R1','PASS','test',folder,2,None,0,[1,2])
            self.assertEqual(ledger.session['results'],{})
            (Path(folder)/'result.json').write_text('{}')
            ledger.record('R1','FAIL','one bridge',folder,2,None,0,[1,2],coverage_confirmed=False)
            self.assertEqual(ledger.completed(),0)
            self.assertEqual(ledger.session['results']['R1']['status'],'FAIL')
            with patch.object(ledger,'save',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):ledger.record('R2','PASS','test',folder,2,None,0,[1,2])
            self.assertNotIn('R2',ledger.session['results'])


if __name__=='__main__':unittest.main()
