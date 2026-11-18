from pathlib import Path
from types import SimpleNamespace
import tempfile
import tkinter as tk
import unittest

import numpy as np

from app import App
from board import COMPONENTS


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.root=tk.Tk()
        self.app=App(self.root,auto_model=False,monitor=False,auto_scan=True,
                     board_store=Path(self.folder.name)/'boards')

    def tearDown(self):
        self.app.close();self.folder.cleanup()

    def test_camera_border_fits_actual_frame_after_resize_and_rotation(self):
        for geometry in ('1440x900','1080x760'):
            self.root.geometry(geometry)
            for width,height in ((640,480),(480,640),(1280,720)):
                self.app.frame=self.app.display_frame=np.zeros((height,width,3),np.uint8)
                self.app.render();self.root.update()
                canvas=self.app.canvas
                cw,ch=canvas.winfo_width(),canvas.winfo_height()
                self.assertLessEqual(cw,self.app.viewport.winfo_width())
                self.assertLessEqual(ch,self.app.viewport.winfo_height())
                self.assertLess(abs(cw-ch*width/height),2)
                ox,oy,sx,sy=self.app.view_transform
                self.assertLessEqual(abs(ox),.5);self.assertLessEqual(abs(oy),.5)
                self.assertAlmostEqual(sx*width,cw,delta=1)
                self.assertAlmostEqual(sy*height,ch,delta=1)

    def test_diagram_uses_saved_results_and_does_not_certify_location_only(self):
        report=Path(self.folder.name)/'report';report.mkdir()
        (report/'result.json').write_text('{}')
        for name,state in (('R1','PASS'),('R2','FAIL'),('R3','REVIEW')):
            self.app.checklist.record(name,state,'test',report,2,None,0,[1,2])
        self.app.auto_targets=[SimpleNamespace(component='R4')]
        self.app.refresh_checklist();self.root.update()
        diagram=self.app.board_layout
        self.assertEqual(set(diagram.hit_boxes),set(COMPONENTS))
        for name,state in (('R1','PASS'),('R2','FAIL'),('R3','REVIEW'),('R4','SEEN'),('R5','PENDING')):
            self.assertEqual(diagram.states[name],state)
        self.assertNotIn('R4',self.app.checklist.session['results'])
        self.app.checklist.new_board();self.app.reset_scan_board();self.app.refresh_checklist()
        self.assertEqual(set(diagram.states.values()),{'PENDING'})

    def test_clicking_diagram_selects_component_without_leaving_auto_mode(self):
        self.root.update()
        diagram=self.app.board_layout
        x1,y1,x2,y2=diagram.hit_boxes['R22']
        diagram.select(SimpleNamespace(x=(x1+x2)/2,y=(y1+y2)/2))
        self.assertEqual(self.app.component.get(),'R22')
        self.assertTrue(self.app.auto_mode.get())
        self.assertEqual(self.app.checklist.session['results'],{})


if __name__=='__main__':unittest.main()
