import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from evidence import crop_evidence,read_evidence,result_evidence,component_detection_image
from scanner import corners,save_auto_report
from inspection import read_image,write_image


class EvidenceTests(unittest.TestCase):
    def test_component_export_preserves_detection_pixels_and_all_ic_capture_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            colours=((20,160,20),(20,20,230))
            for index,colour in enumerate(colours):
                report=root/str(index);report.mkdir()
                (report/'result.json').write_text(json.dumps(dict(component='U1')))
                frame=np.full((120,160,3),colour,np.uint8)
                cv2.rectangle(frame,(30,30),(80,80),(255,210,30),2)
                write_image(report/'result.png',frame)
            entry=dict(status='FAIL',report='1',coverage={
                '1':dict(report='0',confirmed_by='calibrated_ic_pin_reference'),
                '8':dict(report='1',confirmed_by='calibrated_ic_pin_reference')})
            image=component_detection_image('U1',entry,root)
            for colour in colours:self.assertTrue(np.any(np.all(image==colour,axis=2)))
            self.assertGreater(image.shape[0],240)
            with self.assertRaises(ValueError):component_detection_image('U4',entry,root)

    def result(self):
        image=np.zeros((240,400,3),np.uint8)
        image[50:130,80:180]=(20,160,20)
        image[50:130,280:380]=(0,0,255)  # unrelated neighbouring component
        meta=dict(component='R1',roi_quad=corners((80,50,180,130)).tolist(),expected=2)
        return (image,image.copy(),[],'PASS','test',meta)

    def test_component_view_excludes_neighbour_and_keeps_original_frame(self):
        result=self.result();original=result[1].copy()
        cropped=result_evidence(result)
        self.assertLess(cropped.shape[1],150)
        self.assertFalse(np.any(np.all(cropped==(0,0,255),axis=2)))
        np.testing.assert_array_equal(result[1],original)

    def test_rotated_crop_masks_areas_outside_component(self):
        image=np.full((200,200,3),255,np.uint8)
        cropped=crop_evidence(image,[[100,20],[180,100],[100,180],[20,100]],padding=0)
        np.testing.assert_array_equal(cropped[0,0],[16,29,43])
        np.testing.assert_array_equal(cropped[80,80],[255,255,255])

    def test_saved_component_and_each_pin_have_separate_evidence(self):
        result=self.result()
        result[5].update(component='U1',physical_pin_numbers=True,pad_polygons={1:corners((80,50,105,70)).tolist(),
                         2:corners((80,90,105,110)).tolist()},joint_states={1:'PASS',2:'FAIL'})
        with tempfile.TemporaryDirectory() as folder:
            path=save_auto_report(folder,result,result[0][50:130,80:180])
            self.assertEqual(read_image(path/'raw.png').shape,result[0].shape)
            self.assertEqual(read_image(path/'result.png').shape,result[0].shape)
            self.assertLess(read_evidence(path,1).shape[0],read_evidence(path).shape[0])
            self.assertTrue((path/'pin_02.png').is_file())
            (path/'component_result.png').unlink()
            (path/'pin_01.png').unlink()
            self.assertLess(read_evidence(path,1).shape[1],100) # legacy fallback
            with self.assertRaises(ValueError):read_evidence(path,16)

if __name__=='__main__':unittest.main()
