import core.test_engine as infer_engine
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import nnutils.c2 as c2_utils
import nnutils.vis as vis_utils
import numpy as np
from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file


class Detectron:

    def __init__(self):
        c2_utils.import_detectron_ops()
        # OpenCL may be enabled by default in OpenCV3; disable it because it's not
        # thread safe and causes unwanted GPU memory allocations.
        cv2.ocl.setUseOpenCL(False)


        merge_cfg_from_file('/home/king/Documents/measurement/detectron/configs/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml')
        cfg.NUM_GPUS = 1
        weights = '/home/king/Documents/measurement/model_final.pkl'
        assert_and_infer_cfg(cache_urls=False)
        self.model = infer_engine.initialize_model_from_cfg(weights)

    def predict_joints(self, frame):
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.model, frame, None
            )
        # (x, y, logit, prob)
        boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
            cls_boxes, cls_segms, cls_keyps)

        if keypoints != None:
            for i in range(len(keypoints)):
                p = keypoints[i]
                b = boxes[i][4]

                if b > 0.9:
                    p = np.transpose(p, (1, 0))
                    for point in p:
                        cv2.circle(frame, (point[0], point[1]), 3, (255, 0, 255), -1)
                    return keypoints[i]

        return None


