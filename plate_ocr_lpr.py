#coding=utf-8
import cv2
import os
import numpy as np
from tensorflow.contrib import predictor

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class PlateOCR:

    def __init__(self, model_dir='/ssd/wfei/code/Plate_Recognition-LPRnet/model_pb_h5', num_channel=1):
        self.predict_fn = predictor.from_saved_model(model_dir)
        self.num_channel = num_channel

    def __call__(self, img):
        """ Plate OCR with LPR
         results: chars, score
        """
        plate_file = img.strip()

        if not os.path.exists(plate_file.encode('utf-8')):
            print('File does not exist')
            return
        img = cv2.imread(plate_file)
        img = cv2.resize(img, (94, 24))
        if self.num_channel == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images = img[np.newaxis, :, :, np.newaxis]
        else:
            images = img[np.newaxis, :]

        images = images/255.
        images = np.transpose(images, axes=[0, 2, 1, 3])

        inputs = {"Placeholder": images,
                  "Placeholder_4": [24]}
        # "is_train": False}

        predictions = self.predict_fn(inputs)
        #print "Prediction:      ", predictions['code2str_conversion/predicted'][0].decode('utf-8')
        #print "Confidence Score:", predictions['confidence_score'][0]
        chars = predictions['code2str_conversion/predicted'][0].decode('utf-8')
        chars = chars.replace('-', '')
        score = predictions['confidence_score'][0]

        return chars, score

if __name__ == '__main__':

    test_img = '/mnt/soulfs2/wfei/data/plate_sample2.jpg'
    pocr = PlateOCR()
    chars, score = pocr(test_img)
    print chars, score


