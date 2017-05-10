import sys
import os
import dlib
import numpy as np
from skimage import io



sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
faceRec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


def main(image):

    dets = detector(image,1)
    shape = sp(image,dets[0])
    face_features = np.zeros((1,128))
    face_features = faceRec.compute_face_descriptor(image,shape)

    print face_features

if __name__ == "__main__":


    if len(sys.argv) != 2:
        print "must be assign the image path"

    imagePath = sys.argv[1]
    print imagePath
    image = io.imread(imagePath)
    main(image)
