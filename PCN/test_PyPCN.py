#!/usr/bin/python3
import cv2
import sys
import os
import time

from PyPCN import build_init_detector, get_PCN_result, draw_result, get_label_dict


if __name__=="__main__":
    import pdb;pdb.set_trace()
    detector = build_init_detector()
    for i in range(1, 27):
        frame = cv2.imread("imgs/" + str(i) + ".jpg")
        face_count, windows = get_PCN_result(frame, detector)
        frame = draw_result(frame, face_count, windows)
        labels, angles = get_label_dict(face_count, windows)
        cv2.imwrite("./sample_label.jpg", frame)
        print("save sample")
        break
