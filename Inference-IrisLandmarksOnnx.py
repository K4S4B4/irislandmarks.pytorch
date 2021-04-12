import matplotlib.pyplot as plt
import numpy as np
import cv2
import onnxruntime
from irislandmarks import IrisLandmarks

def resize_pad(img):
    """ resize and pad images to be input to the detectors

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio.

    Returns:
        img1: 256x256
        img2: 128x128
        scale: scale factor between original image and 256x256 image
        pad: pixels of padding in the original image
    """

    size0 = img.shape
    if size0[0]>=size0[1]:
        h1 = 512
        w1 = 512 * size0[1] // size0[0]
        padh = 0
        padw = 512 - w1
        scale = size0[1] / w1
    else:
        h1 = 512 * size0[0] // size0[1]
        w1 = 512
        padh = 512 - h1
        padw = 0
        scale = size0[0] / h1
    padh1 = padh//2
    padh2 = padh//2 + padh%2
    padw1 = padw//2
    padw2 = padw//2 + padw%2
    img1 = cv2.resize(img, (w1,h1))
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)))
    pad = (int(padh1 * scale), int(padw1 * scale))
    img2 = cv2.resize(img1, (128,128))
    return img1, img2, scale, pad


onnx_file_name = 'resource/MediaPipe/BlazeIris_B_64_64_BGRxByte.onnx'
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
#sess_options.enable_profiling = True
ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options)

input_name = ort_session.get_inputs()[0].name

WINDOW='test'
cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(2)
hasFrame, frame = capture.read()

while hasFrame:
    img1, img2, scale, pad = resize_pad(frame)
    img = cv2.resize(img1, (64,64))

    #img = cv2.imread("test_eye.jpg")
    #img = cv2.resize(img, (64, 64))
    img_in = np.expand_dims(img, axis=0).astype(np.uint8)
    ort_inputs = {input_name: img_in}

    ort_outs = ort_session.run(None, ort_inputs)

    eye = ort_outs[0]
    iris = ort_outs[1]

    x, y = iris[0, :, 0], iris[0, :, 1]
    for i in range(5):
        cv2.circle(img1, (int(x[i] * 8), int(y[i] * 8)), 5, (255,0,0), -1)

    x, y = eye[0, :, 0], eye[0, :, 1]
    for i in range(71):
        cv2.circle(img1, (int(x[i] * 8), int(y[i] * 8)), 5, (0,0,255), -1)

    cv2.imshow(WINDOW, img1)
    cv2.waitKey(1)

    hasFrame, frame = capture.read()

#torch.onnx.export(
#    net, 
#    (torch.randn(1,3,64,64, device=gpu), ), 
#    "irislandmarks.onnx",
#    input_names=("image", ),
#    output_names=("preds", "conf"),
#    opset_version=9
#)