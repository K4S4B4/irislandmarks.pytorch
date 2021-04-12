import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from irislandmarks import IrisLandmarks

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = IrisLandmarks().to(gpu)
net.load_weights("irislandmarks.pth")

img = cv2.imread("test_eye.jpg")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (64, 64))
x = torch.from_numpy(img).byte().to(gpu).unsqueeze(0)

eye_gpu, iris_gpu = net(x)

eye = eye_gpu.cpu().detach().numpy().copy()
iris = iris_gpu.cpu().detach().numpy().copy()
eye.shape
iris.shape

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img, zorder=1)

x, y = iris[:, :, 0], iris[:, :, 1]
plt.scatter(x, y, zorder=2, s=5.0)

x, y = eye[:, :, 0], eye[:, :, 1]
plt.scatter(x, y, zorder=3, s=5.0)

plt.show()

#torch.onnx.export(
#    net, 
#    (torch.randn(1,3,64,64, device=gpu), ), 
#    "irislandmarks.onnx",
#    input_names=("image", ),
#    output_names=("preds", "conf"),
#    opset_version=9
#)