import torch
from irislandmarks import IrisLandmarks

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = IrisLandmarks().to(gpu)
net.load_weights("irislandmarks.pth")

##############################################################################
batch_size = 1
height = 64
width = 64
x = torch.randn((batch_size, height, width, 3), requires_grad=True).byte().to(gpu)
opset = 12
##############################################################################

input_names = ["input"] #[B,64,64,3],
output_names = ['eye', 'iris'] #[B,71,3], [B,5,3]

onnx_file_name = "BlazeIris_{}x{}x{}xBGRxByte_opset{}.onnx".format(batch_size, height, width, opset)
dynamic_axes = {
    "input": {0: "batch_size"}, 
    "eye": {0: "batch_size"}, 
    "iris": {0: "batch_size"}
    }

torch.onnx.export(net,
                x,
                onnx_file_name,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=input_names, 
                output_names=output_names
                #,dynamic_axes=dynamic_axes
                )
