#!/usr/bin/env python3

import os
import glob
import numpy as np
import tvm
import onnx
from onnx import numpy_helper
from PIL import Image

from tvm.contrib import graph_executor



lib_path = "./outlib/graph.o.so"
param_path = "./outlib/graph.params"
json_path = "./outlib/graph.json"
img_path = "./test/img_10.jpg"

loaded_lib = tvm.runtime.load_module(lib_path)
print(loaded_lib)

# print(loaded_lib.entry_func)

dev = tvm.runtime.cpu()
module = graph_executor.create(open(json_path).read(), loaded_lib, dev)

loaded_param = bytearray(open(param_path, "rb").read())
module.load_params(loaded_param)

# Resize it to 224x224
resized_image = Image.open(img_path).resize((28, 28))
img_data = np.asarray(resized_image).astype("float32")/255
img_data = np.reshape(img_data, (1,1,28,28))


print(loaded_lib)

module.set_input("Input3", img_data)
module.run()

output_shape = (1, 10)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

print(tvm_output)

# out_deploy = module.get_output(0).asnumpy()

