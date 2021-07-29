#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Builds a simple resnet50 graph for testing."""
import argparse
import os
import subprocess
import sys

import onnx
import tvm
from tvm import relay, runtime
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor

from PIL import Image
import numpy as np
import tvm.relay as relay

# https://github.com/onnx/models/tree/master/vision/classification/mnist

# This example uses resnet50-v2-7 model
model_url = "".join(
    [
    "https://github.com/onnx/models/raw/master",
    "/vision/classification/mnist/model/mnist-8.onnx"
    ]
)


def build_graph_lib(opt_level):
    """Compiles the pre-trained model with TVM"""
    out_dir = os.path.join(sys.path[0], "./outlib")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Follow the tutorial to download and compile the model
    model_path = download_testdata(
        model_url, "mnist-1.onnx", module="onnx", overwrite=True)
    # print(model_path)
    onnx_model = onnx.load(model_path)
    # onnx_model = onnx.load("./mnist-8.onnx")

    # img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    # img_path = download_testdata(img_url, "imagenet_cat.png", module="data")
    img_path = "./test/img_10.jpg"

    # Resize it to 224x224
    resized_image = Image.open(img_path).resize((28, 28))
    img_data = np.asarray(resized_image).astype("float32")/255
    # print(img_data)

    img_data = np.reshape(img_data, (1,1,28,28))


    # Normalize according to the ImageNet input specification
    # imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    # imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    # norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

    # # Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
    # img_data = np.expand_dims(norm_img_data, axis=0)

    input_name = "Input3"
    shape_dict = {input_name: img_data.shape}

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    # print(mod.get_source())
    target = "llvm"
    # target = "llvm -mtriple=wasm32-unknown-unknown --system-lib"

    with tvm.transform.PassContext(opt_level=opt_level):
        factory = relay.build(mod, target=target, params=params)



    dev = tvm.device(str(target), 0)
    print(factory)
    print(dev)
    module = graph_executor.GraphModule(factory["default"](dev))

    module.set_input(input_name, img_data)
    module.run()

    output_shape = (1, 10)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

    print(tvm_output)

    # Save the model artifacts to obj_file
    obj_file = os.path.join(out_dir, "graph.o")
    # print(factory.lib.get_source())
    
    # print(factory.get_executor_config())
    factory.get_lib().save(obj_file)

    # Run llvm-ar to archive obj_file into lib_file
    lib_file = os.path.join(out_dir, "libgraph_wasm32.a")
    cmds = [os.environ.get("LLVM_AR", "llvm-ar-10"), "rcs", lib_file, obj_file]
    subprocess.run(cmds)

    # Save the json and params
    with open(os.path.join(out_dir, "graph.json"), "w") as f_graph:
        f_graph.write(factory.get_graph_json())
    with open(os.path.join(out_dir, "graph.params"), "wb") as f_params:
        f_params.write(runtime.save_param_dict(factory.get_params()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX model build example")
    parser.add_argument(
        "-O",
        "--opt-level",
        type=int,
        default=3,
        help="level of optimization. 0 is non-optimized and 3 is the highest level",
    )
    args = parser.parse_args()

    build_graph_lib(args.opt_level)
