import io, os
import torch
import tensorrt as trt
from torch2trt import torch2trt
from inception import Inception3
import time
from PIL import Image
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)),
                                    dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)),
                                     dtype=trt.nptype(trt.float32))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def load_normalized_test_case(input_shape, test_image, pagelocked_buffer, normalization_hint):
    def normalize_image(image):
        c, h, w = input_shape
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1])\
            .astype(trt.nptype(trt.float32)).ravel()
        if (normalization_hint == 0):
            image_arr = (image_arr / 255.0 - 0.45) / 0.225
            return image_arr
        elif (normalization_hint == 1):
            image_arr = (image_arr / 256.0 - 0.5)
            print('Image',image_arr)
            print(image_arr.shape)
            return image_arr

    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image

## Load torch.saved inceptionv3 model
entire_path = 'model_1000.pt'#'./models/inception_v3.entire'
model = torch.load(entire_path).eval()

## torch2trt conversion
engine_path='./tortrt_v3.engine'
x = torch.ones((16, 3, 299, 299)).cuda()
model_trt = torch2trt(model,
                      [x],
                      max_batch_size=16,
                      fp16_mode=False,
                      max_workspace_size=1<<32)

with open(engine_path, "wb") as f:
    f.write(model_trt.engine.serialize())

## Run inference on torch and torch2trt to compare the difference
image_path='./data/tabby_tiger_cat.jpg'
img = Image.open(io.BytesIO(open(image_path, 'rb').read()))
img = img.resize((299,299), Image.ANTIALIAS)
image = np.asarray(img, dtype=float)
image = image / 256.0 - 0.5
print('Image',image.transpose([2, 0, 1]))
print(image.shape)
device=torch.device("cuda:0")
image_tensor = torch.from_numpy(np.array([image.transpose([2, 0, 1])])).to(torch.float).cuda()

y = model(image_tensor)
print(y)
y_trt = model_trt(image_tensor)
print(y_trt)
print(torch.max(torch.abs(y - y_trt)))

## Reload the serialized engine and do the inference 
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

with engine.create_execution_context() as context:
    h_input, d_input, h_output, d_output, stream  = allocate_buffers(engine)
    test_case = load_normalized_test_case((3, 299, 299), image_path, h_input, normalization_hint=1)
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=[d_input, d_output], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    print('output',h_output)
