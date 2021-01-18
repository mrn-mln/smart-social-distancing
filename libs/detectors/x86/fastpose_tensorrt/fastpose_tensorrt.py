import sys

sys.path.append("libs/detectors/x86/fastpose_tensorrt")
import time
import os
import torch
import numpy as np
import cv2
import pycuda.driver as cuda
import PIL
import tensorrt as trt
from libs.detectors.utils.fps_calculator import convert_infr_time_to_fps
from builders import builder
import logging
from convert_results_format import prepare_detection_results, prepare_poses_results

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def allocate_buffers(engine):
    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    bindings = []
    for i in range(engine.num_bindings):
        binding = engine[i]
        size = trt.volume(engine.get_binding_shape(binding)) * \
               engine.max_batch_size
        host_mem = cuda.pagelocked_empty(size, np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    stream = cuda.Stream()  # create a CUDA stream to run inference
    return bindings, host_inputs, cuda_inputs, host_outputs, cuda_outputs, stream


class Detector:
    def _load_engine(self):
        # TODO: add repo path and add script to export trt from onnx
        trtbinpath = 'fastpose_resnet152_duc_256_192.trt'
        if not os.path.exists(trtbinpath):
            os.system('bash /repo/generate_fastpose_tensorrt.bash config-x86-gpu-fastpose-tensorrt.ini')  # TODO
        with open(trtbinpath, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, config):
        self.config = config
        self.fps = None
        self.w, self.h, _ = [int(i) for i in self.config.get_section_dict('Detector')['ImageSize'].split(',')]
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.model_input_size = (self.w, self.h)
        self.device = None  # enter your Gpu id here
        self.cuda_context = None
        self._init_cuda_stuff()
        self.name = config.get_section_dict('Detector')['Name']
        self.detection_model = builder.build_detection_model(self.name, config)

    def _init_cuda_stuff(self):
        cuda.init()
        self.engine = self._load_engine()
        self.device = cuda.Device(0)  # enter your Gpu id here
        self.cuda_context = self.device.make_context()
        self.engine_context = self.engine.create_execution_context()
        bindings, host_inputs, cuda_inputs, host_outputs, cuda_outputs, stream = allocate_buffers(self.engine)
        self.bindings = bindings
        self.host_inputs = host_inputs
        self.host_outputs = host_outputs
        self.cuda_inputs = cuda_inputs
        self.cuda_outputs = cuda_outputs
        self.stream = stream

    def __del__(self):
        """ Free CUDA memory. """

        self.cuda_context.pop()
        del self.cuda_context
        del self.engine_context
        del self.engine

    def inference(self, image):
        """
            inference function sets input tensor to input image and gets the output.
            The model provides corresponding detection output which is used for creating result
            Args:
                image: uint8 numpy array with shape (img_height, img_width, channels)
            Returns:
                result: a dictionary contains of [{"id": 0, "bbox": [y1, x1, y2, x2], "score":s%, "face": [y1, x1, y2, x2]}, {...}, {...}, ...]
        """
        bindings = self.bindings
        host_inputs = self.host_inputs
        host_outputs = self.host_outputs
        cuda_inputs = self.cuda_inputs
        cuda_outputs = self.cuda_outputs
        stream = self.stream

        t_begin = time.perf_counter()
        detections = self.detection_model.inference(image)
        if len(detections) == 0:
            return []
        detections = prepare_detection_results(detections, self.w, self.h)
        inps, cropped_boxes, boxes, scores, ids = self._transform_detections(image, detections)
        # The shape of inps ???!!!
        # TODO: input feeding
        inps = inps.astype(np.float32)
        host_inputs[0] = np.ravel(np.zeros_like(inps))
        self.cuda_context.push()

        np.copyto(host_inputs[0], inps.ravel())
        cuda.memcpy_htod_async(
            cuda_inputs[0], host_inputs[0], stream)

        self.engine_context.execute_async(
            batch_size=1,
            bindings=bindings,
            stream_handle=stream.handle)

        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        output_dict = host_outputs[0]
        # TODO: decode

        inference_time = float(time.perf_counter() - t_begin)
        self.fps = convert_infr_time_to_fps(inference_time)
        return # TODO

    def _transform_detections(self, image, dets):
        if isinstance(dets, int):
            return 0, 0
        dets = dets[dets[:, 0] == 0]
        boxes = dets[:, 1:5]
        scores = dets[:, 5:6]
        ids = torch.zeros(scores.shape)
        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)
        for i, box in enumerate(boxes):
            inps[i], cropped_box = self._transform_single_detection(image, box)
            cropped_boxes[i] = torch.FloatTensor(cropped_box)
        return inps, cropped_boxes, boxes, scores, ids
