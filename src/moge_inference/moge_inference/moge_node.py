#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import std_msgs.msg 
import importlib.resources as pkg_resources

# TensorRT 로거 설정
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference:
    def __init__(self, engine_path):
        print(f"[TRT] Loading engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(TRT_LOGGER)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.input_shape = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            
            if is_input:
                self.input_shape = shape 

            size = trt.volume(shape) * dtype.itemsize
            host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype=trt.nptype(dtype))
            device_mem = cuda.mem_alloc(size)
            self.allocations.append(int(device_mem))

            binding = {
                "index": i,
                "name": name,
                "host": host_mem,
                "device": device_mem,
                "shape": shape,
                "dtype": trt.nptype(dtype)
            }
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def infer(self, image):
        target_h, target_w = self.input_shape[2], self.input_shape[3]
        img_resized = cv2.resize(image, (target_w, target_h))
        
        img_in = img_resized.astype(np.float32) / 255.0
        img_in = (img_in - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_in = img_in.transpose(2, 0, 1)
        img_in = np.expand_dims(img_in, axis=0) 
        
        np.copyto(self.inputs[0]['host'], img_in.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        self.context.execute_v2(self.allocations)
        
        results = {}
        for out in self.outputs:
            cuda.memcpy_dtoh(out['host'], out['device'])
            results[out['name']] = out['host'].reshape(out['shape'])
            
        return results, img_resized

class MoGeNode(Node):
    def __init__(self):
        super().__init__('moge_inference_node')
        
        self.declare_parameter('engine_path', '')
        self.declare_parameter('input_topic', '/camera/color/image_raw')
        self.declare_parameter('output_frame_id', 'camera_color_optical_frame')

        engine_path = self.get_parameter('engine_path').value
        input_topic = self.get_parameter('input_topic').value
        self.frame_id = self.get_parameter('output_frame_id').value

        try:
            if engine_path:
                self.trt_model = TRTInference(engine_path)
            else:
                self.get_logger().info("No engine_path provided. Using packaged engine.")
                self.trt_model = self._load_packaged_engine()
            self.get_logger().info("TensorRT Engine Loaded Successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load engine: {e}")
            exit(1)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, input_topic, self.image_callback, 10)
        
        self.pub_pcd = self.create_publisher(PointCloud2, '/moge/points', 10)
        self.pub_norm_vis = self.create_publisher(Image, '/moge/normal_vis', 10)

        self.get_logger().info(f"Subscribing to {input_topic}...")

    def _load_packaged_engine(self):
        resource_path = 'models/moge2_vits_fp16.engine'
        if hasattr(pkg_resources, 'files') and hasattr(pkg_resources, 'as_file'):
            resource = pkg_resources.files('moge_inference').joinpath(resource_path)
            with pkg_resources.as_file(resource) as engine_file:
                return TRTInference(str(engine_file))

        if hasattr(pkg_resources, 'path'):
            with pkg_resources.path('moge_inference', resource_path) as engine_file:
                return TRTInference(str(engine_file))

        raise RuntimeError('importlib.resources does not support package data access')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        outputs, img_resized = self.trt_model.infer(cv_image)
        
        points_raw = outputs.get('points', None)
        mask_raw = outputs.get('mask', None)
        scale_raw = outputs.get('scale', None)
        normal_raw = outputs.get('normal', None)

        if points_raw is None:
            return

        self.publish_pointcloud(points_raw, mask_raw, scale_raw, img_resized)
        
        if normal_raw is not None:
            self.publish_normal_vis(normal_raw)

    def publish_normal_vis(self, normal_tensor):
        normal_img = normal_tensor[0]
        normal_vis = ((normal_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
        normal_vis = cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR)
        
        msg = self.bridge.cv2_to_imgmsg(normal_vis, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub_norm_vis.publish(msg)

    def publish_pointcloud(self, points, mask, scale, color_img):
        points = points[0]
        if scale is not None:
            points = points * scale[0]

        valid_mask = mask[0, 0] > 0.5 
        valid_points = points[valid_mask] 
        valid_colors = color_img[valid_mask] 

        rgb_packed = np.zeros(valid_points.shape[0], dtype=np.uint32)
        rgb_packed |= (valid_colors[:, 2].astype(np.uint32) << 16) 
        rgb_packed |= (valid_colors[:, 1].astype(np.uint32) << 8)  
        rgb_packed |= (valid_colors[:, 0].astype(np.uint32))       
        rgb_packed_float = rgb_packed.view(np.float32).reshape(-1, 1)

        pc_data = np.hstack((valid_points, rgb_packed_float))

        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        pc_msg = point_cloud2.create_cloud(header, fields, pc_data)
        self.pub_pcd.publish(pc_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MoGeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
