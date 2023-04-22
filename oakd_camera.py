# -*- coding: utf-8 -*-

#--------------------------------------------------------------------------------------------------------------------
# Code for OAK-D camera from Luxonis.
# Robotics Vision Core 2 (RVC2) with 16x SHAVE cores
#  -> Streaming Hybrid Architecture Vector Engine (SHAVE)
# Color camera sensor = 12MP (4032x3040 via ISP stream)
# Depth perception: baseline of 7.5cm
#  -> Ideal range: 70cm - 8m
#  -> MinZ: ~20cm (400P, extended), ~35cm (400P OR 800P, extended), ~70cm (800P)
#  -> MaxZ: ~15 meters with a variance of 10% (depth accuracy evaluation)
# https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1098OAK.html
#
# The code in this file is based on the code from Luxonis Tutorials and Code Samples.
# https://docs.luxonis.com/projects/api/en/latest/tutorials/hello_world/
# https://docs.luxonis.com/projects/api/en/latest/tutorials/code_samples/

#--------------------------------------------------------------------------------------------------------------------
# pip install numpy opencv-python depthai blobconverter
import numpy as np  # numpy package -> manipulate the packet data returned by depthai
import cv2  # opencv-python  package -> display the video stream
import depthai as dai  # depthai package -> access the camera and its data packets
import blobconverter  # blobconverter package -> compile and download MyriadX neural network blobs


#--------------------------------------------------------------------------------------------------------------------
# Pipeline tells DepthAI what operations to perform when running
pipeline = dai.Pipeline()


#--------------------------------------------------------------------------------------------------------------------
# Color camera setup
# https://docs.luxonis.com/projects/api/en/latest/components/nodes/color_camera/?highlight=setPreviewSize
# https://docs.luxonis.com/projects/api/en/latest/tutorials/dispaying_detections/#
# https://docs.luxonis.com/projects/api/en/latest/tutorials/maximize_fov/
cam_rgb = pipeline.createColorCamera()
cam_rgb.setInterleaved(False)
cam_rgb.setFps(10)

# Neural network that will produce the detections
# This is the only way that I know how to create a NN, even if it is not MobileNet.
# Works with pedestrian-and-vehicle-detector-adas-0001
nn = pipeline.createMobileNetDetectionNetwork()

# Filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0..1>.
# Only works with createMobileNetDetectionNetwork()
nn.setConfidenceThreshold(0.5)


#nn = pipeline.createMobileNetSpatialDetectionNetwork()
#nn = pipeline.createNeuralNetwork()
#nn = pipeline.createYoloDetectionNetwork()
#nn = pipeline.createYoloSpatialDetectionNetwork()

#pipeline.createStereoDepth()
#pipeline.createAprilTag()
#pipeline.createFeatureTracker()
#pipeline.createObjectTracker()
#pipeline.createIMU()


#--------------------------------------------------------------------------------------------------------------------
# Configure NN and camera based on the model selected
model = "mobilenet"
#model = "ped-veh"

if model == "mobilenet":
    print("Model: mobilenet-ssd")

    # Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model.
    # We're using a blobconverter tool to retrieve the blob automatically from OpenVINO Model Zoo.
    nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
    labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # Configure camera pipeline
    cam_rgb.setPreviewSize(300, 300) # Preview frame size used to feed the NN --> FOV will be much narrow than raw image
    cam_rgb.setVideoSize(720,720)

elif model == "ped-veh":
    print("Model: pedestrian-and-vehicle-detector-adas-0001")
    nn.setBlobPath(blobconverter.from_zoo(name='pedestrian-and-vehicle-detector-adas-0001', shaves=6))
    labelMap = ["unknown", "vehicle", "pedestrian"]
    cam_rgb.setPreviewSize(672, 384) # Preview frame size used to feed the NN -> FOV will be almost same size as raw image
    cam_rgb.setVideoSize(1260,720)

elif model == "coco":
     print("Model: COCO")
     #nn.
     labelMap = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                 "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                 "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                 "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
                 "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                 "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                 "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                 "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "unknown"]

else:
    print("ERROR: Unknown NN model = " + model)
    exit()


#--------------------------------------------------------------------------------------------------------------------
# higher resolution output image for viewing (must have same aspect ratio as preview for overlay)
# To get the full FOV of a sensor you need to use its max resolution (or 1/N of it, if supported).
# You need to set full 12MP resolution (no 6MP support), then use setIspScale() to downscale to smaller size.
# 4k = 3840x2160 (limitation of "video" stream)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

#cam_rgb.setIspScale(1,2) # 1/2 scale = 1920x1080
cam_rgb.setIspScale(1,3) # 1/3 scale = 1280x720

# Link the camera 'preview' output to the neural network detection input, so that it can produce detections.
cam_rgb.preview.link(nn.input)


#--------------------------------------------------------------------------------------------------------------------
# XLinkOut is a "way out" from the device. Any data you want to transfer to host needs to be sent via XLink.
# 1) CAMERA
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")

# Option A) Stream low res image
#cam_rgb.preview.link(xout_rgb.input) # Linking camera preview to XLink input, so that the frames will be sent to host
#nn.passthrough.link(xout_rgb.input) # alternative to time synch BB and Frame

# Option B) Stream higher res image
cam_rgb.video.link(xout_rgb.input)


# 2) Neural Network
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)


#--------------------------------------------------------------------------------------------------------------------
# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with dai.Device(pipeline) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink

    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")

    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    frame = None    # no valid object
    detections = [] # empty list

    # Since the detections returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
    # receive the actual position of the bounding box on the image
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    # Main host-side application loop
    while True:
        # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            # when data from nn is received, we take the detections array that contains mobilenet-ssd results
            detections = in_nn.detections

        if frame is not None:
            for detection in detections:
                color = (255, 0, 0)

                # for each bounding box, we first normalize it to match the frame size
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                # and then draw a rectangle on the frame to show the actual result
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)

                #print("detection.label " + str(detection.label))
                if detection.label > 80:
                    detection.label = 80

                cv2.putText(frame, labelMap[detection.label] + " " + f'{detection.confidence:.2f}', (bbox[0] + 5, bbox[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            # After all the drawing is finished, we show the frame on the screen
            cv2.imshow("preview", frame)

        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break