# oakd_camera
Python code to setup and run a neural network on the OAK-D Camera and an OpenCV window with detection ROIs on the image.

Code was developed and tested in PyCharm.

Code for OAK-D camera from Luxonis.
Robotics Vision Core 2 (RVC2) with 16x SHAVE cores
 -> Streaming Hybrid Architecture Vector Engine (SHAVE)
Color camera sensor = 12MP (4032x3040 via ISP stream)
Depth perception: baseline of 7.5cm
 -> Ideal range: 70cm - 8m
 -> MinZ: ~20cm (400P, extended), ~35cm (400P OR 800P, extended), ~70cm (800P)
 -> MaxZ: ~15 meters with a variance of 10% (depth accuracy evaluation)
https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1098OAK.html

The code in this file is based on the code from Luxonis Tutorials and Code Samples.
https://docs.luxonis.com/projects/api/en/latest/tutorials/hello_world/
https://docs.luxonis.com/projects/api/en/latest/tutorials/code_samples/

This website provides a good overview of the camera and how to use the NN pipeline.
https://pyimagesearch.com/2022/12/19/oak-d-understanding-and-running-neural-network-inference-with-depthai-api/
