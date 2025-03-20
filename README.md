# Profiler-for-GenICam-camera
This profiler can give real-time gaussian fit of optical spots captured by GenICam camera.

It is recommended to read the Readme.md in another repository 'Profiler for FLIR Camera' first.

In the FLIR profiler, the python SDK is given by Teledyne, you can just download it.
But for this profiler, python package 'harvesters' is used to communicate with GenICam camera.
What we should do is to find the .cti file in the SDK and subsitute it in the python codes.

The tested camera is mv cs060 10uc pro made by Hikrobots. Also the .cti is provided in their SDK files.(Details can be checked in the hikrobots.py)
