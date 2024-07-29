# Video Rendering Application
## Objective 
- Create an an application to recieve a video input encapsulating some real-world 3D space and create a near identical virtual replica in real time.
## Implementation
- This application uses OpenCV to read from a video file and structure the data for processing.
- Then, the structured data is passed into CUDA kernels which determine the vertice points of objects visible in the video.
- These vertice points are next passed to OpenGL to be rendered in a 3D space. Upon vertice render, the same colors, shading, and light sources will be applied to create a realistic representation of the original scene.
- Finally, a series of neural networks will be implemented to find points that the logic based CUDA kernels missed and pass the newfound data to OpenGL. 
## Progress
- The OpenCV code is tentativly finished. OpenCV currently reads frames from a video file, places the pixel data in data structures / containers for use in both the CUDA kernels and neural network(s).
- I am currently desiging and validating logic based CUDA kernels in the 3D real-world domain. I have found object outlines / verticies in the 2D domain, so all that's left is to plot all of these points to scale! I intend to post a more thourough README upon further progress when I have more concrete information. 