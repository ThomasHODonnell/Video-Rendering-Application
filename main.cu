#include <stdio.h>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>

#include "Video.cuh"
#include "Frame.cuh"
#include "auxiliaryFunc.cuh"
#include "Shader.h"
#include"VAO.h"
#include"VBO.h"
#include"EBO.h"
#include"Camera.h"

using namespace cv;
using std::cout;
using std::endl;

const string path = "C:\\Users\\thoma\\source\\repos\\Video-Rendering-Application\\Videos\\Rubix.avi";
const int N = 1920 * 1080;  


VideoCapture capture(path);
Mat frame;

__global__ void testK() {};

int main(int argc, char** argv) { 
	
	// OPENCV INIT
	if (!capture.isOpened()) {
		throw invalid_argument("File upload failed.");
		return -1;
	}
	//

	// VIDEO CLASS INIT
	capture >> frame;
	Video V(frame);
	//

	// GLFW INIT
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow* window = glfwCreateWindow(frame.cols , frame.cols, "Video Input Rendering", NULL, NULL);
	if (window == NULL) 
	{
		cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	gladLoadGL();
	glViewport(0, 0, frame.cols, frame.rows);
	Shader shaderProgram("default.vert", "default.frag");
	VAO VAO1;
	VAO1.Bind();
	GLfloat verts; 
	//

	namedWindow("Video Input", 1);
	bool first = true;
    while (first) {

		capture >> frame;
		
        if (frame.empty())
            break;

		imshow("Video Input", frame);

        // Press 'q' to exit the loop
        if (waitKey(30) >= 0)
            break;

		V.updateSequence(frame);
		Frame* current = new Frame();
		current->processFrame(frame);
		GLfloat* verts = current->getVerts();
		
		VBO VBO1(verts, sizeof(verts));
		//EBO EBO1(indices, sizeof(indices));

		first = false;
    }
	
	

	//OPENCV TERMINATE
	waitKey(0);
	capture.release();
	destroyAllWindows();
	//

	// GLFW TERMINATE 
	glfwDestroyWindow(window);
	glfwTerminate();
	//

	return 0;
}

