#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

int main(void) {
	// Create the two input vectors
	int N = 64;

	float *A = (float *)malloc(sizeof(float) * (N) * (N));
	float *B = (float *)malloc(sizeof(float) * (N) * (N));

	int i, j, k;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			A[i * N + j] = (rand() / (float)RAND_MAX) * (0.5 - 0) + 0.5;
			B[i * N + j] = (rand() / (float)RAND_MAX) * (0 - 0.5) - 0.5;
		}
	}

	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("matmul_HW2.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;   
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, 
			&device_id, &ret_num_devices);

	// Create an OpenCL context
	cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create memory buffers on the device for each vector 
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
			N * N * sizeof(float), NULL, &ret);

	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
			N * N * sizeof(float), NULL, &ret);

	cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
			N * N * sizeof(float), NULL, &ret);

	// Copy the lists A and B to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
			N * N * sizeof(float), A, 0, NULL, NULL);

	ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
			N * N * sizeof(float), B, 0, NULL, NULL);

	clFinish(command_queue);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, 
			(const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "matmul_HW2", &ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(int), &N);
	
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&a_mem_obj);
	
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&b_mem_obj);
	
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&c_mem_obj);
	

	// Execute the OpenCL kernel on the list
	size_t global_item_size[2] = {N, N};

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
			global_item_size, NULL, 0, NULL, NULL);


	clFinish(command_queue);

	// Read the memory buffer C on the device to the local variable C
	float *C = (float *)malloc(sizeof(float) * N * N);
	ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
			N * N * sizeof(float), C, 0, NULL, NULL);

	// Result check, compare match
	float *C_ref = (float *)malloc(sizeof(float) * N * N);
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			float Csub = 0.0f;
			for(k = 0; k < N; k++) {
				Csub += A[i * N + k] * B[j + N * k];
			}
			C_ref[i * N + j] = Csub;
		}
	}

	int res_check = 1;
	for(i = 0; i < N * N; i++) {
		if( (float)(fabs(C_ref[i] - C[i]) / fabs(C_ref[i]))>= 1e-6 ) {
			res_check = 0;
		}
	}

	// printf("Performance: %.9lf sec, result: %s \n", end_time - start_time, res_check ? "PASSED" : "FAILED");
	printf("Performance: result: %s \n", res_check ? "PASSED" : "FAILED");

	// Clean up
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(c_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(A);
	free(B);
	free(C);
	free(C_ref);
	return 0;
}
