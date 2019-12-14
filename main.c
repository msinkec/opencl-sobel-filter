#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>

#include "FreeImage.h"

#define MAX_SOURCE_SIZE 10000

cl_int ret;

char* readKernel(const char* file) {
    FILE *fp;
    size_t source_size;
    char* source_str;

    fp = fopen(file, "r");
    if (!fp) {
        fprintf(stderr, ":-(#\n");
        exit(1);
    }

    source_str = (char*) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);

    return source_str;
}

void printKernelBuildLog(cl_program program, cl_device_id device_id)  {
       size_t build_log_len;
    char *build_log;
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                    0, NULL, &build_log_len);
    build_log = (char *) malloc(sizeof(char) * (build_log_len + 1));
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                    build_log_len, build_log, NULL);
    printf("%s\n", build_log);
    free(build_log);
}

int main(void)
{
	unsigned char *slikaInput;
	unsigned char *slikaOutput;

	FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, "lenna.png", 0);
	FIBITMAP *imageBitmapGrey = FreeImage_ConvertToGreyscale(imageBitmap);

	int width = FreeImage_GetWidth(imageBitmapGrey);
	int height = FreeImage_GetHeight(imageBitmapGrey);

	unsigned char *imageIn = (unsigned char*)malloc(height*width * sizeof(unsigned char));
	unsigned char *imageOut = (unsigned char*)malloc(height*width * sizeof(unsigned char));

	FreeImage_ConvertToRawBits(imageIn, imageBitmapGrey, width, 8, 0xFF, 0xFF, 0xFF, TRUE);

	FreeImage_Unload(imageBitmapGrey);
	FreeImage_Unload(imageBitmap);




    ////////////////// CL ///////////////////
    
    // Define dimensions:
    // break problem down into local WGs, which must evenly fit into the global 2D space
    // that means: height must be divisible by WGx && width must be divisible with WGy
    size_t * global_work_size = (size_t*) malloc(sizeof(size_t)*2);
    size_t * local_work_size = (size_t*) malloc(sizeof(size_t)*2);
    local_work_size[0] = 16;
    local_work_size[1] = 16;

    global_work_size[0] = width;
    while (global_work_size[0] % 16 != 0) {
        global_work_size[0]++;
    }
    
    global_work_size[1] = height;
    while (global_work_size[1] % 16 != 0) {
        global_work_size[1]++;
    }
    
    printf("Global worksize: %ld x %ld\n", (long) global_work_size[0], (long) global_work_size[1]);
    printf("Local worksize: %ld x %ld\n", (long) local_work_size[0], (long) local_work_size[1]);

    // Read kernel
    char* source_str = readKernel("sobel.cl");
    
    // Get platform info for OpenCL
    cl_platform_id    platform_id[10];
    cl_uint            n_platforms;
    ret = clGetPlatformIDs(10, platform_id, &n_platforms);


    // Get GPU device info of first platform
    cl_device_id    device_ids[10];
    cl_uint            n_devices;
    ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10,
                            device_ids, &n_devices);

    // Create context, we'll use only the first GPU
    cl_int ret;
    cl_context context = clCreateContext(NULL, 1, device_ids, NULL, NULL, &ret);

    // Create OpenCL command queue for context
    cl_command_queue command_queue = clCreateCommandQueue(context, device_ids[0], 0, &ret);

    // Memory allocation on the GPU
    size_t atom_buffer_size = height * width * sizeof(unsigned char);
    cl_mem in_img_cl = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                        atom_buffer_size, NULL, &ret);
    cl_mem out_img_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        atom_buffer_size, NULL, &ret);

    // Copy image array to device
    ret = clEnqueueWriteBuffer(command_queue, in_img_cl, CL_TRUE, 0, sizeof(unsigned char) * height * width, imageIn, 0, NULL, NULL);
    
    // Prepare and build kernel program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**) &source_str,
                                                    NULL, &ret);
    ret = clBuildProgram(program, 1, device_ids, NULL, NULL, NULL);

    // Print build log of kernel
    printKernelBuildLog(program, device_ids[0]);

    // Prepare kernel object
    cl_kernel kernel = clCreateKernel(program, "sobel", &ret);

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(in_img_cl), (void*) &in_img_cl);
    ret |= clSetKernelArg(kernel, 1, sizeof(out_img_cl), (void*) &out_img_cl);
    ret |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    ret |= clSetKernelArg(kernel, 3, sizeof(int), &height);

    // Run kernel
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                                global_work_size, local_work_size, 0, NULL, NULL);

    // Wait for the kernel to finish 
    clFinish(command_queue);

    // Retrieve image data from device
    ret = clEnqueueReadBuffer(command_queue, out_img_cl, CL_TRUE, 0,
                                atom_buffer_size, imageOut, 0, NULL, NULL);
   
    // Save image to file
	FIBITMAP *imageOutBitmap = FreeImage_ConvertFromRawBits(imageOut, width, height, width, 8, 0xFF, 0xFF, 0xFF, TRUE);
	FreeImage_Save(FIF_PNG, imageOutBitmap, "result.png", 0);



    //////////////// CLEANUP /////////////////
	/*FreeImage_Unload(imageOutBitmap);
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(in_img_cl);
    ret = clReleaseMemObject(out_img_cl);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(source_str);
    free(global_work_size);
    free(local_work_size);
    free(imageIn);
    free(imageOut);*/

	return 0;
}

