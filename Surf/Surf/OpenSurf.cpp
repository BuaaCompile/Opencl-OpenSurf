/**********************************************************************
Copyright (c) 2014,BUAA COMPILER LAB;
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of the BUAA COMPILER LAB; nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

// standard utilities and system includes
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <CL/cl.h>
#include"cv.h"
#include"highgui.h"
#include<iostream>
#include<iterator>
#include<fstream>

//timer,in linux or windows

#include<time.h>
//#include<unistd.h>

using namespace std;
//#define LINUX
///////////////////////////////////////////////////////////

//Display the process of each step
#define OUTINFO 

//Profile the result of each step to a file 
//#define profile

///////////////////////////////////////////////////////////

//image & step & filter size map 

 float co[10]   = {1, 1, 1, 1, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125};
 int step[10]   = {2, 2, 2, 2, 4, 4, 8, 8, 16, 16};
 int filter[10] = {9, 15, 21, 27, 39, 51, 75, 99, 147, 195};

//OpenCL variables
cl_context context ;
cl_command_queue queue ;
cl_int 		 ciErrNum = CL_SUCCESS;
cl_program	 cpProgram;
cl_kernel 	 ckRowIntegral, ckColIntegral;
cl_kernel	 ckBuildResponseLayer, ckIsExtremum, ckGetOrientation, ckdesReady, ckcomputeDes, cknormalDes;

cl_platform_id   platform = NULL;
cl_device_id	 device = NULL;
cl_ulong	 start, end;
cl_mem		 intImage;
cl_mem 		 d_Input, d_Output;
cl_mem 		 responses, laplacian;
cl_mem		 isExtremum;
cl_mem		 mipts;
cl_mem 		 orientation;
cl_mem		 xs, ys, gauss_s2;
cl_mem		 rrx, rry;
cl_mem 		 des;
cl_mem		 mid, ndes;

///////////////////////////////////////////////////////////
double cRow, cCol, cInt, cBui,  cExt, cOut,  cMov, cRnum, cOri, cDes, cnDes, cCom;
cl_event  RowEvent, ColEvent, BuiEvent, ExtEvent,  WriteOut, WriteMipts,  OriEvent, DesEvent, nDesEvent, comEvent;
///////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////
size_t szParmDataBytes;
size_t szKernelLength;
IplImage* source;
IplImage* img;
///////////////////////////////////////////////////////////

IplImage* Integral(IplImage* img);
void integralHost(IplImage* source,IplImage* intImage);
/////////////////////////////////////////////////////////////////
#ifdef OUTINFO
	#define SHOWERR(x)  do{                                   \
			  	if(ciErrNum != CL_SUCCESS)        \
			   	{                                 \
					printf("\nerr:\t");       \
					printf(#x);               \
					printf("#:%d\n",ciErrNum);\
					printf("\n");             \
					exit(1);                  \
		        	} else {                          \
					printf("done!\t");        \
					printf(#x);		  \
					printf("\n");		  \
				}                                 \
		    	  }while(0) 
#else
	#define SHOWERR(x)
#endif

#ifdef OUTINFO
	#define SHOWINFO(x) do{                                   \
        	                printf("\nnow:\t");               \
				printf(#x);	                  \
				printf("\n");			  \
			      }while(0)
#else
	#define SHOWINFO(x)
#endif      
                                
#define CLEANUP()   do{      								\
    			    if(cpProgram)clReleaseProgram(cpProgram);			\
  			    if(queue)clReleaseCommandQueue(queue);	\
    			    if(context)clReleaseContext(context);		\
			clReleaseKernel(ckBuildResponseLayer);\
			clReleaseKernel(ckIsExtremum);	      \
			clReleaseKernel(ckGetOrientation);    \
			clReleaseKernel(ckdesReady);	      \
			clReleaseKernel(ckcomputeDes);	      \
			clReleaseKernel(cknormalDes);	      \
			clReleaseMemObject(intImage);         \
			clReleaseMemObject(responses);	      \
			clReleaseMemObject(laplacian);	      \
			clReleaseMemObject(isExtremum);	      \
			clReleaseMemObject(orientation);      \
			clReleaseMemObject(mipts);	      \
			clReleaseMemObject(xs);	     	      \
			clReleaseMemObject(ys);		      \
			clReleaseMemObject(gauss_s2);	      \
			clReleaseMemObject(des);	      \
			clReleaseMemObject(mid);	      \
			clReleaseMemObject(ndes);	      \
			clReleaseEvent(RowEvent);	      \
			clReleaseEvent(ColEvent);	      \
			clReleaseEvent(BuiEvent);	      \
			clReleaseEvent(ExtEvent);	      \
			clReleaseEvent(WriteOut);	      \
			clReleaseEvent(WriteMipts);	      \
			clReleaseEvent(OriEvent);	      \
			clReleaseEvent(DesEvent);	      \
			clReleaseEvent(nDesEvent);	      \
			clReleaseEvent(comEvent);	      \
                      }while(0)   
///////////////////////////////////////////////////////////

//Returns profiling time
double executionTime(cl_event &event)
{
    cl_ulong start, end;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    
    return (double)1.0e-6 * (end - start); // convert nanoseconds to min seconds on return
}

// Round Up Division function
size_t shrRoundUp(int group_size, int global_size) 
{
    int r = global_size % group_size;
    if(r == 0) 
    {
        return global_size;
    } else 
    {
        return global_size + group_size - r;
    }
}

// Read source of kernel to string
int convertToString(const char *filename, std::string& s){
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));
	if(f.is_open()){
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);

		str = new char[size+1];
		if(!str){
			f.close();
			return NULL;
		}
		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	printf("Error: Failed to open file %s\n", filename);
	return 1;
}

// Convert image to single channel 32F
IplImage *getGray(const IplImage *img)
{
  IplImage* gray8, * gray32;

  gray32 = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );

  if( img->nChannels == 1 )
  {
        gray8 = (IplImage *) cvClone( img );
  }else{
    gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
    cvCvtColor( img, gray8, CV_BGR2GRAY );
  }
  cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );
  
  cvReleaseImage( &gray8 );
  return gray32;
}

//Program main
int main(int argc,char** argv)
{
	//////////////////////////////////////////////////////////////////////////////
	//Init OpenCL devices
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////

	cl_uint status;

	//Discover and initialize the platform
	cl_uint numPlatforms;
	std::string platformVendor; 
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	//printf("numPlatforms:%d\n",numPlatforms);
	if (0 < numPlatforms) {
		cl_platform_id* platforms = new cl_platform_id[numPlatforms];
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		char platformName[100];
		for (unsigned i = 0; i < numPlatforms; ++i) 
		{
			status = clGetPlatformInfo(platforms[i],
				CL_PLATFORM_VENDOR,
				sizeof(platformName),
				platformName,
				NULL);

			platform = platforms[i];
			platformVendor.assign(platformName);
#ifdef AMD
		if (!strcmp(platformName, "Advanced Micro Devices, Inc.")) {
			break;
			}
#else
#ifdef NVIDIA
			if (!strcmp(platformName, "NVIDIA Corporation"))
				break;
#endif
#endif
		}
		std::cout << "Platform found : " << platformName << "\n";
		delete[] platforms;
	}
	status=clGetDeviceIDs( platform, CL_DEVICE_TYPE_CPU,1,&device,NULL);

//create a context
	context = clCreateContext( NULL,1,&device,NULL, NULL, NULL);
	//create a command queue
	queue = clCreateCommandQueue( context,device,CL_QUEUE_PROFILING_ENABLE, NULL );
	//Read kernel
	const char * filename  =  ".\\SURF.cl"; 
	std::string  sourceStr;
	status = convertToString(filename, sourceStr);
	const char * source1    = sourceStr.c_str();
	size_t sourceSize[]    = { strlen(source1) };
	//Create and compile the program
	cpProgram = clCreateProgramWithSource(context, 1, &source1,sourceSize,NULL);
	status = clBuildProgram( cpProgram, 1, &device, NULL, NULL, NULL );

#ifdef OUTINFO
	if (status!=CL_SUCCESS) {
		size_t log_size;
		clGetProgramBuildInfo(cpProgram, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char *log = (char *) malloc(log_size);
		clGetProgramBuildInfo(cpProgram, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("Error!! %s\n", log);
	}
#endif
	
/*SHOWINFO(clBuildProgram);
	ciErrNum = clBuildProgram(cpProgram,0,NULL,"-cl-mad-enable",NULL,NULL);
	if (ciErrNum != CL_SUCCESS)
    	{
                printf("err!\t");
                printf("clBuildProgram");
                printf("\n");

                //oclLogBuildInfo(cpProgram, oclGetFirstDev(context));
        	//oclLogPtx(cpProgram, oclGetFirstDev(context), "SURF.ptx");
		exit(1);

	} 
*/

	/////////////////////////  Create kernel   ///////////////////////////
#ifdef LINUX
	struct timeval tCreate1, tCreate2;
	gettimeofday(&tCreate1, NULL);
#else
	clock_t tCreate1 = clock();
#endif 

		
        SHOWINFO(clCreateKernel);
        ckRowIntegral = clCreateKernel(cpProgram,"rowIntegral",&ciErrNum);
        SHOWERR(clCreateKernel\t\t\tckRowIntegral);

        SHOWINFO(clCreateKernel);
        ckColIntegral = clCreateKernel(cpProgram,"colIntegral",&ciErrNum);
        SHOWERR(clCreateKernel\t\t\tckBuildResponseLayer);

        SHOWINFO(clCreateKernel);
        ckBuildResponseLayer = clCreateKernel(cpProgram,"BuildResponseLayer",&ciErrNum);
        SHOWERR(clCreateKernel\t\t\tckBuildResponseLayer);

        
	SHOWINFO(clCreateKernel);
        ckIsExtremum = clCreateKernel(cpProgram,"IsExtremum",&ciErrNum);
        SHOWERR(clCreateKernel\t\tckIsExtremum);

        SHOWINFO(clCreateKernel);
        ckGetOrientation= clCreateKernel(cpProgram,"GetOrientation",&ciErrNum);
        SHOWERR(clCreateKernel\t\tckGetOrientation);
        
	SHOWINFO(clCreateKernel);
        ckdesReady= clCreateKernel(cpProgram,"desReady",&ciErrNum);
        SHOWERR(clCreateKernel);
	
	SHOWINFO(clCreateKernel);
        ckcomputeDes= clCreateKernel(cpProgram,"computeDes",&ciErrNum);
        SHOWERR(clCreateKernel);

	SHOWINFO(clCreateKernel);
        cknormalDes= clCreateKernel(cpProgram,"normalDes",&ciErrNum);
        SHOWERR(clCreateKernel);

#ifdef LINUX
	gettimeofday(&tCreate2, NULL);
	double tCr = (tCreate2.tv_sec - tCreate1.tv_sec) * 1000 + (tCreate2.tv_usec - tCreate1.tv_usec) / (float)1000;

#else
	clock_t tCreate2 = clock();
	double tCr =  (double)(tCreate2 - tCreate1) / CLOCKS_PER_SEC*1000;
#endif 


        //////////////////////////////////////////////////////////////////////
	source      =  cvLoadImage(".\\frac_320_240.jpg");	
	//convert the image to single channel 32f
	img=getGray(source);

    int height = img->height;
  	int width = img->width;
	int stride  = img -> widthStep / sizeof(float);
	float *data   = (float *) img->imageData;  
	int imgSize = height * width;


	#ifdef profile
	IplImage* intImgHost;
	intImgHost  =  cvCreateImage(cvGetSize(img), IPL_DEPTH_32F,1); 	

	struct timeval tInt1, tInt2;
        gettimeofday(&tInt1, NULL);

	//Computes the integral image with cpu function
	integralHost(source,intImgHost);
	
	gettimeofday(&tInt2, NULL);
	double tInt = (tInt2.tv_sec - tInt1.tv_sec) * 1000 + (tInt2.tv_usec - tInt1.tv_usec) / (float)1000;

	float *idata   = (float *) img->imageData;  
        FILE* fp = fopen("cpu_imgdata.dat", "w");
        for(int i = 0; i < height; i++){
                for (int j = 0; j < width; j++)
                    fprintf(fp, "%f\t", *idata++);
                fprintf(fp, "\n");
        }
        fclose(fp);

	#endif
	
	//////////////////////////////////////////////////////////////////
#ifdef LINUX
	struct timeval tStart, tEnd;
	gettimeofday(&tStart, NULL);
#else
	clock_t tStart = clock();
#endif 
	/////////////////////////////////////////////////////////////////
	
		
    	unsigned int blockSize = 512; // max size of the thread blocks
    	unsigned int numBlocks =
          max(1, (int)ceil((float)imgSize / (2.f * blockSize)));
    	unsigned int sharedMemSize = 2 * blockSize;
	
	/*
	#ifdef profile
    	printf("imgSize is %d, blockSize is %d, numBlocks is %d\n", imgSize, blockSize, numBlocks);
	#endif
	*/

        d_Input = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, imgSize * sizeof(float), data, &ciErrNum);
//        oclCheckError(ciErrNum, CL_SUCCESS);
        d_Output = clCreateBuffer(context, CL_MEM_READ_WRITE, imgSize * sizeof(float), NULL, &ciErrNum);
//        oclCheckError(ciErrNum, CL_SUCCESS);
        intImage  = clCreateBuffer(context, CL_MEM_READ_WRITE, imgSize * sizeof(float), NULL, &ciErrNum);
//        oclCheckError(ciErrNum, CL_SUCCESS);

	#ifdef profile
        int N = imgSize;
        FILE* hin = fopen("in.dat", "w");

        for(int i = 0; i < height; i++){
	     	for (int j = 0; j < width; j++)
                    fprintf(hin, "%f\t", data[i*width + j]);
                fprintf(hin, "\n");
	}
        fclose(hin);
	#endif

	//RowIntegral kernel: First Compute prefix sum in rows of the source image
	size_t rowLocalWorkSize, rowGlobalWorkSize;

        rowLocalWorkSize = blockSize;
        rowGlobalWorkSize = height * blockSize;

        ciErrNum  = clSetKernelArg(ckRowIntegral, 0, sizeof(cl_mem), (void *)&d_Input);
        ciErrNum |= clSetKernelArg(ckRowIntegral, 1, sizeof(cl_mem), (void *)&d_Output);
        ciErrNum |= clSetKernelArg(ckRowIntegral, 2, sharedMemSize * sizeof(float), NULL);
        ciErrNum |= clSetKernelArg(ckRowIntegral, 3, sizeof(int), (void *)&width);

	/*
	#ifdef profile
        printf("rowGlobalWorkSize is %d, rowLocalWorkSize is %d\n", rowGlobalWorkSize,  rowLocalWorkSize);
	#endif
	*/

        ciErrNum = clEnqueueNDRangeKernel(queue, 
					  ckRowIntegral, 
					  1, NULL,
                                          &rowGlobalWorkSize, 
					  &rowLocalWorkSize, 
					  0, NULL, &RowEvent);
//        oclCheckError(ciErrNum, CL_SUCCESS);

        clWaitForEvents(1, &RowEvent);
	cl_ulong start, end;
	clGetEventProfilingInfo(RowEvent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong),&start, NULL);
	clGetEventProfilingInfo(RowEvent,CL_PROFILING_COMMAND_END, sizeof(cl_ulong),&end, NULL);

        //cRow = executionTime(RowEvent);
	cRow=(end-start)*1e-6;


	#ifdef profile
        float* h_OutputGPU = (float *)malloc(imgSize * sizeof(float));
        ciErrNum = clEnqueueReadBuffer(queue, d_Output, CL_TRUE, 0, N * sizeof(float), h_OutputGPU, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

         FILE* rowout = fopen("Rout.dat", "w");

        for(int i = 0; i < height; i++){
	     	for (int j = 0; j < width; j++)
                    fprintf(rowout, "%f\t", h_OutputGPU[i*width + j]);
                fprintf(rowout, "\n");
	}
	fclose(rowout);
	free(h_OutputGPU);
	#endif

	
	//RowIntegral kernel: Then Compute prefix sum in colunms after RowIntegral kernel 
        size_t colLocalWorkSize, colGlobalWorkSize;
		colLocalWorkSize = blockSize;
        colGlobalWorkSize = width * blockSize;

        ciErrNum  = clSetKernelArg(ckColIntegral, 0, sizeof(cl_mem), (void *)&d_Output);
        ciErrNum |= clSetKernelArg(ckColIntegral, 1, sizeof(cl_mem), (void *)&intImage);
        ciErrNum |= clSetKernelArg(ckColIntegral, 2, sharedMemSize * sizeof(float), NULL);
        ciErrNum |= clSetKernelArg(ckColIntegral, 3, sizeof(int), (void *)&height);
        ciErrNum |= clSetKernelArg(ckColIntegral, 4, sizeof(int), (void *)&width);
	
	/*
	#ifdef profile
        printf("colGlobalWorkSize is %d, colLocalWorkSize is %d\n", colGlobalWorkSize,  colLocalWorkSize);
	#endif
	*/

        ciErrNum = clEnqueueNDRangeKernel(queue, ckColIntegral, 1, NULL,
                                          &colGlobalWorkSize, &colLocalWorkSize, 0, NULL, &ColEvent);
//        oclCheckError(ciErrNum, CL_SUCCESS);

        clWaitForEvents(1, &ColEvent);
       
	clGetEventProfilingInfo(ColEvent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong),&start, NULL);
	clGetEventProfilingInfo(ColEvent,CL_PROFILING_COMMAND_END, sizeof(cl_ulong),&end, NULL);

	// cCol = executionTime(ColEvent);
	cCol=(end-start)*1e-6;

	#ifdef profile
        float* h_ImgputGPU = (float *)malloc(imgSize * sizeof(float));
        ciErrNum = clEnqueueReadBuffer(queue, intImage, CL_TRUE, 0, N * sizeof(float), h_ImgputGPU, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

        FILE* imgout = fopen("intImage.dat", "w");

        for(int i = 0; i < height; i++){
                for (int j = 0; j < width; j++)
                    fprintf(imgout, "%f\t", h_ImgputGPU[i*width + j]);
                fprintf(imgout, "\n");
        }
	fclose(imgout);
	free(h_ImgputGPU);

	#endif

	//BuildResponeselayer kernel: Calculate DoH responses for the layers
	int h =  height / 2;
	int w =  width / 2;
	int s = 2;
	
	SHOWINFO(clCreateBuffer);
	responses = clCreateBuffer(context,
					  CL_MEM_READ_WRITE,
                                          10 * w * h * sizeof(cl_float),
                                          NULL,
                                          &ciErrNum);
	SHOWERR(clCreateBuffer\t\t\tresponses);

        laplacian = clCreateBuffer(context,
                                          CL_MEM_READ_WRITE,
                                          10 * w * h * sizeof(cl_float),
                                          NULL,
                                          &ciErrNum);
	SHOWERR(clCreateBuffer\t\t\tlaplacian);
	
        size_t szBuildLocalWorkSize = 64;
	size_t szBuildGlobalWorkSize = shrRoundUp((int)szBuildLocalWorkSize, h*w);  // rounded up to the nearest multiple of the LocalWorkSize
	
	int k = 0;
        clSetKernelArg(ckBuildResponseLayer, k++, sizeof(cl_mem), (void *)&responses);
        clSetKernelArg(ckBuildResponseLayer, k++, sizeof(cl_mem), (void *)&laplacian);
        clSetKernelArg(ckBuildResponseLayer, k++, sizeof(cl_mem), (void *)&intImage);
        clSetKernelArg(ckBuildResponseLayer, k++, sizeof(int),    (void *)&h);
        clSetKernelArg(ckBuildResponseLayer, k++, sizeof(int),    (void *)&w);
        clSetKernelArg(ckBuildResponseLayer, k++, sizeof(int),    (void *)&s);
        clSetKernelArg(ckBuildResponseLayer, k++, sizeof(int),    (void *)&stride);

        ciErrNum = clEnqueueNDRangeKernel(queue,
                               ckBuildResponseLayer,
                               1, NULL,
                               &szBuildGlobalWorkSize,
                               &szBuildLocalWorkSize,
                               0, NULL, &BuiEvent 
                              );
        SHOWERR(clEnqueueNDRangeKernel\t\tckBuildResponseLayer);
	
	clWaitForEvents(1, &BuiEvent);
	clGetEventProfilingInfo(BuiEvent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong),&start, NULL);
	clGetEventProfilingInfo(BuiEvent,CL_PROFILING_COMMAND_END, sizeof(cl_ulong),&end, NULL);

        //cBui = executionTime(BuiEvent);
	cBui=(end-start)*1e-6;
		

        #ifdef profile
	float* hostResponses = (float*)malloc(10 * h * w * sizeof(float));
	float* hostLaplacian = (float*)malloc(10 * h * w * sizeof(float));

        SHOWINFO(clEnqueueReadBuffer);
        ciErrNum = clEnqueueReadBuffer(queue, responses, CL_TRUE, 0,
					10 *  h * w * sizeof(cl_float), (float *)hostResponses, 0, NULL, NULL);
        SHOWERR(clEnqueueReadBuffer\t\thostResponses <--- responses);

	ciErrNum = clEnqueueReadBuffer(queue, laplacian, CL_TRUE, 0, 
					10 *  h * w * sizeof(cl_float), (float *)hostLaplacian, 0, NULL, NULL);
	SHOWERR(clEnqueueReadBuffer\t\thostLaplacian <--- laplacian);

	
	FILE* resfp = fopen("responses.dat", "w");
	FILE* lapfp = fopen("laplacian.dat", "w");
	
	for(int dptr = 0; dptr < 10 * h * w; dptr++)
	{
		if(dptr % 10 == 0 && dptr != 0)
		{
			fprintf(resfp, "\n");
			fprintf(lapfp, "\n");
		}
		fprintf(resfp, "%f ", hostResponses[dptr]);
		fprintf(lapfp, "%f ", hostLaplacian[dptr]);
	}        
	fclose(lapfp);
	fclose(resfp);

	free(hostResponses);
	free(hostLaplacian);

	#endif

	//IsExtremum kernel: Non Maximal Suppression function
       struct _isExtremum{
                int x,y;
                float scale;
                int lap;
        };

        SHOWINFO(clCreateBuffer);
        isExtremum = clCreateBuffer(context,
                                    CL_MEM_READ_WRITE,
                                    8 * w * h * sizeof(struct _isExtremum),
                                    NULL,
                                    &ciErrNum);
        SHOWERR(clCreateBuffer);

        #define ExtBlockSize 8

        SHOWINFO(clCreateBuffer);
        cl_mem cnum = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              sizeof(int),
                              NULL,
                              &ciErrNum);
        SHOWERR(clCreateBuffer\t\t\tcnum);

        int hnum[] = {0};
        ciErrNum = clEnqueueWriteBuffer(queue, cnum, CL_FALSE, 0, sizeof(int), (int*)hnum, 0, NULL, NULL);

        int hh = shrRoundUp((int)ExtBlockSize, h);
        int ww = shrRoundUp((int)ExtBlockSize, w);
        size_t szExtGlobalWorkSize[] = {hh, ww};
        size_t szExtLocalWorkSize[] = {ExtBlockSize, ExtBlockSize};

        clSetKernelArg(ckIsExtremum, 0, sizeof(cl_mem), (void *)&responses);
        clSetKernelArg(ckIsExtremum, 1, sizeof(cl_mem), (void *)&laplacian);
        clSetKernelArg(ckIsExtremum, 2, sizeof(cl_mem), (void *)&isExtremum);
        clSetKernelArg(ckIsExtremum, 3, sizeof(int),    (void *)&h);
        clSetKernelArg(ckIsExtremum, 4, sizeof(int),    (void *)&w);
        clSetKernelArg(ckIsExtremum, 5, sizeof(int),    (void *)&cnum);

	
	SHOWINFO(clEnqueueNDRangeKernel);
	ciErrNum = clEnqueueNDRangeKernel(queue,
                               ckIsExtremum,
                               2, NULL,
                               szExtGlobalWorkSize,
                               szExtLocalWorkSize,
                               0, NULL, &ExtEvent
                              );
        SHOWERR(clEnqueueNDRangeKernel\t\tckIsExtremum);

	clWaitForEvents(1, &ExtEvent);
	cExt = executionTime(ExtEvent);	

	SHOWINFO(clEnqueueReadBuffer);
        ciErrNum = clEnqueueReadBuffer(queue,
                                         cnum,
                                         CL_TRUE, 0,
                                         sizeof(int),
                                         (int *)hnum,
                                         0, NULL, NULL);
        SHOWERR(clEnqueueReadBuffer\t\tnum <--- cnum);
	
	//cmn is the number of interest point
	int cmn = hnum[0];

	#ifdef profile
        struct _isExtremum *hostExtLocation = (struct _isExtremum *)malloc(8 * w * h * sizeof(struct _isExtremum));
        if(hostExtLocation == NULL)
        {
                printf("\nhostExtLocation fail!!!\n");
        }

        memset(hostExtLocation, 0, 8 * w * h * sizeof(struct _isExtremum));
	
        SHOWINFO(clEnqueueReadBuffer);
        ciErrNum = clEnqueueReadBuffer(queue, isExtremum, CL_TRUE, 0, 
					8 *  h * w * sizeof(struct _isExtremum), (struct _isExtremum *)hostExtLocation, 0, NULL, &WriteOut);
        SHOWERR(clEnqueueReadBuffer\t\thostExtLocation <--- extLocation);

        clWaitForEvents(1, &WriteOut);
        cOut = executionTime(WriteOut);

        FILE* pExtfpi = fopen("extre.dat", "w");
        for(int pi = 0; pi < 8 * w * h; pi++)
        {
               fprintf(pExtfpi, "%d\t%d\t%.6f\t%d\n",   hostExtLocation[pi].x,hostExtLocation[pi].y,
							hostExtLocation[pi].scale,hostExtLocation[pi].lap);
        }
	fclose(pExtfpi);

	free(hostExtLocation);

	#endif

	//GetOrientation kernel: Assign the supplied Ipoint an orientation
        #define ORI_BLOCK 42
        SHOWINFO(clCreateBuffer);
        orientation = clCreateBuffer(context,
                                     CL_MEM_READ_WRITE,
                                     cmn * sizeof(float),
                                     NULL,
                                     &ciErrNum);
        SHOWERR(clCreateBuffer);

/*
        cl_mem test = clCreateBuffer(context,
                                     CL_MEM_READ_WRITE,
                                     cmn * ORI_BLOCK * sizeof(float),
                                     NULL,
                                     &ciErrNum);
*/
        size_t szGetOriLocalWorkSize = ORI_BLOCK;
        size_t szGetOriGlobalWorkSize = cmn * ORI_BLOCK ;  // rounded up to the nearest multiple of the LocalWorkSize

        clSetKernelArg(ckGetOrientation, 0,sizeof(cl_mem), (void *)&isExtremum);
        clSetKernelArg(ckGetOrientation, 1,sizeof(cl_mem), (void *)&intImage);
        clSetKernelArg(ckGetOrientation, 2,sizeof(int),    (void *)&h);
        clSetKernelArg(ckGetOrientation, 3,sizeof(int),    (void *)&w);
        clSetKernelArg(ckGetOrientation, 4,sizeof(int),    (void *)&stride);
        clSetKernelArg(ckGetOrientation, 5,sizeof(cl_mem), (void *)&orientation);
        clSetKernelArg(ckGetOrientation, 6,sizeof(int), (void *)&cmn);
	clSetKernelArg(ckGetOrientation, 7, 109 * sizeof(float), 0);
	clSetKernelArg(ckGetOrientation, 8, 109 * sizeof(float), 0);
	clSetKernelArg(ckGetOrientation, 9, 109 * sizeof(float), 0);
	clSetKernelArg(ckGetOrientation, 10, 48 * sizeof(float), 0);
	clSetKernelArg(ckGetOrientation, 11, 48 * sizeof(float), 0);
//        clSetKernelArg(ckGetOrientation, 12,sizeof(cl_mem), (void *)&test);

        SHOWINFO(clEnqueueNDRangeKernel);
        ciErrNum = clEnqueueNDRangeKernel(queue,
                               ckGetOrientation,
                               1, NULL,
                               &szGetOriGlobalWorkSize,
                               &szGetOriLocalWorkSize,
                               0, NULL, &OriEvent
                              );
        SHOWERR(clEnqueueNDRangeKernel\t\tckGetOrientation);

        clWaitForEvents(1, &OriEvent);
        cOri = executionTime(OriEvent);

        #ifdef profile
        float* hostOri = (float*)malloc(cmn * sizeof(float));

        SHOWINFO(clEnqueueReadBuffer);
        ciErrNum = clEnqueueReadBuffer(queue, orientation, CL_TRUE, 0, cmn * sizeof(float), (float *)hostOri, 0, NULL, NULL);
        SHOWERR(clEnqueueReadBuffer\t\thostOri <--- ori test);

        FILE* orifp = fopen("ori.dat", "w");
        for(int ori = 0; ori < cmn; ori++)
        {
                if(ori % 10 == 0 && ori != 0)
                {
                        fprintf(orifp, "\n");
                }
                fprintf(orifp, "%.6f  ", hostOri[ori]);

        }
        fclose(orifp);
	free(hostOri);
  	
	/*
        float* hostTest = (float*)malloc(cmn * ORI_BLOCK * sizeof(float));
        ciErrNum = clEnqueueReadBuffer(queue, test, CL_TRUE, 0, cmn * ORI_BLOCK * sizeof(float), (float *)hostTest, 0, NULL, NULL);
        FILE* tesfp = fopen("tes.dat", "w");
        for(int ori = 0; ori < cmn * ORI_BLOCK; ori++)
        {
                if(ori % 7 == 0 && ori != 0)
                {
                        fprintf(tesfp, "\n");
                }
                fprintf(tesfp, "%.6f\t", hostTest[ori]);

        }
        fclose(tesfp);
	free(hostTest);
	*/

        #endif
	

	//ckdesReady kernel: compute xs,ys, gauss_s2 to get ready for compute descriptor 
	xs = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           16 * cmn * sizeof(int),
                           NULL,
                           &ciErrNum);
	ys = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           16 * cmn * sizeof(int),
                           NULL,
                           &ciErrNum);
	gauss_s2 = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           16 * cmn * sizeof(float),
                           NULL,
                           &ciErrNum);
 	#define BLOCK_SZ 4

        size_t szdesReadyGlobalWorkSize[] = {cmn * 4, 4};
        size_t szdesReadyLocalWorkSize[] = {BLOCK_SZ, BLOCK_SZ}; 
 
        clSetKernelArg(ckdesReady, 0, sizeof(cl_mem), (void *)&isExtremum);
        clSetKernelArg(ckdesReady, 1, sizeof(cl_mem), (void *)&orientation);
        clSetKernelArg(ckdesReady, 2, sizeof(cl_mem), (void *)&xs);
        clSetKernelArg(ckdesReady, 3, sizeof(cl_mem), (void *)&ys);
        clSetKernelArg(ckdesReady, 4, sizeof(cl_mem), (void *)&gauss_s2);
 
        SHOWINFO(clEnqueueNDRangeKernel);
        ciErrNum = clEnqueueNDRangeKernel(queue,
                               ckdesReady,
                               2, NULL,
                               szdesReadyGlobalWorkSize,
                               szdesReadyLocalWorkSize,
                               0, NULL, &DesEvent
                              );
        SHOWERR(clEnqueueNDRangeKernel\t\tckdesReady);
 
        clWaitForEvents(1, &DesEvent);
        cDes = executionTime(DesEvent);

	#ifdef profile
        int* hostXs = (int*)malloc(16 * cmn * sizeof(int));
        int* hostYs = (int*)malloc(16 * cmn * sizeof(int));
        float* hostGau = (float*)malloc(16 * cmn * sizeof(float));
        SHOWINFO(clEnqueueReadBuffer);
        ciErrNum = clEnqueueReadBuffer(queue, xs, CL_TRUE, 0, 16 * cmn * sizeof(int), (int *)hostXs, 0, NULL, NULL);
        ciErrNum = clEnqueueReadBuffer(queue, ys, CL_TRUE, 0, 16 * cmn * sizeof(int), (int *)hostYs, 0, NULL, NULL);
        ciErrNum = clEnqueueReadBuffer(queue, gauss_s2, CL_TRUE, 0, 16 * cmn * sizeof(float), (float *)hostGau, 0, NULL, NULL);
        SHOWERR(clEnqueueReadBuffer\t\thostDes <--- Xs Ys gauss_s2);
 
 
        FILE* Xsfp = fopen("xs.dat", "w");
        FILE* Ysfp = fopen("ys.dat", "w");
        FILE* Gaufp = fopen("gauss.dat", "w");
        int sc;
        for(sc = 0; sc < 16 * cmn; sc++)
        {
                if(sc % 8 == 0 && sc != 0)
                {
                        fprintf(Xsfp, "\n");
                        fprintf(Ysfp, "\n");
                        fprintf(Gaufp, "\n");
                }
                fprintf(Xsfp, "%d\t", hostXs[sc]);
                fprintf(Ysfp, "%d\t", hostYs[sc]);
                fprintf(Gaufp, "%.6f\t", hostGau[sc]);
        }
        fclose(Xsfp);
        fclose(Ysfp);
        fclose(Gaufp);

	free(hostXs);
	free(hostYs);
	free(hostGau);

        #endif

	//GetDescriptor kernel: Calculate descriptor for the interest points
        des = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           16 * 4 * cmn * sizeof(float),
                           NULL,
                           &ciErrNum);

	/*  
	rry = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           16 * 81 * cmn * sizeof(float),
                           NULL,
                           &ciErrNum);
        rrx = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           16 * 81 * cmn * sizeof(float),
                           NULL,
                           &ciErrNum);
	*/

	#define DES_BLOCK 9
        int localWorkSize = DES_BLOCK*DES_BLOCK;
        int group = cmn * 16;
        size_t szcomputeDesLocalWorkSize[] = {DES_BLOCK, DES_BLOCK};
        size_t xx = shrRoundUp((int)DES_BLOCK, cmn * 16 * 9);  // rounded up to the nearest multiple of the LocalWorkSize
        size_t yy = shrRoundUp((int)DES_BLOCK, 9);  // rounded up to the nearest multiple of the LocalWorkSize
        size_t szcomputeDesGlobalWorkSize[] = {xx,yy};
 
        clSetKernelArg(ckcomputeDes, 0, sizeof(cl_mem), (void *)&isExtremum);
        clSetKernelArg(ckcomputeDes, 1, sizeof(cl_mem), (void *)&orientation);
        clSetKernelArg(ckcomputeDes, 2, sizeof(cl_mem), (void *)&intImage);
        clSetKernelArg(ckcomputeDes, 3, sizeof(cl_mem), (void *)&xs);
        clSetKernelArg(ckcomputeDes, 4, sizeof(cl_mem), (void *)&ys);
        clSetKernelArg(ckcomputeDes, 5, sizeof(int),    (void *)&h);
        clSetKernelArg(ckcomputeDes, 6, sizeof(int),    (void *)&w);
	clSetKernelArg(ckcomputeDes, 7, sizeof(int),    (void *)&stride);
        //clSetKernelArg(ckcomputeDes, 8, sizeof(cl_mem),    (void *)&rrx);
	//clSetKernelArg(ckcomputeDes, 9, sizeof(cl_mem),    (void *)&rry);
	clSetKernelArg(ckcomputeDes, 8, localWorkSize * sizeof(float), 0);
	clSetKernelArg(ckcomputeDes, 9, localWorkSize * sizeof(float), 0);
        clSetKernelArg(ckcomputeDes, 10, sizeof(cl_mem), (void *)&gauss_s2);
        clSetKernelArg(ckcomputeDes, 11, sizeof(cl_mem), (void *)&des);
        clSetKernelArg(ckcomputeDes, 12, sizeof(int), (void *)&group);
       
	SHOWINFO(clEnqueueNDRangeKernel);
        ciErrNum = clEnqueueNDRangeKernel(queue,
                               ckcomputeDes,
                               2, NULL,
                               szcomputeDesGlobalWorkSize,
                               szcomputeDesLocalWorkSize,
                               0, NULL, &comEvent
                              );
        SHOWERR(clEnqueueNDRangeKernel\t\tckcomputeDes);
 
        clWaitForEvents(1, &comEvent);
        cCom = executionTime(comEvent);
	

	#ifdef profile
/*        
	float* hostrrx = (float*)malloc(16 * 81 * cmn * sizeof(float));
        float* hostrry = (float*)malloc(16 * 81 * cmn * sizeof(float));
        SHOWINFO(clEnqueueReadBuffer);
        ciErrNum = clEnqueueReadBuffer(queue, rrx, CL_TRUE, 0, 16 * 81 * cmn * sizeof(float), (float *)hostrrx, 0, NULL, NULL);
        ciErrNum = clEnqueueReadBuffer(queue, rry, CL_TRUE, 0, 16 * 81 * cmn * sizeof(float), (float *)hostrry, 0, NULL, NULL);
        SHOWERR(clEnqueueReadBuffer\t\thostDes <--- rrx rry);


        FILE* rrxfp = fopen("rrx.dat", "w");
        FILE* rryfp = fopen("rry.dat", "w");
        int  rrc;
        for(rrc= 0; rrc< 16 * 81 * cmn; rrc++)
        {
                if(rrc% 8 == 0 && rrc != 0)
                {
                        fprintf(rrxfp, "\n");
                        fprintf(rryfp, "\n");
                }
                fprintf(rrxfp, "%.6f\t", hostrrx[ rrc]);
                fprintf(rryfp, "%.6f\t", hostrry[ rrc]);
        }
        fclose(rrxfp);
        fclose(rryfp);

        free(hostrrx);
        free(hostrry);
*/
        float* hostDes = (float*)malloc(64 * cmn * sizeof(float));
        SHOWINFO(clEnqueueReadBuffer);
        ciErrNum = clEnqueueReadBuffer(queue, des, CL_TRUE, 0, 64 * cmn * sizeof(float), (float *)hostDes, 0, NULL, NULL);
        SHOWERR(clEnqueueReadBuffer\t\thostDes <--- des);

 
        FILE* Desfp = fopen("des.dat", "w");
        int desc;
        for(desc = 0; desc < 64 * cmn; desc++)
        {
                if(desc % 8 == 0 && desc != 0)
                {
                        fprintf(Desfp, "\n");
                }
                fprintf(Desfp, "%.6f\t", hostDes[desc]);
        }
        fclose(Desfp);  

	free(hostDes);

	#endif
	

	//cknormalDes kernel: normalize descriptors 
        ndes = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           64 * cmn * sizeof(float),
                           NULL,
                           &ciErrNum);
	/*
	mid = clCreateBuffer(context,
                           CL_MEM_READ_WRITE,
                           cmn * sizeof(float),
                           NULL,
                           &ciErrNum);
	*/
        #define NDES_BLOCK 64

        size_t szNdesLocalWorkSize =  NDES_BLOCK;
        size_t szNdesGlobalWorkSize = cmn * 64;

        clSetKernelArg(cknormalDes, 0, sizeof(cl_mem), (void *)&des);
        clSetKernelArg(cknormalDes, 1, sizeof(cl_mem), (void *)&ndes);
        clSetKernelArg(cknormalDes, 2, NDES_BLOCK * sizeof(float), 0);

        SHOWINFO(clEnqueueNDRangeKernel);
        ciErrNum = clEnqueueNDRangeKernel(queue,
                               cknormalDes,
                               1, NULL,
                               &szNdesGlobalWorkSize,
                               &szNdesLocalWorkSize,
                               0, NULL, &nDesEvent
                              );
        SHOWERR(clEnqueueNDRangeKernel\t\tcknormalDes);

       	clWaitForEvents(1, &nDesEvent);
     	cnDes = executionTime(nDesEvent);
	
	#ifdef profile
        float* hostnDes = (float*)malloc(64 * cmn * sizeof(float));
        SHOWINFO(clEnqueueReadBuffer);
        ciErrNum = clEnqueueReadBuffer(queue, ndes, CL_TRUE, 0, 
					64 * cmn * sizeof(float), (float *)hostnDes, 0, NULL, NULL);
        SHOWERR(clEnqueueReadBuffer\t\thostnDes <--- ndes);

	FILE* nDesfp = fopen("ndes.dat", "w");
        int ndesc;
        for(ndesc = 0; ndesc < 64 * cmn; ndesc++)
        {
                if(ndesc % 8 == 0 && ndesc != 0)
                {
                        fprintf(nDesfp, "\n");
                }
                fprintf(nDesfp, "%.6f\t", hostnDes[ndesc]);
        }
        fclose(nDesfp);

	free(hostnDes);
	#endif

        ///////////////////////////////////////////////////
		//Calculate the total OpenCL function execution time	
#ifdef LINUX
	gettimeofday(&tEnd, NULL);
	double totalTime = (tEnd.tv_sec - tStart.tv_sec) * 1000 + (tEnd.tv_usec - tStart.tv_usec) / (float)1000;
#else
	clock_t tEnd = clock();
	double totalTime =  (double)(tEnd-tStart) / CLOCKS_PER_SEC*1000;
#endif 


	
	SHOWINFO(cleanup);
	CLEANUP();
	ciErrNum = CL_SUCCESS;
	SHOWERR(cleanup);
	
	cvReleaseImage(&img);	
		
	//Print the number of interest points
	printf("\nOpenCL SURF found:\t %d interest points\n", cmn);	
	fprintf(stderr, "create all kernel:\t %.3f ms\n", tCr);
	#ifdef profile
	fprintf(stderr, "Cpu Integral time:\t %.3f ms\n", tInt);
	#endif
	fprintf(stderr, "Row intImage time:\t %.3f ms\n", cRow);
	fprintf(stderr, "Col intImage time:\t %.3f ms\n", cCol);
	fprintf(stderr, "BuildResponse time:\t %.3f ms\n", cBui);
	fprintf(stderr, "Found i_piont time:\t %.3f ms\n", cExt);
	#ifdef profile
	fprintf(stderr, "Write out i_point time:\t %.3f ms\n", cOut);
	#endif
	fprintf(stderr, "GetOritation time:\t %.3f ms\n", cOri);
	fprintf(stderr, "Ready for des time:\t %.3f ms\n", cDes);
	fprintf(stderr, "Compute Desc time:\t %.3f ms\n", cCom);
	fprintf(stderr, "Normal des time:\t %.3f ms\n", cnDes);

	fprintf(stderr, "=====================================\n");
	fprintf(stderr, "total time:\t\t %.3f ms\n", totalTime);


	FILE* fp_ker;
	fp_ker = fopen("GPUkerTime.dat", "w");
	
	fprintf(fp_ker, "create all kernel:\t %.3f ms\n", tCr);
	#ifdef profile
	fprintf(fp_ker, "Cpu Integral time:\t %.3f ms\n", tInt);
	#endif
	fprintf(fp_ker, "Row intImage time:\t %.3f ms\n", cRow);
	fprintf(fp_ker, "Col intImage time:\t %.3f ms\n", cCol);
	fprintf(fp_ker, "BuildResponse time:\t %.3f ms\n", cBui);
	fprintf(fp_ker, "Found i_piont time:\t\t %.3f ms\n", cExt);
	#ifdef profile
	fprintf(fp_ker, "Write out i_point time:\t %.3f ms\n", cOut);
	#endif
	fprintf(fp_ker, "GetOritation time:\t %.3f ms\n", cOri);
	fprintf(fp_ker, "Ready for des time:\t %.3f ms\n", cDes);
	fprintf(fp_ker, "Compute Desc time:\t %.3f ms\n", cCom);
	fprintf(fp_ker, "Normal des time:\t %.3f ms\n", cnDes);

	fprintf(fp_ker, "=====================================\n");
	fprintf(fp_ker, "total time:\t\t %.3f ms\n", totalTime);
	fclose(fp_ker);

	

}

//compute integral image function
void integralHost(IplImage* source, IplImage* intImageHost)
{
	IplImage *img = getGray(source);
    	int height = img->height;
  	int width = img->width;
  	int step = img->widthStep/sizeof(float);
	float *data   = (float *) img->imageData;  
	float *i_data = (float *) intImageHost->imageData;  
	float rs = 0.0f;
	for(int j=0; j<width; j++) 
	{
		rs += data[j]; 
		i_data[j] = rs;
	}
        
	for(int i=1; i<height; ++i) 
        {
        	rs = 0.0f;

            for(int j=0; j<width; ++j) 
                {
                	rs += data[i*step+j]; 
                        i_data[i*step+j] = rs + i_data[(i-1)*step+j];
                }
        }
   
        cvReleaseImage(&img);
}