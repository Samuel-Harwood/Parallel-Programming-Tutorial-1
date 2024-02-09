#include <iostream>
#include <vector>

#include "Utils.h"

#include <CL/opencl.hpp>

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		//cl::CommandQueue queue(context);

		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
		cl::Event prof_event;


		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 3 - memory allocation
		//host - input
		std::cout << "Enter a length: ";
		int length;
		std::cin >> length;
		//std::vector<float> A = { 3.4, 6.97, 2.11, 2.55, 4, 5, 6, 7, 8.56, 9 }; //C++11 allows this type of initialisation
		//std::vector<float> B = { 4.76, 2.66, 2, 5, 1, 2, 0, 1.6, 2.9, 0 };
		std::vector<int> A(length);
		std::vector<int> B(length); 


		size_t vector_elements = A.size();//number of elements
		size_t vector_size = A.size() * sizeof(int);//size in bytes

		//host - output
		std::vector<float> C(vector_elements);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//Part 4 - device operations

		//4.1 Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0]);



		//queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], NULL, &A_event);		   

		//4.2 Setup and execute the kernel (i.e. device code)
	/*	cl::Kernel kernel_multiadd = cl::Kernel(program, "multadd");
		kernel_multiadd.setArg(0, buffer_A);
		kernel_multiadd.setArg(1, buffer_B); 
		kernel_multiadd.setArg(2, buffer_C); */

		cl::Kernel kernel_addf = cl::Kernel(program, "addf");
		kernel_addf.setArg(0, buffer_A);
		kernel_addf.setArg(1, buffer_B);
		kernel_addf.setArg(2, buffer_C);

		

		queue.enqueueNDRangeKernel(kernel_addf, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		//queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0]);

		//std::cout << "A = " << A << std::endl;
		//std::cout << "B = " << B << std::endl;
		//std::cout << "C = " << C << std::endl;
		std::cout << "Kernel execution time (ns):" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Detailed Event:" << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}