#pragma once
#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>

// call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char* file, const int line) {
	if (cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
#endif

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char* errorMessage, const char* file,
	const int line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		fprintf(stderr,
			"%s(%i) : getLastCudaError() CUDA error :"
			" %s : (%d) %s.\n",
			file, line, errorMessage, static_cast<int>(err),
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

inline const char* _ConvertSMVer2ArchName(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the GPU Arch name)
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), N = SM Major version,
		// and m = SM minor version
		const char* name;
	} sSMtoArchName;

	sSMtoArchName nGpuArchNameSM[] = {
		{0x30, "Kepler"},
		{0x32, "Kepler"},
		{0x35, "Kepler"},
		{0x37, "Kepler"},
		{0x50, "Maxwell"},
		{0x52, "Maxwell"},
		{0x53, "Maxwell"},
		{0x60, "Pascal"},
		{0x61, "Pascal"},
		{0x62, "Pascal"},
		{0x70, "Volta"},
		{0x72, "Xavier"},
		{0x75, "Turing"},
		{0x80, "Ampere"},
		{0x86, "Ampere"},
		{0x87, "Ampere"},
		{0x89, "Ada Lovelace"},
		{0x90, "Hopper"},
		{-1, "Graphics Device"} };

	int index = 0;

	while (nGpuArchNameSM[index].SM != -1) {
		if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchNameSM[index].name;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
		"MapSMtoArchName for SM %d.%d is undefined."
		"  Default to use %s\n",
		major, minor, nGpuArchNameSM[index - 1].name);
	return nGpuArchNameSM[index - 1].name;
}

// Initialization code to find the best CUDA Device
inline int findCudaDevice() {
	int devID = 0;
	if (cudaSetDevice(devID) != cudaSuccess) {
		printf("No GPU devices. \n");
		return -1;
	}
	int major = 0, minor = 0;
	checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
	checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
	printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, _ConvertSMVer2ArchName(major, minor), major, minor);

	return devID;
}

#endif // !CUDA_HELPER_H