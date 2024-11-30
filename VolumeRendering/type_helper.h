#pragma once
#ifndef TYPE_HELPER_H
#define TYPE_HELPER_H

#include "cuda_helper.h"

typedef unsigned int uint;
typedef unsigned char uchar;

typedef unsigned char VolumeType;
// typedef unsigned short VolumeType;

typedef float4 AccBufPixelType;
typedef uint DispBufPixelType;
typedef float4 GradType;

typedef float4 TexPixelType;
typedef float4 TransferFunElemType;

typedef struct { float3 m[4]; } float4x3;

typedef struct { float4 m[4]; } float4x4;

struct CameraData {
	bool changed;		// whether changed
	float fov;			// field of view
	float camFocal;		// focus distance
	float camWidth;		// width of camera
	float camHeight;	// height of camera
	float3 camPos;		// position
	float3 camW;		// vector w
	float3 camU;		// vector u
	float3 camV;		// vector v
};

struct TransferFunCtrl {
	float4 color;
	float pos;
};

constexpr int MAX_K = 8;

struct Path {
	int k = 0;
	float3 pos[MAX_K + 2] = {};
	float3 dir[MAX_K + 2] = {};
	float4 value = {};
};

struct Reservoir {
	uint N = 0;					// sample count
	uint itr = 0;				// iteration count
	float accErr = 0.0f;		// accumulated error
	float repDepth = 0.0f;		// representative depth
	float3 repDepthPos = {};	// epresentative point
	float3 repDir = {};			// representative direction
	float4 pixel = {};			// pixel value
	float4 wSum = {};			// w sum
	float4 pHat = {};			// p hat
	Path path;					// path sample
};

struct KernelParams {
	// Display
	uint2 resolution;					// resolution of rendered image
	float exposureScale;				// exposure scale
	DispBufPixelType* displayBuffer;	// pointer of display buffer

	// Progressive rendering state
	uint d_iteration;					// iteration count
	AccBufPixelType* accumBuffer;		// pointer of accumulated buffer
	Reservoir* reservoirs;				// reservoirs
	Reservoir* prevReservoirs;			// reservoirs in previous frame
	int maxInteractions;				// max interactions of path with volume

	// Camera
	CameraData camera;					// camera data

	// Environment
	int environmentType;				// environment light type
	float4 bgCol1;						// background color 1
	float4 bgCol2;						// background color 2
	cudaTextureObject_t envTex;			// for 2D environment texture

	// Volume definition
	int volumeType;						// volume type (file / built-in)
	float maxExtinction;				// max extinction
	float albedo;						// sigma / kappa
	float2 densityThreshold;			// density threshold
	float3 scale;						// volume scale
	float3 scaleInvHalf;				// half inverse of volume scale
	float g;							// parameter of HG phase function

	// Volume data
	cudaTextureObject_t volumeTex;		// for 3D texture
	cudaTextureObject_t gradTex;		// for gradient texture
	cudaTextureObject_t transFuncTex;	// for 1D transfer function texture

	float repDepthThreshold;			// threshold of representative depth

	// Options
	int method;							// rendering method
	float density;						// whole density
	float brightness;					// brightness

	float transferOffset;				// offset of transfer function
	float transferScale;				// scale of transfer function
	bool transferGrad;					// whether to enable gradient factor
	float transferGradPow;				// power factor of gradient

	bool enableGrad;					// whether to enable gradient to phase function
	float hgFac;						// factor of HG phase function

	bool volumeLinearFiltering;			// whether to enable linear filtering in volume
	bool transferFunLinearFiltering;	// whether to enable linear filtering in transfer function
	bool gammaCorrection;				// whether to enable gamma correction
	int toneMappingMethod;				// method of tone mapping
};

#endif // !TYPE_HELPER_H

