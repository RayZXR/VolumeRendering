#include "math_helper.h"
#include "curand_kernel.h"
#include <cstring>

#define RND (curand_uniform(randState))

constexpr float Inv4PI = 0.25f / PI;

// Phase function
class PhaseFunction {
public:
	__device__ virtual float p(const float3& wo, const float3& wi) const = 0;
	__device__ virtual float sampleDir(const float3& wo, float3* wi, const float* sample) const = 0;
};

// Transform of coordinate system
__device__ inline void coordinateSystem(const float3& v1, float3* v2, float3* v3) {
	if (fabsf(v1.x) > fabsf(v1.y)) {
		*v2 = make_float3(-v1.z, 0, v1.x) / sqrtf(v1.x * v1.x + v1.z * v1.z);
	} else {
		*v2 = make_float3(0, v1.z, -v1.y) / sqrtf(v1.y * v1.y + v1.z * v1.z);
	}
	*v3 = cross(v1, *v2);
}

// Get spherical direction with angles
__device__ inline float3 sphericalDirection(float sinTheta, float cosTheta, float phi,
	const float3& x, const float3& y, const float3& z) {
	return sinTheta * cosf(phi) * x + sinTheta * sinf(phi) * y + cosTheta * z;
}

// PDF of Henyey-Greenstein phase function
__device__ inline float phaseHG(float cosTheta, float g) {
	float denom = 1 + g * g + 2 * g * cosTheta;
	return Inv4PI * (1 - g * g) / (denom * sqrtf(denom));
}

// Henyey-Greenstein phase function
class HenyeyGreenstein : public PhaseFunction {
public:
	__device__ HenyeyGreenstein(float g) : g(g) {}

	__device__ float p(const float3& wo, const float3& wi) const {
		return phaseHG(dot(wo, wi), g);
	}

	__device__ float sampleDir(const float3& wo, float3* wi, const float* sample) const {
		// Compute cos theta for Henyey¨CGreenstein value
		float cosTheta;
		if (fabsf(g) < 1e-3)
			cosTheta = 1 - 2 * sample[0];
		else {
			float sqrTerm = (1 - g * g) / (1 - g + 2 * g * sample[0]);
			cosTheta = (1 + g * g - sqrTerm * sqrTerm) / (2 * g);
		}
		// Compute direction wi for Henyey¨CGreenstein value
		float sinTheta = sqrtf(fmaxf(0.0f, 1 - cosTheta * cosTheta));
		float phi = 2 * PI * sample[1];
		float3 v1, v2;
		coordinateSystem(wo, &v1, &v2);
		*wi = sphericalDirection(sinTheta, cosTheta, phi, v1, v2, wo);
		// PDF
		return phaseHG(cosTheta, g);
	}

private:
	const float g;
};

// Ray
struct Ray {
	float3 o;  // origin
	float3 d;  // direction
};


/* -------- Data -------- */

AccBufPixelType* d_accumBuffer = NULL;	// Accumulated buffer
Reservoir* d_reservoirs;				// Reservoirs
Reservoir* d_prevReservoirs;			// Previous reservoirs
uint2 d_accBufSize;						// Size of accumulated buffer

float* d_volumeArrayOrg = NULL;			// Original volume array
GradType* d_gradArrayOrg = NULL;		// Original gradient array
float* d_gradValueArrayOrg = NULL;		// Original gradient value array

cudaArray* d_volumeArray = NULL;		// Volume array
cudaArray* d_gradArray = NULL;			// gradient array
cudaArray* d_envTexArray = NULL;		// Environment texture array
cudaArray* d_transFuncArray = NULL;		// Transfer function array

curandState* d_randStates = NULL;		// Random states

uint d_iteration = 0;					// Iteration of rendering

cudaTextureObject_t d_volumeTex;		// For 3D texture
cudaTextureObject_t d_gradTex;			// For gradient
cudaTextureObject_t d_transFuncTex;		// For 1D transfer function texture
cudaTextureObject_t d_envTex;			// For 2D environment texture

__device__ KernelParams d_params;		// Parameters in rendering

__device__ AccBufPixelType d_adptVal;	// Adapted value of rednering image

__device__ CameraData d_prevCamera;		// Previous camera data

__device__ float d_maxVoxel;			// Max voxel in volume
__device__ float d_invMaxVoxel;			// Inverse value of max voxel
__device__ float d_maxGrad;				// Max gradient in volume
__device__ float d_invMaxGrad;			// Inverse value of max gradient


/* -------- Random Engine -------- */

// Initialize the random state
__global__ void d_randomStateInit(int width, int height, unsigned long long seed, curandState* d_randStates) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i >= width) || (j >= height)) return;
	int pixelIndex = j * width + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(seed, pixelIndex, 0, &d_randStates[pixelIndex]);
}

// Initialize the random state (external calling)
extern "C" void randomStateInit(dim3 gridSize, dim3 blockSize, int width, int height, unsigned long long seed) {
	unsigned int statesSize = width * height * sizeof(curandState);
	checkCudaErrors(cudaMalloc((void**)&d_randStates, statesSize));
	d_randomStateInit<<<gridSize, blockSize>>>(width, height, seed, d_randStates);
}


/* -------- Public Functions -------- */

// Print a vector
inline __device__ void printVec(const char* s, float4 val) {
	printf("%s: (%f, %f, %f, %f)\n", s, val.x, val.y, val.z, val.w);
}

// Get the initial ray with pixel
inline __device__ Ray getRay(float u, float v) {
	Ray eyeRay;
	eyeRay.o = d_params.camera.camPos;
	eyeRay.d = normalize(d_params.camera.camFocal * d_params.camera.camW + (u - 0.5f) * d_params.camera.camU + (v - 0.5f) * d_params.camera.camV);
	return eyeRay;
}

// Get the pixel in screen of the point in 3d space with camera
inline __device__ float2 getScreenPos(float3 pos, CameraData camera) {
	float3 dir = pos - camera.camPos;
	float t = dot(camera.camW, camera.camFocal * camera.camW) / dot(camera.camW, dir);
	float3 offsetVec = t * dir - camera.camFocal * camera.camW;
	float u = dot(offsetVec, camera.camU) / (camera.camWidth * camera.camWidth) + 0.5f;
	float v = dot(offsetVec, camera.camV) / (camera.camHeight * camera.camHeight) + 0.5f;
	return make_float2(u * d_params.resolution.x, v * d_params.resolution.y);
}

// Determine whether the ray intersects the box
inline __device__ bool intersectBox(Ray r, float3 boxmin, float3 boxmax, float* tnear, float* tfar) {
	// compute intersection of ray with all six bbox planes
	float3 invR = make_float3(1.0f, 1.0f, 1.0f) / r.d;
	float3 tbot = invR * (boxmin - r.o);
	float3 ttop = invR * (boxmax - r.o);

	// re-order intersections to find smallest and largest on each axis
	float3 tmin = fminf(ttop, tbot);
	float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// Determine whether the point is in the box
inline __device__ bool inBox(float3 point, float3 boxmin, float3 boxmax) {
	return point.x >= boxmin.x && point.x <= boxmax.x && point.y >= boxmin.y && point.y <= boxmax.y && point.z >= boxmin.z && point.z <= boxmax.z;
}

// Sample gradient value
inline __device__ float3 sampleGrad(float3 p, float* value) {
	GradType sample = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (d_params.volumeType == 0) {
		sample = tex3D<float4>(d_params.gradTex, p.x * d_params.scaleInvHalf.x + 0.5f,
			p.y * d_params.scaleInvHalf.y + 0.5f, p.z * d_params.scaleInvHalf.z + 0.5f);
	}

	float3 grad = make_float3(sample.x, sample.y, sample.z);
	*value = norm(grad);
	return *value > 0.0f ? grad / *value : grad;
}

// Sample voxel value
inline __device__ float sampleVolume(float3 p) {
	float sample = 0.0f;

	if (d_params.volumeType == 0) {
		sample = tex3D<float>(d_params.volumeTex, p.x * d_params.scaleInvHalf.x + 0.5f,
			p.y * d_params.scaleInvHalf.y + 0.5f, p.z * d_params.scaleInvHalf.z + 0.5f) * d_params.density;
	}
	else if (d_params.volumeType == 1) {
		sample = sqrtf(fabsf(p.x * p.y * p.z)) * 0.8f + 0.1f;
		float3 pos = p;
		const uint steps = 3;
		for (uint i = 0; i < steps; ++i) {
			pos *= 3.0f;
			int s = ((int)pos.x & 1) + ((int)pos.y & 1) + ((int)pos.z & 1);
			if (s >= 2) {
				sample = 0.0f;
				break;
			}
		}
		sample *= d_params.density;
	}
	else if (d_params.volumeType == 2) {
		float r = 0.5f * (0.5f - fabsf(p.y));
		float a = PI * 8.0 * p.y;
		float dx = (cosf(a) * r - p.x) * 2.0f;
		float dy = (sinf(a) * r - p.z) * 2.0f;
		sample = powf(fmaxf(1.0f - dx * dx - dy * dy, 0.0f), 8.0f) * d_params.density;
	}

	if (sample < d_params.densityThreshold.x || sample > d_params.densityThreshold.y)
		sample = 0.0f;

	return sample;
}


/* -------- Ray-Marching -------- */

// Render with ray marching
__global__ void d_renderRM(curandState* d_randStates) {
	const int maxSteps = 500;
	const float tstep = 0.01f;
	const float opacityThreshold = 0.95f;
	const float3 boxMin = make_float3(-1.0f * d_params.scale.x, -1.0f * d_params.scale.y, -1.0f * d_params.scale.z);
	const float3 boxMax = make_float3(1.0f * d_params.scale.x, 1.0f * d_params.scale.y, 1.0f * d_params.scale.z);

	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint pixelIndex = y * d_params.resolution.x + x;
	curandState* randState = d_randStates + pixelIndex;

	if ((x >= d_params.resolution.x) || (y >= d_params.resolution.y)) return;

	float u = (x + RND) / (float)d_params.resolution.x;
	float v = (y + RND) / (float)d_params.resolution.y;

	// calculate eye ray in world space
	Ray eyeRay = getRay(u, v);

	// find intersection with box
	float tmin, tmax;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tmin, &tmax);

	if (!hit)
		return;

	if (tmin < 0.0f)
		tmin = 0.0f;  // clamp to near plane

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float t = tmin;
	float3 pos = eyeRay.o + eyeRay.d * tmin;
	float3 step = eyeRay.d * tstep;

	for (int i = 0; i < maxSteps; i++) {
		// read from 3D texture
		// remap position to [0, 1] coordinates
		float sample = sampleVolume(pos);

		// lookup in transfer function texture
		float4 col = tex1D<float4>(d_params.transFuncTex, (sample - d_params.transferOffset) * d_params.transferScale) * d_params.brightness;

		// "under" operator for back-to-front blending
		// sum = lerp(sum, col, col.w);

		col.w *= clamp(d_params.density, 0.0f, 1.0f);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending
		sum = sum + col * (1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold)
			break;

		t += tstep;

		if (t > tmax)
			break;

		pos += step;
	}

	d_params.accumBuffer[pixelIndex] = sum;
}


/* -------- Path tracing -------- */

// Uniform value on the sphere
inline __device__ float3 sphereSample(curandState* randState) {
	float theta = 2.0f * PI * RND;
	float cos_phi = 1.0f - 2.0f * RND;
	float sin_phi = sqrtf(1.0f - cos_phi * cos_phi);
	return make_float3(sin_phi * cosf(theta), sin_phi * sinf(theta), cos_phi);
}

// Henyey-Greenstein phase function
inline __device__ float3 henyeyGreenstein(float3 dir, float density, float3 normGrad, float gradVal, float* pdf, curandState* randState) {
	float3 dirOut = dir;
	if (d_params.enableGrad) {
		float3 reflectDir = dir;
		if (gradVal > 0.0f) {
			reflectDir = reflect(dir, normGrad);
		}
		float g = d_params.hgFac * gradVal;
		g = clamp(g, EPSILON - 1.0f, 1.0f - EPSILON);
		HenyeyGreenstein hg(g);
		float rnd2[] = { RND, RND };
		*pdf = hg.sampleDir(reflectDir, &dirOut, rnd2);
	} else {
		HenyeyGreenstein hg(d_params.g);
		float rnd2[] = { RND, RND };
		*pdf = hg.sampleDir(dir, &dirOut, rnd2);
	}
	return dirOut;
}

// Sample scattering direction
inline __device__ float3 sampleDirection(float3 dir, float density, float3 normGrad, float gradVal, float* pdf, curandState* randState) {
	// return sphereSample(randState);
	return henyeyGreenstein(dir, density, normGrad, gradVal, pdf, randState);
}

// Absorption coefficient
inline __device__ float4 sigma_a(float val, float grad) {
	return (d_params.transferGrad ? pow(grad * d_invMaxGrad, d_params.transferGradPow) : 1.0f)
		* d_invMaxVoxel * make_float4(val, val, val, val) 
		+ make_float4(EPSILON, EPSILON, EPSILON, EPSILON);
}

// Scattering coefficient
inline __device__ float4 sigma_s(float val, float grad) {
	return (d_params.transferGrad ? pow((1.0f - grad * d_invMaxGrad), d_params.transferGradPow) : 1.0f) 
		* tex1D<float4>(d_params.transFuncTex, (val * d_invMaxVoxel + d_params.transferOffset) * d_params.transferScale)
		+ make_float4(EPSILON, EPSILON, EPSILON, EPSILON);
}

// Extinction coefficient
inline __device__ float4 sigma_t(float val, float grad) {
	return (sigma_a(val, grad) + sigma_s(val, grad));
}

// Get environment light
inline __device__ float4 getEnvL(float3 dir) {
	float4 res;
	if (d_params.environmentType == 0) {
		float t = 0.5f * (dir.y + 1.0f);
		res = ((1.0f - t) * d_params.bgCol1 + t * d_params.bgCol2);
	} else {
		float x = atan2f(dir.z, dir.x) * (0.5f * inv_PI) + 0.5f;
		float y = acosf(fmaxf(fminf(dir.y, 1.0f), -1.0f)) * inv_PI;
		res = tex2D<TexPixelType>(d_params.envTex, x, y);
	}
	return res;
}

// Estimate transmittance with ray marching
__device__ float4 estimateTransmittance(Ray& ray, float tMax) {
	float4 T = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	float delta_t = 0.01f, t = 0.0f;
	float3 pos = ray.o, delta_v = delta_t * ray.d;
	float gradVal;
	for (t = 0.0f; t < tMax; t += delta_t) {
		pos += delta_v;
		sampleGrad(pos, &gradVal);
		T += delta_t * sigma_t(sampleVolume(pos), gradVal) * d_invMaxVoxel / d_params.maxExtinction;
	}
	if (t < tMax) {
		pos = ray.o + tMax * ray.d;
		sampleGrad(pos, &gradVal);
		T += (tMax - t) * sigma_t(sampleVolume(pos), gradVal) * d_invMaxVoxel / d_params.maxExtinction;
	}
	return make_float4(exp(-T.x), exp(-T.y), exp(-T.z), exp(-T.w));
}

// Ratio tracking (estimate transmittance)
__device__ float4 ratioTracking(float3 o, float3 d, float s, curandState* randState) {
	float t = 0.0f;
	float4 T = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

	float invMaxDensity = d_invMaxVoxel;
	float scaleFac = 1.0f / d_params.maxExtinction;
	do {
		t -= logf(1.0f - RND) * invMaxDensity * scaleFac;
		if (t >= s) {
			break;
		}
		float3 pos = o + t * d;
		float value = sampleVolume(pos);
		float gradVal;
		sampleGrad(pos, &gradVal);
		T *= make_float4(1.0f, 1.0f, 1.0f, 1.0f) - sigma_t(value, gradVal) * invMaxDensity * scaleFac;
	} while (true);
	return T;
}

// Delta tracking (sample scattering points)
__device__ bool deltaTracking(Ray& ray, float3 boxMin, float3 boxMax, float& value, float4& T, float& tMax, curandState* randState) {
	float tMin;
	float t = 0.0f;
	float3 pos;
	T = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

	if (intersectBox(ray, boxMin, boxMax, &tMin, &tMax)) {
		float tMin0 = fmax(tMin, 0.0f);
		ray.o += tMin0 * ray.d;
		tMax -= tMin0;

		float invMaxDensity = d_invMaxVoxel;
		float scaleFac = 1.0f / d_params.maxExtinction;
		do {
			t -= logf(1.0f - RND) * invMaxDensity * scaleFac;
			if (t > tMax) {
				t = tMax;
				pos = ray.o + t * ray.d;
				break;
			}
			pos = ray.o + t * ray.d;
			value = sampleVolume(pos);
			float gradVal;
			sampleGrad(pos, &gradVal);
			T *= make_float4(1.0f, 1.0f, 1.0f, 1.0f) - sigma_t(value, gradVal) * invMaxDensity * scaleFac;
		} while (RND > value * invMaxDensity);

		ray.o = pos;
		tMax -= t;
		return tMax > 0.0f;
	} else {
		tMax = -1.0f;
		return false;
	}
}

// Trace one path in volume
__device__ float4 traceVolume(Ray ray, float3 boxMin, float3 boxMax, bool& hasIntr, curandState* randState) {
	float value, tMin, tMax;
	float4 T = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	float4 L = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

	hasIntr = false;
	uint num_interactions = 0;
	while (deltaTracking(ray, boxMin, boxMax, value, T, tMax, randState)) {
		hasIntr = true;
		float gradVal;
		float3 gradNorm = sampleGrad(ray.o, &gradVal);
		float gradValNorm = gradVal * d_invMaxGrad;
		float rho;
		ray.d = sampleDirection(ray.d, value, gradNorm, gradValNorm, &rho, randState);

		float4 volumeExt = d_params.brightness * rho * (T * sigma_s(value, gradVal));
		L *= volumeExt;
		T = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

		if (++num_interactions >= MAX_K) {
			intersectBox(ray, boxMin, boxMax, &tMin, &tMax);
			break;
		}
	}

	if (tMax > 0.0f) {
		L *= ratioTracking(ray.o, ray.d, tMax, randState);
	}

	L *= getEnvL(ray.d);

	return L;
}

// Render with path tracing
__global__ void d_renderPT(curandState* d_randStates) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= d_params.resolution.x) || (y >= d_params.resolution.y)) return;

	const float3 boxMin = make_float3(-1.0f * d_params.scale.x, -1.0f * d_params.scale.y, -1.0f * d_params.scale.z);
	const float3 boxMax = make_float3(1.0f * d_params.scale.x, 1.0f * d_params.scale.y, 1.0f * d_params.scale.z);

	uint pixelIndex = y * d_params.resolution.x + x;
	curandState* randState = d_randStates + pixelIndex;

	float u = (x + RND) / (float)d_params.resolution.x;
	float v = (y + RND) / (float)d_params.resolution.y;

	// calculate initial ray in world space
	Ray ray = getRay(u, v);

	// Result
	float4 value;
	bool hasIntr = false;
	value = traceVolume(ray, boxMin, boxMax, hasIntr, randState);
	value.w = 1.0f;

	// For test
	//if (hasIntr) {
	//	float fac = 1.0f / N_min;
	//	value = fac * make_float4(1.0f, 0.0f, 0.0f, 1.0f) + (1.0f - fac) * make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	//}

	// Set buffer
	d_params.accumBuffer[pixelIndex] = value;
}


/* -------- Volumetric ReSTIR -------- */

// Representative depth
__device__ float3 representativeDepth(Ray ray, float3 boxMin, float3 boxMax, float& depth) {
	float T = 0.0f;
	float delta_t = 0.01f, t = 0.0f, tMin, tMax;
	depth = 0.0f;
	if (intersectBox(ray, boxMin, boxMax, &tMin, &tMax)) {
		depth = t = tMin;
		float3 pos = ray.o + t * ray.d, maxPos = pos;
		float3 delta_v = delta_t * ray.d;
		float maxT = 0.0f;
		while (t < tMax) {
			float Ti = sampleVolume(pos) * d_invMaxVoxel;
			if (Ti > maxT) {
				maxT = Ti;
				maxPos = pos;
				depth = t;
			}
			T += Ti;
			if (T >= d_params.repDepthThreshold) {
				depth = t;
				return pos;
			}
			t += delta_t;
			pos += delta_v;
		}
		return maxPos;
	}
	return make_float3(INFINITY, INFINITY, INFINITY);
}

// Render with volumetric ReSTIR
__global__ void d_renderReSTIR(curandState* d_randStates) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= d_params.resolution.x) || (y >= d_params.resolution.y)) return;

	uint pixelIndex = y * d_params.resolution.x + x;
	curandState* randState = d_randStates + pixelIndex;

	float u = (x + RND) / (float)d_params.resolution.x;
	float v = (y + RND) / (float)d_params.resolution.y;

	const float3 boxMin = make_float3(-1.0f * d_params.scale.x, -1.0f * d_params.scale.y, -1.0f * d_params.scale.z);
	const float3 boxMax = make_float3(1.0f * d_params.scale.x, 1.0f * d_params.scale.y, 1.0f * d_params.scale.z);

	// Calculate initial ray in world space
	Ray ray = getRay(u, v);

	Reservoir& reservoir = d_params.reservoirs[pixelIndex];

	reservoir.repDir = ray.d;
	reservoir.repDepthPos = representativeDepth(ray, boxMin, boxMax, reservoir.repDepth);

	bool hasIntr = false;

	// Accumulate
	if (d_params.d_iteration == 0 || d_params.camera.changed == false) {
		// No camera moving

		int sampleN = 0;
		float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if (reservoir.N < N_max) {
			sampleN = max(N_min - (int)reservoir.N, min(N_max - (int)reservoir.N, N_norm));
			for (int i = 0; i < sampleN; ++i) {
				value += traceVolume(ray, boxMin, boxMax, hasIntr, randState);
			}
			value /= (float)sampleN;
			value.w = 1.0f;

			double alpha = (double)reservoir.N / (reservoir.N + sampleN);
			reservoir.pixel = (1.0 - alpha) * value + alpha * reservoir.pixel;
			reservoir.N += sampleN;
			// ++reservoir.itr;
		}

		// For test
		//value = traceVolume(ray, boxMin, boxMax, hasIntr, randState);
		//if (hasIntr) {
		//	float fac = (float)sampleN / N_min;
		//	reservoir.pixel = fac * make_float4(1.0f, 0.0f, 0.0f, 1.0f) + (1.0f - fac) * make_float4(1.0f, 1.0f, 1.0f, 1.0f);
		//}

	} else {
		// Camera moved

		int reuse = 0;		// Reuse flag

		if (isfinite(reservoir.repDepthPos.x)) {
			float2 relPixel = getScreenPos(reservoir.repDepthPos, d_prevCamera);	// Get relative pixel in previous frame

			if ((int)relPixel.x >= 0 && (int)relPixel.x < d_params.resolution.x && (int)relPixel.y >= 0 && (int)relPixel.y < d_params.resolution.y) {
				uint relPixelIndex = (uint)relPixel.y * d_params.resolution.x + (uint)relPixel.x;
				Reservoir& relReservoir = d_params.prevReservoirs[relPixelIndex];
				float err = fabs(relReservoir.repDepth - reservoir.repDepth) + 0.08f * (1 - dot(relReservoir.repDir, reservoir.repDir));
				reservoir.accErr = relReservoir.accErr + err;

				if (err > 0.038f) {
					// Accumulated error reaches threshold

					reuse = 0;

				} else if (reservoir.accErr > 0.076f) {
					// Single error reaches threshold

					reuse = 2;

					relReservoir.N /= 2;
					reservoir.accErr = relReservoir.accErr / 2.0f + err;
				} else {
					// Totally reuse

					reuse = 1;
				}

				if (reuse) {
					int sampleN = 0;
					float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
					if (relReservoir.N < N_max) {
						sampleN = max(N_min - (int)relReservoir.N, min(N_max - (int)relReservoir.N, N_norm));
						for (int i = 0; i < sampleN; ++i) {
							value += traceVolume(ray, boxMin, boxMax, hasIntr, randState);
						}
						value /= (float)sampleN;
						value.w = 1.0f;

						double alpha = (double)relReservoir.N / (relReservoir.N + sampleN);
						reservoir.pixel = (1.0 - alpha) * value + alpha * relReservoir.pixel;
						reservoir.N = relReservoir.N + sampleN;
					}

					// For test
					//value = traceVolume(ray, boxMin, boxMax, hasIntr, randState);
					//if (hasIntr) {
					//	float fac = (float)sampleN / N_min;
					//	reservoir.pixel = fac * make_float4(1.0f, 0.0f, 0.0f, 1.0f) + (1.0f - fac) * make_float4(1.0f, 1.0f, 1.0f, 1.0f);
					//}

					double alphaErr = (double)(relReservoir.N) / (relReservoir.N + max(sampleN, 1));
					reservoir.repDepthPos = (1.0 - alphaErr) * reservoir.repDepthPos + alphaErr * relReservoir.repDepthPos;
					reservoir.repDepth = (1.0 - alphaErr) * reservoir.repDepth + alphaErr * relReservoir.repDepth;
					reservoir.repDir = (1.0 - alphaErr) * reservoir.repDir + alphaErr * relReservoir.repDir;
				}
			}
		}

		if (reuse == 0) {
			// Don't reuse

			uint sampleN = N_min;
			float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			for (uint i = 0; i < sampleN; ++i) {
				value += traceVolume(ray, boxMin, boxMax, hasIntr, randState);
			}
			value /= (float)sampleN;
			value.w = 1.0f;

			// For test
			//value = traceVolume(ray, boxMin, boxMax, hasIntr, randState);
			//if (hasIntr) {
			//	float fac = (float)sampleN / N_min;
			//	value = fac * make_float4(1.0f, 0.0f, 0.0f, 1.0f) + (1.0f - fac) * make_float4(1.0f, 1.0f, 1.0f, 1.0f);
			//}

			reservoir.pixel = value;
			reservoir.N = sampleN;
		}

		// For test
		//if (hasIntr) {
		//	reservoir.pixel = (reuse == 1) ? make_float4(0.0f, 1.0f, 0.0f, 1.0f) : ((reuse == 2) ? make_float4(0.0f, 0.5f, 1.0f, 1.0f) : make_float4(1.0f, 0.0f, 0.0f, 1.0f));
		//}
	}

	d_params.accumBuffer[pixelIndex] = reservoir.pixel;
}

// Save reservoirs to previous reservoirs
__global__ void d_saveReservoirs() {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= d_params.resolution.x) || (y >= d_params.resolution.y)) return;

	uint pixelIndex = y * d_params.resolution.x + x;

	d_params.prevReservoirs[pixelIndex] = d_params.reservoirs[pixelIndex];
}


/* -------- Display processing -------- */

// Reduction of pixels
__global__ void d_maxPixelReduc2D(AccBufPixelType* d_array, AccBufPixelType* d_result, uint2 imageSize) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= imageSize.x) return;
	uint index = x;

	uint step = imageSize.x;
	uint cap = imageSize.x * imageSize.y;

	AccBufPixelType maxVal = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	for (uint i = index; i < cap; i += step) {
		maxVal = fmaxf(d_array[i], maxVal);
	}

	d_result[index] = maxVal;
}

__global__ void d_maxPixelReduc1D(AccBufPixelType* d_array, AccBufPixelType* d_result, uint2 imageSize) {
	uint cap = imageSize.x;

	AccBufPixelType maxVal = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	for (uint i = 0; i < cap; ++i) {
		maxVal = fmaxf(d_array[i], maxVal);
	}

	*d_result = maxVal;
}

__global__ void d_maxPixelReducRes(AccBufPixelType* result) {
	d_adptVal = *result;
}

void imageReduction() {
	AccBufPixelType* d_buffer1D = NULL, * d_bufferRes = NULL;
	checkCudaErrors(cudaMalloc(&d_buffer1D, sizeof(AccBufPixelType) * d_accBufSize.x));
	checkCudaErrors(cudaMalloc(&d_bufferRes, sizeof(AccBufPixelType)));

	dim3 reduc2DBlockSize(8);
	dim3 reduc2DGridSize(iDivUp((int)d_accBufSize.x, reduc2DBlockSize.x));

	d_maxPixelReduc2D<<<reduc2DGridSize, reduc2DBlockSize>>>(d_accumBuffer, d_buffer1D, d_accBufSize);
	d_maxPixelReduc1D<<<1, 1>>>(d_buffer1D, d_bufferRes, d_accBufSize);
	d_maxPixelReducRes<<<1, 1>>>(d_bufferRes);

	if (d_buffer1D) {
		checkCudaErrors(cudaFree(d_buffer1D));
	}
	if (d_bufferRes) {
		checkCudaErrors(cudaFree(d_bufferRes));
	}
}

// Convert pixels from float to uint8 (Tone mapping)
__global__ void d_convDisp() {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= d_params.resolution.x) || (y >= d_params.resolution.y)) return;
	unsigned int pixelIndex = y * d_params.resolution.x + x;
	d_params.displayBuffer[pixelIndex] = rgbaFloatToInt(d_params.accumBuffer[pixelIndex] * d_params.exposureScale,
		make_float4(1.0f, 1.0f, 1.0f, 1.0f), d_params.toneMappingMethod, d_params.gammaCorrection);
}


/* -------- Data processing -------- */

// Create Volume Texture
void createVolTex(void* hVolume, cudaExtent volumeSize) {
	// create volume texture
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
	checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(hVolume, volumeSize.width * sizeof(VolumeType), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// texture description
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_volumeArray;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = true;  // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear;  // linear interpolation

	texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeNormalizedFloat;

	checkCudaErrors(cudaCreateTextureObject(&d_volumeTex, &texRes, &texDescr, NULL));
}

// Create Volume data in float
__global__ void d_calculateNormVolume(float* d_volumeArrayOrg, cudaTextureObject_t d_volumeTex, cudaExtent volumeSize) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth) return;
	uint voxelIndex = z * volumeSize.width * volumeSize.height + y * volumeSize.width + x;
	float xf = (float)x / volumeSize.width, yf = (float)y / volumeSize.height, zf = (float)z / volumeSize.depth;
	d_volumeArrayOrg[voxelIndex] = tex3D<float>(d_volumeTex, xf, yf, zf);
}

// Get voxel value in float
__device__ float getVoxel(float* vol, uint x, uint y, uint z, cudaExtent volumeSize) {
	return vol[clamp(z, 0, volumeSize.depth - 1) * volumeSize.width * volumeSize.height + clamp(y, 0, volumeSize.height - 1) * volumeSize.width + clamp(x, 0, volumeSize.width - 1)];
}

// Calculate gradient of single voxel
__device__ float calGrad(float* vol, uint x, uint y, uint z, cudaExtent volumeSize, float S[3][3][3]) {
	float grad = 0.0f;
	for (int k = -1; k <= 1; ++k) {
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				grad += getVoxel(vol, x + j, y + i, z + k, volumeSize) * S[k + 1][i + 1][j + 1];
			}
		}
	}
	return grad;
}

// Calculate gradient of volume
__global__ void d_calculateVolumeGrad(GradType* d_gradArrayOrg, float* d_gradValueArrayOrg, float* d_volumeArrayOrg, cudaExtent volumeSize) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth) return;
	uint voxelIndex = z * volumeSize.width * volumeSize.height + y * volumeSize.width + x;
	float Sx[3][3][3] = {{{1, 0, -1}, {2, 0, -2}, {1, 0, -1}}, {{2, 0, -2}, {4, 0, -4}, {2, 0, -2}}, {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}}};
	float Sy[3][3][3] = {{{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}}, {{2, 4, 2}, {0, 0, 0}, {-2, -4, -2}}, {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}}};
	float Sz[3][3][3] = {{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}, {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, {{-1, -2, -1}, {-2, -4, -2}, {-1, -2, -1}}};
	d_gradArrayOrg[voxelIndex] = make_float4(calGrad(d_volumeArrayOrg, x, y, z, volumeSize, Sx),
		calGrad(d_volumeArrayOrg, x, y, z, volumeSize, Sy),
		calGrad(d_volumeArrayOrg, x, y, z, volumeSize, Sz), 0.0f);
	d_gradValueArrayOrg[voxelIndex] = norm(d_gradArrayOrg[voxelIndex]);
}

// Calculate max value in volume (reduction of 3 times)
// Reduction functions
__global__ void d_maxVolumeValueReduc3D(float* d_array, float* d_result, cudaExtent volumeSize) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= volumeSize.width || y >= volumeSize.height) return;
	uint index = y * volumeSize.width + x;

	uint step = volumeSize.width * volumeSize.height;
	uint cap = volumeSize.width * volumeSize.height * volumeSize.depth;

	float maxVal = 0.0f;
	for (uint i = index; i < cap; i += step) {
		maxVal = max(d_array[i], maxVal);
	}

	d_result[index] = maxVal;
}

__global__ void d_maxVolumeValueReduc2D(float* d_array, float* d_result, cudaExtent volumeSize) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= volumeSize.width) return;
	uint index = x;

	uint step = volumeSize.width;
	uint cap = volumeSize.width * volumeSize.height;

	float maxVal = 0.0f;
	for (uint i = index; i < cap; i += step) {
		maxVal = max(d_array[i], maxVal);
	}

	d_result[index] = maxVal;
}

__global__ void d_maxVolumeValueReduc1D(float* d_array, float* d_result, cudaExtent volumeSize) {
	uint cap = volumeSize.width;

	float maxVal = 0.0f;
	for (uint i = 0; i < cap; ++i) {
		maxVal = max(d_array[i], maxVal);
	}

	*d_result = maxVal;
}

__global__ void d_maxVoxelReducRes(float* result) {
	d_maxVoxel = *result;
	d_invMaxVoxel = 1.0f / d_maxVoxel;
	printf("Max voxel: %f\n", d_maxVoxel);
}

__global__ void d_maxGradReducRes(float* result) {
	d_maxGrad = *result;
	d_invMaxGrad = 1.0f / d_maxGrad;
	printf("Max gradient: %f\n", d_maxGrad);
}

// Calculate gradient of whole volume and the max value
void calculateVolumeGrad(cudaExtent volumeSize) {
	checkCudaErrors(cudaMalloc(&d_volumeArrayOrg, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth));
	checkCudaErrors(cudaMalloc(&d_gradArrayOrg, sizeof(GradType) * volumeSize.width * volumeSize.height * volumeSize.depth));
	checkCudaErrors(cudaMalloc(&d_gradValueArrayOrg, sizeof(float) * volumeSize.width * volumeSize.height * volumeSize.depth));

	// calculate volume gradient
	dim3 blockSize(8, 8, 8);
	dim3 gridSize(iDivUp((int)volumeSize.width, blockSize.x), iDivUp((int)volumeSize.height, blockSize.y), iDivUp((int)volumeSize.depth, blockSize.z));

	d_calculateNormVolume<<<gridSize, blockSize>>>(d_volumeArrayOrg, d_volumeTex, volumeSize);
	d_calculateVolumeGrad<<<gridSize, blockSize>>>(d_gradArrayOrg, d_gradValueArrayOrg, d_volumeArrayOrg, volumeSize);

	// calculate max value of voxel and gradient
	float* d_buffer2D = NULL, * d_buffer1D = NULL, * d_bufferRes = NULL;
	checkCudaErrors(cudaMalloc(&d_buffer2D, sizeof(float) * volumeSize.width * volumeSize.height));
	checkCudaErrors(cudaMalloc(&d_buffer1D, sizeof(float) * volumeSize.width));
	checkCudaErrors(cudaMalloc(&d_bufferRes, sizeof(float)));

	dim3 reduc3DBlockSize(8, 8);
	dim3 reduc3DGridSize(iDivUp((int)volumeSize.width, reduc3DBlockSize.x), iDivUp((int)volumeSize.height, reduc3DBlockSize.y));
	dim3 reduc2DBlockSize(8);
	dim3 reduc2DGridSize(iDivUp((int)volumeSize.width, reduc2DBlockSize.x));

	d_maxVolumeValueReduc3D<<<reduc3DGridSize, reduc3DBlockSize>>>(d_volumeArrayOrg, d_buffer2D, volumeSize);
	d_maxVolumeValueReduc2D<<<reduc2DGridSize, reduc2DBlockSize>>>(d_buffer2D, d_buffer1D, volumeSize);
	d_maxVolumeValueReduc1D<<<1, 1>>>(d_buffer1D, d_bufferRes, volumeSize);
	d_maxVoxelReducRes <<<1, 1>>>(d_bufferRes);

	d_maxVolumeValueReduc3D<<<reduc3DGridSize, reduc3DBlockSize>>>(d_gradValueArrayOrg, d_buffer2D, volumeSize);
	d_maxVolumeValueReduc2D<<<reduc2DGridSize, reduc2DBlockSize>>>(d_buffer2D, d_buffer1D, volumeSize);
	d_maxVolumeValueReduc1D<<<1, 1>>>(d_buffer1D, d_bufferRes, volumeSize);
	d_maxGradReducRes<<<1, 1>>>(d_bufferRes);

	if (d_buffer2D) {
		checkCudaErrors(cudaFree(d_buffer2D));
	}
	if (d_buffer1D) {
		checkCudaErrors(cudaFree(d_buffer1D));
	}
	if (d_bufferRes) {
		checkCudaErrors(cudaFree(d_bufferRes));
	}
}

// Create gradient texture
void createGradTex(cudaExtent volumeSize) {
	// create gradient texture
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<GradType>();
	checkCudaErrors(cudaMalloc3DArray(&d_gradArray, &channelDesc, volumeSize));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr(d_gradArrayOrg, sizeof(GradType) * volumeSize.width, volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_gradArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// texture description
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_gradArray;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = true;  // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear;  // linear interpolation

	texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(&d_gradTex, &texRes, &texDescr, NULL));
}

// Save camera state in previous frame
__global__ void d_savePrevCameraState() {
	d_prevCamera = d_params.camera;
}


/* -------- Functions for external calling -------- */

// Copy kernel parameters to GPU
extern "C" void copyKernelParams(KernelParams* params, size_t size) {
	checkCudaErrors(cudaMemcpyToSymbol(d_params, params, size));
}

// Create volume data
extern "C" void createVolume(void* volumeData, cudaExtent volumeSize) {
	if (d_volumeTex) {
		checkCudaErrors(cudaDestroyTextureObject(d_volumeTex));
	}
	if (d_gradTex) {
		checkCudaErrors(cudaDestroyTextureObject(d_gradTex));
	}
	if (d_volumeArray) {
		checkCudaErrors(cudaFreeArray(d_volumeArray));
	}
	if (d_gradArray) {
		checkCudaErrors(cudaFreeArray(d_gradArray));
	}

	createVolTex(volumeData, volumeSize);
	calculateVolumeGrad(volumeSize);
	createGradTex(volumeSize);

	if (d_volumeArrayOrg) {
		checkCudaErrors(cudaFree(d_volumeArrayOrg));
		d_volumeArrayOrg = NULL;
	}
	if (d_gradArrayOrg) {
		checkCudaErrors(cudaFree(d_gradArrayOrg));
		d_gradArrayOrg = NULL;
	}
	if (d_gradValueArrayOrg) {
		checkCudaErrors(cudaFree(d_gradValueArrayOrg));
		d_gradValueArrayOrg = NULL;
	}
}

// Create environment texture
extern "C" void createEnvTex(void* hEnvTex, cudaExtent envTexSize) {
	if (d_envTexArray) {
		checkCudaErrors(cudaFreeArray(d_envTexArray));
	}
	if (d_envTex) {
		checkCudaErrors(cudaDestroyTextureObject(d_envTex));
	}

	// create environment texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<TexPixelType>();
	checkCudaErrors(cudaMallocArray(&d_envTexArray, &channelDesc, envTexSize.width, envTexSize.height));

	const size_t bytesPerElem = sizeof(TexPixelType);
	checkCudaErrors(cudaMemcpy2DToArray(d_envTexArray, 0, 0, hEnvTex, envTexSize.width * bytesPerElem, envTexSize.width * bytesPerElem, envTexSize.height, cudaMemcpyHostToDevice));

	// texture description
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_envTexArray;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = true;  // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear;  // linear interpolation

	texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	texDescr.addressMode[1] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(&d_envTex, &texRes, &texDescr, NULL));
}

// Create transfer function texture
extern "C" void createTransTex(void* hTransFunc, uint len) {
	if (d_transFuncArray) {
		checkCudaErrors(cudaFreeArray(d_transFuncArray));
	}
	if (d_transFuncTex) {
		checkCudaErrors(cudaDestroyTextureObject(d_transFuncTex));
	}

	// create transfer function texture
	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<TransferFunElemType>();
	checkCudaErrors(cudaMallocArray(&d_transFuncArray, &channelDesc2, len, 1));
	checkCudaErrors(cudaMemcpy2DToArray(d_transFuncArray, 0, 0, hTransFunc, 0, len * sizeof(TransferFunElemType), 1, cudaMemcpyHostToDevice));

	// texture description
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_transFuncArray;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = true;  // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
	texDescr.readMode = cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(&d_transFuncTex, &texRes, &texDescr, NULL));
}

// Initialize data
extern "C" void initCuda(void* volumeData, cudaExtent volumeSize, void* envTexData, cudaExtent envTexSize, void* transferFuncData, uint transferFuncSize) {
	createVolume(volumeData, volumeSize);
	createEnvTex(envTexData, envTexSize);
	createTransTex(transferFuncData, transferFuncSize);
}

// Allocate accumulated buffer
extern "C" void allocAccumBuffer(uint width, uint height) {
	if (d_accumBuffer != NULL) {
		checkCudaErrors(cudaFree(d_accumBuffer));
	}
	d_accBufSize.x = width;
	d_accBufSize.y = height;
	checkCudaErrors(cudaMalloc(&d_accumBuffer, sizeof(AccBufPixelType) * width * height));
}

// Allocate reservoirs
extern "C" void allocReservoirs(uint width, uint height) {
	if (d_reservoirs != NULL) {
		checkCudaErrors(cudaFree(d_reservoirs));
	}
	if (d_prevReservoirs != NULL) {
		checkCudaErrors(cudaFree(d_prevReservoirs));
	}
	d_accBufSize.x = width;
	d_accBufSize.y = height;
	checkCudaErrors(cudaMalloc(&d_reservoirs, sizeof(Reservoir) * width * height));
	checkCudaErrors(cudaMalloc(&d_prevReservoirs, sizeof(Reservoir) * width * height));
}

// Reset accumulated buffer
extern "C" void resetAccumBuffer() {
	d_iteration = 0;
	if (d_accumBuffer != NULL) {
		checkCudaErrors(cudaMemset(d_accumBuffer, 0, sizeof(AccBufPixelType) * d_accBufSize.x * d_accBufSize.y));
	}
}

// Reset reservoirs
extern "C" void resetReservoirs() {
	d_iteration = 0;
	if (d_reservoirs != NULL) {
		checkCudaErrors(cudaMemset(d_reservoirs, 0, sizeof(Reservoir) * d_accBufSize.x * d_accBufSize.y));
	}
	if (d_prevReservoirs != NULL) {
		checkCudaErrors(cudaMemset(d_prevReservoirs, 0, sizeof(Reservoir) * d_accBufSize.x * d_accBufSize.y));
	}
}

// Copy accumulated buffer to host for display
extern "C" void copyAccumBufferToHost(void*& buffer, uint& width, uint& height) {
	if (d_accumBuffer == NULL) {
		return;
	}
	width = d_accBufSize.x;
	height = d_accBufSize.y;
	size_t bufSize = sizeof(AccBufPixelType) * width * height;
	buffer = malloc(bufSize);
	if (buffer == NULL) {
		return;
	}
	checkCudaErrors(cudaMemcpy(buffer, d_accumBuffer, bufSize, cudaMemcpyDeviceToHost));
}

// Free all buffers
extern "C" void freeCudaBuffers() {
	checkCudaErrors(cudaDestroyTextureObject(d_volumeTex));
	checkCudaErrors(cudaDestroyTextureObject(d_gradTex));
	checkCudaErrors(cudaDestroyTextureObject(d_envTex));
	checkCudaErrors(cudaDestroyTextureObject(d_transFuncTex));
	checkCudaErrors(cudaFreeArray(d_volumeArray));
	checkCudaErrors(cudaFreeArray(d_gradArray));
	checkCudaErrors(cudaFreeArray(d_envTexArray));
	checkCudaErrors(cudaFreeArray(d_transFuncArray));
	checkCudaErrors(cudaFree(d_accumBuffer));
	checkCudaErrors(cudaFree(d_reservoirs));
	checkCudaErrors(cudaFree(d_prevReservoirs));
}

// Set transfer function filter mode (linear / point)
extern "C" void d_setTransferFilterMode(bool bLinearFilter) {
	if (d_transFuncTex) {
		checkCudaErrors(cudaDestroyTextureObject(d_transFuncTex));
	}
	// texture description
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_transFuncArray;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = true;  // access with normalized texture coordinates
	texDescr.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
	texDescr.readMode = cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(&d_transFuncTex, &texRes, &texDescr, NULL));
}

// Set volume filter mode (linear / point)
extern "C" void d_setVolumeFilterMode(bool bLinearFilter) {
	if (d_volumeTex) {
		checkCudaErrors(cudaDestroyTextureObject(d_volumeTex));
	}
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_volumeArray;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = true;
	texDescr.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;

	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeNormalizedFloat;

	checkCudaErrors(cudaCreateTextureObject(&d_volumeTex, &texRes, &texDescr, NULL));
}

// Enter of rendering
extern "C" void renderGPU(KernelParams* params, dim3 gridSize, dim3 blockSize) {
	params->scaleInvHalf = 0.5f / params->scale;

	params->volumeTex = d_volumeTex;
	params->gradTex = d_gradTex;
	params->envTex = d_envTex;
	params->transFuncTex = d_transFuncTex;

	params->d_iteration = d_iteration;
	params->accumBuffer = d_accumBuffer;
	params->reservoirs = d_reservoirs;
	params->prevReservoirs = d_prevReservoirs;

	copyKernelParams(params, sizeof(KernelParams));

	if (params->method == 0) {
		d_renderRM<<<gridSize, blockSize>>>(d_randStates);
	} else if (params->method == 1) {
		d_renderPT<<<gridSize, blockSize>>>(d_randStates);
	} else if (params->method == 2) {
		d_renderReSTIR<<<gridSize, blockSize>>>(d_randStates);
		checkCudaErrors(cudaMemcpy(d_prevReservoirs, d_reservoirs, sizeof(Reservoir) * d_accBufSize.x * d_accBufSize.y, cudaMemcpyDeviceToDevice));
	}

	// imageReduction();
	d_convDisp<<<gridSize, blockSize>>>();
	d_savePrevCameraState<<<1, 1>>>();

	++d_iteration;
}
