#pragma once
#ifndef MATH_HELPER_H
#define MATH_HELPER_H

#include "type_helper.h"
#include "cuda_helper.h"
#include <cmath>
using namespace std;

constexpr float PI = 3.14159265358979323846f;
constexpr float inv_PI = 1.0f / PI;
constexpr float inv_3 = 1.0f / 3.0f;
constexpr float DEG_TO_RAD = PI / 180.0f;
constexpr float EPSILON = 1e-3f;

constexpr int N_min = 4;
constexpr int N_max = 256;
constexpr int N_norm = 1;

inline __device__ __host__ int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// Returns a power of two size for the given target capacity.
inline __device__ __host__ int tableSizeFor(int cap) {
	int n = cap - 1;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	return (n < 0) ? 1 : n + 1;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float4 operator+(float4 a, float4 b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float3& a, float3 b) {
	a.x += b.x; a.y += b.y; a.z += b.z;
}
inline __host__ __device__ void operator+=(float4& a, float4 b) {
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator-(float3 a) {
	return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(float4 a, float4 b) {
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ float4 operator-(float4 a) {
	return make_float4(-a.x, -a.y, -a.z, -a.w);
}

inline __host__ __device__ float3 operator*(float3 a, float3 b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float4 operator*(float4 a, float4 b) {
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ float3 operator*(float3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a) {
	return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ float4 operator*(float4 a, float b) {
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a) {
	return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float3& a, float b) {
	a.x *= b; a.y *= b; a.z *= b;
}
inline __host__ __device__ void operator*=(float4& a, float b) {
	a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}
inline __host__ __device__ void operator*=(float4& a, float4 b) {
	a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}

inline __host__ __device__ float3 operator/(float3 a, float3 b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float4 operator/(float4 a, float4 b) {
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ float3 operator/(float b, float3 a) {
	return make_float3(b / a.x, b / a.y, b / a.z);
}
inline __host__ __device__ float3 operator/(float3 a, float b) {
	float invB = 1.0f / b;
	return make_float3(a.x * invB, a.y * invB, a.z * invB);
}
inline __host__ __device__ float4 operator/(float4 a, float b) {
	float invB = 1.0f / b;
	return make_float4(a.x * invB, a.y * invB, a.z * invB, a.w * invB);
}
inline __host__ __device__ void operator/=(float4& a, float b) {
	float invB = 1.0f / b;
	a.x *= invB; a.y *= invB; a.z *= invB; a.w *= invB;
}
inline __host__ __device__ void operator/=(float4& a, float4 b) {
	a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}

inline __host__ __device__ float4x4 operator*(float4x4 A, float4x4 B) {
	float4x4 res;
	for (int i = 0; i < 4; ++i) {
		res.m[i] = make_float4(A.m[i].x * B.m[0].x + A.m[i].y * B.m[1].x + A.m[i].z * B.m[2].x + A.m[i].w * B.m[3].x,
			A.m[i].x * B.m[0].y + A.m[i].y * B.m[1].y + A.m[i].z * B.m[2].y + A.m[i].w * B.m[3].y,
			A.m[i].x * B.m[0].z + A.m[i].y * B.m[1].z + A.m[i].z * B.m[2].z + A.m[i].w * B.m[3].z,
			A.m[i].x * B.m[0].w + A.m[i].y * B.m[1].w + A.m[i].z * B.m[2].w + A.m[i].w * B.m[3].w);
	}
	return res;
}

inline __host__ __device__ float4 operator*(float4x4 A, float4 v) {
	return make_float4(A.m[0].x * v.x + A.m[0].y * v.y + A.m[0].z * v.z + A.m[0].w * v.w,
		A.m[1].x * v.x + A.m[1].y * v.y + A.m[1].z * v.z + A.m[1].w * v.w,
		A.m[2].x * v.x + A.m[2].y * v.y + A.m[2].z * v.z + A.m[2].w * v.w,
		A.m[3].x * v.x + A.m[3].y * v.y + A.m[3].z * v.z + A.m[3].w * v.w);
}

inline __host__ __device__ float3 to_float3(float4 v) {
	return make_float3(v.x, v.y, v.z);
}

inline __host__ __device__ float4 exp(float4 v) {
	return make_float4(exp(v.x), exp(v.y), exp(v.z), exp(v.w));
}

inline __host__ __device__ float4 homogeneous(float3 a) {
	return make_float4(a.x, a.y, a.z, 1.0f);
}

inline __host__ __device__ float3 fminf(float3 a, float3 b) {
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

inline __host__ __device__ float4 fminf(float4 a, float4 b) {
	return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

inline __host__ __device__ float3 fmaxf(float3 a, float3 b) {
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

inline __host__ __device__ float4 fmaxf(float4 a, float4 b) {
	return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

inline __host__ __device__ float dot(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float dot(float4 a, float4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ float norm(float3 v) {
	return sqrtf(dot(v, v));
}
inline __host__ __device__ float norm(float4 v) {
	return sqrtf(dot(v, v));
}
inline __host__ __device__ float sqrNorm(float3 v) {
	return dot(v, v);
}
inline __host__ __device__ float3 normalize(float3 v) {
	float invLen = 1.0f / sqrtf(dot(v, v));
	return v * invLen;
}

inline __host__ __device__ float3 cross(float3 a, float3 b) {
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline __host__ __device__ float3 reflect(float3 vecIn, float3 normal) {
	return vecIn - 2.0f * dot(vecIn, normal) * normal;
}

inline __device__ __host__ uint clamp(uint v, uint min_v, uint max_v) {
	return v > min_v ? (v < max_v ? v : max_v) : min_v;
}

inline __device__ __host__ float clamp(float v, float min_v, float max_v) {
	return v > min_v ? (v < max_v ? v : max_v) : min_v;
}

inline __device__ __host__ float3 clamp(float3 v, float min_v, float max_v) {
	return make_float3(clamp(v.x, min_v, max_v), clamp(v.y, min_v, max_v), clamp(v.z, min_v, max_v));
}

inline __device__ __host__ float3 rotateWithAxis(float3 v, float3 axis, float angle) {
	if (angle == 0.0f || dot(axis, axis) == 0.0f)
		return v;
	axis = normalize(axis);
	float cosV = cosf(angle * DEG_TO_RAD), sinV = sinf(angle * DEG_TO_RAD), omcv = 1.0f - cosV;
	float3 r1 = make_float3(cosV + omcv * axis.x * axis.x, omcv * axis.x * axis.y - sinV * axis.z, omcv * axis.x * axis.z + sinV * axis.y);
	float3 r2 = make_float3(omcv * axis.x * axis.y + sinV * axis.z, cosV + omcv * axis.y * axis.y, omcv * axis.y * axis.z - sinV * axis.x);
	float3 r3 = make_float3(omcv * axis.x * axis.z - sinV * axis.y, omcv * axis.y * axis.z + sinV * axis.z, cosV + omcv * axis.z * axis.z);
	return normalize(make_float3(dot(r1, v), dot(r2, v), dot(r3, v)));
}

inline __device__ __host__ float avg(float4 val) {
	return 0.25f * (val.x + val.y + val.z + val.w);
}

inline __device__ __host__ float F(const float& x) {
	const float A = 0.22f;
	const float B = 0.30f;
	const float C = 0.10f;
	const float D = 0.20f;
	const float E = 0.01f;
	const float F = 0.30f;

	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

inline __device__ __host__ float toneMapping(float val, const float& adapted_lum = 1.0f, const int& method = 0) {
	if (method == 0) {
		return val * (1.0f + val * 0.1f) / (1.0f + val);
	} else if (method == 1) {
		return val / adapted_lum;
	} else if (method == 2) {
		const float MIDDLE_GREY = 1;
		val *= MIDDLE_GREY / adapted_lum;
		return val / (1.0f + val);
	} else if (method == 3) {
		return 1.0f - exp(-adapted_lum * val);
	} else if (method == 4) {
		const float WHITE = 11.2f;
		return F(1.6f * adapted_lum * val) / F(WHITE);
	} else if (method == 5) {
		const float A = 2.51f;
		const float B = 0.03f;
		const float C = 2.43f;
		const float D = 0.59f;
		const float E = 0.14f;

		val *= adapted_lum;
		return (val * (A * val + B)) / (val * (C * val + D) + E);
	}

	return val;
}

inline __device__ __host__ uint rgbaFloatToInt(float4 val, float4 adapted_lum, int method = 0, bool gammaCorrection = true) {
	// Tone Mapping
	val.x = toneMapping(val.x, adapted_lum.x, method);
	val.y = toneMapping(val.y, adapted_lum.y, method);
	val.z = toneMapping(val.z, adapted_lum.z, method);
	uint r, g, b, a;
	if (gammaCorrection) {
		float fac = (float)(1.0 / 2.2);
		r = (unsigned int)(255.0f * fminf(powf(fmaxf(val.x, 0.0f), fac), 1.0f));
		g = (unsigned int)(255.0f * fminf(powf(fmaxf(val.y, 0.0f), fac), 1.0f));
		b = (unsigned int)(255.0f * fminf(powf(fmaxf(val.z, 0.0f), fac), 1.0f));
	} else {
		r = (unsigned int)(255.0f * clamp(val.x, 0.0f, 1.0f));
		g = (unsigned int)(255.0f * clamp(val.y, 0.0f, 1.0f));
		b = (unsigned int)(255.0f * clamp(val.z, 0.0f, 1.0f));
	}
	a = uint(clamp(val.w, 0.0f, 1.0f) * 255.0f);
	return (a << 24) | (b << 16) | (g << 8) | r;
}

inline bool cmp(TransferFunCtrl& ctrl1, TransferFunCtrl& ctrl2) {
	return ctrl1.pos < ctrl2.pos;
}

#endif // !MATH_HELPER_H
