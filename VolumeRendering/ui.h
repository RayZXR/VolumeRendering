#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include "type_helper.h"

#ifndef uint
typedef unsigned int uint;
#endif

#ifndef GLFWwindow
struct GLFWwindow;
#endif

#ifndef GLuint
typedef unsigned int GLuint;
#endif

class UIInterface {
public:
    static const uint maxStrLen = 256;              // max string length
    char volumeFilename[maxStrLen] = {};            // volume file name
    uint3 volumeSize;                               // volume size
    char envTexFileName[maxStrLen] = {};            // environment file name
    char saveImgFileName[maxStrLen] = {};           // save image file name
    bool isVolumeFileExist;                         // whether volume file exist
    bool isEnvTexFileExist;                         // whether environment texture file exist

    float avgFps;                                   // average fps
    static const uint rcnFpsLen = 120;              // recent fps length
    float rcnFpsList[rcnFpsLen] = {};               // recent fps list
    bool needResetBuffer;                           // whether need to reset buffer

    // camera data
    float cameraDist, defaultCameraDist;
    float3 cameraCenter, defaultCameraCenter;
    float3 cameraRotation, defaultCameraRotation;
    float3 u, v, w;
    bool cameraChanged;

    uint frameCount = 0, saveFrameCount = 0;        // frame count
    bool canSaveImage = false;
    bool continuousSave = false;
    bool conv = false;

    UIInterface();

    void createNewVolume();
    void createNewEnvTex();
    void createNewTransFuncTex();
    void setVolumeFilterMode(bool bLinearFilter);
    void setTransferFilterMode(bool bLinearFilter);
    bool saveImage(char* fileName = NULL);
    void resetCamera();
};

class UI {
private:
	uint w_width, w_height, d_width, d_height, m_width, m_height;
	GLFWwindow* window;
    UIInterface& uii;
public:
	UI(GLFWwindow* window, uint w_width, uint w_height, uint d_width, uint d_height, UIInterface& uii);
	~UI();
	void display(GLuint tex, KernelParams& params);
	void destroy();

	void setStyle();
	void sceneWindow(GLuint tex);
	void consoleWindow(KernelParams& params);
	void status(KernelParams& params);
	void renderingMethod(KernelParams& params);
	void volume(KernelParams& params);
	void volumeProperties(KernelParams& params);
	void algorithmProperties(KernelParams& params);
	void rayMarchingProperties(KernelParams& params);
	void pathTracingProperties(KernelParams& params);
	void camera(KernelParams& params);
	void result(KernelParams& params);
	void statusBar();
	void handleEvent();
};
