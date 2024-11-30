#include "math_helper.h"
#include "ui.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "cuda_gl_interop.h"
#include "cuda_profiler_api.h"
#include "curand_kernel.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib>
#include <cstring>
#include <ctime>
using namespace std;


/* Data */
// char volumeFilePrefix[maxStrLen] = "CThead.";
const char volumeDir[256] = "Assets/Models/";
const char envTexDir[256] = "Assets/Textures/";
const char saveImgDir[256] = "Images/PT_RE2/";

TransferFunElemType* transferFunc;
uint transferFuncLen = 101;

/* Config */
KernelParams params;

/* System */
uint r_width = 1080, r_height = 1080;
uint d_width = 1080, d_height = 1080, m_width = 384, m_height = 40, w_width = d_width + m_width, w_height = d_height + m_height;
dim3 blockSize(16, 16);
dim3 gridSize;

/* Camera */
float3 cameraForwardInit(0.0f, 0.0f, -1.0f);

float3 cameraForward(0.0f, 0.0f, -1.0f);
float3 cameraUp(0.0f, 1.0f, 0.0f);
float3 cameraRight(1.0f, 0.0f, 0.0f);
float focusDist = 0.1f;
float fov = 45.0f;

float aspect = (float)r_width / r_height;                                       // screen's aspect
float s_height = focusDist * 2.0f * tanf(0.5f * fov * DEG_TO_RAD);			    // screen's height
float s_width = aspect * s_height;								                // screen's width

UIInterface uii;                // UI interface

GLFWwindow* window = NULL;

unsigned int frameCount = 0;
clock_t lt = 0, ct = 0;

GLuint pbo = 0;  // OpenGL pixel buffer object
GLuint tex = 0;  // OpenGL texture object
struct cudaGraphicsResource* cuda_pbo_resource;  // CUDA Graphics Resource (to transfer PBO)

void* volumeData = NULL;
void* envTexData = NULL;


/* Functions in cuda kernel file */
// Initialize cuda
extern "C" void initCuda(void* volumeData, cudaExtent volumeSize, void* envTexData, cudaExtent envTexSize, void* transferFuncData, uint transferFuncSize);
// Initialize random state
extern "C" void randomStateInit(dim3 gridSize, dim3 blockSize, int width, int height, unsigned long long seed);
// Copy kernel parameters to GPU
extern "C" void copyKernelParams(KernelParams* params, size_t size);
// Allocate accumulated buffer
extern "C" void allocAccumBuffer(uint width, uint height);
// Allocate reservoirs
extern "C" void allocReservoirs(uint width, uint height);
// Reset accumulated buffer
extern "C" void resetAccumBuffer();
// Reset reservoirs
extern "C" void resetReservoirs();
// Copy accumulated buffer to host
extern "C" void copyAccumBufferToHost(void*& buffer, uint& width, uint& height);
// Free all cuda buffers
extern "C" void freeCudaBuffers();
// Render with GPU
extern "C" void renderGPU(KernelParams* params, dim3 gridSize, dim3 blockSize);
// Set volume filter mode
extern "C" void d_setVolumeFilterMode(bool bLinearFilter);
// Set transfer function mode
extern "C" void d_setTransferFilterMode(bool bLinearFilter);
// Create volume data
extern "C" void createVolume(void* volumeData, cudaExtent volumeSize);
// Create environment texture
extern "C" void createEnvTex(void* hEnvTex, cudaExtent envTexSize);
// Create transfer function texture
extern "C" void createTransTex(void* hTransFunc, uint len);


/* Functions for rendering */
// Computer frame per second
void computeFPS() {
    frameCount++;

    float curFps = CLOCKS_PER_SEC / fmaxf(1.0f, (float)(ct - lt));
    float sumFps = 0.0f;
    for (int i = 0; i < uii.rcnFpsLen - 1; ++i) {
        uii.rcnFpsList[i] = uii.rcnFpsList[i + 1];
        sumFps += uii.rcnFpsList[i];
    }
    uii.rcnFpsList[uii.rcnFpsLen - 1] = curFps;
    sumFps += curFps;
    uii.avgFps = sumFps / uii.rcnFpsLen;

    // char fps[256];
    // snprintf(fps, 256, "Volume Render: %0.2f fps", avgFps);

    // glfwSetWindowTitle(window, fps);
}

// Switch endian
void switchEndian(void* data, size_t singleSize, size_t size) {
    if (singleSize == 2) {
        size >>= 1;
        for (size_t i = 0; i < size; ++i) {
            ((unsigned short*)data)[i] = (((unsigned short*)data)[i] << 8) | (((unsigned short*)data)[i] >> 8);
        }
    }
    else if (singleSize == 4) {
        size >>= 2;
        for (size_t i = 0; i < size; ++i) {
            ((unsigned int*)data)[i] = ((((unsigned int*)data)[i] >> 24) & 0xff)
                | ((((unsigned int*)data)[i] >> 8) & 0xFF00)
                | ((((unsigned int*)data)[i] << 8) & 0xFF0000)
                | ((((unsigned int*)data)[i] << 24));
        }
    }
}

// Load raw data from disk
void* loadRawFile(const char* filename, size_t size, bool endian = false) {
    char filePath[256] = {};
    strcpy(filePath, volumeDir);
    strcat(filePath, filename);

    FILE* fp;
    fopen_s(&fp, filePath, "rb");

    if (!fp) {
        uii.isVolumeFileExist = false;
        fprintf(stderr, "Error opening file '%s'\n", filePath);
        return NULL;
    }

    void* data = malloc(size);
    if (!data) {
        fclose(fp);
        return NULL;
    }
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    if (endian) {
        switchEndian(data, sizeof(VolumeType), size);
    }

    printf("Read '%s', %zu bytes\n", filename, read);
    uii.isVolumeFileExist = true;

    return data;
}

// Load texture data
void* loadTexture(const char* filename, uint& width, uint& height, uint& channels) {
    char filePath[256] = {};
    strcpy(filePath, envTexDir);
    strcat(filePath, filename);

    cv::Mat image = cv::imread(filePath, cv::IMREAD_UNCHANGED);
    // perror(cv::getBuildInformation().c_str());

    if (image.empty()) {
        uii.isEnvTexFileExist = false;
        fprintf(stderr, "Error opening file '%s'\n", filePath);
        return NULL;
    }

    width = image.cols;
    height = image.rows;
    channels = image.channels();

    float* src = (float*)image.ptr();
    size_t size = sizeof(TexPixelType) * width * height;
    TexPixelType* data = (TexPixelType*)malloc(size);
    if (!data) {
        return NULL;
    }

    for (uint i = 0; i < height; ++i) {
        for (uint j = 0; j < width; ++j) {
            data[i * width + j].z = *src++;
            data[i * width + j].y = *src++;
            data[i * width + j].x = *src++;
            if (channels == 4) {
                data[i * width + j].w = *src++;
            } else {
                data[i * width + j].w = 1.0f;
            }
        }
    }

    printf("Read '%s', %zu bytes (%d x %d x %d)\n", uii.envTexFileName, sizeof(float) * width * height * channels, width, height, channels);
    uii.isEnvTexFileExist = true;

    return data;
}

// Load volume in multi-files
void* loadMultiFile(const char* dirName, const char* filePrefix, uint fileNum, size_t singleFileSize, size_t size, bool endian = false) {
    void* data = malloc(size);
    if (!data) {
        return NULL;
    }

    char filePath[256] = {};
    strcpy(filePath, volumeDir);
    strcat(filePath, dirName);

    FILE* fp;
    char fileName[256];
    size_t totalSize = 0;
    for (unsigned int i = 0; i < fileNum; ++i) {
        snprintf(fileName, 256, "%s%s%d", filePath, filePrefix, i + 1);
        fopen_s(&fp, fileName, "rb");
        if (!fp) {
            free(data);
            uii.isVolumeFileExist = false;
            fprintf(stderr, "Error opening file '%s'\n", fileName);
            return NULL;
        }
        size_t read = fread((uchar*)data + totalSize, 1, singleFileSize, fp);
        totalSize += singleFileSize;
        fclose(fp);
        printf("Read '%s', %zu bytes\n", fileName, read);
    }
    if (endian) {
        switchEndian(data, sizeof(VolumeType), size);
    }
    uii.isVolumeFileExist = true;
    return data;
}

// Calculate volume size (scale)
void calVolumeSize() {
    uint maxDim = max(max(uii.volumeSize.x, uii.volumeSize.y), uii.volumeSize.z);
    params.scale = make_float3((float)uii.volumeSize.x / maxDim, (float)uii.volumeSize.y / maxDim, (float)uii.volumeSize.z / maxDim);
    // d_params.scale = *((float3*)&params.scale);
}

// Convert texture channel
template <typename T>
void convertTexChannel(void* data, uint width, uint height) {
    for (uint i = 0; i < height; ++i) {
        for (uint j = 0; j < width; ++j) {
            swap(((T*)data)[i * width + j].x, ((T*)data)[i * width + j].z);
        }
    }
}

// Convert pixel type
void convertPixelType(void*& data, uint width, uint height) {
    uint* convData = (uint*)malloc(sizeof(uint) * width * height);
    for (uint i = 0; i < height; ++i) {
        for (uint j = 0; j < width; ++j) {
            convData[i * width + j] = rgbaFloatToInt(((float4*)data)[i * width + j], make_float4(1.0f, 1.0f, 1.0f, 1.0f), 
                params.toneMappingMethod, params.gammaCorrection);
        }
    }
    free(data);
    data = convData;
}

// Fill transfer function with control points
void fillTransferfunction(TransferFunCtrl ctrl[], uint n) {
    sort(ctrl, ctrl + n, cmp);
    uint pos1, pos2;
    for (uint i = 0; i < n; ++i) {
        if (i == 0) {
            pos1 = uint(ctrl[0].pos * 100 + 0.5);
            for (uint j = 0; j < pos1; ++j) {
                transferFunc[j] = ctrl[0].color;
            }
        }
        if (i == n - 1) {
            pos2 = uint(ctrl[n - 1].pos * 100 + 0.5);
            for (uint j = pos2; j < transferFuncLen; ++j) {
                transferFunc[j] = ctrl[n - 1].color;
            }
        } else {
            pos1 = uint(ctrl[i].pos * 100 + 0.5);
            pos2 = uint(ctrl[i + 1].pos * 100 + 0.5);
            float step = 1.0f / (pos2 - pos1);
            float4 diff = ctrl[i + 1].color - ctrl[i].color;
            for (uint j = pos1; j < pos2; ++j) {
                transferFunc[j] = ctrl[i].color + (float)(j - pos1) * step * diff;
            }
        }
    }
}

// Initialize GL
void initGL() {
    // Initialize GLFW
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW.\n");
        exit(EXIT_FAILURE);
    }
    // glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    window = glfwCreateWindow(w_width, w_height, "Volume Rendering", NULL, NULL);
    if (window == NULL)
    {
        fprintf(stderr, "Failed to create GLFW window.\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwSetWindowPos(window, (mode->width - w_width) >> 1, (mode->height - w_height) >> 1);
    glfwMakeContextCurrent(window);
    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        fprintf(stderr, "Failed to initialize GLAD.\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    /*
    // Initialize ImGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.Fonts->AddFontFromFileTTF("Assets/Fonts/NotoSansHans-Regular.otf", 16.0f, NULL, io.Fonts->GetGlyphRangesChineseSimplifiedCommon());


    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();
    ImGui::StyleColorsLight();

    ImGui::GetStyle().WindowBorderSize = 0.0f;
    */

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
}

// Initialize data
void initData() {
    // init config
    params.resolution.x = r_width;
    params.resolution.y = r_height;
    params.scale = make_float3(1.0f, 1.0f, 1.0f);
    params.density = 1.0f;
    params.brightness = 15.0f;

    params.transferOffset = 0.0f;
    params.transferScale = 1.0f;
    params.transferGrad = true;
    params.transferGradPow = 1.0f;

    params.densityThreshold = make_float2(0.1f, 1.0f);
    params.volumeLinearFiltering = true;
    params.transferFunLinearFiltering = true;
    params.method = 2;
    params.environmentType = 0;
    params.bgCol1 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    params.bgCol2 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    // params.bgCol2 = make_float4(0.5f, 0.7f, 1.0f, 1.0f);
    params.gammaCorrection = true;
    params.toneMappingMethod = 5;
    params.g = 0.0f;
    params.enableGrad = true;
    params.hgFac = 1.0f;
    
    params.exposureScale = 1.0f;
    params.maxInteractions = MAX_K;
    params.maxExtinction = 64.0;
    params.albedo = 0.8f;

    params.repDepthThreshold = 0.9f;

    // calculate new grid size
    gridSize = dim3(iDivUp(r_width, blockSize.x), iDivUp(r_height, blockSize.y));

    // load volume data
    size_t singleSize = uii.volumeSize.x * uii.volumeSize.y * sizeof(VolumeType);
    size_t size = uii.volumeSize.z * uii.volumeSize.x * uii.volumeSize.y * sizeof(VolumeType);
    volumeData = loadRawFile(uii.volumeFilename, size);
    // void* h_volume = loadMultiFile(volumeFilename, volumeFilePrefix, volumeSize.depth, singleSize, size, false);

    // load environment texture
    uint width, height, channels;
    envTexData = loadTexture(uii.envTexFileName, width, height, channels);

    // load transfer function texture
    transferFunc = new TransferFunElemType[transferFuncLen];

#if 0
    TransferFunCtrl ctrl[] = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 0.125f, 0.125f},
        {1.0f, 0.5f, 0.0f, 0.25f, 0.25f},
        {1.0f, 1.0f, 0.0f, 0.375f, 0.375f},
        {0.0f, 1.0f, 0.0f, 0.5f, 0.5f},
        {0.0f, 1.0f, 1.0f, 0.625f, 0.625f},
        {0.0f, 0.0f, 1.0f, 0.75f, 0.75f},
        {1.0f, 0.0f, 1.0f, 0.875f, 0.875f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f} };
#else
    TransferFunCtrl ctrl[] = { 
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.5f, 0.0f, 0.0f, 0.2f, 0.046f},
        {0.515f, 0.143f, 0.084f, 0.6f, 0.41f},
        {0.8f, 0.8f, 0.8f, 0.8f, 0.7f},
        {0.757f, 0.731f, 0.613f, 1.0f, 1.0f} };
#endif

    fillTransferfunction(ctrl, sizeof(ctrl) / sizeof(TransferFunCtrl));

    if (!uii.isVolumeFileExist || !uii.isEnvTexFileExist) {
        free(volumeData);
        free(envTexData);
        volumeData = NULL;
        envTexData = NULL;
    } else {
        initCuda(volumeData, make_cudaExtent(uii.volumeSize.x, uii.volumeSize.y, uii.volumeSize.z),
            envTexData, make_cudaExtent(width, height, 4), transferFunc, transferFuncLen);
    }

    calVolumeSize();

    allocAccumBuffer(r_width, r_height);
    allocReservoirs(r_width, r_height);
    randomStateInit(gridSize, blockSize, r_width, r_height, rand());
}

// Initialize pixel buffer
void initPixelBuffer() {
    if (pbo) {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(DispBufPixelType) * r_width * r_height, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, r_width, r_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// Calculate camera property
void calculateCameraProperty() {
    cameraForward = rotateWithAxis(rotateWithAxis(cameraForwardInit, make_float3(1.0f, 0.0f, 0.0f), uii.cameraRotation.x),
        make_float3(0.0f, 1.0f, 0.0f), uii.cameraRotation.y);
    cameraUp = rotateWithAxis(make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), uii.cameraRotation.z);
    cameraRight = normalize(cross(cameraForward, cameraUp));

    uii.w = cameraForward;                       // Unit vector from camera to looking center
    uii.u = cameraRight;                         // Unit vector of screen's right
    uii.v = normalize(cross(uii.u, uii.w));      // Unit vector of screen's up

    float3 cameraPosition = uii.cameraCenter - uii.cameraDist * cameraForward;

    params.camera.changed = uii.cameraChanged;
    params.camera.fov = fov;
    params.camera.camFocal = focusDist;
    params.camera.camWidth = s_width;
    params.camera.camHeight = s_height;
    params.camera.camPos = cameraPosition;
    params.camera.camW = cameraForward;
    params.camera.camU = s_width * uii.u;
    params.camera.camV = s_height * uii.v;
}

// Render image using CUDA
void render() {
    if (uii.needResetBuffer) {
        resetAccumBuffer();
        resetReservoirs();
        uii.needResetBuffer = false;
        uii.saveFrameCount = 0;
    }

    calculateCameraProperty();
    // map PBO to get CUDA device pointer
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&params.displayBuffer, &num_bytes, cuda_pbo_resource));
    // printf("CUDA mapped PBO: May access %zu bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(params.displayBuffer, 0, sizeof(DispBufPixelType) * r_width * r_height));

    // call CUDA kernel, writing results to PBO
    renderGPU(&params, gridSize, blockSize);
    uii.cameraChanged = false;

    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// Display results using OpenGL (called by GLUT)
void display() {
    // Volume Rendering
    lt = clock();
    render();
    ct = clock();
    computeFPS();
    ++uii.frameCount;
    ++uii.saveFrameCount;

    // Display results
    // glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // draw image from PBO
#if 0
    // draw using glDrawPixels (slower)
    glRasterPos2i(0, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
#else
    // draw using texture

    // copy from pbo to texture
    // glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, r_width, r_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(1, 0);
    glVertex2f(1, -1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(-1, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);
#endif

    if (uii.continuousSave) {
        int n = strrchr(uii.saveImgFileName, '.') - uii.saveImgFileName;
        char buf[UIInterface::maxStrLen] = {};
        strncpy_s(buf, uii.saveImgFileName, n);
        snprintf(buf + n, UIInterface::maxStrLen - n - 5, "_%04u%s", uii.saveFrameCount, uii.saveImgFileName + n);
        uii.saveImage(buf);
    }
}


/* Functions for UI control */
// Create new volume
void UIInterface::createNewVolume() {
    // load volume data
    size_t singleSize = uii.volumeSize.x * uii.volumeSize.y * sizeof(VolumeType);
    size_t size = uii.volumeSize.z * uii.volumeSize.x * uii.volumeSize.y * sizeof(VolumeType);
    if (volumeData != NULL) {
        free(volumeData);
    }
    volumeData = loadRawFile(uii.volumeFilename, size);
    if (!uii.isVolumeFileExist) {
        return;
    }
    // void* volumeData = loadMultiFile(volumeFilename, volumeFilePrefix, volumeSize.depth, singleSize, size, false);

    cudaExtent volumeSize = make_cudaExtent(uii.volumeSize.x, uii.volumeSize.y, uii.volumeSize.z);
    createVolume(volumeData, volumeSize);
    calVolumeSize();
}

// Create new environment texture
void UIInterface::createNewEnvTex() {
    // load environment texture
    uint width, height, channels;
    if (envTexData != NULL) {
        free(envTexData);
    }
    envTexData = loadTexture(uii.envTexFileName, width, height, channels);
    if (!uii.isEnvTexFileExist) {
        return;
    }

    createEnvTex(envTexData, make_cudaExtent(width, height, 4));
}

// Create new transfer function texture
void UIInterface::createNewTransFuncTex() {
    createTransTex(transferFunc, transferFuncLen);
}

// Set volume filter mode
void UIInterface::setVolumeFilterMode(bool bLinearFilter) {
    d_setVolumeFilterMode(bLinearFilter);
}

// Set transfer function filter mode
void UIInterface::setTransferFilterMode(bool bLinearFilter) {
    d_setTransferFilterMode(bLinearFilter);
}

// Save rendered image
bool UIInterface::saveImage(char* fileName) {
    char filePath[256] = {};
    strcpy(filePath, saveImgDir);
    if (fileName == NULL) {
        strcat(filePath, uii.saveImgFileName);
    } else {
        strcat(filePath, fileName);
    }

    void* buffer;
    uint width, height;
    copyAccumBufferToHost(buffer, width, height);
    cv::Mat image;
    if (!uii.conv) {
        convertTexChannel<AccBufPixelType>(buffer, width, height);
        image = cv::Mat(height, width, CV_32FC4, buffer);
    } else {
        convertPixelType(buffer, width, height);
        convertTexChannel<uchar4>(buffer, width, height);
        image = cv::Mat(height, width, CV_8UC4, buffer);
    }
    cv::Mat imageConv;
    cv::flip(image, imageConv, 0);
    bool flag = false;
    cv::imwrite(filePath, imageConv);
    free(buffer);
    flag = true;
    printf("Save image: %s\n", filePath);
    return flag;
}

// Reset camera
void UIInterface::resetCamera() {
    uii.cameraDist = uii.defaultCameraDist;
    uii.cameraRotation = defaultCameraRotation;
    uii.cameraCenter = defaultCameraCenter;

    uii.needResetBuffer = true;
    uii.cameraChanged = true;
}


/* UI events */
double ox, oy;
int buttonState = 0;
bool firstMouse = true;

// On frame buffer size changing
void onFrameBufferSize(GLFWwindow* window, int w, int h) {
    w_width = max((uint)w, m_width);
    w_height = max((uint)h, m_height);
    d_width = w_width - m_width;
    d_height = w_height - m_height;
    /*
    initPixelBuffer();
    h_allocAccumBuffer(r_width, r_height);
    if (gpuAvailable) {
        allocAccumBuffer(r_width, r_height);
    }

    // calculate new grid size
    if (gpuAvailable) {
        gridSize = dim3(iDivUp(r_width, blockSize.x), iDivUp(r_height, blockSize.y));
    }

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    */
}

// On error occurring
void onError(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}

// On window closing
void onClose(GLFWwindow* window) {
    freeCudaBuffers();
    delete[] transferFunc;

    if (pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // Calling cudaProfilerStop causes all profile data to be flushed before the application exits
    checkCudaErrors(cudaProfilerStop());
}


int main() {
    // Initialize OpenGL
    initGL();
    // Find CUDA Device
    if (findCudaDevice() < 0) {
        fprintf(stderr, "Error: No available GPU device!\n");
        return 0;
    }
    // Initialize Data
    initData();
    // Initialize pixel buffer
    initPixelBuffer();
    // Register events
    glfwSetErrorCallback(onError);
    glfwSetFramebufferSizeCallback(window, onFrameBufferSize);
    glfwSetWindowCloseCallback(window, onClose);

    UI ui(window, w_width, w_height, d_width, d_height, uii);

    // Loop
    while (!glfwWindowShouldClose(window)) {
        display();
        ui.display(tex, params);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Terminate
    ui.destroy();
    glfwDestroyWindow(window);
    glfwTerminate();

	return 0;
}

