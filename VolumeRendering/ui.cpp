#include "ui.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "math_helper.h"
#include <stdio.h>
#include <ctype.h>

int margin = 16;
float inputWidth = 160.0f;
ImVec4 imgBgCol(1.0f, 1.0f, 1.0f, 1.0f);
bool imageFocused = false, isLeftDown = false, isMiddleDown = false;
ImVec2 lastMousePos;
float xRotSenst = 1.0f, yRotSenst = 1.0f;
float xMoveSenst = 0.01f, yMoveSenst = 0.01f;
float scrollSenst = 0.1f;
const char* volumeTypes[] = { "From file", "Built-in 1", "Built-in 2" };
const char* toneMappingMethods[] = { "Default", "Linear", "Reinhard", "CryEngine2", "Filmic", "ACES" };
uint statusBarHeight = 32;
char infoStr[256] = "";
ImVec4 infoColor(0.0f, 0.0f, 0.0f, 1.0f);

bool endWith(const char* a, const char* b) {
    int i = strlen(a) - 1, j = strlen(b) - 1;
    while (i >= 0 && j >= 0) {
        if (tolower(a[i]) != tolower(b[j])) {
            return false;
        }
        --i;
        --j;
    }
    return true;
}

UIInterface::UIInterface() {
    strcpy(volumeFilename, "mri_ventricles_256x256x124_uint8.raw");
    strcpy(envTexFileName, "blue_photo_studio_4k.tif");
    strcpy(saveImgFileName, "01.png");
    conv = !(endWith(saveImgFileName, ".tif") || endWith(saveImgFileName, ".tiff") ||
        endWith(saveImgFileName, ".exr") || endWith(saveImgFileName, ".hdr"));
    volumeSize = make_uint3(256, 256, 124);

    isVolumeFileExist = false;
    isEnvTexFileExist = false;

    avgFps = 0.0f;

    needResetBuffer = false;

    defaultCameraDist = cameraDist = 3.0f;
    defaultCameraCenter = cameraCenter = make_float3(0.0f, 0.0f, 0.0f);
    defaultCameraRotation = cameraRotation = make_float3(-20.0f, -45.0f, 0.0f);
    u = make_float3(0.0f, 0.0f, 0.0f);
    v = make_float3(0.0f, 0.0f, 0.0f);
    w = make_float3(0.0f, 0.0f, 0.0f);
    cameraChanged = false;
}

UI::UI(GLFWwindow* window, uint w_width, uint w_height, uint d_width, uint d_height, UIInterface& uii) :
    w_width(w_width), w_height(w_height), d_width(d_width), d_height(d_height), window(window), uii(uii) {
    m_width = w_width - d_width;
    m_height = w_height - d_height;
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
}

UI::~UI() { }

void UI::destroy() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void helpMarker(const char* desc) {
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) && ImGui::BeginTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void setInfo(const char* info, const ImVec4& color) {
    strcpy_s(infoStr, info);
    infoColor = color;
}

void UI::setStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    float rounding = 8.0f;
    style.ChildRounding = rounding;
    style.FrameRounding = rounding;
    style.PopupRounding = rounding;
    style.ScrollbarRounding = rounding;
    style.GrabRounding = rounding;
    style.TabRounding = rounding;

    style.WindowTitleAlign.x = 0.5f;
    style.SeparatorTextAlign.x = 0.5f;
}

void UI::sceneWindow(GLuint tex) {
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2((float)(d_width + margin), (float)w_height));
    ImGui::PushID(0);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, imgBgCol);
    ImGui::Begin("Scene", 0, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    {
        ImGui::Image((ImTextureID)tex, ImVec2((float)d_width, (float)d_height), ImVec2(0, 1), ImVec2(1, 0));
        if (ImGui::IsItemHovered()) {
            imageFocused = true;
        } else {
            imageFocused = false;
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopID();
}

void UI::consoleWindow(KernelParams& params) {
    ImGui::SetNextWindowPos(ImVec2((float)(d_width + margin), 0.0f));
    ImGui::SetNextWindowSize(ImVec2((float)(w_width - d_width - margin), (float)(w_height - statusBarHeight)));
    ImGui::Begin("Console", 0, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    {
        // Status
        status(params);
        // Rendering Method
        renderingMethod(params);
        // Volume
        volume(params);
        // Algorithm Properties
        algorithmProperties(params);
        // Camera
        camera(params);
        // Result
        result(params);
    }
    ImGui::End();
}

void UI::status(KernelParams& params) {
    if (ImGui::CollapsingHeader("Status", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TreePush("Status");
        ImGui::Unindent();
        char avgFpsStr[256];
        snprintf(avgFpsStr, 256, "Avg FPS: %0.2f", uii.avgFps);
        ImGui::SetNextItemWidth((float)(w_width - d_width - margin - 16));
        ImGui::PlotLines("FPS", uii.rcnFpsList, uii.rcnFpsLen, 0, avgFpsStr);
        ImGui::TreePop();
        ImGui::Indent();
    }
}

void UI::renderingMethod(KernelParams& params) {
    if (ImGui::CollapsingHeader("Rendering Method", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TreePush("Rendering Method");
        ImGui::Unindent();
        if (ImGui::RadioButton("Ray-marching", &params.method, 0)) {
            imgBgCol = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
            uii.needResetBuffer = true;
        }
        if (ImGui::RadioButton("Path-tracing", &params.method, 1)) {
            imgBgCol = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
            uii.needResetBuffer = true;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("ReSTIR", &params.method, 2)) {
            imgBgCol = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
            uii.needResetBuffer = true;
        }
        ImGui::TreePop();
        ImGui::Indent();
    }
}

void UI::volumeProperties(KernelParams& params) {
    ImGui::SeparatorText("Volume Properties");
    ImGui::SetNextItemWidth(inputWidth);
    if (ImGui::SliderFloat3("Scale", (float*)&params.scale, 0.0f, 1.0f)) {
        uii.needResetBuffer = true;
    }
    ImGui::SameLine(); helpMarker("Scale of the box.\nCTRL+click to input value.");

    ImGui::SetNextItemWidth(inputWidth);
    if (ImGui::DragFloat("Density", &params.density, 0.001f, 0.0f)) {
        params.density = clamp(params.density, 0.0f, params.density);
        uii.needResetBuffer = true;
    }
    ImGui::SameLine(); helpMarker("Density of the volume.\nDouble click or CTRL+click to input value.");

    ImGui::SetNextItemWidth(inputWidth);
    if (ImGui::DragFloatRange2("Threshold", &params.densityThreshold.x, &params.densityThreshold.y, 0.001f, 0.0f, 1.0f)) {
        params.densityThreshold.x = clamp(params.densityThreshold.x, 0.0f, 1.0f);
        params.densityThreshold.y = clamp(params.densityThreshold.y, 0.0f, 1.0f);
        uii.needResetBuffer = true;
    }
    ImGui::SameLine(); helpMarker("Specific density range.\n[Min, Max]. Double click or CTRL+click to input value.");

    if (params.volumeType == 0) {
        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::Checkbox("Linear Filtering", &params.volumeLinearFiltering)) {
            uii.setVolumeFilterMode(params.volumeLinearFiltering);
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Whether to turn on linear filtering of the volume data.");
    }
}

void UI::volume(KernelParams& params) {
    if (ImGui::CollapsingHeader("Volume", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TreePush("Volume");
        ImGui::Unindent();
        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::Combo("Type", &params.volumeType, volumeTypes, IM_ARRAYSIZE(volumeTypes))) {
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Type of the volume.");
        if (params.volumeType == 0) {
            ImGui::SetNextItemWidth(inputWidth);
            ImGui::InputText("Path", uii.volumeFilename, uii.maxStrLen);
            ImGui::SameLine(); helpMarker("Volume file path.");

            ImGui::SetNextItemWidth(inputWidth);
            int volSize[3] = { uii.volumeSize.x, uii.volumeSize.y, uii.volumeSize.z };
            if (ImGui::InputInt3("Size", volSize)) {
                uii.volumeSize.x = volSize[0];
                uii.volumeSize.y = volSize[1];
                uii.volumeSize.z = volSize[2];
            }
            ImGui::SameLine(); helpMarker("3-D(xyz) size of the volume data.");
            if (ImGui::Button("Load")) {
                uii.createNewVolume();
                uii.needResetBuffer = true;

                if (uii.isVolumeFileExist) {
                    setInfo("Load volume successfully.", ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
                } else {
                    setInfo("Volume file does not exist!", ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
                }
            }
        }

        // Volume Properties
        volumeProperties(params);

        ImGui::TreePop();
        ImGui::Indent();
    }
}

void UI::rayMarchingProperties(KernelParams& params) {
    if (ImGui::CollapsingHeader("Ray-marching Configuration", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TreePush("Ray-marching Configuration");
        ImGui::Unindent();
        ImGui::SeparatorText("Transfer Function");
        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::SliderFloat("Brightness", &params.brightness, 0.0f, 10.0f)) {
            params.brightness = clamp(params.brightness, 0.0f, 10.0f);
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Brightness factor.\nCTRL+click to input value.");

        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::DragFloat("Offset", &params.transferOffset, 0.001f, 0.0f)) {
            params.transferOffset = clamp(params.transferOffset, 0.0f, params.transferOffset);
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Offset of the transfer function.\nDouble click or CTRL+click to input value.");

        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::DragFloat("Scale", &params.transferScale, 0.001f, 0.0f)) {
            params.transferScale = clamp(params.transferScale, 0.0f, params.transferScale);
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Scale of the transfer function.\nDouble click or CTRL+click to input value.");

        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::Checkbox("Linear Filtering", &params.transferFunLinearFiltering)) {
            uii.setTransferFilterMode(params.transferFunLinearFiltering);
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Whether to turn on linear filtering of the transfer function.");

        ImGui::TreePop();
        ImGui::Indent();
    }
}

void UI::pathTracingProperties(KernelParams& params) {
    if (ImGui::CollapsingHeader("Path-tracing Configuration", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TreePush("Path-tracing Configuration");
        ImGui::Unindent();
        ImGui::SeparatorText("Background Type");
        if (ImGui::RadioButton("Environment Map", &params.environmentType, 1)) {
            uii.needResetBuffer = true;
        }
        if (params.environmentType == 1) {
            ImGui::SameLine();
            if (ImGui::Button("Load")) {
                uii.createNewEnvTex();
                uii.needResetBuffer = true;

                if (uii.isEnvTexFileExist) {
                    setInfo("Load environment map successfully.", ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
                } else {
                    setInfo("Environment map file does not exist!", ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
                }
            }

            ImGui::SetNextItemWidth(inputWidth);
            ImGui::InputText("Path", uii.envTexFileName, uii.maxStrLen);
            ImGui::SameLine(); helpMarker("Environment map file path.");
        }
        if (ImGui::RadioButton("Gradient Color", &params.environmentType, 0)) {
            uii.needResetBuffer = true;
        }
        if (params.environmentType == 0) {
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 gradient_size = ImVec2(ImGui::CalcItemWidth(), ImGui::GetFrameHeight());
            {
                ImVec2 p0 = ImGui::GetCursorScreenPos();
                ImVec2 p1 = ImVec2(p0.x + gradient_size.x, p0.y + gradient_size.y);
                ImU32 col_a = ImGui::GetColorU32(*(ImVec4*)&params.bgCol2);
                ImU32 col_b = ImGui::GetColorU32(*(ImVec4*)&params.bgCol1);
                draw_list->AddRectFilledMultiColor(p0, p1, col_a, col_b, col_b, col_a);
                ImGui::InvisibleButton("##gradient1", gradient_size);
            }
            if (ImGui::ColorEdit4("Bg Color 1", (float*)&params.bgCol2)) {
                uii.needResetBuffer = true;
            }
            if (ImGui::ColorEdit4("Bg Color 2", (float*)&params.bgCol1)) {
                uii.needResetBuffer = true;
            }
        }

        /* Transfer Function */
        ImGui::SeparatorText("Transfer Function");

        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::Checkbox("TF Grad", &params.transferGrad)) {
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Grad dimension in transfer function.");

        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::DragFloat("TF Grad Pow", &params.transferGradPow, 0.001f)) {
            params.transferGradPow = fmax(params.transferGradPow, 0.0f);
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Grad power in transfer function.");

        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::DragFloat("Brightness", &params.brightness, 0.001f, 0.0f)) {
            params.brightness = fmax(params.brightness, 0.0f);
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Brightness factor.\nCTRL+click to input value.");

        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::DragFloat("Offset", &params.transferOffset, 0.001f, -1.0f, 1.0f)) {
            params.transferOffset = clamp(params.transferOffset, -1.0f, 1.0f);
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Offset of the transfer function.\nDouble click or CTRL+click to input value.");

        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::DragFloat("Scale", &params.transferScale, 0.001f, 0.0f)) {
            params.transferScale = clamp(params.transferScale, 0.0f, params.transferScale);
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Scale of the transfer function.\nDouble click or CTRL+click to input value.");

        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::Checkbox("Linear Filtering", &params.transferFunLinearFiltering)) {
            uii.setTransferFilterMode(params.transferFunLinearFiltering);
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Whether to turn on linear filtering of the transfer function.");

        ImGui::SeparatorText("Phase Function");
        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::Checkbox("Enable gradient", &params.enableGrad)) {
            uii.needResetBuffer = true;
        }
        ImGui::SameLine(); helpMarker("Whether to turn on gradient control of parameter g");
        if (params.enableGrad) {
            ImGui::SetNextItemWidth(inputWidth);
            if (ImGui::SliderFloat("Grad Fac", &params.hgFac, 0.0f, 10.0f)) {
                params.hgFac = clamp(params.hgFac, 0.0f, 10.0f);
                uii.needResetBuffer = true;
            }
            ImGui::SameLine(); helpMarker("Factor of parameter g.");
        } else {
            ImGui::SetNextItemWidth(inputWidth);
            if (ImGui::SliderFloat("g", &params.g, -1.0f + EPSILON, 1.0f - EPSILON)) {
                params.g = clamp(params.g, -1.0f + EPSILON, 1.0f - EPSILON);
                uii.needResetBuffer = true;
            }
            ImGui::SameLine(); helpMarker("Parameter g of the Henyey-Greenstein phase function.\nCTRL+click to input value.");
        }

        ImGui::TreePop();
        ImGui::Indent();
    }
}

void UI::algorithmProperties(KernelParams& params) {
    if (params.method == 0) {
        // Ray-marching Properties
        rayMarchingProperties(params);
    } else if (params.method == 1 || params.method == 2) {
        // Path-tracing Properties
        pathTracingProperties(params);
    }
}

void UI::camera(KernelParams& params) {
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TreePush("Camera");
        ImGui::Unindent();
        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::DragFloat("Distance", &uii.cameraDist, 0.001f, 0.5f)) {
            uii.cameraDist = clamp(uii.cameraDist, 0.001f, uii.cameraDist);
            // uii.needResetBuffer = true;
            uii.cameraChanged = true;
        }
        ImGui::SameLine(); helpMarker("Distance from camera to center.\nDouble click or CTRL+click to input value.");

        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::DragFloat3("Center", (float*)&uii.cameraCenter, 0.001f)) {
            // uii.needResetBuffer = true;
            uii.cameraChanged = true;
        }
        ImGui::SameLine(); helpMarker("Camera center.\nDouble click or CTRL+click to input value.");

        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::DragFloat3("Rotation", (float*)&uii.cameraRotation, 0.001f)) {
            uii.cameraRotation.x = clamp(uii.cameraRotation.x, -89.9999f, 89.9999f);
            // uii.needResetBuffer = true;
            uii.cameraChanged = true;
        }
        ImGui::SameLine(); helpMarker("Camera rotation.\nDouble click or CTRL+click to input value.");

        ImGui::TreePop();
        ImGui::Indent();
    }
}

void UI::result(KernelParams& params) {
    if (ImGui::CollapsingHeader("Rendering Result", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TreePush("Rendering Result");
        ImGui::Unindent();
        if (ImGui::Checkbox("Gamma Correction", &params.gammaCorrection)) {}
        ImGui::SetNextItemWidth(inputWidth);
        if (ImGui::Combo("Tone Mapping", &params.toneMappingMethod, toneMappingMethods, IM_ARRAYSIZE(toneMappingMethods))) {}
        ImGui::SetNextItemWidth(inputWidth);
        ImGui::InputText("Path", uii.saveImgFileName, uii.maxStrLen);
        ImGui::SameLine(); helpMarker("Saving image file path.");
        ImGui::SameLine();
        if (ImGui::Button("Save")) {
            uii.conv = !(endWith(uii.saveImgFileName, ".tif") || endWith(uii.saveImgFileName, ".tiff") ||
                endWith(uii.saveImgFileName, ".exr") || endWith(uii.saveImgFileName, ".hdr"));
            if (uii.saveImage()) {
                setInfo("Save image successfully.", ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
            } else {
                setInfo("Failed to save image!", ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
            }
        }
        ImGui::TreePop();
        ImGui::Indent();
    }
}

void UI::statusBar() {
    ImGui::SetNextWindowPos(ImVec2((float)(d_width + margin), (float)(w_height - statusBarHeight)));
    ImGui::SetNextWindowSize(ImVec2((float)(w_width - d_width - margin), (float)statusBarHeight));
    ImGui::Begin("Status", 0, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);
    {
        ImGui::TextColored(infoColor, infoStr);
    }
    ImGui::End();
}

void UI::handleEvent() {
    ImGuiIO& io = ImGui::GetIO();

    if (imageFocused) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            isLeftDown = true;
            lastMousePos = ImGui::GetMousePos();
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            isLeftDown = false;
        }
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
            isMiddleDown = true;
            lastMousePos = ImGui::GetMousePos();
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
            isMiddleDown = false;
        }

        if (isLeftDown) {
            ImVec2 newMousePos = ImGui::GetMousePos();
            float dx = newMousePos.x - lastMousePos.x, dy = newMousePos.y - lastMousePos.y;
            if (dx != 0 || dy != 0) {
                uii.cameraRotation.y -= dx * xRotSenst;
                uii.cameraRotation.x -= dy * yRotSenst;
                uii.cameraRotation.x = clamp(uii.cameraRotation.x, -89.9999f, 89.9999f);
                // uii.needResetBuffer = true;
                uii.cameraChanged = true;
            }
            lastMousePos = newMousePos;
        }
        if (isMiddleDown) {
            ImVec2 newMousePos = ImGui::GetMousePos();
            float dx = newMousePos.x - lastMousePos.x, dy = newMousePos.y - lastMousePos.y;
            if (dx != 0 || dy != 0) {
                uii.cameraCenter += -dx * xMoveSenst * uii.u + dy * yMoveSenst * uii.v;
                // uii.needResetBuffer = true;
                uii.cameraChanged = true;
            }
            lastMousePos = newMousePos;
        }
        if (fabsf(io.MouseWheel) > 0.0f) {
            uii.cameraDist -= io.MouseWheel * scrollSenst;
            uii.cameraDist = clamp(uii.cameraDist, 0.5f, uii.cameraDist);
            // uii.needResetBuffer = true;
            uii.cameraChanged = true;
        }

        if (ImGui::IsKeyPressed(ImGuiKey_C, false)) {
            uii.defaultCameraDist = uii.cameraDist;
            uii.defaultCameraCenter = uii.cameraCenter;
            uii.defaultCameraRotation = uii.cameraRotation;
        }

        if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
            uii.resetCamera();
        }

        if (ImGui::IsKeyPressed(ImGuiKey_S, false)) {
            uii.conv = !(endWith(uii.saveImgFileName, ".tif") || endWith(uii.saveImgFileName, ".tiff") ||
                endWith(uii.saveImgFileName, ".exr") || endWith(uii.saveImgFileName, ".hdr"));
            if (uii.saveImage()) {
                setInfo("Save image successfully.", ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
            } else {
                setInfo("Failed to save image!", ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
            }
        }

        if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
            if (uii.continuousSave) {
                setInfo("End to save continuously.", ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
                uii.continuousSave = false;
            } else {
                setInfo("Start to save continuously.", ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
                uii.continuousSave = true;
                uii.conv = !(endWith(uii.saveImgFileName, ".tif") || endWith(uii.saveImgFileName, ".tiff") ||
                    endWith(uii.saveImgFileName, ".exr") || endWith(uii.saveImgFileName, ".hdr"));
            }
        }

        if (ImGui::IsKeyPressed(ImGuiKey_X, false)) {
            if (uii.continuousSave) {
                setInfo("End to save continuously.", ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
                uii.continuousSave = false;
            } else {
                setInfo("Start to save continuously.", ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
                uii.saveFrameCount = 0;
                uii.needResetBuffer = true;
                uii.continuousSave = true;
                uii.conv = !(endWith(uii.saveImgFileName, ".tif") || endWith(uii.saveImgFileName, ".tiff") ||
                    endWith(uii.saveImgFileName, ".exr") || endWith(uii.saveImgFileName, ".hdr"));
            }
        }
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void UI::display(GLuint tex, KernelParams& params) {
    // New frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    // Set style
    setStyle();
    // Show GUI
    {
        sceneWindow(tex);
        consoleWindow(params);
        statusBar();
        // ImGui::ShowDemoWindow();
    }
    // Handle Event
    handleEvent();
    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
