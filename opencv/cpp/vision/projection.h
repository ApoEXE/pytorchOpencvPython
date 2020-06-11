#ifndef PROJECTION_H
#define PROJECTION_H
#include "include.h"
class projection
{
private:
    int cap_height;
    int cap_width;
    cv::Mat black;
    cv::Mat black_or;
    int frame_h, frame_w, new_frame_h, new_frame_w, black_h, black_w;
    //CANNY TRHESHOLDS
    uint8_t lowerThr;
    uint8_t higherThr;
    uint8_t new_x, new_y;
    cv::cuda::GpuMat resizeGpu;      //SAVE RESIZE FRAME BUT GPU
    cv::cuda::GpuMat blackGpu;       //SAVE RESIZE FRAME BUT GPU
    cv::cuda::GpuMat black_orGpu;    //SAVE RESIZE FRAME BUT GPU
    cv::cuda::GpuMat frameCuda;      //SAVE FRAME FRAME BUT GPU
    cv::cuda::GpuMat imageCanny_gpu; //SAVE CANNY LINES FRAME BUT GPU
    cv::cuda::GpuMat gray_gpu;       //SAVE GRAYSCALE FRAME BUT GPU

public:
    projection();
    ~projection();
    cv::Mat projectionFrame(cv::Mat frame);
    void setValues(int cap_height, int cap_width);
    void updateHxW(int new_frame_h, int new_frame_w);
    void updateXY(int new_x, int new_y);
};

#endif