#ifndef PROJECTION_H
#define PROJECTION_H

#include <iostream>
#include <unistd.h>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>

#define top 119        //w
#define bottom 115     //s
#define left 97        //a
#define right 100      //d
#define scale_up 120   //x
#define scale_down 122 //z
#define esc 27         //esc
#define ratio 0.2      // w and h ratio
#define minSizePorce 0.1
#define low_low 108 // l minus lower
#define low_high 107 //  + lower
#define high_high 103 // h + higer
#define high_low 104 // h minus higer
class projection
{
private:
    int cap_height;
    int cap_width;
    cv::Mat black;
    cv::Mat black_or;


    cv::cuda::GpuMat resizeGpu;      //SAVE RESIZE FRAME BUT GPU
    cv::cuda::GpuMat blackGpu;       //SAVE RESIZE FRAME BUT GPU
    cv::cuda::GpuMat black_orGpu;    //SAVE RESIZE FRAME BUT GPU
    cv::cuda::GpuMat frameCuda;      //SAVE FRAME FRAME BUT GPU
    cv::cuda::GpuMat imageCanny_gpu; //SAVE CANNY LINES FRAME BUT GPU
    cv::cuda::GpuMat gray_gpu;       //SAVE GRAYSCALE FRAME BUT GPU


    std::vector<std::string> split(const std::string& str, const std::string& delim);

public:
    uint16_t new_x,new_y;
    int frame_h, frame_w, new_frame_h, new_frame_w, black_h, black_w;
    //CANNY TRHESHOLDS
    uint16_t lowerThr;
    uint16_t higherThr;


    projection();
    ~projection();
    cv::Mat projectionFrame(cv::Mat frame);
    void setValues(int cap_height, int cap_width, int black_w, int black_h);
    void saveConfig();
    void uploadConfig();
    /*
    static void* getKey_wrapper(void* object){
        reinterpret_cast<projection*>(object)->getKey();
        return 0;
    }
    void getKey();
    */
};

#endif