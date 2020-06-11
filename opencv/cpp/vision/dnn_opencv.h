#ifndef DNN_OPENCV_H
#define DNN_OPENCV_H
#include "include.h"

class dnn_opencv
{
private:
    /* data */
    cv::dnn::Net net; //load model here
    const float confidence_threshold = 0.5;
    const float nms_threshold = 0.4;
    const int num_classes = 3;
    // colors for bounding boxes
    const cv::Scalar colors[4] = {
        {0, 255, 255},
        {255, 255, 0},
        {0, 255, 0},
        {255, 0, 0}};
    const int num_colors = sizeof(colors) / sizeof(colors[0]);

public:
    dnn_opencv(/* args */);
    ~dnn_opencv();
    void load_model(std::string conf, std::string weights, std::string classes);
    cv::Mat inference(cv::Mat frame);
};

#endif