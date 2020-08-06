#ifndef DNN_OPENCV_H
#define DNN_OPENCV_H
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

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>


class dnn_opencv
{
private:
    /* data */
    cv::dnn::Net net; //load model here
    const float confidence_threshold = 0.5;
    const float nms_threshold = 0.4;
    const int num_classes = 6;

    std::string model = "yolov3";

    std::string classes;
    std::string weights;
    std::string conf;
    cv::Size size_model = cv::Size(320,320);

    // detection
    cv::Mat blob;
    std::vector<cv::Mat> detections;
    std::vector<cv::Rect> boxes;
    std::vector<int> class_id;
    std::vector<float> scores;
    std::vector<cv::String> output_names;
    std::vector<std::string> class_names;
    std::vector<int> indices;
    float inference_fps;
    double total_start;
    
    // Box propeties
    cv::Scalar color_bounding_box = cv::Scalar(255, 0, 0);
    cv::Scalar color_text_box = cv::Scalar(0, 0, 0);
    cv::Scalar color_fill_box = cv::Scalar(0, 0, 0);
    cv::Scalar color_edges = cv::Scalar(255, 0, 0);
    int type_text_box = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double scale_text_box = 1;
    
    //FPS - stats
    std::string stats; 
    cv::Scalar color_text_stats = cv::Scalar(0, 0, 0);
    cv::Scalar color_fill_box_stats = cv::Scalar(255, 255, 255);
    int type_text_stats = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double scale_text_stats = 0.55;

private:
    void border(cv::Mat &frame, cv::Mat &mask);

public:
    dnn_opencv(/* args */);
    ~dnn_opencv();
    void load_model(std::string conf, std::string weights, std::string classes);
    void inference(cv::Mat &frame);
    void drawDetection(cv::Mat &frame, bool fill_detection);
    void drawFPS(cv::Mat &frame);
    void getDetections(cv::Mat &frame);
    void processFrame(cv::Mat &frame, bool fill_detection);
};

#endif