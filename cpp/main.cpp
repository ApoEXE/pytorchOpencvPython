#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

#include "C++/chooseser.h"


const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.4;
const cv::Scalar meanVal(104.0, 177.0, 123.0);
float confidence;

const std::string caffeConfigFile = "deploy.prototxt.txt";
const std::string caffeWeightFile = "res10_300x300_ssd_iter_140000.caffemodel";
const std::string embeddedModel = "openface_nn4.small2.v1.t7";
const std::string recognizerPickle = "output/recognizer.pickle";
const std::string labelPickle = "output/le.pickle";



/
int main(int argc, const char **argv)
{
    
    cv::dnn::dnn4_v20200310::Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    cv::dnn::dnn4_v20200310::Net embedder = cv::dnn::readNetFromTorch(embeddedModel);
    Val recognizer;
    LoadValFromFile(recognizerPickle,recognizer,OC::OC::SERIALIZE_P0); 
    cv::VideoCapture source;

    source.open(0, cv::CAP_V4L);

    cv::Mat frame;

    double tt_opencvDNN = 0;
    double fpsOpencvDNN = 0;
    while (1)
    {
        source >> frame;
        if (frame.empty())
            break;
        double tim = cv::getTickCount();
        //detectFaceOpenCVDNN(net, frame);
        auto tt_opencvDNN = ((double)cv::getTickCount() - tim) / cv::getTickFrequency();
        auto fpsOpencvDNN = 1 / tt_opencvDNN;
        cv::putText(frame, cv::format("FPS = %.2f Confidence: %0.2f", fpsOpencvDNN, confidence), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        cv::imshow("OpenCV - DNN Face Detection", frame);
        int k = cv::waitKey(1);
    }
}
