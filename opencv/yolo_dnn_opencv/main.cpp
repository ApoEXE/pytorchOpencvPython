//g++ main.cpp dnn_opencv.cpp  -o test  `pkg-config opencv --cflags --libs`
#include <opencv2/highgui.hpp>
#include "dnn_opencv.h"
#include <bits/stdc++.h>
using namespace std;


std::string classes = "customYolov3-tiny-6classes-320x320/obj.names";
std::string weights = "customYolov3-tiny-6classes-320x320/yolov3-tiny-obj-320x320-c-1.weights";
std::string conf = "customYolov3-tiny-6classes-320x320/yolov3-tiny-obj-320x320-c-1.cfg";


dnn_opencv dnn = dnn_opencv();
int main(int argc, char const *argv[]){
    dnn.load_model(conf, weights, classes);
    cv::VideoCapture cap("../video2.mp4");
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::VideoWriter outVideo("out.avi", fourcc, 30, cv::Size(frame_width,frame_height), true);
    

    cv::Mat frame;
    while (cap.isOpened()){
        cap >> frame;
        if (frame.empty()){
            break;
        }
        
        cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
        dnn.processFrame(frame,false);
    
        
        outVideo.write(frame);
        cv::imshow("detections", frame);
        if (cv::waitKey(1) > 0){
            break;
        }
    }
    cap.release();
    outVideo.release();
    return 0;
}