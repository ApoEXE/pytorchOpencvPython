//g++ main.cpp dnn_opencv.cpp  -o test  `pkg-config opencv --cflags --libs`
#include <opencv2/highgui.hpp>
#include "dnn_opencv.h"
int main(int argc, char const *argv[])
{
    dnn_opencv dnn = dnn_opencv();
    dnn.load_model("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel", "classes");
    cv::VideoCapture cap = cv::VideoCapture(0, cv::CAP_V4L);
    cv::Mat frame;
    while (cap.isOpened())
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        cv::resize(frame,frame,cv::Size(300,300));
        cv::Mat detectionMat = dnn.inference(frame);
       // dnn.getDetections(frame);
        frame = dnn.drawDetection(detectionMat,frame);
        frame = dnn.drawFPS(frame); //Draw FPS

        cv::imshow("detections", frame);
        if (cv::waitKey(1) > 0)
        {
            break;
        }
    }
    cap.release();
    return 0;
}
