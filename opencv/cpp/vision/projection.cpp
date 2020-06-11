#include "projection.h"

projection::projection()
{
    new_x = 0;
    new_y = 0;

    //CANNY TRHESHOLDS
    lowerThr = 30;
    higherThr = lowerThr * 3;
}

projection::~projection()
{
}

void projection::setValues(int cap_height, int cap_width)
{
    frame_h = cap_height; //ORIGINAL H
    frame_w = cap_width;  //ORIGINAL W

    new_frame_h = frame_h; //START WITH ORIGINAL CAMERA H
    new_frame_w = frame_w; //START WITH ORIGINAL CAMERA W
    //LOAD A BLACK WALL PAPER AS BACKGROUND
    black = cv::imread(lookPath);
    black_or = cv::imread(lookPath);
    //SAVE BLACK PARAMETERS H W
    black_h = black.rows; //BLACK H LIMIT FOR MOVING THE FRAME AROUND
    black_w = black.cols; //BLACK W LIMIT FOR MOVING THE FRAME AROUND
    //UPLOAD FRAME TO GPU
    blackGpu.upload(black);
    black_orGpu.upload(black_or);
}
void projection::updateHxW(int lnew_frame_h, int lnew_frame_w)
{
    new_frame_h = lnew_frame_h;
    new_frame_w = lnew_frame_w;
}
void projection::updateXY(int lnew_x, int lnew_y)
{
    new_x = lnew_x;
    new_y = lnew_y;
}

cv::Mat projection::projectionFrame(cv::Mat lframe)
{
    cv::Mat downFrame;
    //CUDA canny
    auto start = std::chrono::high_resolution_clock::now();
    frameCuda.upload(lframe); //paso de CPU a GPU
    //GPU
    cv::cuda::cvtColor(frameCuda, gray_gpu, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edg = cv::cuda::createCannyEdgeDetector(lowerThr, higherThr, 3, true);
    canny_edg->detect(gray_gpu, imageCanny_gpu);
    frameCuda.setTo(cv::Scalar(0, 255, 0), imageCanny_gpu);
    //DO CONFIG OF SCREEN BEFORE SHOWING
    frameCuda.copyTo(resizeGpu);
    //std::cout << "CONFIG----------------------" << new_frame_w << " " << new_frame_h;
    cv::cuda::resize(resizeGpu, resizeGpu, cv::Size(new_frame_w, new_frame_h));
    black_orGpu.copyTo(blackGpu);
    resizeGpu.copyTo(blackGpu(cv::Rect(new_x, new_y, resizeGpu.cols, resizeGpu.rows)));
    blackGpu.download(downFrame);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = finish - start;
    std::cout << "GPU CUDA Elapsed Time: " << elapsed_time.count() * 1000 << " msecs"
              << "\n"
              << std::endl;
    return downFrame;
}