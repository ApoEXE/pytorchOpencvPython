#include "border.h"


border::border(/* args */)
{
}

border::~border()
{
}


void border(cv::Mat frame,int height,int width)
{

    frame_h = height;
    frame_w = width;
    new_frame_h = frame_h;
    new_frame_w = frame_w;
    cv::Mat black = cv::imread(lookPath);
    cv::Mat black_or = cv::imread(lookPath);
    black_h = black.rows;
    black_w = black.cols;

    cv::cuda::GpuMat frameGpu;
    cv::cuda::GpuMat resizeGpu;
    cv::cuda::GpuMat blackGpu;
    cv::cuda::GpuMat black_orGpu;
    blackGpu.upload(black);
    black_orGpu.upload(black_or);
    cv::cuda::GpuMat frameCuda;
    cv::cuda::GpuMat imageCanny_gpu;
    cv::cuda::GpuMat gray_gpu;
    cv::Mat downFrame;

    //CUDA canny

    auto start = std::chrono::high_resolution_clock::now();
    frameCuda.upload(frame); //paso de CPU a GPU
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
    resizeGpu.copyTo(blackGpu(cv::Rect(x, y, resizeGpu.cols, resizeGpu.rows)));
    blackGpu.download(downFrame);

    //frameCuda.download(downFrame);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = finish - start;
    std::cout << "GPU CUDA Elapsed Time: " << elapsed_time.count() * 1000 << " msecs"
              << "\n"
              << std::endl;
}