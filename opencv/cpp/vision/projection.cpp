#include "projection.h"

projection::projection()
{
    new_x = 0;
    new_y = 0;

    //CANNY TRHESHOLDS
    lowerThr = 50;
    higherThr = 200;
}

projection::~projection()
{
}

void projection::setValues(int cap_height, int cap_width, int lblack_w, int lblack_h)
{
    frame_h = cap_height; //ORIGINAL H
    frame_w = cap_width;  //ORIGINAL W

    new_frame_h = frame_h; //START WITH ORIGINAL CAMERA H
    new_frame_w = frame_w; //START WITH ORIGINAL CAMERA W
    //LOAD A BLACK WALL PAPER AS BACKGROUND
    lblack_w = 1920; //I have preseted
    cv::Mat image(lblack_h, lblack_w, CV_8UC3, cv::Scalar(0, 0, 0));
    black = image;
    black_or = image;
    //SAVE BLACK PARAMETERS H W
    black_h = lblack_h; //BLACK H LIMIT FOR MOVING THE FRAME AROUND
    black_w = lblack_w; //BLACK W LIMIT FOR MOVING THE FRAME AROUND
    //printf("Black: width %d, height %d \n", black_w, black_h);
    //UPLOAD FRAME TO GPU
    blackGpu.upload(black);
    black_orGpu.upload(black_or);
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
    /*
    std::cout << "GPU CUDA Elapsed Time: " << elapsed_time.count() * 1000 << " msecs"
              << "\n"
              << std::endl;
              */
    return downFrame;
}

void projection::saveConfig()
{
    std::ofstream file;
    file.open("screen.conf", std::ios::trunc | std::ios::in);
    if (file.is_open())
    {

        file << "x_position," << new_x << "\n";
        file << "y_position," << new_y << "\n";
        file << "width," << new_frame_w << "\n";
        file << "hight," << new_frame_h << "\n";
        //printf("lower SAVECONFIG %d",lowerThr);
        file << "lowThr," << lowerThr << "\n";
        //printf("  HIGH SAVECONFIG %d \n",higherThr);
        file << "highThr," << higherThr;
        file.close();
    }
}

void projection::uploadConfig()
{
    std::fstream file;

    file.open("screen.conf", std::ios::in | std::ios::out);

    if (file.is_open())
    {

        std::string conf;
        while (!file.eof())
        {
            file >> conf;
            std::vector<std::string> token = split(conf, ",");
            //std::cout << token[0] << std::endl;
            //std::cout << token[1] << std::endl;
            if (strcmp(token[0].c_str(), "x_position") == 0)
            {
                new_x = std::stoi(token[1]);
            }
            if (strcmp(token[0].c_str(), "y_position") == 0)
            {
                new_y = std::stoi(token[1]);
            }
            if (strcmp(token[0].c_str(), "width") == 0)
            {
                new_frame_w = std::stoi(token[1]);
            }
            if (strcmp(token[0].c_str(), "hight") == 0)
            {
                new_frame_h = std::stoi(token[1]);
            }
            if (strcmp(token[0].c_str(), "lowThr") == 0)
            {
                lowerThr = std::stoi(token[1]);
            }
            if (strcmp(token[0].c_str(), "highThr") == 0)
            {
                higherThr = std::stoi(token[1]);
            }
        }
    }
    file.close();
}

std::vector<std::string> projection::split(const std::string &str, const std::string &delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos)
            pos = str.length();
        std::string token = str.substr(prev, pos - prev);
        if (!token.empty())
            tokens.push_back(token);
        prev = pos + delim.length();
    } while (pos < str.length() && prev < str.length());
    return tokens;
}