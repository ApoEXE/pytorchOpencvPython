#include "dnn_opencv.h"
#include <bits/stdc++.h>
using namespace std;

dnn_opencv::dnn_opencv(/* args */) { }
dnn_opencv::~dnn_opencv() { }

double getTime(){
    return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();
}

void dnn_opencv::load_model(std::string lconf, std::string lweights, std::string lclasses){
    conf = lconf;
    weights = lweights;
    classes = lclasses;
    net = cv::dnn::readNetFromDarknet(conf, weights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::ifstream class_file(classes);
    if (!class_file){
        std::cerr << "failed to open " << classes << "\n";
    }
    class_names.assign(std::istream_iterator<std::string>(class_file), {});

    output_names = net.getUnconnectedOutLayersNames();
    printf("passed loading model \n");
}

void dnn_opencv::inference(cv::Mat &frame){
    total_start = getTime();
    cv::dnn::blobFromImage(frame, blob, 0.00392, size_model, cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);

    double dnn_start = getTime();
    net.forward(detections, output_names);
    double dnn_end = getTime();
    inference_fps = 1000.0 / (dnn_end - dnn_start);
}

void dnn_opencv::getDetections(cv::Mat &frame){
    boxes.clear();
    class_id.clear();
    scores.clear();
    for (auto &output : detections){
        const auto num_boxes = output.rows;
        for (size_t i = 0; i < num_boxes; i++){
            auto itr = std::max_element(output.ptr<float>(i, 5), output.ptr<float>(i, 5 + num_classes));
            auto confidence = *itr;
            auto classid = itr - output.ptr<float>(i, 5);
            if (confidence >= confidence_threshold){
                auto x = output.at<float>(i, 0) * frame.cols;
                auto y = output.at<float>(i, 1) * frame.rows;
                auto width = output.at<float>(i, 2) * frame.cols;
                auto height = output.at<float>(i, 3) * frame.rows;
                cv::Rect rect(x - width / 2, y - height / 2, width, height);

                boxes.push_back(rect);
                class_id.push_back(classid);
                scores.push_back(confidence);
            }
        }
    }
}

void dnn_opencv::drawDetection(cv::Mat &frame, bool fill_detection){
    cv::dnn::NMSBoxes(boxes, scores, 0.0, nms_threshold, indices);
    
    int baseline = 0;
    for (size_t i = 0; i < indices.size(); ++i){
        auto idx = indices[i];
        const auto &rect = boxes[idx];

        std::ostringstream label_ss;
        label_ss << class_names[class_id[idx]];
        string label = label_ss.str();
        
        cv::Point upper_left_box = cv::Point(rect.x, rect.y);
        cv::Point bottom_right_box = cv::Point(rect.x + rect.width, rect.y + rect.height);
        
        cv::Size size_text = cv::getTextSize(label.c_str(), type_text_box, scale_text_box, 0.5, &baseline);
        cv::Point text = cv::Point(rect.x, rect.y + rect.height + size_text.height / 2 + 5);
        cv::Point upper_left_text = cv::Point(rect.x, rect.y + rect.height);
        cv::Point bottom_right_text = cv::Point(rect.x + size_text.width, rect.y + rect.height + size_text.height + 5);
        
        if(fill_detection){
            cv::Mat mask(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            cv::Mat mask1(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            cv::rectangle(mask1, upper_left_box, bottom_right_box, cv::Scalar(255, 255, 255), cv::FILLED);
            cv::bitwise_or(frame, mask, mask);
            border(frame,mask1);
            cv::rectangle(mask, upper_left_box, bottom_right_box, color_fill_box, cv::FILLED);
            float lalfa = 0.3;
            cv::addWeighted(mask, lalfa, frame, 1 - lalfa, 0, frame);
        }
        
        //BOX DETECTION
        cv::rectangle(frame, upper_left_box, bottom_right_box, color_bounding_box, 1);
        //BACKGROUIND FOR TEXT
        cv::rectangle(frame, upper_left_text, bottom_right_text, color_bounding_box, cv::FILLED);
        cv::putText(frame, label, text, type_text_box, scale_text_box, color_text_box);
    }
}

void dnn_opencv::drawFPS(cv::Mat &frame){
    double total_end = getTime();
    double total_fps = 1000.0 / (total_end - total_start);
    std::ostringstream stats_ss;
    stats_ss << std::fixed << std::setprecision(2);
    stats_ss << model << " Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
    stats = stats_ss.str();
    int baseline = 0;
    cv::Size size_text = cv::getTextSize(stats.c_str(), type_text_stats, scale_text_stats, 1, &baseline);
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(size_text.width, size_text.height + size_text.height / 2), color_fill_box_stats, cv::FILLED);
    cv::putText(frame, stats, cv::Point(0, size_text.height), type_text_stats, scale_text_stats, color_text_stats);
}

void dnn_opencv::border(cv::Mat &frame, cv::Mat &mask){
    int lowerThr = 50;
    int higherThr = 200;
    cv::cuda::GpuMat frameGpu, gray_gpu, imageCanny_gpu, mask_gpu;
    frameGpu.upload(frame);
    mask_gpu.upload(mask);
    //cv::cuda::cvtColor(frameGpu,frameGpu,cv::COLOR_RGB2GRAY);
    //cv::cuda::cvtColor(frameGpu,frameGpu,cv::COLOR_GRAY2RGB);
    cv::cuda::cvtColor(frameGpu, gray_gpu, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edg = cv::cuda::createCannyEdgeDetector(lowerThr, higherThr, 3, true);
    canny_edg->detect(gray_gpu, imageCanny_gpu);
    //DELETE borders from MASK
    if (!mask_gpu.empty()) {
        //std::cout << "Entry 14" << std::endl;
        //std::cout << "Entry 18 imageCanny_gpu " << imageCanny_gpu.size() << "mask1_gpu " << mask1_gpu.size() << std::endl;
        if (mask_gpu.channels() == 3){
            cv::cuda::cvtColor(mask_gpu, mask_gpu, cv::COLOR_RGB2GRAY);
        }
        //std::cout << "Entry 19 imageCanny_gpu " << imageCanny_gpu.channels() << " mask1_gpu " << mask1_gpu.channels() << std::endl;
        cv::cuda::GpuMat tempo; //mask with cross out section
        cv::cuda::resize(mask_gpu, tempo, imageCanny_gpu.size());

        cv::cuda::bitwise_and(imageCanny_gpu, tempo, imageCanny_gpu);
        //std::cout << "Entry 4" << std::endl;
        //std::cout << "Entry 15" << std::endl;
    }
    frameGpu.setTo(color_edges, imageCanny_gpu);
    frameGpu.download(frame);
}

void dnn_opencv::processFrame(cv::Mat &frame, bool fill_detection){
    inference(frame);
    getDetections(frame);
    drawDetection(frame,fill_detection);
    drawFPS(frame);
}