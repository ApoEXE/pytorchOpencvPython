// For more information and tips to improve inference FPS, visit https://github.com/opencv/opencv/pull/14827#issuecomment-568156546
//g++ dnn_opencv.cpp -o test -pthread -lncurses `pkg-config opencv --cflags --libs`
//./test 1 ~/wsl/images_videos/prueba.mp4

#include "dnn_opencv.h"
dnn_opencv::dnn_opencv(/* args */)
{
}

dnn_opencv::~dnn_opencv()
{
}
void dnn_opencv::load_model(std::string lconf, std::string lweights, std::string lclasses)
{
    conf = lconf;
    weights = lweights;
    classes = lclasses;
    net = cv::dnn::DetectionModel(conf, weights);
    //net = cv::dnn::readNet(conf, weights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    printf("passed loading model \n");
}

cv::Mat dnn_opencv::inference(cv::Mat frame)
{

    std::vector<std::string> class_names;
    {
        std::ifstream class_file(classes);
        if (!class_file)
        {
            std::cerr << "failed to open " << classes << "\n";
        }
        class_names.assign(std::istream_iterator<std::string>(class_file), {});
    }

    std::vector<std::string> output_names = net.getUnconnectedOutLayersNames();
    cv::Mat blob;
    std::vector<cv::Mat> detections;

    if (frame.empty())
    {
        cv::waitKey();
        std::cout << "missing Frame in dnn_opencv" << std::endl;
    }
    else
    {

        auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
        net.setInput(blob);

        auto dnn_start = std::chrono::steady_clock::now();
        net.forward(detections, output_names);
        auto dnn_end = std::chrono::steady_clock::now();

        std::vector<cv::Rect> boxes;
        std::vector<int> class_id;
        std::vector<float> scores;

        for (auto &output : detections)
        {
            const auto num_boxes = output.rows;
            for (size_t i = 0; i < num_boxes; i++)
            {
                auto itr = std::max_element(output.ptr<float>(i, 5), output.ptr<float>(i, 5 + num_classes));
                auto confidence = *itr;
                auto classid = itr - output.ptr<float>(i, 5);
                if (confidence >= confidence_threshold)
                {
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

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, 0.0, nms_threshold, indices);

        for (size_t i = 0; i < indices.size(); ++i)
        {
            const auto color = colors[i % num_colors];

            auto idx = indices[i];
            const auto &rect = boxes[idx];
            cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

            std::ostringstream label_ss;
            label_ss << class_names[class_id[idx]] << ": " << std::fixed << std::setprecision(2) << scores[idx];
            auto label = label_ss.str();

            int baseline;
            auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
            cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
        }

        auto total_end = std::chrono::steady_clock::now();

        float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
        float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        std::ostringstream stats_ss;
        stats_ss << std::fixed << std::setprecision(2);
        stats_ss << model << " Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();

        int baseline;
        auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));
        return frame;
    }
}
