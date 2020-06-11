//g++ main.cpp dnn_opencv.cpp projection.cpp -o test -pthread -lncurses `pkg-config opencv --cflags --libs`
//./test 1 ~/wsl/images_videos/prueba.mp4
#include "include.h"
#include "dnn_opencv.h"
#include "projection.h"

std::string model = "yolov3";

std::string absPath_weights = "/home/jav/wsl/weights/";
std::string absPath_img = "/home/jav/wsl/images_videos/";
std::string classes = absPath_weights + "default/coco.names";
std::string weights = absPath_weights + "default/yolov3.weights";
std::string conf = absPath_weights + "default/yolov3.cfg";
//std::string videoPath = absPath_img+"busystreet.mp4";
//std::string videoPath = absPath_img+"thermalDriving.mp4";
std::string videoPath = absPath_img + "prueba.mp4";
std::string lookPath = absPath_img + "blackest.jpg";

bool FileExists(const std::string &Filename)
{
    bool check = access(Filename.c_str(), 0) == 0;
    return check;
}

void pathInit()
{
    //custom
    conf = absPath_weights + "custom/yolov3-obj.cfg";

    weights = absPath_weights + "custom/yolov3-obj_final.weights";
    //weights = "custom/thermal8.weights";

    classes = absPath_weights + "custom/obj.names";
    //model = "yolov3 cus";

    if (!FileExists(weights) || !FileExists(conf) || !FileExists(videoPath))
    {
        std::cerr << "check which one is missing: " << weights << " " << conf << " " << videoPath << "\n";
        return exit(0);
    }
    else
    {
        std::cout << weights << "\n " << conf << " \n"
                  << videoPath << "\n";
    }
}
int main(int argc, char const *argv[])
{
    //CHECK CONF WEIGHTS CLASSES PATH IF THEY EXIST
    pathInit();
    //OBJECT FOR INFERENCE CLASS AND LOADING THE MODEL
    dnn_opencv dnn = dnn_opencv();
    dnn.load_model(conf, weights, classes);

    // LETS GET THE FRAME 0 FOR INPUT:2 FOR THERMAL INPUT:0 FOR WEBCAM INPUT:1 FOR VIDEO.MP4 FOLLOWED BY THE PATH
    cv::VideoCapture cap;
    cv::Mat frame;

    int input = std::stoi(argv[1]);
    if (argc >= 2 && input == 2)
    {
        printf("Thermal ");
        cap.open(2, cv::CAP_V4L);
    }
    if (argc >= 2 && input == 0)
    {
        printf("webcam ");
        cap.open(0, cv::CAP_V4L);
    }
    if (argc >= 3 && input == 1)
    {
        printf("video Path ");
        cap.open(argv[2], cv::CAP_FFMPEG);
    }

    //PROJECTION OBJ
    projection pro = projection();
    pro.setValues(cap.get(cv::CAP_PROP_FRAME_HEIGHT), cap.get(cv::CAP_PROP_FRAME_WIDTH));

    //VIDEO CAPTURE
    cv::Mat yoloFrame, projectFrame;
    while (cap.isOpened() > 0)
    {
        cap >> frame;
        frame.copyTo(yoloFrame);
        if (frame.empty())
        {
            cv::waitKey();
            break;
        }
        //GET DETECTION BOX ON FRAME
        yoloFrame = dnn.inference(yoloFrame);
        projectFrame = pro.projectionFrame(yoloFrame);

        cv::namedWindow("output");
        cv::imshow("output", projectFrame);
        cv::waitKey(1);
    }
    return 0;
}
