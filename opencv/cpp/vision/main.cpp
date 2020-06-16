//g++ main.cpp dnn_opencv.cpp projection.cpp -o test -pthread -lncurses -lX11 `pkg-config opencv --cflags --libs`
//./test 1 ~/wsl/images_videos/video1.mp4
#include "dnn_opencv.h"
#include "projection.h"
#include <X11/Xlib.h>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include "conio.h"

projection pro = projection();
dnn_opencv dnn = dnn_opencv();

int screen_num;    //number of display
int screen_width;  //width of display
int screen_height; //height of display
Window root_window;
unsigned long white_pixel;
unsigned long black_pixel;

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
bool closeKeyGet = false;

bool FileExists(const std::string &Filename);
cv::VideoCapture arguments(int argc, char const *argv[]);
int getDisplayInfo();
void *getKey(void *t_id);
//********************MAIN**************************************************
int main(int argc, char const *argv[])
{

    cv::VideoCapture cap = arguments(argc, argv); //check if files exist, else exit program

    //GET SCREEN DISPLAY INFO
    getDisplayInfo();
    //OBJECT FOR INFERENCE CLASS AND LOADING THE MODEL
    dnn.load_model(conf, weights, classes);
    //PROJECTION OBJ

    pro.setValues(cap.get(cv::CAP_PROP_FRAME_HEIGHT), cap.get(cv::CAP_PROP_FRAME_WIDTH), screen_width, screen_height);
    pro.uploadConfig();
    //create thread for reading keyinputs

    pthread_t t;
    int rc = pthread_create(&t, NULL, getKey, (void *)1);
    if (rc)
    {
        std::cout << "Error:unable to create thread," << rc << std::endl;
        exit(-1);
    }

    //VIDEO CAPTURE
    cv::Mat yoloFrame, projectFrame, frame;
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

        cv::namedWindow("output", cv::WINDOW_NORMAL);
        cv::setWindowProperty("output", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        cv::imshow("output", projectFrame);
        cv::waitKey(1);
        //if (cv::waitKey(1) >= 0)
        //break;
        if (closeKeyGet == true)
        {

            break;
        }
    }
    pthread_exit(NULL);
    cv::destroyAllWindows();
    return 0;
}

//*********************************************END MAIN**********************

bool FileExists(const std::string &Filename)
{
    bool check = access(Filename.c_str(), 0) == 0;
    return check;
}

cv::VideoCapture arguments(int argc, char const *argv[])
{
    // LETS GET THE FRAME 0 FOR INPUT:2 FOR THERMAL INPUT:0 FOR WEBCAM INPUT:1 FOR VIDEO.MP4 FOLLOWED BY THE PATH
    cv::VideoCapture cap;

    int input = std::stoi(argv[1]);

    //custom
    conf = absPath_weights + "custom/yolov3-obj.cfg";

    weights = absPath_weights + "custom/yolov3-obj_final.weights";
    //weights = "custom/thermal8.weights";

    classes = absPath_weights + "custom/obj.names";
    //model = "yolov3 cus";

    videoPath = argv[2];

    if (!FileExists(weights) || !FileExists(conf) || !FileExists(videoPath))
    {
        std::cerr << "check which one is missing: " << weights << " " << conf << " " << videoPath << "\n";
        exit(0);
    }
    else
    {
        std::cout << weights << "\n"
                  << conf << "\n"
                  << videoPath << "\n";
    }

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

        printf("video\n");

        cap.open(argv[2], cv::CAP_FFMPEG);
    }

    return cap;
}

int getDisplayInfo()
{
    //GET SCREEN RESOLUTION
    Display *display = XOpenDisplay(NULL);

    /* check the number of the default screen for our X server. */
    screen_num = DefaultScreen(display);
    //printf("number of screen %d \n", screen_num);
    /* find the width of the default screen of our X server, in pixels. */
    screen_width = DisplayWidth(display, screen_num);

    /* find the height of the default screen of our X server, in pixels. */
    screen_height = DisplayHeight(display, screen_num);

    /* find the ID of the root window of the screen. */
    root_window = RootWindow(display, screen_num);

    /* find the value of a white pixel on this screen. */
    white_pixel = WhitePixel(display, screen_num);

    /* find the value of a black pixel on this screen. */
    black_pixel = BlackPixel(display, screen_num);
    //printf("width, height display %d %d \n", screen_width, screen_height);
    return 0;
}

void *getKey(void *t_id)
{
    while (true)
    {
        if (kbhit() != 0)
        {
            int key = getch();
            //std::cout << key << std::endl;

            switch (key)
            {
            case top:
                //std::cout << "TOP" << std::endl;
                if (pro.new_y > 0)
                    pro.new_y--;
                break;
            case bottom:
                //std::cout << "BOTTOM" << std::endl;
                if ((pro.new_y + pro.new_frame_h) < pro.black_h)
                    pro.new_y++;
                break;
            case left:
                //std::cout << "LEFT" << std::endl;
                if (pro.new_x > 0)
                    pro.new_x--;
                break;
            case right:
                //std::cout << "RIGHT" << std::endl;
                if ((pro.new_x + pro.new_frame_w) < pro.black_w)
                    pro.new_x++;
                break;
            case scale_up:

                if (pro.new_frame_w < pro.black_w && ((pro.new_frame_w + pro.new_x) + (pro.new_frame_w * ratio)) < pro.black_w && pro.new_frame_h < pro.black_h && ((pro.new_frame_h + pro.new_y) + (pro.new_frame_h * ratio)) < pro.black_h)
                {
                    //std::cout << "scale up" << std::endl;
                    pro.new_frame_w += pro.new_frame_w * ratio;
                    pro.new_frame_h += pro.new_frame_h * ratio;
                }

                break;
            case scale_down:

                if (pro.new_frame_w > (minSizePorce * pro.frame_w))
                {
                    //std::cout << "scale down" << std::endl;
                    pro.new_frame_w -= pro.new_frame_w * ratio;
                    pro.new_frame_h -= pro.new_frame_h * ratio;
                }
                break;
            case low_low:

                if (pro.lowerThr > 0)
                {
                    //std::cout << "Low threshold Canny" << std::endl;
                    pro.lowerThr -= 1;
                }

                break;
            case low_high:

                if (pro.lowerThr < 255)
                {
                    //std::cout << "Low threshold Canny" << std::endl;
                    pro.lowerThr += 1;
                }
                break;
            case high_low:

                if (pro.higherThr > 0)
                {
                    //std::cout << "Higher threshold Canny" << std::endl;
                    pro.higherThr -= 1;
                }

                break;
            case high_high:

                if (pro.higherThr < 999)
                {
                    //std::cout << "Higher threshold Canny" << std::endl;
                    pro.higherThr += 1;
                }
                break;
            case esc:
                closeKeyGet = true;
                exit(0);
                break;

            default:
                std::cout << "KEY NOT MAPPED" << std::endl;
                break;
            }
            pro.saveConfig();
        }
    }
}