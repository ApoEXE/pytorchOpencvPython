#include "screenConfig.h"
void initThread(){
        //Thread
    uint8_t id = 1;
    int rc = 0;
    pthread_t t;
    rc = pthread_create(&t, NULL, getKey, (void *)id);
    if (rc)
    {
        std::cout << "Error:unable to create thread," << rc << std::endl;
        exit(-1);
    }
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
                std::cout << "TOP" << std::endl;
                if (y > 0)
                    y--;
                break;
            case bottom:
                std::cout << "BOTTOM" << std::endl;
                if ((y + new_frame_h) < black_h)
                    y++;
                break;
            case left:
                std::cout << "LEFT" << std::endl;
                if (x > 0)
                    x--;
                break;
            case right:
                std::cout << "RIGHT" << std::endl;
                if ((x + new_frame_w) < black_w)
                    x++;
                break;
            case scale_up:

                if (new_frame_w < black_w && ((new_frame_w + x) + (new_frame_w * ratio)) < black_w && new_frame_h < black_h && ((new_frame_h + y) + (new_frame_h * ratio)) < black_h)
                {
                    std::cout << "scale up" << std::endl;
                    new_frame_w += new_frame_w * ratio;
                    new_frame_h += new_frame_h * ratio;
                }

                break;
            case scale_down:

                if (new_frame_w > (minSizePorce * frame_w))
                {
                    std::cout << "scale down" << std::endl;
                    new_frame_w -= new_frame_w * ratio;
                    new_frame_h -= new_frame_h * ratio;
                }
                break;
            case l:

                if (lowerThr > 0)
                {
                    std::cout << "Low threshold Canny" << std::endl;
                    lowerThr -= 1;
                }

                break;
            case k:

                if (lowerThr < 255)
                {
                    std::cout << "Low threshold Canny" << std::endl;
                    lowerThr += 1;
                }
                break;
            case h:

                if (higherThr > 0)
                {
                    std::cout << "Higher threshold Canny" << std::endl;
                    higherThr -= 1;
                }

                break;
            case g:

                if (higherThr < 999)
                {
                    std::cout << "Higher threshold Canny" << std::endl;
                    higherThr += 1;
                }
                break;
            case esc:
                exit(0);
                break;

            default:
                std::cout << "KEY NOT MAPPED" << std::endl;
                break;
            }
        }
    }
}