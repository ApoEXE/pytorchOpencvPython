//g++ main.cpp -o test  `pkg-config opencv --cflags --libs`
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <iostream>
using namespace std;
using namespace cv;
#define USE_VTK
#ifdef USE_VTK

void KeyboardViz3d(const viz::KeyboardEvent &w, void *t)
{
    viz::Viz3d *fen = (viz::Viz3d *)t;
    if (w.action)
        cout << "you pressed " << w.code << " = " << w.symbol << " in viz window " << fen->getWindowName() << "\n";
}

int main(int argc, char **argv)
{

    Mat img = imread("1186.png", IMREAD_COLOR);
    Mat imgh;
    cvtColor(img, imgh, CV_BGR2GRAY);
    
    viz::Viz3d visualizer("Viz window");
    visualizer.registerKeyboardCallback(KeyboardViz3d, &visualizer);
    visualizer.setBackgroundColor(cv::viz::Color::black());
    while (!visualizer.wasStopped())
    {
        visualizer.spinOnce(1,true);
    }
    

    /*
    while(true){
        imshow("img",img);
        waitKey(1);

    }
/*
    int nbF = 0;
    vector<Point3d> cloud;
    vector<int> polygon;
    vector<Vec3b> couleur;
    double k = 1;
    int nbPix = 0;
    for (int i = 1; i < img.rows; i++)
        for (int j = 1; j < img.cols; j++)
        {
            Vec3d p1(j / k, i / k, imgh.at<uchar>(i, j) / k);
            Vec3d p2((j - 1) / k, i / k, imgh.at<uchar>(i, j - 1) / k);
            Vec3d p3(j / k, (i - 1) / k, imgh.at<uchar>(i - 1, j) / k);
            Vec3d p4((j - 1) / k, (i - 1) / k, imgh.at<uchar>(i - 1, j - 1) / k);
            cloud.push_back(p1);
            cloud.push_back(p2);
            cloud.push_back(p3);
            cloud.push_back(p4);
            couleur.push_back(img.at<Vec3b>(i, j));
            couleur.push_back(img.at<Vec3b>(i, j));
            couleur.push_back(img.at<Vec3b>(i, j));
            couleur.push_back(img.at<Vec3b>(i, j));
            polygon.push_back(3);
            polygon.push_back(nbPix);
            polygon.push_back(nbPix + 1);
            polygon.push_back(nbPix + 2);
            polygon.push_back(3);
            polygon.push_back(nbPix + 2);
            polygon.push_back(nbPix + 3);
            polygon.push_back(nbPix);
            nbPix += 4;
        }
    viz::WMesh grille(cloud, polygon, couleur);
    fen3D.showWidget("I3d", grille);
    fen3D.spin();
*/
    return 0;
}

#else

int main(int argc, char **argv)
{
    cout << " you need VIZ module\n";
    return 0;
}
#endif