#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>

#include <sys/stat.h>
#include <sys/types.h>
#include <string>
#include <time.h>

#include "PGRDevice.h"

#include <opencv2/cudafilters.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace uav::plvision;
using namespace cv;
using namespace cv::cuda;

int main(int argc, char** argv)
{
    cv::Mat g0(5,1,CV_32F);
    cv::Mat g1(5,1,CV_32F);
    
    cv::Mat img, grey;
    cv::cuda::GpuMat d_img, d_inputImgF;
    
    cv::VideoCapture cap("myvideo.avi");
    
    cv::cuda::GpuMat md_dfdg0;
    
    Ptr<Filter> f_g0_g1 = cv::cuda::createSeparableLinearFilter(CV_32F, CV_32F, g0, g1);
    
    std::chrono::time_point<std::chrono::steady_clock> mPrevCapTime;
    
    int c = -1;
    while(c==-1)
    {
        cap.read(img);
        cv::cvtColor(img, grey, CV_BGR2GRAY);
        
        d_img.upload(grey);
        
        d_img.convertTo(d_inputImgF, CV_32F, 1.0);
        
        f_g0_g1->apply(d_inputImgF, md_dfdg0);
        
        auto t1 = std::chrono::steady_clock::now();
    
        std::cout << static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - mPrevCapTime).count()) << std::endl;
    
        mPrevCapTime = t1;
    
        c = cv::waitKey(5);
    }

	return 0;
};

