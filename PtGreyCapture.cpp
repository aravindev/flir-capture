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

using namespace std;
using namespace uav::plvision;
using namespace cv;
using namespace cv::cuda;

int main(int argc, char** argv)
{
    cv::Mat g0(5,1,CV_32F);
    cv::Mat g1(5,1,CV_32F);
    
    cv::Mat img = cv::Mat::zeros(1000,1000,CV_32F);
    cv::cuda::GpuMat d_inputImgF;
    d_inputImgF.upload(img);
    
    cv::cuda::GpuMat md_dfdg0;
    
    // Comment the next 2 lines to get highr frame rate
    //Ptr<Filter> f_g0_g1 = cv::cuda::createSeparableLinearFilter(CV_32F, CV_32F, g0, g1);
    //f_g0_g1->apply(d_inputImgF, md_dfdg0);
    //

	PGRDevice device;
	   
    device.configure();
    
    device.start();
    
  	cout << "Done! Press Enter to exit..." << endl;
	cin.ignore();

	device.stop();
	
	device.cleanup();

	return 0;
};

