#ifndef PLVISION_PGR_DEVICE_H
#define PLVISION_PGR_DEVICE_H

#include <vector>
#include <deque>
#include <memory>
#include <atomic>
#include <chrono>

#include "flycapture/FlyCapture2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace uav
{
namespace plvision
{

/**
 * Device that capture video from an Point Grey Research camera.
 */
class PGRDevice
{
public:

    PGRDevice();
    virtual ~PGRDevice();

    const char* getName() const;

    void configure();
    void start();
    void stop();
    void cleanup();

    void pause(bool enabled);
    void next();

    int getImageWidth() const;
    int getImageHeight() const;
    int getImageType() const;
    const cv::Rect& getValidPixROI() const;
    const cv::Mat& getNewCameraMatrix() const;

private:

    std::atomic<bool> mIsRunning{false};
    bool mConfigured{false};

    std::unique_ptr<FlyCapture2::Camera> mpCam{nullptr};

    int mImageWidth{0};
    int mImageHeight{0};

    const FlyCapture2::Mode cstFmt7Mode{FlyCapture2::MODE_0};
    //PIXEL_FORMAT_MONO8;   PIXEL_FORMAT_RGB8   PIXEL_FORMAT_422YUV8    PIXEL_FORMAT_444YUV8
    const FlyCapture2::PixelFormat cstFmt7PixFmt{FlyCapture2::PIXEL_FORMAT_MONO8};

    cv::Mat mImgFullSize;
    cv::Mat mImgHalfSize;

    cv::cuda::GpuMat md_img;
    cv::cuda::GpuMat md_undistordedImage;

    cv::Mat mNewCameraMatrix;
    cv::Mat mMap1, mMap2;
    cv::cuda::GpuMat md_map1, md_map2;
    cv::Rect mValidPixROI;

    // Thread functions that handle the event of the camera
    static void imageProducerWrapper(FlyCapture2::Image* pImage, const void* pCallbackData);
    void imageProducer(FlyCapture2::Image* pImage, const void* pCallbackData);
    void deviceUnplugged();

    void PrintError( FlyCapture2::Error error );
    void PrintFormat7Capabilities( FlyCapture2::Format7Info fmt7Info );
    void PrintCameraInfo( FlyCapture2::CameraInfo* pCamInfo );
    void PrintBuildInfo();
    
    std::chrono::time_point<std::chrono::steady_clock> mPrevCapTime;
};

} // namespace plvision

} // namespace uav

#endif // PLVISION_PGR_DEVICE_H
