#include "PGRDevice.h"

#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudawarping.hpp>

#ifdef WIN32
#include <windows.h>
#endif // WIN32

using namespace uav::plvision;
using namespace FlyCapture2;

void* myPt2ObjectPGR;

PGRDevice::PGRDevice()
{

}

PGRDevice::~PGRDevice()
{
    cleanup();
}

const char *PGRDevice::getName() const
{
    return "d";
}

void PGRDevice::configure()
{
    if (mConfigured)
    {
        cleanup();
    }

    // Assign global variable which is used in the static wrapper function
    // important: never forget to do this!!
    myPt2ObjectPGR = (void*)this;

    // Start the acquisition from the PTGrey camera
    PrintBuildInfo();

    Error error;

    BusManager busMgr;
    unsigned int numCameras;
    error = busMgr.GetNumOfCameras(&numCameras);
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        throw std::runtime_error("Failed to configure PGRDevice: can not get the number of cameras");
    }

    std::cout << "Number of cameras detected: " << numCameras << std::endl;

    if ( numCameras < 1 )
    {
        throw std::runtime_error("Failed to configure PGRDevice: no camera detected");
    }

    // Take the first detected camera
    PGRGuid guid;
    error = busMgr.GetCameraFromIndex(0, &guid);
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        std::runtime_error("Failed to configure PGRDevice: can not get camera from index");
    }

    mpCam.reset(new Camera);
    if(!mpCam)
        std::runtime_error("Failed to configure PGRDevice: can not create camera object");

    // Connect to a camera
    error = mpCam->Connect(&guid);
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        std::runtime_error("Failed to configure PGRDevice: can not connect to the camera");
    }

    // Get the camera information
    CameraInfo camInfo;
    error = mpCam->GetCameraInfo(&camInfo);
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        std::runtime_error("Failed to configure PGRDevice: can not get camera info");
    }

    PrintCameraInfo(&camInfo);

    // Check if the camera supports the FRAME_RATE property
    PropertyInfo propInfo;
    propInfo.type = FRAME_RATE;
    error = mpCam->GetPropertyInfo(&propInfo);
    if (error != PGRERROR_OK)
    {
        PrintError(error);
        throw std::runtime_error("Camera doesn't support frame rate property");
    }

    // Set the frame rate property of the camera
    if (propInfo.present == true)
    {
        // Set the frame rate
        Property prop;
        prop.type = FRAME_RATE;
        prop.absValue = 50;
        prop.absControl = true; // use absolute value
        prop.onOff = true; // on
        prop.autoManualMode = false; // manual mode
        error = mpCam->SetProperty(&prop);
        if (error != PGRERROR_OK)
        {
            PrintError(error);
            throw std::runtime_error("Failed to set manual frame rate.");
        }
    }

    // Query effective frame rate
    Property frameRateProp;
    frameRateProp.type = FRAME_RATE;
    error = mpCam->GetProperty(&frameRateProp);
    if(error == PGRERROR_OK)
    {
        std::cout << "Effective frame rate is " << frameRateProp.absValue << " fps." << std::endl;
    }

    // Query for available Format 7 modes
    Format7Info fmt7Info;
    bool supported;
    fmt7Info.mode = cstFmt7Mode;
    error = mpCam->GetFormat7Info(&fmt7Info, &supported);
    if (error != PGRERROR_OK)
    {
        PrintError(error);
        throw std::runtime_error(
                "Failed to configure camera: can not get camera format capabilities");
    }

    PrintFormat7Capabilities(fmt7Info);

    if ((cstFmt7PixFmt & fmt7Info.pixelFormatBitField) == 0)
    {
        throw std::runtime_error(
                "Failed to configure camera: pixel format is not supported");
    }

    Format7ImageSettings fmt7ImageSettings;
    fmt7ImageSettings.mode = cstFmt7Mode;
    fmt7ImageSettings.offsetX = 0;
    fmt7ImageSettings.offsetY = 0;
    fmt7ImageSettings.width = fmt7Info.maxWidth;
    fmt7ImageSettings.height = fmt7Info.maxHeight;
    fmt7ImageSettings.pixelFormat = cstFmt7PixFmt;

    bool valid;
    Format7PacketInfo fmt7PacketInfo;

    // Validate the settings
    error = mpCam->ValidateFormat7Settings(&fmt7ImageSettings, &valid, &fmt7PacketInfo );
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        std::runtime_error("Failed to configure PGRDevice: can not validate pixel format");
    }

    if ( !valid )
    {
        std::runtime_error("Failed to configure PGRDevice: format settings not valid");
    }

    // Set the settings to the camera
    error = mpCam->SetFormat7Configuration(&fmt7ImageSettings, fmt7PacketInfo.recommendedBytesPerPacket);
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        std::runtime_error("Failed to configure PGRDevice: can not change camera settings");
    }

    int width = fmt7Info.maxWidth;
    int height = fmt7Info.maxHeight;
    cv::Size imgSize(width, height);

    mImgFullSize = cv::Mat::zeros(imgSize, CV_8U);
    mImgHalfSize = cv::Mat::zeros(imgSize/2, CV_8U);
    
    // Camera intrinsic parameters used to undistort the image
    cv::Mat cameraMatrix = (cv::Mat_<float>(3,3) << 
                5.7138457606243685e+02, 0., 6.3056992211432373e+02,
                0., 5.7138457606243685e+02, 4.8243715809754491e+02,
                0., 0., 1.);
    cv::Mat distCoeffs = (cv::Mat_<float>(8,1) << 6.5032225125971521e-01, 4.6613895806078394e-02,
                2.0072639760212272e-05, 1.0875897003264509e-04,
                -1.7202605781646435e-04, 9.3334245713397113e-01,
                1.5401785519310726e-01, 1.7124368020013511e-03);

    std::cout << "Camera matrix: " << cameraMatrix << std::endl << std::endl;

    std::cout << "Distortion coefficients: " << distCoeffs << std::endl << std::endl;

    bool centerPrincipalPoint = false;
    double alpha = 1.0;

    // This function will compute the new camera matrix and image size after undistortion
    mNewCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs,
                                                     imgSize,
                                                     alpha,
                                                     imgSize,
                                                     &mValidPixROI,
                                                     centerPrincipalPoint);

    std::cout << "New camera matrix: " << mNewCameraMatrix << std::endl << std::endl;

    cv::initUndistortRectifyMap(cameraMatrix,
                                distCoeffs,
                                cv::Mat(),
                                mNewCameraMatrix,
                                imgSize,
                                CV_32FC1,
                                mMap1,
                                mMap2);

    md_map1.upload(mMap1);
    md_map2.upload(mMap2);

    // This is the size of image after undistort + crop valid part
    mImageWidth = mValidPixROI.width;
    mImageHeight = mValidPixROI.height;

    mConfigured = true;
}

void PGRDevice::start()
{
    if (!mConfigured) std::runtime_error("Device not configured");

    if (mIsRunning)
    {
        stop();
    }

    mIsRunning = true;

    // Start capturing images
    Error error = mpCam->StartCapture(PGRDevice::imageProducerWrapper);
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        mIsRunning = false;
        if (!mIsRunning) std::runtime_error("Failed to start PGRDevice");
    }

    // Retrieve frame rate property
    /*Property frmRate;
    frmRate.type = FRAME_RATE;
    error = cam.GetProperty( &frmRate );
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        return -1;
    }

    cout << "Frame rate is " << fixed << setprecision(2) << frmRate.absValue << " fps" << endl;//*/
}

void PGRDevice::stop()
{
    // Do not throw exception if the device isn't configured. Just
    // silently return.
    if (!mConfigured) return;

    // Stop capturing images
    Error error = mpCam->StopCapture();
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        throw std::runtime_error("Failed to stop PGRDevice");
    }

    mIsRunning = false;
}

void PGRDevice::cleanup()
{
    if (mIsRunning)
    {
        stop();
    }

    // Disconnect the camera
    Error error = mpCam->Disconnect();
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        throw std::runtime_error("Failed to disconnect from PGRDevice");
    }

    mConfigured = false;
}

void PGRDevice::pause(bool enabled)
{

}

void PGRDevice::next()
{

}

int PGRDevice::getImageWidth() const
{
    if (!mConfigured) throw std::runtime_error("Device not configured");

    return mImageWidth;
}

int PGRDevice::getImageHeight() const
{
    if (!mConfigured) throw std::runtime_error("Device not configured");

    return mImageHeight;
}

int PGRDevice::getImageType() const
{
    if (!mConfigured) throw std::runtime_error("Device not configured");

    if(cstFmt7PixFmt == PIXEL_FORMAT_MONO8)
        return CV_8U;

    return CV_8UC3;
}

const cv::Rect& PGRDevice::getValidPixROI() const
{
    if(!mConfigured) throw std::runtime_error("Device not configured");

    return mValidPixROI;
}

const cv::Mat& PGRDevice::getNewCameraMatrix() const
{
    if(!mConfigured) throw std::runtime_error("Device not configured");

    return mNewCameraMatrix;
}

void PGRDevice::imageProducerWrapper(Image* pImage, const void* pCallbackData)
{
    // explicitly cast global variable <pt2Object> to a pointer to TClassB
    // warning: <pt2Object> MUST point to an appropriate object!
    PGRDevice* mySelf = (PGRDevice*) myPt2ObjectPGR;

    // call member
    mySelf->imageProducer(pImage, pCallbackData);
}

void PGRDevice::imageProducer(Image* pImage, const void* pCallbackData)
{
    auto t1 = std::chrono::steady_clock::now();
    
    std::cout << static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - mPrevCapTime).count()) << std::endl;
    
    mPrevCapTime = t1;

    memcpy(mImgFullSize.data, pImage->GetData(), pImage->GetDataSize());

    // Undistort image
    md_img.upload(mImgFullSize);
    cv::cuda::remap(md_img, md_undistordedImage, md_map1, md_map2, cv::INTER_LINEAR); //cv::INTER_LINEAR cv::INTER_NEAREST
}

void PGRDevice::deviceUnplugged()
{

}

void PGRDevice::PrintBuildInfo()
{
    FC2Version fc2Version;
    Utilities::GetLibraryVersion( &fc2Version );

    std::cout << "FlyCapture2 library version: " << std::to_string(fc2Version.major)
                   << "." << std::to_string(fc2Version.minor)
                   << "." << std::to_string(fc2Version.type)
                   << "." << std::to_string(fc2Version.build) << std::endl;

    std::cout << "Application build date: "
                   << __DATE__ << " "
                   << __TIME__ << std::endl;
}

void PGRDevice::PrintCameraInfo( CameraInfo* pCamInfo )
{
    std::cout << "*** CAMERA INFORMATION ***" << std::endl;
    std::cout << "Serial number -" << pCamInfo->serialNumber << std::endl;
    std::cout << "Camera model - " << pCamInfo->modelName << std::endl;
    std::cout << "Camera vendor - " << pCamInfo->vendorName << std::endl;
    std::cout << "Sensor - " << pCamInfo->sensorInfo << std::endl;
    std::cout << "Resolution - " << pCamInfo->sensorResolution << std::endl;
    std::cout << "Firmware version - " << pCamInfo->firmwareVersion << std::endl;
    std::cout << "Firmware build time - " << pCamInfo->firmwareBuildTime << std::endl;
}

void PGRDevice::PrintFormat7Capabilities( Format7Info fmt7Info )
{
    std::cout << "Max image pixels: (" << fmt7Info.maxWidth << ", " << fmt7Info.maxHeight << ")" << std::endl;
    std::cout << "Image Unit size: (" << fmt7Info.imageHStepSize << ", " << fmt7Info.imageVStepSize << ")"  << std::endl;
    std::cout << "Offset Unit size: (" << fmt7Info.offsetHStepSize << ", " << fmt7Info.offsetVStepSize << ")" << std::endl;
    std::cout << "Pixel format bitfield: 0x" << fmt7Info.pixelFormatBitField << std::endl;
}

void PGRDevice::PrintError( Error error )
{
    error.PrintErrorTrace();
}
