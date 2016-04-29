#ifndef VIDEOPROCESSOR_H
#define VIDEOPROCESSOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>

class VideoProcessor
{
private:
    cv::VideoCapture *video;
    int frameCounter;

public:
    VideoProcessor(std::string);
    bool getNextFrame(cv::OutputArray);
};

#endif // VIDEOPROCESSOR_H
