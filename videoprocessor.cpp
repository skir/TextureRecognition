#include "videoprocessor.h"

using namespace cv;

VideoProcessor::VideoProcessor(std::string fileName) {
    this->video = new VideoCapture();
    video->open(fileName);
    frameCounter = video->get(CV_CAP_PROP_FRAME_COUNT);
//    video->set(CV_CAP_PROP_FPS, 5.0);
//    std::cout<<video->get(CV_CAP_PROP_FRAME_COUNT)<<std::endl<<video->get(CV_CAP_PROP_FPS);
}

bool VideoProcessor::getNextFrame(OutputArray result) {
    Mat mat;
    bool ok = false;
    #pragma omp critical
    {
        for (int i = 0; i < 30; i++) {      //read only every 30th frame
            frameCounter--;
            if (frameCounter > 0) {
                ok = video->read(mat);
            }
        }
        if (mat.channels() >= 3) {
            cvtColor(mat, mat, CV_BGR2GRAY);
        }
        mat.copyTo(result);
    }
    return ok;
}
