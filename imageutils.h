#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

using namespace cv;

class ImageUtils
{
public:
    ImageUtils();

    static Mat substract(Mat source, Mat sub);
    static double getAverageValue(Mat image);
    static bool radius(Mat source, Mat sub, int row, int col, int radius);
    static double distance(int x, int y, int centerX, int centerY);
    static std::string typeToString(int type);
};

#endif // IMAGEUTILS_H
