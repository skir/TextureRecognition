#ifndef KERNELS_H
#define KERNELS_H

#include <opencv2/core/core.hpp>
#include <math.h>
#include <vector>
#include <iostream>

using namespace cv;

class Kernels
{
public:

    enum FunctionType {Even, Odd, Radial};

    Kernels();

    static Mat kernel(int size, double angle, double sigma1, double sigma2, FunctionType type);
    static double kernelEvenFunction(double x, double y, double sigma1, double sigma2);
    static double kernelOddFunction(double x, double y, double sigma1, double sigma2);
    static double kernelRadialFunction(double x, double y, double sigma1, double sigma2);
    static Mat convertVectorToMat(vector<Mat> kernelsVector);

    static Mat normalization(Mat mat);
    static Mat normalization2(Mat mat);
};

#endif // KERNELS_H
