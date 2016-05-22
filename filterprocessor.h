#ifndef FILTERPROCESSOR_H
#define FILTERPROCESSOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>

#include "imageutils.h"
#include "constants.h"

using namespace cv;
using namespace std;

class FilterProcessor
{
public:
    FilterProcessor();

    static void filterImage(Mat image, vector<Mat> kernels, OutputArray vectors, OutputArray centers, InputOutputArray labels);
    static float distance(Mat vector1, Mat vector2);
    static vector<Mat> mapPixelToTexton(Mat image, Mat vectors, Mat labels);
    static vector<Mat> getTextonsVector(Mat centers, Mat kernels);

};

#endif // FILTERPROCESSOR_H
