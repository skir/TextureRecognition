#ifndef MFS_H
#define MFS_H

#include <opencv2/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>
#include <vector>
#include <iostream>

#include "imageutils.h"
#include "constants.h"

using namespace cv;

class MFS
{
public:
    MFS();

    static Mat density(Mat image, int levels);

    static void coverWithMFS(Mat image, int windowSize, OutputArray vectors, OutputArray centers, InputOutputArray labels);
};

#endif // MFS_H
