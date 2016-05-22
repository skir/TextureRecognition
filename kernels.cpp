#include "kernels.h"

Kernels::Kernels()
{

}

Mat Kernels::kernel(int size, double angle, double sigma1, double sigma2, FunctionType type) {
    Mat k = Mat::zeros(size, size, CV_64F);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int x = i - (int)(size/2);
            int y = j - (int)(size/2);

            switch (type) {
                case Even:
                    k.at<double>(i, j) = kernelEvenFunction(x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle) , sigma1, sigma2);
                    break;
                case Odd:
                    k.at<double>(i, j) = kernelOddFunction(x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle) , sigma1, sigma2);
                    break;
                case Radial:
                    k.at<double>(i, j) = kernelRadialFunction(x, y, sigma1, sigma2);
                    break;
            }
        }
    }

//    k = normalization(k);

    return k;
}

Mat Kernels::normalization(Mat k) {
    double meanS = 0;
    double sum = 0;
    double min = 0;

    for (int i = 0; i < k.rows; i++) {
        for (int j = 0; j < k.cols; j++) {
            meanS += k.at<double>(i, j);
            if (k.at<double>(i, j) < min) {
                min = k.at<double>(i, j);
            }
        }
    }

    meanS /= k.cols * k.rows;
    for (int i = 0; i < k.rows; i++) {
        for (int j = 0; j < k.cols; j++) {
            k.at<double>(i, j) -= meanS;            //zero mean
//            k.at<double>(i, j) -= min;
            sum += fabs(k.at<double>(i, j));
        }
    }

    for (int i = 0; i < k.rows; i++) {
        for (int j = 0; j < k.cols; j++) {
            k.at<double>(i ,j) /= sum;
//            std::cout << k.at<double>(i, j) << '\t';
        }
    }
    return k;
}

Mat Kernels::normalization2(Mat k) {
    double sum = 0.0;

    for (int i = 0; i < k.rows; i++) {
        for (int j = 0; j < k.cols; j++) {
            if (k.at<double>(i, j) < 0.00001) {
                k.at<double>(i, j) = 0;
            }
            sum += k.at<double>(i, j);
        }
    }

    for (int i = 0; i < k.rows; i++) {
        for (int j = 0; j < k.cols; j++) {
            k.at<double>(i, j) /= sum;
        }
    }
    return k;
}

double Kernels::kernelEvenFunction(double x, double y, double sigma1, double sigma2) {
    double result = exp(- pow(x, 2) / (2 * pow(sigma2, 2))) / (sigma2 * pow(2 * M_PI, 0.5)) * (
                - exp(- pow(y, 2) / (2 * pow(sigma1, 2))) / (pow(sigma1, 3) * pow(2 * M_PI, 0.5)) +
                (exp(- pow(y, 2) / (2 * pow(sigma1, 2))) * pow(y, 2)) / (pow(sigma1, 5) * pow(2 * M_PI, 0.5)));
//    cout<<x<<'\t'<<y<<'\t'<<result<<endl;
    return result;  //*10
}

double Kernels::kernelOddFunction(double x, double y, double sigma1, double sigma2) {
    return - exp(- x*x / (2*sigma2*sigma2) - y*y / (2*sigma1*sigma1)) * y / (2 * M_PI *sigma1*sigma1*sigma1 * sigma2);  //*10
}

double Kernels::kernelRadialFunction(double x, double y, double sigma1, double sigma2) {
//    double x1 = x * 0.5;
//    double y1 = y * 0.5;
//    return exp(-(x1*x1+y1*y1)/(2*sigma1*sigma1))/(2*M_PI*sigma1*sigma1) -
//            exp(-(x*x+y*y)/(2*sigma2*sigma2))/(2*M_PI*sigma2*sigma2);
    return - exp(- (x*x + y*y) / (2 * sigma1 * sigma1)) / (M_PI * sigma1 * sigma1 * sigma1 * sigma1) *
            (1 - (x*x + y*y) / (2 * sigma1 * sigma1));
}

Mat Kernels::convertVectorToMat(std::vector<Mat> kernelsVector) {
    Mat kernels(kernelsVector[0].rows * kernelsVector[0].cols, kernelsVector.size(), CV_64F);
    for ( int i = 0; i < kernelsVector.size(); i++) {
        int s = 0;
        for (int j = 0; j < kernelsVector[i].cols; j++) {
            for (int k = 0; k < kernelsVector[i].rows; k++){
                kernels.at<double>(s, i) = kernelsVector[i].at<double>(k, j);
                s++;
            }
        }
    }
    return kernels;
}
