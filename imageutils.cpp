#include "imageutils.h"

ImageUtils::ImageUtils()
{

}

double ImageUtils::getAverageValue(Mat image) {
   double sum = 0;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            sum += image.at<unsigned char>(i, j);
        }
    }
    return sum / (image.rows * image.cols);
}

Mat ImageUtils::substract(Mat source, Mat sub) {
    unsigned char averageColor = (unsigned char) getAverageValue(source);
    Mat result = Mat::zeros(source.rows, source.cols, CV_8U);
    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {    
//            unsigned char dif = source.at<unsigned char>(i, j) - sub.at<unsigned char>(i, j);
            if (radius(source, sub, i, j, 5)) {//dif <= 5) {
                result.at<unsigned char>(i, j) = averageColor;
            } else {
                result.at<unsigned char>(i, j) = source.at<unsigned char>(i, j);
            }
        }
    }
    return result;
}


bool ImageUtils::radius(Mat source, Mat sub, int row, int col, int radius) {
    for (int i = 0; i < radius; i++) {
        int startRow = (row - i) <= 0 ? 0 : (row - i);
        int endRow = (row + i) < source.rows ? (row + i) : source.rows;
        int endCol = (col + i) < source.cols ? (col + i) : source.cols;
        int startCol = (col - i) <= 0 ? 0 : (col - i);
        for (int k = startRow; k < endRow; k++) {
            if ((distance(k, startCol, row, col) < radius*radius) && (sub.at<unsigned char>(k, startCol) > 1)) {
                return true;
            }
        }
        for (int k = startCol; k < endCol; k++) {
            if ((distance(endRow, k, row, col) < radius*radius) && sub.at<unsigned char>(endRow, k) >1) {
                return true;
            }
        }
        for (int k = endRow; k > startRow; k--) {
            if ((distance(k, endCol, row, col) < radius*radius) && sub.at<unsigned char>(k, endCol) >1) {
                return true;
            }
        }
        for (int k = endCol; k > startCol; k--) {
            if ((distance(endCol, k, row, col) < radius*radius) && sub.at<unsigned char>(startRow, k) >1) {
                return true;
            }
        }
    }
    return false;
}

double ImageUtils::distance(int x, int y, int centerX, int centerY) {
    return (x - centerX)*(x - centerX) + (y - centerY)*(y - centerY);
}

std::string ImageUtils::typeToString(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
