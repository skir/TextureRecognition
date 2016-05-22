#include "mfs.h"

using namespace cv;
using namespace std;

MFS::MFS()
{

}

void MFS::coverWithMFS(Mat image, int windowSize, OutputArray vectors, OutputArray centers, InputOutputArray labels) {
    Mat densities = Mat::zeros(image.cols * image.rows, Constants::MFS_DIMENSION, CV_32F);    //CV_32F is for weak kmeans;
    int step = 10;
#pragma omp parallel for
    for (int i = 0; i < image.rows; i += step) {
        int y = i - windowSize / 2;
        int height = windowSize;
        if (y < 0) {
            height = windowSize + y;
            y = 0;
        }
        if (y + height >= image.rows) {
            height = image.rows - y;
        }
        for (int j = 0; j < image.cols; j += step) {
            int width = windowSize;
            int x = j - windowSize / 2;
            if (x < 0) {
                width = windowSize + x;
                x = 0;
            }
            if (x + width >= image.cols) {
                width = image.cols - x;
            }
            Mat densityForCurrentWindow = density(image(Rect(x, y, width, height)), 0);

            int endI = i + step < image.rows ? i + step : image.rows;
            int endJ = j + step < image.cols ? j + step : image.cols;
            for (int s = i; s < endI; s++) {
                for (int m = j; m < endJ; m++) {
                    for (int k = 0; k < densityForCurrentWindow.cols; k++) {
                        densities.at<float>(s*image.cols + m, k) = (float) densityForCurrentWindow.at<double>(0, k);
                    }
                }
            }

//            cout<<i<<'\t'<<j<<endl;
        }
    }

    densities.copyTo(vectors);

    int attempts = 5;
    int clusterNumber = Constants::CLUSTER_NUMBER;

    kmeans(densities,
           clusterNumber,
           labels,
           TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
           attempts,
           KMEANS_PP_CENTERS,
           centers);
}

Mat MFS::density(Mat image, int levels) {
    int ind_num = 8; //density counting levels
    int f_num = Constants::MFS_DIMENSION;  //the dimension of MFS
    int ite_num = 8; //Box counting levels
image.convertTo(image, CV_64F);
    double min, max;
    minMaxLoc(image, &min, &max);
    image = (image - min) / (max - min);
    image = image.mul(255) + 1;

    // Estimating density function of the image
    // by solving least squares for D in  the equation
    // log10(bw) = D*log10(c) + b

    double r = 1.0 / std::max(image.rows, image.cols);
    Mat c = Mat::zeros(1, ind_num, CV_64F);
    for (int i = 0; i < ind_num; i++) {
        c.at<double>(0, i) = log10((i + 1) *r);
    }

    vector<Mat> bw;
    bw.push_back(image);
    for (int k = 2; k <= ind_num; k++) {
        Mat temp;
        int size = k;
        if (size % 2 != 1) {
            size++;
        }
        GaussianBlur(image, temp, Size(size, size), k/2, k/2);
        temp = temp.mul(k*k);
        temp.convertTo(temp, CV_64F);
        bw.push_back(temp);
    }

    Mat ss = Mat::zeros(image.rows, image.cols, CV_64F);
    for (int k = 0; k < bw.size(); k++) {
        for (int i = 0; i < bw[k].rows; i++) {
            for (int j = 0; j < bw[k].cols; j++) {
                bw[k].at<double>(i, j) = log10(bw[k].at<double>(i, j));
                ss.at<double>(i, j) += bw[k].at<double>(i, j) / ind_num;
            }
        }
    }

    double x = sum(c)[0] / ind_num;
    Mat D = Mat::zeros(image.rows, image.cols, CV_64F);
    for (int i = 0; i < ind_num; i++) {
        Mat temp = Mat::zeros(bw[i].rows, bw[i].cols, CV_64F);
        for (int j = 0; j < temp.rows; j++) {
            for (int k = 0; k < temp.cols; k++) {
                temp.at<double>(j, k) = bw[i].at<double>(j, k) - ss.at<double>(j, k);
            }
        }
//        temp = bw[i] - ss;
        temp = temp.mul(c.at<double>(0, i) - x);
        D = D + temp;
    }

    c = c - x;
    D = D / sum(c.mul(c))[0];
    double minD, maxD;
    minMaxLoc(D, &minD, &maxD);
    int D_MAX = 4;
    int D_MIN = 1;

    if (maxD < 0) {
        D = -D;
    }

    for (int i = 0; i < D.rows; i++) {
        for (int j = 0; j < D.cols; j++) {
            if ((D.at<double>(i, j) < D_MIN) || (D.at<double>(i ,j) > D_MAX)) {
                D.at<double>(i, j) = -1;
            }
        }
    }

    D = D(Rect(ind_num, ind_num, D.cols - 2*ind_num, D.rows - 2*ind_num)).clone();

    Mat im = Mat::zeros(D.rows, D.cols, CV_64F);

    double gap = ((double)(D_MAX - D_MIN)) / (double)f_num;
//    cout<<gap<<endl;
    Mat center = Mat::zeros(1, f_num, CV_64F);

    for (int k = 1; k <= f_num; k++) {
        double bin_min = ((double)(k-1)) * gap + (double)D_MIN;
        double bin_max = ((double)k) * gap + (double)D_MIN;
        center.at<double>(0, k - 1) = (bin_min + bin_max) / 2;
        for (int i = 0; i < D.rows; i++) {
            for (int j = 0; j < D.cols; j++) {
                if ((D.at<double>(i, j) <= bin_max) && (D.at<double>(i, j) > bin_min)) {
                    im.at<double>(i, j) = center.at<double>(0, k - 1);
                }
            }
        }
    }

    Mat Idx_IM = Mat::zeros(im.rows, im.cols, CV_64F);
    for (int k = 1; k <= f_num; k++) {
        for (int i = 0; i < im.rows; i++) {
            for (int j = 0; j < im.cols; j++) {
                if (im.at<double>(i, j) == center.at<double>(0, k-1)) {
                    Idx_IM.at<double>(i, j) = k;
                }
            }
        }
    }

    Mat rMat = Mat::zeros(1, ite_num, CV_64F);
    int maxSize = image.cols < image.rows ? image.rows : image.cols;
    for (int i = 0; i < ite_num; i++) {
        rMat.at<double>(0, i) = log10(maxSize / (i + 1));
    }
    rMat = rMat - sum(rMat)[0] / ite_num;

    Mat num = Mat::zeros(1, ite_num, CV_64F);
    Mat MFS = Mat::zeros(1, f_num, CV_64F);


    //main logic
    for (int k = 1; k <= f_num; k++){
        Mat level = Mat::zeros(Idx_IM.rows, Idx_IM.cols, CV_64F);
        for (int i = 0; i < level.rows; i++) {
            for (int j = 0; j < level.cols; j++) {
                if (Idx_IM.at<double>(i, j) == k) {
                    level.at<double>(i, j) =  1;
                }
            }
        }

        double temp = sum(level)[0];
        if (temp == 0.0) { // for the case where no points.
            MFS.at<double>(0, k) = 0;
        } else {
            num.at<double>(0, 0) = log10(temp);
            for (int j = 2; j <= ite_num; j++) {
                Mat mask = Mat::ones(j, j, CV_64F);
                Mat conv;
                filter2D(level, conv, CV_64F,mask);
                int countPositive = 0;
                for (int i = j-1; i < conv.rows; i = i + j) {
                    for (int m = j-1; m < conv.cols; m = m + j) {
                        if (conv.at<double>(i, m) > 0) {
                            countPositive++;
                        }
                    }
                }
                num.at<double>(0, j-1) = log10( countPositive > 1 ? countPositive : 1);
            }
//            cout<<num<<endl;
            num = num - sum(num)[0] / ite_num;
            MFS.at<double>(0, k) = sum(num.mul(rMat))[0]/sum(rMat.mul(rMat))[0];
        }
    }
//    ALPHA = center;

    return MFS;
}
