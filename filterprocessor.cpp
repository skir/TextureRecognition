#include "filterprocessor.h"

using namespace cv;
using namespace std;

FilterProcessor::FilterProcessor()
{

}

void FilterProcessor::filterImage(Mat image, vector<Mat> kernelsVector, OutputArray vectors, OutputArray centers, OutputArray labels) {

    Mat dst, dst32;
    Mat vectorsTemp = Mat::zeros(image.cols * image.rows, kernelsVector.size(), CV_32F);    //CV_32F is for weak kmeans
//    imwrite("im.png",kernel(11, M_PI / 4 ) * 255);

    for ( int i = 0; i < kernelsVector.size(); i++) {
        filter2D(image, dst, CV_32F, kernelsVector[i]);
//        dst32 = imread("dst" + to_string(i) + ".png");
        int s = 0;
//        dst.convertTo(dst, CV_32F);
        for (int j = 0; j < dst.rows; j++) {
            for (int k = 0; k < dst.cols; k++) {
                vectorsTemp.at<float>(s, i) = dst.at<float>(j, k);
//                cout<<typeToString(dst32.type())<<endl;
                s++;
            }
        }

        imwrite("dst" + to_string(i) + ".png", dst);
    }

//    vectorsTemp.convertTo(dst, CV_8U);
    vectorsTemp.copyTo(vectors);
//    imwrite("vectors.png", dst);


    int attempts = 5;
    int clusterNumber = 10;

    kmeans(vectorsTemp,
           clusterNumber,
           labels,
           TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
           attempts,
           KMEANS_PP_CENTERS,
           centers);

    cout<<"ok"<<endl;
//    imwrite("centers.png", centers);
}

vector<Mat> FilterProcessor::getTextonsVector(Mat centers, Mat kernels) {
    vector<Mat> textons;
    centers.convertTo(centers, CV_64F);
    for (int i = 0; i < centers.rows; i++) {
        cout<<kernels.cols<<'\t'<<kernels.rows<<'\t'<<centers.row(i).cols<<endl;
        Mat result = centers.row(i) * kernels;
        Mat image(Constants::KERNEL_SIZE, Constants::KERNEL_SIZE, CV_64F);
        for (int j = 0; j < image.rows; j++) {
            for (int k = 0; k < image.cols; k++) {
                image.at<double>(j,k) = result.at<double>(0, j * Constants::KERNEL_SIZE + k);
            }
        }
        image.convertTo(image, CV_8U);
        textons.push_back(image);
        imwrite("texton" + to_string(i) +".png", image);
    }
    return textons;
}

vector<Mat> FilterProcessor::mapPixelToTexton(Mat centers, Mat image, Mat vectors, Mat labels) {
    vector<Mat> result;

    for (int i = 0; i < centers.rows; i++) {
        Mat mat = Mat::ones(image.rows, image.cols, CV_8U);
        result.push_back(mat);
    }

    cout<<endl<<vectors.rows<<'\t'<<vectors.cols<<endl;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
//            float minDistance = distance(centers.row(0), vectors.row(i * image.cols + j));
//            int cluster = 0;
//            for (int k = 1; k < centers.rows; k++) {
//                float curDistance = distance(centers.row(k), vectors.row(i * image.cols + j));
//                if ( curDistance < minDistance) {
//                    minDistance = curDistance;
//                    cluster = k;
//                }
//            }
            result[labels.at<int>(i * image.cols + j)].at<unsigned char>(i, j) = image.at<unsigned char>(i, j);
        }
    }

    for (int i = 0; i < centers.rows; i ++) {
        imwrite("mat" + to_string(i) + ".png", result[i]);
    }
    return result;
}

float FilterProcessor::distance(Mat vector1, Mat vector2) {
    float result = 0.0;
    for (int i = 0; i < vector1.cols; i++) {
        result += pow(vector1.at<float>(0, i) - vector2.at<float>(0, i), 2);
    }
    return sqrt(result);
}
