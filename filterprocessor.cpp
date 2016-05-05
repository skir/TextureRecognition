#include "filterprocessor.h"

using namespace cv;
using namespace std;

FilterProcessor::FilterProcessor()
{

}

void FilterProcessor::filterImage(Mat image, vector<Mat> kernelsVector, OutputArray vectors, OutputArray centers, OutputArray labels) {

    Mat dst, dst32;
    Mat vectorsTemp = Mat::zeros(image.cols * image.rows, kernelsVector.size() * image.channels(), CV_32F);    //CV_32F is for weak kmeans

    vector<Mat> channels;
    split(image, channels);
#pragma omp parallel for
    for (int channel = 0; channel < channels.size(); channel++) {
        for (int i = 0; i < kernelsVector.size(); i++) {
            filter2D(channels[channel], dst, CV_32F, kernelsVector[i]);

            int s = 0;
            for (int j = 0; j < dst.rows; j++) {
                for (int k = 0; k < dst.cols; k++) {
                    vectorsTemp.at<float>(s, i + kernelsVector.size() * channel) = dst.at<float>(j, k);
                    s++;
                }
            }

//            imwrite("dst" + to_string(i) + ".png", dst);
        }
    }

    vectorsTemp.copyTo(vectors);

    int attempts = 5;
    int clusterNumber = Constants::CLUSTER_NUMBER;

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
    int channels = centers.cols / kernels.rows;
    centers.convertTo(centers, CV_64F);
    for (int i = 0; i < centers.rows; i++) {
        Mat row = centers.row(i);
        vector<Mat> subTextons;
        for (int channel = 0; channel < channels; channel++) {
            Mat result = row(Range::all(), Range(channel * kernels.rows, (channel + 1) * kernels.rows)) * kernels;
            Mat image(Constants::KERNEL_SIZE, Constants::KERNEL_SIZE, CV_64F);
            for (int j = 0; j < image.rows; j++) {
                for (int k = 0; k < image.cols; k++) {
                    image.at<double>(j,k) = result.at<double>(0, j * Constants::KERNEL_SIZE + k);
                }
            }
            image.convertTo(image, CV_8U);
            subTextons.push_back(image);
        }
        Mat merged;
        merge(subTextons, merged);
        textons.push_back(merged);
        imwrite("texton" + to_string(i) +".png", merged);
    }
    return textons;
}

vector<Mat> FilterProcessor::mapPixelToTexton(Mat image, Mat vectors, Mat labels) {
    vector<Mat> result;
    int type;
    if (image.channels() == 1) {
        type = CV_8U;
    } else {
        type = CV_8UC3;
    }

    for (int i = 0; i < Constants::CLUSTER_NUMBER; i++) {
        Mat mat = Mat::ones(image.rows, image.cols, type);
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
            if (image.channels() == 1) {
                result[labels.at<int>(i * image.cols + j)].at<uchar>(i, j) = image.at<uchar>(i, j);
            } else {
                result[labels.at<int>(i * image.cols + j)].at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
            }
        }
    }

    for (int i = 0; i < Constants::CLUSTER_NUMBER; i ++) {
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
