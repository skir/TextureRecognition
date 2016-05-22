#include <QCoreApplication>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <stddef.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "kernels.h"
#include "filterprocessor.h"
#include "imageutils.h"
#include "videoprocessor.h"
#include "constants.h"
#include "mfs.h"

#define CNN_USE_OMP

#include "tiny_cnn/tiny_cnn.h"

using namespace cv;
using namespace std;
using namespace tiny_cnn;

network<mse, adagrad> constructCNN(int size);
void train(network<mse, adagrad> net, Mat vectors, Mat labelsMat);
int predict(network<mse, adagrad> net, Mat center);

const char *files[] = {"DJI00177.png", "image0.png", "image1.png", "DJI00175.png", "test2.png"};
const double SIGMAS[] = {1.0, sqrt(2), 2.0, 2.0 * sqrt(2), 4.0, 4.0 * sqrt(2), 8.0};
                       //1.0, sqrt(2), 2.0

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    namedWindow( "window_name", CV_WINDOW_AUTOSIZE );
//    namedWindow( "window_name2", CV_WINDOW_AUTOSIZE );

    Mat image = imread(files[3], CV_LOAD_IMAGE_GRAYSCALE);
//    bitwise_not(image, image);        //invert
    imwrite("image.png", image);

    VideoProcessor* video = new VideoProcessor("/media/d/Dropbox/Camera Uploads/2015-08-09 12.03.42.mp4");//"/media/d/Videos/diploma/DJI00177.MP4");

//    vector<Mat> kernelsVector;
//    Mat k = Mat::zeros(Constants::KERNEL_SIZE, Constants::KERNEL_SIZE, CV_64F);
//    for (int i = 0; i < 4; i++) {
//        k = Kernels::kernel(Constants::KERNEL_SIZE, 0, SIGMAS[i], 0, Kernels::Radial);
//        kernelsVector.push_back(k);
//        imwrite("kernel00" + to_string(i) + ".png", k *1000);
//    }
//    for (int i = 0; i < 4; i++) {
//        k = Kernels::kernel(Constants::KERNEL_SIZE, 0, 3 * SIGMAS[i], 0, Kernels::Radial);
//        kernelsVector.push_back(k);
//        imwrite("kernel01" + to_string(i) + ".png", k *1000);
//    }
//    for (int j = 0; j < 3; j++) {
//        for (int i = 0; i < 6; i++){
//            k = Kernels::kernel(Constants::KERNEL_SIZE, i * M_PI / 6, SIGMAS[j], 3 * SIGMAS[j], Kernels::Even);   //[1, 3]; [1.5, 2]
//            kernelsVector.push_back(k);
//            imwrite("kernel1" + to_string(j) + to_string(i) + ".png", k * 1000);
//        }
//        for (int i = 0; i < 6; i++){
//            k = Kernels::kernel(Constants::KERNEL_SIZE, i * M_PI / 6, SIGMAS[j], 3 * SIGMAS[j], Kernels::Odd);
//            kernelsVector.push_back(k);
//            imwrite("kernel2" + to_string(j) + to_string(i) + ".png", k *1000);
//        }
//    }
//M_PI / 2 - M_PI / 18 + i * M_PI / 18

    Mat vectors, centers, labels;

//    FilterProcessor::filterImage(image, kernelsVector, vectors, centers, labels);
    MFS::coverWithMFS(image, 100, vectors, centers, labels);

    for (int i = 0; i < centers.rows; i++) {
        for (int j = 0; j < centers.cols; j++) {
            cout<<centers.at<float>(i,j)<<',';
        }
        cout<<endl;
    }

    ofstream vec_out("vectors.txt");
    for (int i = 0; i < vectors.rows; i += 100) {
        for (int j = 0; j < vectors.cols; j++) {
            vec_out << vectors.at<float>(i,j) << '\t';
        }
        vec_out << endl;
    }

//    network<mse, adagrad> net = constructCNN(vectors.cols);
//    train(net, vectors, labels);
//    cout << "label " << predict(net, centers.row(0));
//    cout << "label " << predict(net, centers.row(1));
//    cout << "label " << predict(net, centers.row(2));

    vector<Mat> maps = FilterProcessor::mapPixelToTexton(image, vectors, labels);
    for (int i = 0; i < maps.size(); i++) {
        imwrite("sub" + to_string(i) + ".png", ImageUtils::substract(image, maps[i]));
    }

//    Mat kernels = Kernels::convertVectorToMat(kernelsVector);
//    cout<<typeToString(centers.type()) + " " + typeToString(kernels.type())<<endl;
//    FilterProcessor::getTextonsVector(centers, kernels.inv(DECOMP_SVD));

    return a.exec();
}

network<mse, adagrad> constructCNN(int vectorsSize) {
    network<mse, adagrad> net;
    net << fully_connected_layer<activation::sigmoid>(vectorsSize, 10)     //maybe 10          //?
        << fully_connected_layer<activation::sigmoid>(10, 3);
    return net;
}

void train(network<mse, adagrad> net, Mat vectors, Mat labelsMat) {
    vector<vec_t> data;
    for (int i = 0; i < vectors.rows; i++) {
        vec_t temp;
        for (int j = 0; j < vectors.cols; j++) {
            temp.push_back(vectors.at<float>(i, j));
        }
        data.push_back(temp);
    }

    vector<label_t> labels;
    for (int i = 0; i < labelsMat.rows; i++) {
        label_t label = labelsMat.at<int>(i,0);
        labels.push_back(label);
    }

    net.train(data, labels, 20, 10);
    ofstream output("nets.txt");
    output << net;
}

int predict(network<mse, adagrad> net, Mat center) {

    vec_t in;
    for (int i = 0; i < center.cols; i++) {
        in.push_back(center.at<float>(0,i));
    }

    ifstream input("nets.txt");
    input >> net;
    return net.predict_label(in);
}

vector<Mat> processVideo(VideoProcessor *video, vector<Mat> kernelsVector) {
    vector<Mat> centersVector;
    int frameCounter = 0;
    #pragma omp parallel
    {
        Mat videoFrame;

        while (video->getNextFrame(videoFrame)) {

            #pragma omp critical
            {
            frameCounter++;
            cout<<frameCounter<<endl;
            }

            Mat vectors, centers, labels;

            FilterProcessor::filterImage(videoFrame, kernelsVector, vectors, centers, labels);
            #pragma omp critical
            centersVector.push_back(centers);
        }
    }
    cout<<"finished"<<endl;
    return centersVector;
}
