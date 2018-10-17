#include <stddef.h>

#include <unistd.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "zbar.h"
#include <iostream>

using namespace std;
using namespace zbar;  //添加zbar名称空间
using namespace cv;

Rect DrawXYProjection(const Mat image, Mat &imageOut, const int threshodValue, const int binaryzationValue);

///////////////////////////
int main(int argc, char*argv[])
{
    printf("Build time[%s, %s]\n", __DATE__, __TIME__);

    if(argc != 2){
        cout << "Please Usage: " << argv[0] << " [xxx.jpg]" << endl;
        return -1;
    }

    if( access(argv[1], F_OK ) == -1 ){
        cout << "File : " << argv[1] << " do not exist." << endl;
        return -2;
    }

    Mat image = imread(argv[1]);

    Mat imageCopy = image.clone();

    Mat imageGray, imagOut;
    cvtColor(image, imageGray, CV_RGB2GRAY);

    ////////////////
    Rect rect(0, 0, 0, 0);
    rect = DrawXYProjection(image, imagOut, image.rows/10, 100);

    Mat roi = image(rect);

    //画出条形码的矩形框
    rectangle(imageCopy, Point(rect.x,rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(0,0,255), 2);

    imshow("Source Image", image);
    imshow("Source Image Rect", imageCopy);

    imshow("水平垂直投影", imagOut);
    imshow("Output Image", roi);

    waitKey();      

    return 0;
}

//***********************************************
// 函数通过水平和垂直方向投影, 找到两个方向上投影的交叉矩形, 定位到条形码/二维码
// int threshodValue 投影的最少像素单位
// int binaryzationValue  原图像阈值分割值
//***********************************************
Rect DrawXYProjection(const Mat image, Mat &imageOut, const int threshodValue, const int binaryzationValue)
{
    Mat img = image.clone();
    printf("img-channels=%d\n", img.channels());
    if(img.channels() > 1)
    {
        cvtColor(img, img, CV_RGB2GRAY);
    }

    //zgj 01;
    Mat out(img.size(), img.type(), Scalar(255));
    imageOut = out;

    /*
     * zgj 02;
     *
     * 1. 对每一个传入的图片做灰度归一化, 以便使用同一套阈值参数
     */
    normalize(img, img, 0, 255, NORM_MINMAX);
    /*
     * 2. 垂直方向投影, 构建数组向量, 包含img.cols维度 
     */
    vector<int> vectorVertical(img.cols, 0);
    for(int i=0; i<img.cols; i++)
    {
        for(int j=0; j<img.rows; j++)
        {
            if(img.at<uchar>(j,i)<binaryzationValue)
            {
                vectorVertical[i]++;
            }
        }
    }

    /* 
     * 3. 列值归一化
     */
    int high=img.rows/6;
    normalize(vectorVertical,vectorVertical,0,high,NORM_MINMAX);
    for(int i=0;i<img.cols;i++)
    {
        for(int j=0;j<img.rows;j++)
        {
            if(vectorVertical[i]>threshodValue)
            {
                line(imageOut,Point(i,img.rows),Point(i,img.rows-vectorVertical[i]),Scalar(0));
            }
        }
    }

    /*
     * 4.水平方向投影, 构建数组向量, 包含img.rows维度 
     */
    vector<int> vectorHorizontal(img.rows,0);
    for(int i=0;i<img.rows;i++)
    {
        for(int j=0;j<img.cols;j++)
        {
            if(img.at<uchar>(i,j)<binaryzationValue)
            {
                vectorHorizontal[i]++;
            }
        }
    }

    normalize(vectorHorizontal,vectorHorizontal,0,high,NORM_MINMAX);
    for(int i=0;i<img.rows;i++)
    {
        for(int j=0;j<img.cols;j++)
        {
            if(vectorHorizontal[i]>threshodValue)
            {
                line(imageOut,Point(img.cols-vectorHorizontal[i],i),Point(img.cols,i),Scalar(0));
            }
        }
    }

    /*
     * 5. 找到投影四个角点坐标
     */
    vector<int>::iterator beginV=vectorVertical.begin();
    vector<int>::iterator beginH=vectorHorizontal.begin();
    vector<int>::iterator endV=vectorVertical.end()-1;
    vector<int>::iterator endH=vectorHorizontal.end()-1;
    int widthV=0;
    int widthH=0;
    int highV=0;
    int highH=0;
    while(*beginV<threshodValue)
    {
        beginV++;
        widthV++;
    }
    while(*endV<threshodValue)
    {
        endV--;
        widthH++;
    }
    while(*beginH<threshodValue)
    {
        beginH++;
        highV++;
    }
    while(*endH<threshodValue)
    {
        endH--;
        highH++;
    }

    /*
     * 6. 投影矩形
     */
    Rect rect(widthV,highV,img.cols-widthH-widthV,img.rows-highH-highV);

    return rect;
}

