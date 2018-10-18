#include <stddef.h>

#include <unistd.h>

#include "zbar.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc,char *argv[])
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

    Mat image,imageGray,imageGuussian;
    Mat imageSobelX,imageSobelY,imageSobelOut;
    image=imread(argv[1]);

    //1. 原图像大小调整，提高运算效率  
    resize(image,image,Size(500,300));  
    imshow("1.原图像",image);  

    //2. 转化为灰度图  
    cvtColor(image,imageGray,CV_RGB2GRAY);  
    imshow("2.灰度图",imageGray);  

    //3. 高斯平滑滤波  
    GaussianBlur(imageGray,imageGuussian,Size(3,3),0);  
    imshow("3.高斯平衡滤波",imageGuussian);  

    //4.求得水平和垂直方向灰度图像的梯度差,使用Sobel算子  
    Mat imageX16S,imageY16S;  
    Sobel(imageGuussian,imageX16S,CV_16S,1,0,3,1,0,4);  
    Sobel(imageGuussian,imageY16S,CV_16S,0,1,3,1,0,4);  
    convertScaleAbs(imageX16S,imageSobelX,1,0);  
    convertScaleAbs(imageY16S,imageSobelY,1,0);  
    imageSobelOut=imageSobelX-imageSobelY;  
    imshow("4.X方向梯度",imageSobelX);  
    imshow("4.Y方向梯度",imageSobelY);  
    imshow("4.XY方向梯度差",imageSobelOut);    

    //5.均值滤波，消除高频噪声  
    blur(imageSobelOut,imageSobelOut,Size(3,3));  
    imshow("5.均值滤波",imageSobelOut);   

    //6.二值化  
    Mat imageSobleOutThreshold;  
    threshold(imageSobelOut,imageSobleOutThreshold,180,255,CV_THRESH_BINARY);     
    imshow("6.二值化",imageSobleOutThreshold);  

    //7.闭运算，填充条形码间隙  
    Mat  element=getStructuringElement(0,Size(7,7));  
    morphologyEx(imageSobleOutThreshold,imageSobleOutThreshold,MORPH_CLOSE,element);      
    imshow("7.闭运算",imageSobleOutThreshold);  

    //8. 腐蚀，去除孤立的点  
    erode(imageSobleOutThreshold,imageSobleOutThreshold,element);  
    imshow("8.腐蚀",imageSobleOutThreshold);  

    //9. 膨胀，填充条形码间空隙，根据核的大小，有可能需要2~3次膨胀操作  
    dilate(imageSobleOutThreshold,imageSobleOutThreshold,element);  
    dilate(imageSobleOutThreshold,imageSobleOutThreshold,element);  
    dilate(imageSobleOutThreshold,imageSobleOutThreshold,element);  
    imshow("9.膨胀",imageSobleOutThreshold);        

    vector<vector<Point> > contours;  
    vector<Vec4i> hiera;  

    //10.通过findContours找到条形码区域的矩形边界  
    findContours(imageSobleOutThreshold,contours,hiera,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);  
    for(int i=0;i<contours.size();i++)  
    {  
        Rect rect=boundingRect((Mat)contours[i]);  
        rectangle(image,rect,Scalar(255),2);      
    }     
    imshow("10.找出二维码矩形区域",image);  

    waitKey();  
} 

