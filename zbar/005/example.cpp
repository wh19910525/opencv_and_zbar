#include <stddef.h>

#include <unistd.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#include "zbar.h"

using namespace std;
using namespace zbar;
using namespace cv;

int main(int argc, char *argv[])
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

    Mat image, imageGray, imageGuussian;
    Mat imageSobelX,imageSobelY,imageSobelOut;

    /*
     * 1. Read Source Image;
     */
    imageGray = imread(argv[1], 0);
    imageGray.copyTo(image);
    imshow("1. Source Image", image);

    GaussianBlur(imageGray, imageGuussian, Size(3,3), 0);
    /*
     * 2. 水平和垂直方向灰度图像的梯度和, 突出图像边缘信息, 使用Sobel算子
     */
    Mat imageX16S,imageY16S;
    Sobel(imageGuussian,imageX16S,CV_16S,1,0,3,1,0,4);
    Sobel(imageGuussian,imageY16S,CV_16S,0,1,3,1,0,4);
    convertScaleAbs(imageX16S,imageSobelX,1,0);
    convertScaleAbs(imageY16S,imageSobelY,1,0);
    imageSobelOut=imageSobelX+imageSobelY;
    imshow("2. XY方向梯度和", imageSobelOut);

    Mat srcImg = imageSobelOut;
    //宽高扩充，非必须，特定的宽高可以提高傅里叶运算效率
    Mat padded;
    int opWidth = getOptimalDFTSize(srcImg.rows);
    int opHeight = getOptimalDFTSize(srcImg.cols);
    copyMakeBorder(srcImg, padded, 0, opWidth-srcImg.rows, 0, opHeight-srcImg.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat comImg;
    //通道融合，融合成一个2通道的图像
    merge(planes, 2, comImg);
    dft(comImg, comImg);
    split(comImg, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magMat = planes[0];
    magMat += Scalar::all(1);
    log(magMat, magMat);     //对数变换，方便显示
    magMat = magMat(Rect(0, 0, magMat.cols & -2, magMat.rows & -2));

    //以下把傅里叶频谱图的四个角落移动到图像中心
    int cx = magMat.cols/2;  
    int cy = magMat.rows/2;   
    Mat q0(magMat, Rect(0, 0, cx, cy));  
    Mat q1(magMat, Rect(0, cy, cx, cy));  
    Mat q2(magMat, Rect(cx, cy, cx, cy));  
    Mat q3(magMat, Rect(cx, 0, cx, cy));   
    Mat tmp;  
    q0.copyTo(tmp);  
    q2.copyTo(q0);  
    tmp.copyTo(q2);   
    q1.copyTo(tmp);  
    q3.copyTo(q1);  
    tmp.copyTo(q3);  
    normalize(magMat, magMat, 0, 1, CV_MINMAX);  
    Mat magImg(magMat.size(), CV_8UC1);  
    magMat.convertTo(magImg,CV_8UC1,255,0);  
    imshow("3. 傅里叶频谱", magImg); 

    //HoughLines查找傅里叶频谱的直线，该直线跟原图的一维码方向相互垂直
    threshold(magImg, magImg, 180, 255, CV_THRESH_BINARY);
    imshow("4. 二值化", magImg);

    vector<Vec2f> lines;
    float pi180 = (float)CV_PI/180;  
    Mat linImg(magImg.size(),CV_8UC3);  
    HoughLines(magImg,lines,1,pi180,100,0,0);  
    int numLines = lines.size();  
    float theta;
    for(int l=0; l<numLines; l++)  
    {
        float rho = lines[l][0];
        theta = lines[l][1];  
        float aa=(theta/CV_PI)*180;
        Point pt1, pt2;  
        double a = cos(theta), b = sin(theta);  
        double x0 = a*rho, y0 = b*rho;  
        pt1.x = cvRound(x0 + 1000*(-b));  
        pt1.y = cvRound(y0 + 1000*(a));  
        pt2.x = cvRound(x0 - 1000*(-b));  
        pt2.y = cvRound(y0 - 1000*(a));  
        line(linImg,pt1,pt2,Scalar(255,0,0),3,8,0);     
    }  
    imshow("5. Hough直线", linImg);  

    /*
     * 6. 校正角度计算
     */
    float angelD = 180*theta/CV_PI-90;
    Point center(image.cols/2, image.rows/2);//
    Mat rotMat = getRotationMatrix2D(center, angelD, 1.0);//
    Mat imageSource = Mat::ones(image.size(), CV_8UC3);
    warpAffine(image, imageSource, rotMat, image.size(), 1, 0, Scalar(255,255,255));//仿射变换校正图像
    imshow("6. 角度校正", imageSource);

    //Zbar一维码识别
    ImageScanner scanner;        
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);   
    int width1 = imageSource.cols;        
    int height1 = imageSource.rows;        
    uchar *raw = (uchar *)imageSource.data;           
    Image imageZbar(width1, height1, "Y800", raw, width1 * height1);          
    scanner.scan(imageZbar); //扫描条码      
    Image::SymbolIterator symbol = imageZbar.symbol_begin();    
    if(imageZbar.symbol_begin()==imageZbar.symbol_end())    
    {    
        cout<<"查询条码失败，请检查图片！"<<endl;    
    }    
    for(;symbol != imageZbar.symbol_end();++symbol)      
    {        
        cout<<"类型："<<endl<<symbol->get_type_name()<<endl<<endl;      
        cout<<"条码："<<endl<<symbol->get_data()<<endl<<endl;         
    }        

    waitKey();      
    imageZbar.set_data(NULL,0);    

    return 0;    
}  

