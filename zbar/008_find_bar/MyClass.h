#include <opencv2/opencv.hpp>
#include <iostream>
#include <zbar.h>

#define NAME_LEN (100)

using namespace cv;
using namespace std;
using namespace zbar;

class MyClass
{
public:
    MyClass();
    ~MyClass();
    void QrRun();

    Mat getRotate(Mat image,double angle);
    bool IsCorrect(Point point[]);//判断是否正对
    Point Center_cal(vector<vector<Point> > contours, int i); 



    MyClass(char* argv);

    void Run();

    Mat getGray(Mat image, bool show = false);//获取灰度图

    Mat getGass(Mat image, bool show = false);//高斯平滑滤波

    Mat getSobel(Mat image, bool show = false);//Sobel x―y梯度差

    Mat getBlur(Mat image, bool show = false);//均值滤波除高频噪声

    Mat getThold(Mat image, bool show = false);//二值化

    Mat getBys(Mat image, bool show = false);//闭运算

    Mat getErode(Mat image, bool show = false);//腐蚀

    Mat getDilate(Mat image, bool show = false);//膨胀

    Mat getRect(Mat image, Mat simage, bool show = false);//获取范围

    void Dis_code(Mat image, bool show = false);

    void set_img_name(char * name);
private:
    char img_name[NAME_LEN];
    char decode_filename[NAME_LEN];
    int show_img_num;
    Mat srcimage;//原图
    Mat element;//核
};


