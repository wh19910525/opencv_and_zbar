#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#include <unistd.h>

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
    if(argc != 2){
        cout << "Please Usage: " << argv[0] << " [xxx.jpg]" << endl;
        return -1;
    }

    if( access(argv[1], F_OK ) == -1 ){
        cout << "File : " << argv[1] << " do not exist." << endl;
        return -2;
    }

    // 载入图像
    Mat myMat = imread(argv[1], 1);

    // 创建一个窗口
    //namedWindow("Opencv Image, 01", WINDOW_AUTOSIZE);
    //namedWindow("Opencv Image, 01", WINDOW_NORMAL);

    // 显示图像
    imshow("Opencv Image, 02", myMat);

    // 等待按键, 延时 ms
    int keynum = waitKey(0);
    printf("key value = %d\n", keynum);

    return 0;
}
