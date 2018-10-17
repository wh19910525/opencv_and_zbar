#include <stddef.h>

#include <unistd.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "zbar.h" 
#include <iostream>

using namespace std;
using namespace cv;
using namespace zbar;  //添加zbar名称空间

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

    ImageScanner scanner;
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

    Mat image = imread(argv[1]);
    Mat imageGray;
    //RGB颜色空间 转换到 灰度空间;
    cvtColor(image, imageGray, CV_RGB2GRAY);
    int width = imageGray.cols;
    int height = imageGray.rows;
    uchar *raw = (uchar *)imageGray.data;

    Image imageZbar(width, height, "Y800", raw, width * height);

    //扫描条码;
    scanner.scan(imageZbar);
    Image::SymbolIterator symbol = imageZbar.symbol_begin();
    if(imageZbar.symbol_begin()==imageZbar.symbol_end())
    {
        cout<<"查询条码失败，请检查图片！"<<endl;
    }

    for(;symbol != imageZbar.symbol_end(); ++symbol)
    {
        cout << "    Type ：" << symbol->get_type_name() << endl;
        cout << "    Value：" << symbol->get_data()<<endl<<endl;
    }

    //imshow("Source Image", image);
    imshow("Source Image", imageGray);

    waitKey();

    imageZbar.set_data(NULL, 0);

    return 0;
} 

