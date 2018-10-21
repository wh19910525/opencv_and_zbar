#include <stdint.h>
#include <sys/types.h>
#include <math.h>
//#include <utils/misc.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <assert.h>
#include <unistd.h>
#include <sys/file.h>
#include <sys/stat.h>


#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#include <unistd.h>

using namespace cv;
using namespace std;

#define Width 720
#define Height 480
#define Only_Y_size (Width*Height)

char data[Only_Y_size];

int readData(char * filename, int datalen){
    int ret = -1; 
//    char *data = (char *)malloc(datalen);
#ifndef O_BINARY
#  define O_BINARY  0
#endif
    int mode_fd = ::open(filename, O_RDONLY | O_BINARY);
    if (mode_fd < 0)
    {   
        printf("Unable to open file[%s]:%s\n", filename, strerror(errno));
        return -1;
    }else{
        printf("Open file[%s], success.\n", filename);
    } 

    bzero(data, sizeof(data));

    ret = ::read(mode_fd, data, datalen);
    if(ret > 0){
        printf("len=%d\n", ret);
    }   

    ::close(mode_fd);

    return ret;
}

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
    //Mat myMat = imread(argv[1], 1);
    readData(argv[1], Only_Y_size);
    Mat myMat(Height, Width, CV_8UC1, (unsigned char *) data);

    // 显示图像
    imshow("Opencv Image", myMat);

    // 等待按键, 延时 ms
    int keynum = waitKey(0);
    printf("key value = %d\n", keynum);

    return 0;
}
