#include "MyClass.h"

MyClass::MyClass()
{
    srcimage = imread("qr.png");//"F:\Pictures\qr.png""F:\Pictures\条形码.png"
    //srcimage = imread("F:\\Pictures\\qr测试.png");
    //srcimage = imread("条码.jpg");
    if (srcimage.empty()){
        printf("文件不存在");
        exit(1);
    }
    //resize(srcimage, srcimage, Size(srcimage.size().width/2, srcimage.size().height/2));
    element = getStructuringElement(0, Size(7, 7));
}
/******************************************************
  函数名称： QrRun
  函数功能： 开始
  传入参数：
  返 回 值：
  建立时间： 2018-05-19
  修改时间：
  建 立 人： 范泽华
  修 改 人：
  其它说明：
 ******************************************************/
void MyClass::QrRun(){
    RNG rng(12345);
    //imshow("原图", srcimage);
    Mat src_all = srcimage.clone();
    Mat src_gray;
    //灰度处理
    src_gray = getBlur(getGray(srcimage));

    Scalar color = Scalar(1, 1, 255);
    Mat threshold_output;
    vector<vector<Point> > contours, contours2;
    vector<Vec4i> hierarchy;
    Mat drawing = Mat::zeros(srcimage.size(), CV_8UC3);
    Mat drawing2 = Mat::zeros(srcimage.size(), CV_8UC3);
    Mat drawingAllContours = Mat::zeros(srcimage.size(), CV_8UC3);

    threshold_output = getThold(src_gray);

    findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

    int c = 0, ic = 0, k = 0, area = 0;
    // 边缘检测 
    //通过黑色定位角作为父轮廓，有两个子轮廓的特点，筛选出三个定位角  
    int parentIdx = -1;
    for (int i = 0; i< contours.size(); i++)
    {
        //画出所以轮廓图  
        drawContours(drawingAllContours, contours, parentIdx, CV_RGB(255, 255, 255), 1, 8);
        if (hierarchy[i][2] != -1 && ic == 0)
        {
            parentIdx = i;
            ic++;
        }
        else if (hierarchy[i][2] != -1)
        {
            ic++;
        }
        else if (hierarchy[i][2] == -1)
        {
            ic = 0;
            parentIdx = -1;
        }
        //特征轮廓检测 - 》
        //有两个子轮廓  
        if (ic >= 2)
        {
            //保存找到的三个黑色定位角  
            contours2.push_back(contours[parentIdx]);
            //画出三个黑色定位角的轮廓  
            drawContours(drawing, contours, parentIdx, CV_RGB(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1, 8);
            ic = 0;
            parentIdx = -1;
        }
    }
    //提取特征点 
    //填充的方式画出黑色定位角的轮廓  
    for (int i = 0; i<contours2.size(); i++)
        drawContours(drawing2, contours2, i, CV_RGB(rng.uniform(100, 255), rng.uniform(100, 255), rng.uniform(100, 255)), -1, 4, hierarchy[k][2], 0, Point());

    //获取定位角的中心坐标  
    vector<Point> pointfind;
    for (int i = 0; i<contours2.size(); i++)
    {
        pointfind.push_back(Center_cal(contours2, i));
    }
    //排除干扰点
    Mat dst;
    Point point[3]; double angle; Mat rot_mat;
    ///选择合适的点-核心筛选
    if (pointfind.size()>3){
        double lengthA = 10000000000000000, lengthB = 10000000000000000000;
        for (int i = 0; i < pointfind.size(); i++){
            for (int j = 0; j < pointfind.size(); j++){
                for (int k = 0; k < pointfind.size(); k++){
                    if (i != j&&j != k&&i != k){
                        double dxa, dxb,dya,dyb;
                        double k1, k2, wa, wb;
                        dxa = pointfind[i].x - pointfind[j].x;
                        dxb = pointfind[i].x - pointfind[k].x;
                        dya = pointfind[i].y - pointfind[j].y;
                        dyb = pointfind[i].y - pointfind[k].y;
                        if (dxa == 0 || dxb == 0)continue;
                        k1 = dya/dxa;
                        k2 = dyb/dxb ;
                        wa = sqrt(pow(dya, 2) + pow(dya, 2));
                        wb = sqrt(pow(dyb, 2) + pow(dxb, 2));
                        double anglea = abs(atan(k1) * 180 / CV_PI) + abs(atan(k2) * 180 / CV_PI);
                        if (int(anglea)>=85&&int(anglea)<=95&&wa<=lengthA&&wb<=lengthB){
                            lengthA = wa;
                            lengthB = wb;
                            point[0] = pointfind[i];
                            point[1] = pointfind[j];
                            point[2] = pointfind[k];
                        }
                    }
                }
            }
        }
    }
    else{
        for (int i = 0; i < 3; i++){
            point[i] = pointfind[i];
        }
    }
    //绘制直角三角形 
    //计算轮廓的面积，计算定位角的面积，从而计算出边长  
    area = contourArea(contours2[0]);
    int area_side = cvRound(sqrt(double(area)));
    for (int i = 0; i < 3; i++){
        line(drawing2, point[i], point[(i + 1)%3], color, area_side / 2, 8);
    }

    //纠正旋转
    //判断是否正对
    if (!IsCorrect(point)){
        //进入修正环节
        double angle; Mat rot_mat;
        int start = 0;
        for (int i = 0; i < 3; i++){
            double k1, k2,kk;
            k1 = (point[i].y - point[(i + 1) % 3].y) / (point[i].x - point[(i + 1) % 3].x);
            k2 = (point[i].y - point[(i + 2) % 3].y) / (point[i].x - point[(i + 2) % 3].x);
            kk = k1*k2;
            if (k1*k2 <0)
                start = i;
        }
        double ax, ay, bx, by;
        ax = point[(start + 1) % 3].x;
        ay = point[(start + 1) % 3].y;
        bx = point[(start + 2) % 3].x;
        by = point[(start + 2) % 3].y;
        Point2f center(abs(ax - bx) / 2, abs(ay -by)/ 2);
        double dy = ay - by;
        double dx = ax - bx;
        double k3 = dy / dx;
        angle =atan(k3) * 180 / CV_PI;//转化角度
        rot_mat = getRotationMatrix2D(center, angle, 1.0);

        warpAffine(src_all, dst, rot_mat, src_all.size(), 1, 0, 0);//旋转原图查看
        warpAffine(drawing2, drawing2, rot_mat, src_all.size(), 1, 0, 0);//旋转连线图
        warpAffine(src_all, src_all, rot_mat, src_all.size(), 1, 0, 0);//旋转原图

        namedWindow("Dst");
        imshow("Dst", dst);
    }

    namedWindow("DrawingAllContours");
    imshow("DrawingAllContours", drawingAllContours);

    namedWindow("Drawing2");
    imshow("Drawing2", drawing2);

    namedWindow("Drawing");
    imshow("Drawing", drawing);

    //提取ROI
    //接下来要框出这整个二维码  
    Mat gray_all, threshold_output_all;
    vector<vector<Point> > contours_all;
    vector<Vec4i> hierarchy_all;
    cvtColor(drawing2, gray_all, CV_BGR2GRAY);


    threshold(gray_all, threshold_output_all, 45, 255, THRESH_BINARY);
    findContours(threshold_output_all, contours_all, hierarchy_all, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));//RETR_EXTERNAL表示只寻找最外层轮廓  


    Point2f fourPoint2f[4];
    //求最小包围矩形  
    RotatedRect rectPoint = minAreaRect(contours_all[1]);//pointfind.size()-3

    //将rectPoint变量中存储的坐标值放到 fourPoint的数组中  
    rectPoint.points(fourPoint2f);

    int maxx = 0, maxy = 0, minx = 100000, miny = 100000;
    for (int i = 0; i < 4; i++)
    {
        if (maxx < fourPoint2f[i].x)maxx = fourPoint2f[i].x;
        if (maxy < fourPoint2f[i].y)maxy = fourPoint2f[i].y;
        if (minx > fourPoint2f[i].x)minx = fourPoint2f[i].x;
        if (miny > fourPoint2f[i].y)miny = fourPoint2f[i].y;
        line(src_all, fourPoint2f[i % 4], fourPoint2f[(i + 1) % 4]
                , Scalar(0), 3);
    }
    namedWindow("Src_all");
    ///边际处理
    int set_inter = 5;
    while (true)
    {
        minx -= set_inter;
        miny -= set_inter;
        maxx += set_inter;
        maxy += set_inter;
        if (maxx > srcimage.size().width || maxy > srcimage.size().height || minx < 0 || miny < 0){
            minx += set_inter;
            miny += set_inter;
            maxx -= set_inter;
            maxy -= set_inter;
            set_inter--;
        }
        else
        {
            break;
        }
    }
    imshow("Src_all", src_all(Rect(minx, miny, maxx - minx, maxy - miny)));//ROI
    Mat fout = src_all(Rect(minx, miny, maxx - minx, maxy - miny));//ROI

    //识别
    Dis_code(fout, true);

    waitKey(0);
    destroyAllWindows();
}
/******************************************************
  函数名称： get Rect
  函数功能： 获取轮廓的中心点
  传入参数： vector<vector<Point> > contours, int i
  返 回 值：
  建立时间： 2018-05-19
  修改时间：
  建 立 人： 范泽华
  修 改 人：
  其它说明：
 ******************************************************/
Point MyClass::Center_cal(vector<vector<Point> > contours, int i)
{
    int centerx = 0, centery = 0, n = contours[i].size();
    //在提取的小正方形的边界上每隔周长个像素提取一个点的坐标，  
    //求所提取四个点的平均坐标（即为小正方形的大致中心）  
    centerx = (contours[i][n / 4].x + contours[i][n * 2 / 4].x + contours[i][3 * n / 4].x + contours[i][n - 1].x) / 4;
    centery = (contours[i][n / 4].y + contours[i][n * 2 / 4].y + contours[i][3 * n / 4].y + contours[i][n - 1].y) / 4;
    Point point1 = Point(centerx, centery);
    return point1;
}
/******************************************************
  函数名称： getRotate
  函数功能： 旋转保留原画
  传入参数： 
  返 回 值：
  建立时间： 2018-05-19
  修改时间：
  建 立 人： 范泽华
  修 改 人：
  其它说明：
 ******************************************************/
Mat MyClass::getRotate(Mat image, double angle){
    IplImage imgTmp = image;
    IplImage *img = cvCloneImage(&imgTmp);
    double a = sin(angle), b = cos(angle);
    int width = img->width, height = img->height;
    //旋转后的新图尺寸   
    int width_rotate = int(height * fabs(a) + width * fabs(b));
    int height_rotate = int(width * fabs(a) + height * fabs(b));
    IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), img->depth, img->nChannels);
    cvZero(img_rotate);
    //保证原图可以任意角度旋转的最小尺寸    
    int tempLength = sqrt((double)width * width + (double)height *height) + 10;
    int tempX = (tempLength + 1) / 2 - width / 2;
    int tempY = (tempLength + 1) / 2 - height / 2;
    IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), img->depth, img->nChannels);
    cvZero(temp);
    //将原图复制到临时图像tmp中心    
    cvSetImageROI(temp, cvRect(tempX, tempY, width, height));
    cvCopy(img, temp, NULL);
    cvResetImageROI(temp);
    //旋转数组map    
    // [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]    
    // [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]    
    float m[6];
    int w = temp->width;
    int h = temp->height;
    m[0] = b;
    m[1] = a;
    m[3] = -m[1];
    m[4] = m[0];
    // 将旋转中心移至图像中间    
    m[2] = w * 0.5f;
    m[5] = h * 0.5f;
    CvMat M = cvMat(2, 3, CV_32F, m);
    cvGetQuadrangleSubPix(temp, img_rotate, &M);
    cvReleaseImage(&temp);
    Mat out=cvarrToMat(img_rotate);
    return out;
}
/******************************************************
  函数名称： IsCorrect
  函数功能： 旋转保留原画
  传入参数：
  返 回 值：
  建立时间： 2018-05-19
  修改时间：
  建 立 人： 范泽华
  修 改 人：
  其它说明：
 ******************************************************/
bool MyClass::IsCorrect(Point point[]){
    for (int i = 0; i < 3; i++){
        if (point[i].x == point[(i + 1) % 3].x&&point[i].y == point[(i + 2) % 3].y)
            return true;
        if (point[i].y == point[(i + 1) % 3].y&&point[i].x == point[(i + 2) % 3].x)
            return true;
    }
    return false;
}

//////////////////////////////////////////
MyClass::~MyClass()
{
}

MyClass::MyClass(char* filename)
{
    srcimage = imread(filename);
    if (srcimage.empty()){
        printf("file[%s] no exist.\n", filename);
        exit(1);
    }

    printf("01 cols=%d, rows=%d\n", srcimage.cols, srcimage.rows);
    resize(srcimage, srcimage, Size(500, 500));
    printf("02 cols=%d, rows=%d\n", srcimage.cols, srcimage.rows);

    /*
     * 
     */
    element = getStructuringElement(MORPH_RECT, Size(7, 7));
    show_img_num = 1;

    memset(decode_filename, 0, NAME_LEN);
    memcpy(decode_filename, filename, strlen(filename));
}

void MyClass::Run(){
    Mat image;

    //1. 获取灰度图;
    image = getGray(srcimage, true);

    //2. 高斯平滑滤波;
    image = getGass(image, true);

    //3. Sobel x―y 梯度差;
    image = getSobel(image, true);

    /*
     * 4. 均值滤波除高频噪声;
     *      this step, you can use the other filters;
     */
    image = getBlur(image, true);

    //5. 二值化;
    image = getThold(image, true);

    //6. 闭运算;
    image = getBys(image, true, 1);

    //7. 腐蚀;
    image = getErode(image, true, 1);

    //8. 膨胀;
    image = getDilate(image, true, 3);

    //9. 获取ROI;
    image = getRect(image, srcimage, true);

    //10. start Zbar decode;
    Dis_code(image, true);

    waitKey();
}

void MyClass::set_img_name(char * name){
    memset(img_name, 0, NAME_LEN);
    sprintf(img_name, "%02d. %s", show_img_num++, name);
}

Mat MyClass::getGray(Mat image, bool show){
    Mat cimage;
    cvtColor(image, cimage, CV_RGBA2GRAY);
    if (show){
        set_img_name("Gary");
        imshow(img_name, cimage);
    }
    return cimage;
}

Mat MyClass::getGass(Mat image, bool show){
    Mat cimage;
    GaussianBlur(image, cimage, Size(3, 3), 0);
    if (show){
        set_img_name("Gaussian filter");
        imshow(img_name, cimage);
    }

    return cimage;
}

Mat MyClass::getSobel(Mat image, bool show){
    Mat cimageX16s, cimageY16s, imageSobelX, imageSobelY, out;

#if 1
    //zgj, why use this CV_16S ? 
    Sobel(image, cimageX16s, CV_16S, 1, 0, 3, 1, 0, 4);
    Sobel(image, cimageY16s, CV_16S, 0, 1, 3, 1, 0, 4);
#else
    Sobel(image, cimageX16s, -1, 1, 0, 3, 1, 0, 4);
    Sobel(image, cimageY16s, -1, 0, 1, 3, 1, 0, 4);
#endif 
    convertScaleAbs(cimageX16s, imageSobelX, 1, 0);
    convertScaleAbs(cimageY16s, imageSobelY, 1, 0);

#if 1
    //if the codebar is standard and horizontal, the effect is ok.
    out = imageSobelX - imageSobelY;
#else
    //out = imageSobelX;
    out = imageSobelY;
#endif
    if (show){
        set_img_name("Sobel x-y");
        imshow(img_name, out);
    }

    return out;
}

Mat MyClass::getBlur(Mat image, bool show){
    Mat cimage;
    blur(image, cimage, Size(3, 3));
    if (show){
        set_img_name("Blur filter");
        imshow(img_name, cimage);
    }

    return cimage;
}

Mat MyClass::getThold(Mat image, bool show){
    Mat cimage;
    //zgj, the thres is ?
    threshold(image, cimage, 112, 255, CV_THRESH_BINARY);
    if (show){
        set_img_name("Thres-hold");
        imshow(img_name, cimage);
    }

    return cimage;
}

Mat MyClass::getBys(Mat image, bool show, int times){
    for (int i = 0; i < times; i++){
        morphologyEx(image, image, MORPH_CLOSE, element);
    }
    if (show){
        set_img_name("Close");
        imshow(img_name, image);
    }

    return image;
}

Mat MyClass::getErode(Mat image, bool show, int times){
    for (int i = 0; i < times; i++){
        erode(image, image, element);
    }
    if (show){
        set_img_name("Erode");
        imshow(img_name, image);
    }

    return image;
}

Mat MyClass::getDilate(Mat image, bool show, int times){
    for (int i = 0; i < times; i++){
        dilate(image, image, element);
    }

    if (show){
        set_img_name("Dilate");
        imshow(img_name, image);
    }

    return image;
}

Mat MyClass::getRect(Mat image, Mat simage, bool show){
    vector<vector<Point> > contours;
    vector<Vec4i> hiera;
    Mat cimage;

    /*
     * 1. 查找物体的轮廓;
     */
    findContours(image, contours, hiera, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    /*
     * 2. 计算 每一个轮廓的面积;
     */
    vector<float> saveAllContourArea;
    for (int i = 0; i < contours.size(); i++)
    {
        saveAllContourArea.push_back(cv::contourArea(contours[i]));
    }

    /*
     * 3. 找出面积最大的轮廓;
     */
    double maxValue; Point maxLoc;
    minMaxLoc(saveAllContourArea, NULL, &maxValue, NULL, &maxLoc);

    /*
     * 4. 计算面积最大的轮廓的最小的外包矩形
     */
    RotatedRect minRect = minAreaRect(contours[maxLoc.x]);
    printf("angle=%f\n", minRect.angle);

    //为了防止找错,要检查这个矩形的偏斜角度不能超标
    //如果超标, 那就是没找到;
    if (minRect.angle < 2.0)
    {
        //找到了矩形的角度, 但是这是一个旋转矩形, 所以还要重新获得一个外包最小矩形
        Rect myRect = boundingRect(contours[maxLoc.x]);
#if 0
        //把这个矩形在源图像中画出来
        rectangle(srcimage, myRect, Scalar(0,255,255), 3, LINE_AA);

        //看看显示效果,找的对不对
        imshow("find Rect", srcimage);

        waitKey(0);
#endif
        /*
         * 将扫描的图像裁剪下来, 并保存为相应的结果, 用来解码;
         *     保留一些X方向的边界, 对rect进行一定的扩张;
         */
        //myRect.x = myRect.x - (myRect.width / 20);
        //myRect.width = myRect.width*1.1;
        Mat resultImage = Mat(srcimage, myRect);

        if (show){
            set_img_name("Cut Src Rect");
            imshow(img_name, resultImage);
        }

        return resultImage;
    }

    for (int i = 0; i<contours.size(); i++)
    {
        Rect rect = boundingRect((Mat)contours[i]);
        //cimage = simage(rect);
        rectangle(simage, rect, Scalar(0), 2);
        if (show)
            imshow("转变图", simage);
    }

    return simage;
}

void MyClass::Dis_code(Mat image, bool show){
    //定义一个扫描仪  
    ImageScanner scanner;
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

    // 所转化成的灰度图像 
    Mat imageGray;

    cvtColor(image, imageGray, CV_RGB2GRAY);

    if (show){
        set_img_name("Decode img");
        imshow(img_name, imageGray);
    }

    // 获取所摄取图像的长和宽  
    int width = imageGray.cols;
    int height = imageGray.rows;
    /*
     * 在Zbar中进行扫描时候，需要将OpenCV中的Mat类型转换为（uchar *）类型,
     *     raw中存放的是图像的地址; 对应的图像需要转成Zbar中对应的图像zbar::Image  
     */
    uchar *raw = (uchar *)imageGray.data;
    Image imageZbar(width, height, "Y800", raw, width * height);

    cout << " file : " << decode_filename << endl;

    // 扫描相应的图像imageZbar(imageZbar是zbar::Image类型，存储着读入的图像)  
    scanner.scan(imageZbar); //扫描条码      
    Image::SymbolIterator symbol = imageZbar.symbol_begin();
    if (imageZbar.symbol_begin() == imageZbar.symbol_end())
    {
        cout << "    失败, 请检查图片！" << endl;
    }
    for (; symbol != imageZbar.symbol_end(); ++symbol)
    {
        cout << "    类型：" << symbol->get_type_name() << endl;
        cout << "    条码：" << symbol->get_data() << endl;
    }

    //等待按下;
    waitKey();

    // 将图像中的数据置为0  
    imageZbar.set_data(NULL, 0);
}

//////////////////////////////////////////
int main(int argc, char ** argv)
{
    if (argc < 2){
        MyClass *myclass = new MyClass();
        myclass->QrRun();
        delete myclass;
    }
    else
    {
        MyClass *myclass = new MyClass(argv[1]);
        myclass->Run();
        //myclass->QrRun();
        delete myclass;
    }
    return 0;
}

