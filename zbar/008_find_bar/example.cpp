/******************************************************
�ļ���   :main.cpp
��  ��   :�����룬��ά���ʶ��
��  ��   :
��  ��   :����
��  ��   :
��  ��   :2018-05-19
˵  ��   :��ҪZbar��֧��
******************************************************/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <zbar.h>

using namespace cv;
using namespace std;
using namespace zbar;

class MyClass
{
public:
	MyClass();
	MyClass(char* argv);
	~MyClass();
	void Dis_code(Mat image);
	void Run();
	void QrRun();
	Mat getGray(Mat image, bool show = false);//��ȡ�Ҷ�ͼ
	Mat getGass(Mat image, bool show = false);//��˹ƽ���˲�
	Mat getSobel(Mat image, bool show = false);//Sobel x��y�ݶȲ�
	Mat getBlur(Mat image, bool show = false);//��ֵ�˲�����Ƶ����
	Mat getThold(Mat image, bool show = false);//��ֵ��
	Mat getBys(Mat image, bool show = false);//������
	Mat getErode(Mat image, bool show = false);//��ʴ
	Mat getDilate(Mat image, bool show = false);//����
	Mat getRect(Mat image, Mat simage, bool show = false);//��ȡ��Χ
	Mat getRotate(Mat image,double angle);
	bool IsCorrect(Point point[]);//�ж��Ƿ�����
	Point Center_cal(vector<vector<Point> > contours, int i);
private:
	Mat srcimage;//ԭͼ
	Mat element;//��
};
/******************************************************
�������ƣ� MyClass
�������ܣ� ��ʼ��
���������
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
MyClass::MyClass()
{
	srcimage = imread("qr.png");//"F:\Pictures\qr.png""F:\Pictures\������.png"
	//srcimage = imread("F:\\Pictures\\qr����.png");
	//srcimage = imread("����.jpg");
	if (srcimage.empty()){
		printf("�ļ�������");
		exit(1);
	}
	//resize(srcimage, srcimage, Size(srcimage.size().width/2, srcimage.size().height/2));
	element = getStructuringElement(0, Size(7, 7));
	//Dis_code(srcimage);
}
/******************************************************
�������ƣ� MyClass
�������ܣ� ��ʼ��
��������� char* argv
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
MyClass::MyClass(char* argv)
{
	srcimage = imread(argv);
	if (srcimage.empty()){
		printf("�ļ�������");
		exit(1);
	}
	resize(srcimage, srcimage, Size(500, 500));
	element = getStructuringElement(0, Size(7, 7));
}
/******************************************************
�������ƣ� ~MyClass
�������ܣ� �ͷſռ�
���������
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
MyClass::~MyClass()
{
}
/******************************************************
�������ƣ� Dis_Barcode
�������ܣ� ʶ��������Ͷ�ά��
���������
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵���������ǽ�������˵Ĵ��룺
ԭ�����ӣ�https://www.cnblogs.com/dengxiaojun/p/5278679.html
���´����Ǿ����Ķ���
******************************************************/
void MyClass::Dis_code(Mat image){
	Mat imageGray;  // ��ת���ɵĻҶ�ͼ�� 
	//����һ��ɨ����  
	ImageScanner scanner;
	scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

	cvtColor(image, imageGray, CV_RGB2GRAY);
	imshow("�Ҷ�ͼ", imageGray);
	// ��ȡ����ȡͼ��ĳ��Ϳ�  
	int width = imageGray.cols;
	int height = imageGray.rows;
	// ��Zbar�н���ɨ��ʱ����Ҫ��OpenCV�е�Mat����ת��Ϊ��uchar *�����ͣ�raw�д�ŵ���ͼ��ĵ�ַ����Ӧ��ͼ����Ҫת��Zbar�ж�Ӧ��ͼ��zbar::Image  
	uchar *raw = (uchar *)imageGray.data;
	Image imageZbar(width, height, "Y800", raw, width * height);
	// ɨ����Ӧ��ͼ��imageZbar(imageZbar��zbar::Image���ͣ��洢�Ŷ����ͼ��)  
	scanner.scan(imageZbar); //ɨ������      
	Image::SymbolIterator symbol = imageZbar.symbol_begin();
	if (imageZbar.symbol_begin() == imageZbar.symbol_end())
	{
		cout << "��ѯ����ʧ�ܣ�����ͼƬ��" << endl;
	}
	for (; symbol != imageZbar.symbol_end(); ++symbol)
	{
		cout << "���ͣ�" << endl << symbol->get_type_name() << endl << endl;
		cout << "���룺" << endl << symbol->get_data() << endl << endl;
	}

	waitKey(); // �ȴ�����esc��������Ҫ��ʱ1s�����waitKey(1000);  

	// ��ͼ���е�������Ϊ0  
	imageZbar.set_data(NULL, 0);
	system("pause");
}
/******************************************************
�������ƣ� Run
�������ܣ� ��ʼ
���������
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
void MyClass::Run(){
	Mat image;
	image = getGray(srcimage,true);//��ȡ�Ҷ�ͼ
	image = getGass(image, true);//��˹ƽ���˲�
	image = getSobel(image, true);//Sobel x��y�ݶȲ�
	image = getBlur(image, true);//��ֵ�˲�����Ƶ����
	image = getThold(image, true);//��ֵ��
	image = getBys(image, true);//������
	image = getErode(image, true);//��ʴ
	image = getDilate(image, true);//����
	image = getRect(image, srcimage, true);//��ȡROI
	imshow("����ͼ", image);
    Dis_code(image);
	waitKey();
}
/******************************************************
�������ƣ� QrRun
�������ܣ� ��ʼ
���������
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
void MyClass::QrRun(){
	RNG rng(12345);
	//imshow("ԭͼ", srcimage);
	Mat src_all = srcimage.clone();
	Mat src_gray;
	//�Ҷȴ���
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
	// ��Ե��� 
	//ͨ����ɫ��λ����Ϊ�����������������������ص㣬ɸѡ��������λ��  
	int parentIdx = -1;
	for (int i = 0; i< contours.size(); i++)
	{
		//������������ͼ  
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
		//����������� - ��
		//������������  
		if (ic >= 2)
		{
			//�����ҵ���������ɫ��λ��  
			contours2.push_back(contours[parentIdx]);
			//����������ɫ��λ�ǵ�����  
			drawContours(drawing, contours, parentIdx, CV_RGB(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1, 8);
			ic = 0;
			parentIdx = -1;
		}
	}
	//��ȡ������ 
	//���ķ�ʽ������ɫ��λ�ǵ�����  
	for (int i = 0; i<contours2.size(); i++)
		drawContours(drawing2, contours2, i, CV_RGB(rng.uniform(100, 255), rng.uniform(100, 255), rng.uniform(100, 255)), -1, 4, hierarchy[k][2], 0, Point());

	//��ȡ��λ�ǵ���������  
	vector<Point> pointfind;
	for (int i = 0; i<contours2.size(); i++)
	{
		pointfind.push_back(Center_cal(contours2, i));
	}
	//�ų����ŵ�
	Mat dst;
	Point point[3]; double angle; Mat rot_mat;
	///ѡ����ʵĵ�-����ɸѡ
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
	//����ֱ�������� 
	//������������������㶨λ�ǵ�������Ӷ�������߳�  
	area = contourArea(contours2[0]);
	int area_side = cvRound(sqrt(double(area)));
	for (int i = 0; i < 3; i++){
		line(drawing2, point[i], point[(i + 1)%3], color, area_side / 2, 8);
	}

	//������ת
	//�ж��Ƿ�����
	if (!IsCorrect(point)){
	//������������
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
		angle =atan(k3) * 180 / CV_PI;//ת���Ƕ�
		rot_mat = getRotationMatrix2D(center, angle, 1.0);
		
		warpAffine(src_all, dst, rot_mat, src_all.size(), 1, 0, 0);//��תԭͼ�鿴
		warpAffine(drawing2, drawing2, rot_mat, src_all.size(), 1, 0, 0);//��ת����ͼ
		warpAffine(src_all, src_all, rot_mat, src_all.size(), 1, 0, 0);//��תԭͼ

		namedWindow("Dst");
		imshow("Dst", dst);
	}

	namedWindow("DrawingAllContours");
	imshow("DrawingAllContours", drawingAllContours);

	namedWindow("Drawing2");
	imshow("Drawing2", drawing2);

	namedWindow("Drawing");
	imshow("Drawing", drawing);

	//��ȡROI
	//������Ҫ�����������ά��  
	Mat gray_all, threshold_output_all;
	vector<vector<Point> > contours_all;
	vector<Vec4i> hierarchy_all;
	cvtColor(drawing2, gray_all, CV_BGR2GRAY);


	threshold(gray_all, threshold_output_all, 45, 255, THRESH_BINARY);
	findContours(threshold_output_all, contours_all, hierarchy_all, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));//RETR_EXTERNAL��ʾֻѰ�����������  


	Point2f fourPoint2f[4];
	//����С��Χ����  
	RotatedRect rectPoint = minAreaRect(contours_all[1]);//pointfind.size()-3

	//��rectPoint�����д洢������ֵ�ŵ� fourPoint��������  
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
	///�߼ʴ���
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

	//ʶ��
    Dis_code(fout);

	waitKey(0);
	destroyAllWindows();
}
/******************************************************
�������ƣ� getGray
�������ܣ� �Ҷȴ���
��������� Mat image
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
Mat MyClass::getGray(Mat image, bool show){
	Mat cimage;
	cvtColor(image, cimage, CV_RGBA2GRAY);
	if (show)
	imshow("�Ҷ�ͼ", cimage);
	return cimage;
}
/******************************************************
�������ƣ� getGass
�������ܣ� ��˹�˲�����
��������� Mat image
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
Mat MyClass::getGass(Mat image, bool show){
	Mat cimage;
	GaussianBlur(image, cimage, Size(3, 3), 0);
	if (show)
	imshow("��˹�˲�ͼ", cimage);
	return cimage;
}
/******************************************************
�������ƣ� getSobel
�������ܣ� Sobel����
��������� Mat image
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
Mat MyClass::getSobel(Mat image, bool show){
	Mat cimageX16s, cimageY16s, imageSobelX, imageSobelY, out;
	Sobel(image, cimageX16s, CV_16S, 1, 0, 3, 1, 0, 4);
	Sobel(image, cimageY16s, CV_16S, 0, 1, 3, 1, 0, 4);
	convertScaleAbs(cimageX16s, imageSobelX, 1, 0);
	convertScaleAbs(cimageY16s, imageSobelY, 1, 0);
	out = imageSobelX - imageSobelY;
	if (show)
	imshow("Sobelx-y�� ͼ", out);
	return out;
}
/******************************************************
�������ƣ� getThold
�������ܣ� ��ֵ������
��������� Mat image
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
Mat MyClass::getThold(Mat image, bool show){
	Mat cimage;
	threshold(image, cimage, 112, 255, CV_THRESH_BINARY);
	if (show)
	imshow("��ֵ��ͼ", cimage);
	return cimage;
}
/******************************************************
�������ƣ� getBys
�������ܣ� �����㴦��
��������� Mat image
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
Mat MyClass::getBys(Mat image, bool show){
	morphologyEx(image, image, MORPH_CLOSE, element);
	if (show)
	imshow("������ͼ", image);
	return image;
}
/******************************************************
�������ƣ� getBlur
�������ܣ� ��ֵ�˲�����
��������� Mat image
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
Mat MyClass::getBlur(Mat image, bool show){
	Mat cimage;
	blur(image, cimage, Size(3, 3));
	if (show)
	imshow("��ֵ�˲�ͼ", cimage);
	return cimage;
}
/******************************************************
�������ƣ� getErode
�������ܣ� ��ʴ����
��������� Mat image
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
Mat MyClass::getErode(Mat image, bool show){
	//Mat cimage;
	erode(image, image, element);
	if (show)
	imshow("��ʴͼ", image);
	return image;
}
/******************************************************
�������ƣ� getDilate
�������ܣ� ���ʹ���
��������� Mat image
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
Mat MyClass::getDilate(Mat image, bool show){
	for (int i = 0; i < 3; i++)
		dilate(image, image, element);
	if (show)
	imshow("����ͼ", image);
	return image;
}
/******************************************************
�������ƣ� getRect
�������ܣ� ��ȡ�������
��������� Mat image�� Mat simageԭͼ
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
Mat MyClass::getRect(Mat image, Mat simage, bool show){
	vector<vector<Point> > contours;
	vector<Vec4i> hiera;
	Mat cimage;
	findContours(image, contours, hiera, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<float>contourArea;
	for (int i = 0; i < contours.size(); i++)
	{
		contourArea.push_back(cv::contourArea(contours[i]));
	}
	//�ҳ������������
	double maxValue; Point maxLoc;
	minMaxLoc(contourArea, NULL, &maxValue, NULL, &maxLoc);
	//�������������������С���������
	RotatedRect minRect = minAreaRect(contours[maxLoc.x]);
	//Ϊ�˷�ֹ�Ҵ�,Ҫ���������ε�ƫб�ǶȲ��ܳ���
	//�������,�Ǿ���û�ҵ�
	if (minRect.angle<2.0)
	{
		//�ҵ��˾��εĽǶ�,��������һ����ת����,���Ի�Ҫ���»��һ�������С����
		Rect myRect = boundingRect(contours[maxLoc.x]);
		//�����������Դͼ���л�����
		//rectangle(srcImage,myRect,Scalar(0,255,255),3,LINE_AA);
		//������ʾЧ��,�ҵĶԲ���
		//imshow(windowNameString,srcImage);
		//��ɨ���ͼ��ü�����,������Ϊ��Ӧ�Ľ��,����һЩX����ı߽�,���Զ�rect����һ��������
		myRect.x = myRect.x - (myRect.width / 20);
		myRect.width = myRect.width*1.1;
		Mat resultImage = Mat(srcimage, myRect);
		return resultImage;
	}

	for (int i = 0; i<contours.size(); i++)
	{
		Rect rect = boundingRect((Mat)contours[i]);
		//cimage = simage(rect);
	    rectangle(simage, rect, Scalar(0), 2);
		if (show)
		imshow("ת��ͼ", simage);
	}
	return simage;
}
/******************************************************
�������ƣ� getRect
�������ܣ� ��ȡ���������ĵ�
��������� vector<vector<Point> > contours, int i
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
Point MyClass::Center_cal(vector<vector<Point> > contours, int i)
{
	int centerx = 0, centery = 0, n = contours[i].size();
	//����ȡ��С�����εı߽���ÿ���ܳ���������ȡһ��������꣬  
	//������ȡ�ĸ����ƽ�����꣨��ΪС�����εĴ������ģ�  
	centerx = (contours[i][n / 4].x + contours[i][n * 2 / 4].x + contours[i][3 * n / 4].x + contours[i][n - 1].x) / 4;
	centery = (contours[i][n / 4].y + contours[i][n * 2 / 4].y + contours[i][3 * n / 4].y + contours[i][n - 1].y) / 4;
	Point point1 = Point(centerx, centery);
	return point1;
}
/******************************************************
�������ƣ� getRotate
�������ܣ� ��ת����ԭ��
��������� 
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
******************************************************/
Mat MyClass::getRotate(Mat image, double angle){
	IplImage imgTmp = image;
	IplImage *img = cvCloneImage(&imgTmp);
	double a = sin(angle), b = cos(angle);
	int width = img->width, height = img->height;
	//��ת�����ͼ�ߴ�   
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), img->depth, img->nChannels);
	cvZero(img_rotate);
	//��֤ԭͼ��������Ƕ���ת����С�ߴ�    
	int tempLength = sqrt((double)width * width + (double)height *height) + 10;
	int tempX = (tempLength + 1) / 2 - width / 2;
	int tempY = (tempLength + 1) / 2 - height / 2;
	IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), img->depth, img->nChannels);
	cvZero(temp);
	//��ԭͼ���Ƶ���ʱͼ��tmp����    
	cvSetImageROI(temp, cvRect(tempX, tempY, width, height));
	cvCopy(img, temp, NULL);
	cvResetImageROI(temp);
	//��ת����map    
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]    
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]    
	float m[6];
	int w = temp->width;
	int h = temp->height;
	m[0] = b;
	m[1] = a;
	m[3] = -m[1];
	m[4] = m[0];
	// ����ת��������ͼ���м�    
	m[2] = w * 0.5f;
	m[5] = h * 0.5f;
	CvMat M = cvMat(2, 3, CV_32F, m);
	cvGetQuadrangleSubPix(temp, img_rotate, &M);
	cvReleaseImage(&temp);
	Mat out=cvarrToMat(img_rotate);
	return out;
}
/******************************************************
�������ƣ� IsCorrect
�������ܣ� ��ת����ԭ��
���������
�� �� ֵ��
����ʱ�䣺 2018-05-19
�޸�ʱ�䣺
�� �� �ˣ� ����
�� �� �ˣ�
����˵����
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
int main(int argc, char *argv)
{
	if (argc < 2){
		MyClass *myclass = new MyClass();
		//myclass->Run();
		myclass->QrRun();
		delete myclass;
	}
	else
	{
		MyClass *myclass = new MyClass(argv);
		myclass->Run();
		myclass->QrRun();
		delete myclass;
	}
	return 0;
}
