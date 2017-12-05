#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include "opencv2/video/tracking.hpp"

#include "cameraParameters.h"
#include "pointDefinition.h"

#include <iostream>

using namespace std;
using namespace cv;

bool systemInited = false;
bool isOddFrame = true;
double timeCur, timeLast;

const int imagePixelNum = imageHeight * imageWidth;
int rows = imageHeight;
int cols = imageWidth;
CvSize imgSize = cvSize(imageWidth, imageHeight);

IplImage *imageCur = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
IplImage *imageLast = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
IplImage *imageHarris = cvCreateImage(imgSize, IPL_DEPTH_32F, 1);
IplImage *imageHarrisNorm = cvCreateImage(imgSize, IPL_DEPTH_32F, 1);
IplImage *imageHarrisNormScaled = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
IplImage *mask;

Mat image0, image1;

Mat mapxMap, mapyMap;

int showCount = 0;
const int showSkipNum = 0;
const int showDSRate = 1;
CvSize showSize = cvSize(imageWidth / showDSRate, imageHeight / showDSRate);

IplImage *imageShow = cvCreateImage(showSize, IPL_DEPTH_8U, 1);
IplImage *harrisLast = cvCreateImage(showSize, IPL_DEPTH_32F, 1);

CvMat kMat = cvMat(3, 3, CV_64FC1, kImage);
CvMat dMat = cvMat(4, 1, CV_64FC1, dImage);
Mat kMatMat = Mat(3, 3, CV_64FC1, kImage);
Mat dMatMat = Mat(4, 1, CV_64FC1, dImage);

Mat imageShowMat;


IplImage *mapx, *mapy;

const int maxFeatureNumPerSubregion = 16;
const int xSubregionNum = 12;
const int ySubregionNum = 8;
const int totalSubregionNum = xSubregionNum * ySubregionNum;
const int MAXFEATURENUM = maxFeatureNumPerSubregion * totalSubregionNum;

const int xBoundary = 20;
const int yBoundary = 20;
const double subregionWidth = (double)(imageWidth - 2 * xBoundary) / (double)xSubregionNum;
const double subregionHeight = (double)(imageHeight - 2 * yBoundary) / (double)ySubregionNum;

const double maxTrackDis = 100;
const int winSize = 21;
const int lktPyramid = 4;

IplImage *imageEig, *imageTmp, *pyrCur, *pyrLast;

CvPoint2D32f *featuresCur = new CvPoint2D32f[2 * MAXFEATURENUM];
CvPoint2D32f *featuresLast = new CvPoint2D32f[2 * MAXFEATURENUM];
char featuresFound[2 * MAXFEATURENUM];
float featuresError[2 * MAXFEATURENUM];

vector<Point2f> featuresCurVec;
vector<Point2f> featuresLastVec;
vector<Point2f> featuresSubVec;
vector<unsigned char> featuresStatus;
vector<float> featuresErr;
vector<int> featuresIndVec;

 cv::Ptr<cv::GFTTDetector> gftt = cv::GFTTDetector::create();

int featuresIndFromStart = 0;
int featuresInd[2 * MAXFEATURENUM] = {0};

int totalFeatureNum = 0;
int subregionFeatureNum[2 * totalSubregionNum] = {0};

pcl::PointCloud<ImagePoint>::Ptr imagePointsCur(new pcl::PointCloud<ImagePoint>());
pcl::PointCloud<ImagePoint>::Ptr imagePointsLast(new pcl::PointCloud<ImagePoint>());

ros::Publisher *imagePointsLastPubPointer;
ros::Publisher *imageShowPubPointer;
cv_bridge::CvImage bridge;
cv_bridge::CvImagePtr cv_ptr;

void imageDataHandler(const sensor_msgs::Image::ConstPtr& imageData) 
{
  timeLast = timeCur;
  timeCur = imageData->header.stamp.toSec() - 0.1163;

  cv_bridge::CvImageConstPtr imageDataCv = cv_bridge::toCvShare(imageData, "mono8");

  if (!systemInited) {
    remap(imageDataCv->image, image0, mapxMap, mapyMap, CV_INTER_LINEAR);
    systemInited = true;

    return;
  }

   Mat imageLastMat, imageCurMat;
   if (isOddFrame) {
    remap(imageDataCv->image, image1, mapxMap, mapyMap, CV_INTER_LINEAR);

    imageLastMat = image0;
    imageCurMat = image1;

  } else {
    remap(imageDataCv->image, image0, mapxMap, mapyMap, CV_INTER_LINEAR);
    
    imageLastMat = image1;
    imageCurMat = image0;

  }

  isOddFrame = !isOddFrame;

  IplImage *imageTemp = imageLast;
  imageLast = imageCur;
  imageCur = imageTemp;

  //Mat imageTempMat;
  //imageLastMat = imageCurMat;



  //for (int i = 0; i < imagePixelNum; i++) {
  //  imageCur->imageData[i] = (char)imageData->data[i];
  //}

  cv_ptr = cv_bridge::toCvCopy(imageData, sensor_msgs::image_encodings::MONO8);
  //cout << cv_ptr->image.size() << "\n";
  IplImage copy(cv_ptr->image);

  imageCur = &copy;
  imageCurMat = cv_ptr->image;

  //cout << "imageCur.size: " <<  imageCur->height << " " << imageCur->width << "\n";

  //cv::imshow("Image", cv_ptr->image);
  //cv::waitKey(1);

  //Mat imgShow = imageData->image;

  //cvNamedWindow( "Image", CV_WINDOW_AUTOSIZE );
	//cvShowImage("Image", imageCur);
	//cvWaitKey(1);

  //imageShowMat = cvarrToMat(imageCur);

  //  cout << imageShowMat.size() << "\n";
  //	if(imageShowMat.size().width > 0)
  //		imshow("Trajectory", imageShowMat);

  IplImage *t = cvCloneImage(imageCur);
  cvRemap(t, imageCur, mapx, mapy);
  //cvEqualizeHist(imageCur, imageCur);
  cvReleaseImage(&t);

  cvCornerHarris(imageCur, imageHarris, 3);
  cvNormalize(
	imageHarris,
	imageHarrisNorm,
	0,
	255,
	CV_MINMAX,
	mask
	);

  cvConvertScaleAbs(imageHarrisNorm, imageHarrisNormScaled);
  int thresh = 100;

  cvResize(imageLast, imageShow);
  cvCornerHarris(imageShow, harrisLast, 3);
  //Mat imageHarrisNormMat = cvarrToMat(imageHarrisNorm);
  //Mat imageHarrisNormScaledMat = cvarrToMat(imageHarrisNormScaled);
  int cornersNum = 0;

  Mat oclImageShow, oclHarrisLast;
  Mat imageHarrisNormMat;
  Mat imageHarrisNormScaledMat;
  cornerHarris(imageLastMat, oclHarrisLast, 2, 3, 0.04);
  normalize( oclHarrisLast, imageHarrisNormMat, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs( imageHarrisNormMat, imageHarrisNormScaledMat );

  for( int j = 0; j < imageHarrisNormMat.rows ; j++ )
    { 
    	for( int i = 0; i < imageHarrisNormMat.cols; i++ )
    	{
        	if( (int) imageHarrisNormMat.at<float>(j,i) > thresh )
        		{
        			cornersNum++;
           			circle( cv_ptr->image, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
        		}
    	}
    }
   cout << "cornersNum: " << cornersNum << "\n";
   cv::imshow("Image", cv_ptr->image);
   cv::waitKey(1);

   //cvShowImage("Image", imageHarrisNormScaled);
   //cvWaitKey(1);
  
  vector<Point2f> featuresTempVec = featuresLastVec;
  featuresLastVec = featuresCurVec;
  featuresCurVec = featuresTempVec;


  CvPoint2D32f *featuresTemp = featuresLast;
  featuresLast = featuresCur;
  featuresCur = featuresTemp;

  pcl::PointCloud<ImagePoint>::Ptr imagePointsTemp = imagePointsLast;
  imagePointsLast = imagePointsCur;
  imagePointsCur = imagePointsTemp;
  imagePointsCur->clear();


  int recordFeatureNum = totalFeatureNum;
  vector<KeyPoint> subFeatures;
  for (int i = 0; i < ySubregionNum; i++) {
    for (int j = 0; j < xSubregionNum; j++) {
      int ind = xSubregionNum * i + j;
      int numToFind = maxFeatureNumPerSubregion - subregionFeatureNum[ind];

      if (numToFind > 0) {
        int subregionLeft = xBoundary + (int)(subregionWidth * j);
        int subregionTop = yBoundary + (int)(subregionHeight * i);
        CvRect subregion = cvRect(subregionLeft, subregionTop, (int)subregionWidth, (int)subregionHeight);
        cvSetImageROI(imageLast, subregion);

        Rect subregionMy = Rect(subregionLeft, subregionTop, (int)subregionWidth, (int)subregionHeight);
        Mat image_roi = cv_ptr->image(subregionMy);
        gftt->setMaxFeatures(numToFind);
        gftt->detect(image_roi, subFeatures);

        KeyPoint::convert(subFeatures, featuresSubVec);

        //cout << "featuresSubVec: " << featuresSubVec.size() << "\n";

        //for(auto i = subFeatures.begin(); i != subFeatures.end(); i++)
        //{
       // 	featuresSubVec.push_back(i->_pt);
        //}

        /*
        cvGoodFeaturesToTrack(imageLast, imageEig, imageTmp, featuresLast + totalFeatureNum,
                              &numToFind, 0.1, 5.0, NULL, 3, 1, 0.04);

        int numFound = 0;
        for(int k = 0; k < numToFind; k++) {
          featuresLast[totalFeatureNum + k].x += subregionLeft;
          featuresLast[totalFeatureNum + k].y += subregionTop;

          featuresSubVec[k].x += subregionLeft;
		  featuresSubVec[k].y += subregionTop;

		  int xIndVec = (featuresSubVec[k].x + 0.5) / showDSRate;
		  int yIndVec = (featuresSubVec[k].y + 0.5) / showDSRate;

          int xInd = (featuresLast[totalFeatureNum + k].x + 0.5) / showDSRate;
          int yInd = (featuresLast[totalFeatureNum + k].y + 0.5) / showDSRate;


          if (((float*)(harrisLast->imageData + harrisLast->widthStep * yInd))[xInd] > 1e-7 &&
                oclHarrisLast.at<float>(yIndVec, xIndVec) > 1e-7) {
            featuresLast[totalFeatureNum + numFound].x = featuresLast[totalFeatureNum + k].x;
            featuresLast[totalFeatureNum + numFound].y = featuresLast[totalFeatureNum + k].y;
            featuresInd[totalFeatureNum + numFound] = featuresIndFromStart;

            featuresLastVec->push_back(featuresSubVec[k]);
			featuresIndVec.push_back(featuresIndFromStart);

            numFound++;
            featuresIndFromStart++;
          }
        }
        */
          int numFound = 0;
          for(int k = 0; k < numToFind; k++) {
            featuresSubVec[k].x += subregionLeft;
            featuresSubVec[k].y += subregionTop;

            int xInd = (featuresSubVec[k].x + 0.5) / showDSRate;
            int yInd = (featuresSubVec[k].y + 0.5) / showDSRate;

            if (oclHarrisLast.at<float>(yInd, xInd) > 1e-7) {
              featuresLastVec.push_back(featuresSubVec[k]);
              featuresIndVec.push_back(featuresIndFromStart);

              numFound++;
              featuresIndFromStart++;
            }
}
        totalFeatureNum += numFound;
        subregionFeatureNum[ind] += numFound;

        cvResetImageROI(imageLast);
      }
    }
  }

  //cout << "totalFeatureNum: " << totalFeatureNum << "\n";

  cvCalcOpticalFlowPyrLK(imageLast, imageCur, pyrLast, pyrCur,
                         featuresLast, featuresCur, totalFeatureNum, cvSize(winSize, winSize), 
                         3, featuresFound, featuresError, 
                         cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01), 0);

  cv::calcOpticalFlowPyrLK(imageLastMat, imageCurMat, featuresLastVec, featuresCurVec,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
            			   lktPyramid,
           				   cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
           				   	                30, 0.01), 0);

  //cout << "featuresLastVec.size: " << featuresLastVec.size() << "\n";
  //cout <<  "featuresCurVec.size: " << featuresCurVec.size() << "\n";

  totalFeatureNum = featuresCurVec.size();

  for (int i = 0; i < totalSubregionNum; i++) {
    subregionFeatureNum[i] = 0;
  }

  ImagePoint point;
  int featureCount = 0;
  double meanShiftX = 0, meanShiftY = 0;
  for (int i = 0; i < totalFeatureNum; i++) {
    double trackDis = sqrt((featuresLastVec[i].x - featuresCurVec[i].x) 
                    * (featuresLastVec[i].x - featuresCurVec[i].x)
                    + (featuresLastVec[i].y - featuresCurVec[i].y) 
                    * (featuresLastVec[i].y - featuresCurVec[i].y));

    if (!(trackDis > maxTrackDis || featuresCurVec[i].x < xBoundary || 
      featuresCurVec[i].x > imageWidth - xBoundary || featuresCurVec[i].y < yBoundary || 
      featuresCurVec[i].y > imageHeight - yBoundary)) {

      int xInd = (int)((featuresLastVec[i].x - xBoundary) / subregionWidth);
      int yInd = (int)((featuresLastVec[i].y - yBoundary) / subregionHeight);
      int ind = xSubregionNum * yInd + xInd;

      if (subregionFeatureNum[ind] < maxFeatureNumPerSubregion) {
        featuresCurVec[featureCount].x = featuresCurVec[i].x;
        featuresCurVec[featureCount].y = featuresCurVec[i].y;
        featuresLastVec[featureCount].x = featuresLastVec[i].x;
        featuresLastVec[featureCount].y = featuresLastVec[i].y;
        featuresIndVec[featureCount] = featuresIndVec[i];

        //cout << "featuresCur[featureCount].x :" << featuresCur[featureCount].x << "\n " 
        //					<< "featuresCur[featureCount].y: " << featuresCur[featureCount].y << " \n";

        point.u = -(featuresCur[featureCount].x - kImage[2]) / kImage[0];
        point.v = -(featuresCur[featureCount].y - kImage[5]) / kImage[4];
        point.ind = featuresInd[featureCount];
        imagePointsCur->push_back(point);

        //cout << "draw point :" << point.u << " " << point.v <<  " \n";

        cv::circle(cv_ptr->image, cv::Point(point.u, point.v), 10, CV_RGB(255,255,0));

        if (i >= recordFeatureNum) {
          point.u = -(featuresLast[featureCount].x - kImage[2]) / kImage[0];
          point.v = -(featuresLast[featureCount].y - kImage[5]) / kImage[4];
          imagePointsLast->push_back(point);
        }

        //meanShiftX += fabs((featuresCur[featureCount].x - featuresLast[featureCount].x) / kImage[0]);
        //meanShiftY += fabs((featuresCur[featureCount].y - featuresLast[featureCount].y) / kImage[4]);

        featureCount++;
        subregionFeatureNum[ind]++;
      }
    }
  }

   //cout << "imagePointsLast.size: " << imagePointsLast->size() << "\n";
  //cout <<  "imagePointsCur.size: " << imagePointsCur->size() << "\n";

  totalFeatureNum = featureCount;
  featuresCurVec.resize(totalFeatureNum);
  featuresLastVec.resize(totalFeatureNum);
  featuresIndVec.resize(totalFeatureNum);
  //meanShiftX /= totalFeatureNum;
  //meanShiftY /= totalFeatureNum;

  sensor_msgs::PointCloud2 imagePointsLast2;
  pcl::toROSMsg(*imagePointsLast, imagePointsLast2);
  imagePointsLast2.header.stamp = ros::Time().fromSec(timeLast);
  imagePointsLastPubPointer->publish(imagePointsLast2);

  showCount = (showCount + 1) % (showSkipNum + 1);
  
  if (showCount == showSkipNum) {
    imageShowMat = cvarrToMat(imageShow);
    //cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);
    //cout << imageShowMat.size() << "\n";
  	//if(imageShowMat.size().width > 0)
  	//cvShowImage("Image", imageHarris);
	//cvWaitKey(1);
  	//imshow("Image", cv_ptr->image);
  	//cv::waitKey(1);
    
    //cout << imageShowMat.size() << "\n";
    bridge.image = imageShowMat;
    bridge.encoding = sensor_msgs::image_encodings::MONO8;
    sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
    imageShowPubPointer->publish(imageShowPointer);
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "featureTracking");
  ros::NodeHandle nh;
  

  mapx = cvCreateImage(imgSize, IPL_DEPTH_32F, 1);
  mapy = cvCreateImage(imgSize, IPL_DEPTH_32F, 1);
  cvInitUndistortMap(&kMat, &dMat, mapx, mapy);

  mapxMap.create(imgSize, CV_32FC1);
  mapyMap.create(imgSize, CV_32FC1);
  initUndistortRectifyMap(kMatMat, dMatMat, Mat(), kMatMat, imgSize, CV_32FC1, mapxMap, mapyMap);
  

  CvSize subregionSize = cvSize((int)subregionWidth, (int)subregionHeight);
  imageEig = cvCreateImage(subregionSize, IPL_DEPTH_32F, 1);
  imageTmp = cvCreateImage(subregionSize, IPL_DEPTH_32F, 1);

  CvSize pyrSize = cvSize(imageWidth + 8, imageHeight / 3);
  pyrCur = cvCreateImage(pyrSize, IPL_DEPTH_32F, 1);
  pyrLast = cvCreateImage(pyrSize, IPL_DEPTH_32F, 1);
  //cv::namedWindow("Image");
  cvNamedWindow( "Image", CV_WINDOW_AUTOSIZE );

  ros::Subscriber imageDataSub = nh.subscribe<sensor_msgs::Image>("/kitti/camera_gray_left/image_raw", 1, imageDataHandler);

  ros::Publisher imagePointsLastPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_last", 5);
  imagePointsLastPubPointer = &imagePointsLastPub;

  ros::Publisher imageShowPub = nh.advertise<sensor_msgs::Image>("/image/show", 1);
  imageShowPubPointer = &imageShowPub;


  ros::spin();

  return 0;
}