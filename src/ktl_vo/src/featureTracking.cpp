#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "opencv2/video/tracking.hpp"
#include <opencv2/ximgproc/disparity_filter.hpp>

#include "cameraParameters.h"
#include "pointDefinition.h"

#include <iostream>

using namespace std;
using namespace cv;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image> sync_pol;

bool systemInited = false;
bool systemInitedDepth = false;
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
Mat image0_left, image1_left, image0_right, image1_right;
Mat imageLastMat, imageCurMat;
Mat imageLastLeft, imageLastRight, imageCurLeft, imageCurRight;

Mat imageLastDepth, imageCurDepth;

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

const int maxFeatureNumPerSubregion = 40;
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

//pcl::PointCloud<ImagePoint>::Ptr imagePointsCur(new pcl::PointCloud<ImagePoint>());
//pcl::PointCloud<ImagePoint>::Ptr imagePointsLast(new pcl::PointCloud<ImagePoint>());

ros::Publisher *imagePointsLastPubPointer;
ros::Publisher *imagePointsCurPubPointer;
ros::Publisher *imageShowPubPointer;
ros::Publisher *imageDepthPubPointer;
cv_bridge::CvImage bridge;
cv_bridge::CvImagePtr cv_ptr;
cv_bridge::CvImage depthBridge;

void ComputeDepthMap(const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat &filtered_disp, cv::Mat &filtered_disp_vis)
{
    cv::Mat left_for_matcher, right_for_matcher;
    cv::Mat left_disp, right_disp;
    //cv::Mat filtered_disp_in;
    cv::Mat conf_map = cv::Mat(imLeft.rows, imLeft.cols, CV_8U);
    conf_map = cv::Scalar(255);
    cv::Rect ROI;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
    int wsize = 3;
    double wls_lambda = 8000.0;
    double wls_sigma = 1.5;

    left_for_matcher = imLeft.clone();
    right_for_matcher = imRight.clone();

    cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(0, 160, wsize,  0, 0, 0,
                                                                  0,  0, 0, 0, cv::StereoSGBM::MODE_HH);

    left_matcher->setP1(24*wsize*wsize);
    left_matcher->setP2(96*wsize*wsize);
    left_matcher->setPreFilterCap(63);
    left_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);

    cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

    left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
    right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);


    wls_filter->setLambda(wls_lambda);
    wls_filter->setSigmaColor(wls_sigma);
    wls_filter->filter(left_disp, imLeft, filtered_disp, right_disp);

    //std::cout << "Disparity map 1 : " << filtered_disp.size() << std::endl;

    conf_map = wls_filter->getConfidenceMap();
    ROI = wls_filter->getROI();


    //cv::Mat raw_disp_vis, filtered_disp_vis;
    double vis_mult = 1.0;

    //cv::ximgproc::getDisparityVis(left_disp, raw_disp_vis, vis_mult);
    //cv::namedWindow("raw disparity", cv::WINDOW_AUTOSIZE);
    //cv::imshow("raw disparity", raw_disp_vis);

    cv::ximgproc::getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);

    //cv::imshow("filtered disparity", filtered_disp_vis);
    //cv::waitKey(1);

}

void imageDepthHandler(const sensor_msgs::Image::ConstPtr& imageData)
{

	cv_bridge::CvImageConstPtr imageDataCv = cv_bridge::toCvShare(imageData, "mono8");

	if (!systemInitedDepth) {
    	imageLastDepth = imageDataCv->image;
    	systemInitedDepth = true;

    return;
  }


   Mat filtered_disp_vis;
   Mat depth;
   std::vector<Point2f> featuresLastDepth;

   bool answ = imageLastDepth.size() ==  imageLastMat.size();

   
   //cout << "imageLastDepth.size: " << imageLastDepth.size() << "\n";
   //cout << "imageLastMat.size: " << imageLastMat.size() << "\n";
   //cout << "answ: " << answ << "\n";

   if(!imageLastDepth.empty() && !imageLastMat.empty() && !featuresLastVec.empty())
   {
  
  	cv::calcOpticalFlowPyrLK(imageLastMat, imageLastDepth, featuresLastVec, featuresLastDepth,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
            			   lktPyramid,
           				   cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
           				   	                30, 0.01), 0);

  	for(int i = 0; i < featuresLastDepth.size(); i++)
  	{
  		float disparity = (featuresLastVec[i].x-featuresLastDepth[i].x);
  		//std::cout << "disparity: " << disparity << "\n";
  		cv::arrowedLine(imageDataCv->image, featuresLastVec[i], featuresLastDepth[i],
        				Scalar(0), 2, CV_AA);

  	}

  	//cv::imshow("filtered disparity", imageLastDepth);
    //cv::waitKey(1);

    cv::imshow("OpticalFlow", imageDataCv->image);
   	cv::waitKey(1);
   	cout << "featuresLastDepth: " << featuresLastDepth.size() << "\n"; 

   	ComputeDepthMap(imageLastMat, imageLastDepth, depth, filtered_disp_vis);

  	imageLastDepth = imageDataCv->image;

   	depthBridge.image = depth;
    depthBridge.encoding = sensor_msgs::image_encodings::TYPE_16SC1;
    sensor_msgs::Image::Ptr imageDepthPointer = depthBridge.toImageMsg();
    imageDepthPubPointer->publish(imageDepthPointer);

   }
  	
}


void imageDataHandler(const sensor_msgs::Image::ConstPtr& imageData) 
{

  pcl::PointCloud<ImagePoint>::Ptr imagePointsCur(new pcl::PointCloud<ImagePoint>());
  pcl::PointCloud<ImagePoint>::Ptr imagePointsLast(new pcl::PointCloud<ImagePoint>());

  timeLast = timeCur;
  timeCur = imageData->header.stamp.toSec() - 0.1163;

  //cv_bridge::CvImageConstPtr imageDataCv = cv_bridge::toCvShare(imageData, "mono8");
  cv_bridge::CvImagePtr imageDataCv = cv_bridge::toCvCopy(imageData, sensor_msgs::image_encodings::MONO8);

  if (!systemInited) {
  	image0 = imageDataCv->image;
    //remap(imageDataCv->image, image0, mapxMap, mapyMap, CV_INTER_LINEAR);
    systemInited = true;

    return;
  }

   
   if (isOddFrame) {
    //remap(imageDataCv->image, image1, mapxMap, mapyMap, CV_INTER_LINEAR);

    image1 = imageDataCv->image;
    imageLastMat = image0;
    imageCurMat = image1;

  } else {
    //remap(imageDataCv->image, image0, mapxMap, mapyMap, CV_INTER_LINEAR);
    
    image0 = imageDataCv->image;
    imageLastMat = image1;
    imageCurMat = image0;

  }

  isOddFrame = !isOddFrame;

  IplImage *imageTemp = imageLast;
  imageLast = imageCur;
  imageCur = imageTemp;

  featuresCurVec.clear();
  featuresLastVec.clear();
  featuresIndVec.clear();
  //cout << "imageLastMat.depth: " << imageLastMat.depth() << "\n";

  //Mat imageTempMat;
  //imageLastMat = imageCurMat;



  //for (int i = 0; i < imagePixelNum; i++) {
  //  imageCur->imageData[i] = (char)imageData->data[i];
  //}

  cv_ptr = cv_bridge::toCvCopy(imageData, sensor_msgs::image_encodings::MONO8);
  //cout << cv_ptr->image.size() << "\n";
  IplImage copy(cv_ptr->image);

  imageCur = &copy;
  //imageCurMat = cv_ptr->image;

  cout << "imageCur.size: " <<  imageCur->height << " " << imageCur->width << "\n";

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
  cvEqualizeHist(imageCur, imageCur);
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
/*
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
    */
   //cout << "cornersNum: " << cornersNum << "\n";
   //cv::imshow("Image", cv_ptr->image);
   //cv::waitKey(1);

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

  gftt->setMaxFeatures(MAXFEATURENUM);
  gftt->detect(imageLastMat, subFeatures);

  KeyPoint::convert(subFeatures, featuresLastVec);

  std::cout << "featuresLastVec: " << featuresLastVec.size() << "\n";

/*
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
*/
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

  //Consistency check
  vector<Point2f> featuresLastVecConstCheck;

  cv::calcOpticalFlowPyrLK(imageCurMat, imageLastMat, featuresCurVec, featuresLastVecConstCheck,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
            			   lktPyramid,
           				   cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
           				   	                30, 0.01), 0);

  cout << "featuresLastVec.size: " << featuresLastVec.size() << "\n";
  cout <<  "featuresCurVec.size: " << featuresCurVec.size() << "\n";
  cout <<  "featuresLastVecConstCheck.size: " << featuresLastVecConstCheck.size() << "\n";

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
      featuresCurVec[i].y > imageHeight - yBoundary ||
      featuresLastVec[i].x - featuresLastVecConstCheck[i].x > 5 || 
      featuresLastVec[i].y - featuresLastVecConstCheck[i].y > 5)) 
    {

      int xInd = (int)((featuresLastVec[i].x - xBoundary) / subregionWidth);
      int yInd = (int)((featuresLastVec[i].y - yBoundary) / subregionHeight);
      int ind = xSubregionNum * yInd + xInd;

        point.u = featuresCurVec[i].x;
        point.v = featuresCurVec[i].y;
        point.ind = featuresInd[featureCount];
        imagePointsCur->push_back(point);

        //cout << "draw point :" << point.u << " " << point.v <<  " \n";

        //cv::circle(cv_ptr->image, cv::Point(point.u, point.v), 10, CV_RGB(255,255,0));

          //point.u = -(featuresLastVec[featureCount].x - kImage[2]) / kImage[0];
          //point.v = -(featuresLastVec[featureCount].y - kImage[5]) / kImage[4];

          point.u = featuresLastVec[i].x;
          point.v = featuresLastVec[i].y;
          imagePointsLast->push_back(point);

        //meanShiftX += fabs((featuresCur[featureCount].x - featuresLast[featureCount].x) / kImage[0]);
        //meanShiftY += fabs((featuresCur[featureCount].y - featuresLast[featureCount].y) / kImage[4]);

        featureCount++;
        subregionFeatureNum[ind]++;

        //cout << "featuresCurVec[featureCount]: " << featuresCurVec[featureCount].x << " " << featuresCurVec[featureCount].y << "\n";
        //cout <<  

        cv::arrowedLine(cv_ptr->image, featuresLastVec[featureCount], featuresCurVec[featureCount],
        				 Scalar(0), 2, CV_AA);

/*
      if (subregionFeatureNum[ind] < maxFeatureNumPerSubregion) {
        //featuresCurVec[featureCount].x = featuresCurVec[i].x;
        //featuresCurVec[featureCount].y = featuresCurVec[i].y;
        //featuresLastVec[featureCount].x = featuresLastVec[i].x;
        //featuresLastVec[featureCount].y = featuresLastVec[i].y;
        //featuresIndVec[featureCount] = featuresIndVec[i];

        //cout << "featuresCur[featureCount].x :" << featuresCur[featureCount].x << "\n " 
        //					<< "featuresCur[featureCount].y: " << featuresCur[featureCount].y << " \n";

        //point.u = -(featuresCurVec[featureCount].x - kImage[2]) / kImage[0];
        //point.v = -(featuresCurVec[featureCount].y - kImage[5]) / kImage[4];
        point.u = featuresCurVec[i].x;
        point.v = featuresCurVec[i].y;
        point.ind = featuresInd[featureCount];
        imagePointsCur->push_back(point);

        //cout << "draw point :" << point.u << " " << point.v <<  " \n";

        //cv::circle(cv_ptr->image, cv::Point(point.u, point.v), 10, CV_RGB(255,255,0));

          //point.u = -(featuresLastVec[featureCount].x - kImage[2]) / kImage[0];
          //point.v = -(featuresLastVec[featureCount].y - kImage[5]) / kImage[4];

          point.u = featuresLastVec[i].x;
          point.v = featuresLastVec[i].y;
          imagePointsLast->push_back(point);

        //meanShiftX += fabs((featuresCur[featureCount].x - featuresLast[featureCount].x) / kImage[0]);
        //meanShiftY += fabs((featuresCur[featureCount].y - featuresLast[featureCount].y) / kImage[4]);

        featureCount++;
        subregionFeatureNum[ind]++;

        //cout << "featuresCurVec[featureCount]: " << featuresCurVec[featureCount].x << " " << featuresCurVec[featureCount].y << "\n";
        //cout <<  

        cv::arrowedLine(cv_ptr->image, featuresLastVec[featureCount], featuresCurVec[featureCount],
        				 Scalar(0), 2, CV_AA);
      }
      */
    }
  }
             //cv::imshow("OpticalFlow", cv_ptr->image);
   			//cv::waitKey(1);

   cout << "imagePointsLast.size: " << imagePointsLast->size() << "\n";
   cout <<  "imagePointsCur.size: " << imagePointsCur->size() << "\n";

  totalFeatureNum = featureCount;

  //meanShiftX /= totalFeatureNum;
  //meanShiftY /= totalFeatureNum;

  sensor_msgs::PointCloud2 imagePointsLast2;
  pcl::toROSMsg(*imagePointsLast, imagePointsLast2);
  imagePointsLast2.header.stamp = ros::Time().fromSec(timeLast);
  imagePointsLastPubPointer->publish(imagePointsLast2);

  sensor_msgs::PointCloud2 imagePointsCur2;
  pcl::toROSMsg(*imagePointsCur, imagePointsCur2);
  imagePointsCur2.header.stamp = ros::Time().fromSec(timeCur);
  imagePointsCurPubPointer->publish(imagePointsCur2);


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

void getOpticalFlow(const cv::Mat &image_1, const cv::Mat &image_2,
					const std::vector<Point2f> &features_1,
					const std::vector<Point2f> &features_depth,
					std::vector<Point2f> &features_1_selected,
					std::vector<Point2f> &features_2_selected,
					std::vector<Point2f> &features_3_selected,
					int y_threshold, bool temporal_flow=false)
{
	vector<unsigned char> featuresStatus;
	vector<float> featuresErr;
	vector<int> featuresIndVec;

	const double maxTrackDis = 100;
	const int winSize = 21;
	const int lktPyramid = 4;

	std::vector<Point2f> features_2;
	std::vector<Point2f> features_1_checked;

	cv::Mat imageShow;
	image_1.copyTo(imageShow);

	cv::calcOpticalFlowPyrLK(image_1, image_2, features_1, features_2,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
            			   lktPyramid,
           				   cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
           				   	                30, 0.01), 0);

	//Consistency check
    vector<Point2f> featuresLastVecConstCheck;

    cv::calcOpticalFlowPyrLK(image_2, image_1, features_2, features_1_checked,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
            			   lktPyramid,
           				   cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
           				   	                30, 0.01), 0);

    //std::cout << "features_1: " <<  features_1.size() << "\n";
    //std::cout << "features_2: " << features_2.size() << "\n";

    for (int i = 0; i < features_1.size(); i++) {
    	double trackDis = sqrt((features_1[i].x - features_2[i].x) 
                    * (features_1[i].x - features_2[i].x)
                    + (features_1[i].y - features_2[i].y) 
                    * (features_1[i].y - features_2[i].y));


    	if (!(trackDis > maxTrackDis || features_2[i].x < xBoundary || 
      		features_2[i].x > imageWidth - xBoundary || features_2[i].y < yBoundary || 
      		features_2[i].y > imageHeight - yBoundary ||
      		features_1[i].x - features_1_checked[i].x > 5 || 
      		features_1[i].y - features_1_checked[i].y > y_threshold)) 
    	{
    		cv::arrowedLine(imageShow, features_1[i], features_2[i],
        				 Scalar(0), 2, CV_AA);

    		features_1_selected.push_back(features_1[i]);
    		features_2_selected.push_back(features_2[i]);
    		if(temporal_flow)
    			features_3_selected.push_back(features_depth[i]);

    	}

	}

	//cv::imshow("OpticalFlow", imageShow);
   // cv::waitKey(1);

    //std::cout << "features_1_selected: " << features_1_selected.size() << "\n";
    //std::cout << "features_2_selected: " << features_2_selected.size() << "\n";

}

void stereoImageCallback( const sensor_msgs::ImageConstPtr& msg_left,
						  const sensor_msgs::ImageConstPtr& msg_right)
{
  cout << "start callback 1  \n";
  timeLast = timeCur;
  timeCur = msg_left->header.stamp.toSec() - 0.1163;

  //cv_bridge::CvImageConstPtr imageDataCv = cv_bridge::toCvShare(imageData, "mono8");
  cv_bridge::CvImagePtr imageDataLeft = cv_bridge::toCvCopy(msg_left, sensor_msgs::image_encodings::MONO8);
  cv_bridge::CvImagePtr imageDataRight = cv_bridge::toCvCopy(msg_right, sensor_msgs::image_encodings::MONO8);

  if (!systemInited)
  {
  	imageCurLeft = imageDataLeft->image;
  	imageCurRight = imageDataRight->image;
    //remap(imageDataCv->image, image0, mapxMap, mapyMap, CV_INTER_LINEAR);
    systemInited = true;
    return;
  }

  cout << "start callback 2  \n";
  Mat imageTempLeft = imageCurLeft;
  Mat imageTempRight = imageCurRight;

  imageLastLeft = imageCurLeft;
  imageLastRight = imageCurRight;

  imageCurLeft = imageDataLeft->image;
  imageCurRight = imageDataRight->image;

  // get points from last left image 
  std::vector<KeyPoint> subFeatures;
  std::vector<Point2f> featuresLastLeft;
  std::vector<Point2f> featuresLastLeftTemp;
  std::vector<Point2f> featuresLastLeftSelect; 
  std::vector<Point2f> featuresLastRightTemp;
  std::vector<Point2f> featuresLastRightSelect;
  std::vector<Point2f> featuresTemp;
  std::vector<Point2f> featuresCurLeftTemp;
  std::vector<Point2f> featuresCurLeftSelect;

  pcl::PointCloud<ImagePoint>::Ptr imagePointsCur(new pcl::PointCloud<ImagePoint>());
  pcl::PointCloud<DepthPoint>::Ptr imagePointsLast(new pcl::PointCloud<DepthPoint>());


  gftt->setMaxFeatures(MAXFEATURENUM);
  gftt->detect(imageLastLeft, subFeatures);

  KeyPoint::convert(subFeatures, featuresLastLeft);

  //getOpticalFlow(imageLastLeft, imageCurLeft, featuresLastLeft, featuresTemp,
	//			 featuresLastLeftTemp, featuresCurLeftTemp, featuresTemp, 5);

  getOpticalFlow(imageLastLeft, imageLastRight, featuresLastLeft, featuresTemp,
  				 featuresLastLeftTemp, featuresLastRightTemp, featuresTemp, 1);

  getOpticalFlow(imageLastLeft, imageCurLeft, featuresLastLeftTemp, featuresLastRightTemp,
				 featuresLastLeftSelect, featuresCurLeftSelect, featuresLastRightSelect, 5, true);

  std::cout << "featuresLastLeftSelect: " << featuresLastLeftSelect.size() << "\n";
  std::cout << "featuresCurLeftSelect: " << featuresCurLeftSelect.size() << "\n";
  std::cout << "featuresLastRightSelect: " << featuresLastRightSelect.size() << "\n";

  ImagePoint point;
  DepthPoint depth_point;
  for(int i =0; i < featuresLastRightSelect.size(); i++)
  {

  	float disparity = (featuresLastLeftSelect[i].x - featuresLastRightSelect[i].x);
  	if(disparity > 0)
  	{
  		depth_point.u = featuresLastLeftSelect[i].x;
        depth_point.v = featuresLastLeftSelect[i].y;
        depth_point.depth = bf / disparity;
        depth_point.ind = i;
        imagePointsLast->push_back(depth_point);

        //std::cout << "depth: " << depth_point.depth << "\n";

        point.u = featuresCurLeftSelect[i].x;
        point.v = featuresCurLeftSelect[i].y;
        point.ind = i;
        imagePointsCur->push_back(point);

  	}

    cv::arrowedLine(imageTempLeft, featuresLastLeftSelect[i], featuresCurLeftSelect[i],
        				 Scalar(0), 2, CV_AA);

  }

  std::cout << "imagePointsLast: " << imagePointsLast->size() << "\n";
  std::cout << "imagePointsCur: " << imagePointsCur->size() << "\n";
  
  cv::imshow("OpticalFlow", imageTempLeft);
  cv::waitKey(1);

  sensor_msgs::PointCloud2 imagePointsLast2;
  pcl::toROSMsg(*imagePointsLast, imagePointsLast2);
  imagePointsLast2.header.stamp = ros::Time().fromSec(timeLast);
  imagePointsLastPubPointer->publish(imagePointsLast2);

  sensor_msgs::PointCloud2 imagePointsCur2;
  pcl::toROSMsg(*imagePointsCur, imagePointsCur2);
  imagePointsCur2.header.stamp = ros::Time().fromSec(timeLast);
  imagePointsCurPubPointer->publish(imagePointsCur2);

  // calculate left-right optical flow
/*
  cv::calcOpticalFlowPyrLK(imageLastLeft, imageLastRight, featuresLastLeft, featuresLastRight,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
            			   lktPyramid,
           				   cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
           				   	                30, 0.01), 0);

   //Consistency check
  vector<Point2f> featuresLastVecConstCheck;

  cv::calcOpticalFlowPyrLK(imageLastRight, imageLastLeft, featuresLastRight, featuresLastVecConstCheck,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
            			   lktPyramid,
           				   cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
           				   	                30, 0.01), 0);

  for (int i = 0; i < totalFeatureNum; i++) {
    double trackDis = sqrt((featuresLastVec[i].x - featuresCurVec[i].x) 
                    * (featuresLastVec[i].x - featuresCurVec[i].x)
                    + (featuresLastVec[i].y - featuresCurVec[i].y) 
                    * (featuresLastVec[i].y - featuresCurVec[i].y));


    if (!(trackDis > maxTrackDis || featuresCurVec[i].x < xBoundary || 
      featuresCurVec[i].x > imageWidth - xBoundary || featuresCurVec[i].y < yBoundary || 
      featuresCurVec[i].y > imageHeight - yBoundary ||
      featuresLastVec[i].x - featuresLastVecConstCheck[i].x > 5 || 
      featuresLastVec[i].y - featuresLastVecConstCheck[i].y > 5)) 
    {
*/


/*
  cv::calcOpticalFlowPyrLK(imageLastLeft, imageCurLeft, featuresLastVec, featuresCurVec,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
            			   lktPyramid,
           				   cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
           				   	                30, 0.01), 0);

  //Consistency check
  //vector<Point2f> featuresLastVecConstCheck;

  cv::calcOpticalFlowPyrLK(imageCurLeft, imageLastLeft, featuresCurVec, featuresLastVecConstCheck,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
            			   lktPyramid,
           				   cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
           				   	                30, 0.01), 0);

  cout << "featuresLastVec.size: " << featuresLastVec.size() << "\n";
  cout <<  "featuresCurVec.size: " << featuresCurVec.size() << "\n";
  cout <<  "featuresLastVecConstCheck.size: " << featuresLastVecConstCheck.size() << "\n";

  totalFeatureNum = featuresCurVec.size();
*/
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
  cvNamedWindow( "OpticalFlow", CV_WINDOW_AUTOSIZE );
  cv::namedWindow("filtered disparity", cv::WINDOW_AUTOSIZE);

  //ros::Subscriber imageDataSub = 
  //				nh.subscribe<sensor_msgs::Image>("/kitti/camera_gray_left/image_raw", 1, imageDataHandler);
  //ros::Subscriber imageDepthSub =
  //			    nh.subscribe<sensor_msgs::Image>("/kitti/camera_gray_right/image_raw", 1, imageDepthHandler);

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> left_sub_ = 
      std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>(
                        new message_filters::Subscriber<sensor_msgs::Image>(nh, "/kitti/camera_gray_left/image_raw", 1));
  
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> right_sub_ = 
      std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>(
      					new message_filters::Subscriber<sensor_msgs::Image>(nh, "/kitti/camera_gray_right/image_raw", 1));

  std::shared_ptr<message_filters::Synchronizer<sync_pol>> sync_ = 
  			sync_ = std::shared_ptr<message_filters::Synchronizer<sync_pol>>(
      			new message_filters::Synchronizer<sync_pol>(sync_pol(10), *left_sub_, *right_sub_));

  sync_->registerCallback(boost::bind(stereoImageCallback, _1, _2));

  ros::Publisher imagePointsLastPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_last", 5);
  imagePointsLastPubPointer = &imagePointsLastPub;

  ros::Publisher imagePointsCurPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_cur", 5);
  imagePointsCurPubPointer = &imagePointsCurPub;

  ros::Publisher imageShowPub = nh.advertise<sensor_msgs::Image>("/image/show", 1);
  imageShowPubPointer = &imageShowPub;

  ros::Publisher imageDepthPub = nh.advertise<sensor_msgs::Image>("/image/depth", 1);
  imageDepthPubPointer = &imageDepthPub;


  ros::spin();

  return 0;
}