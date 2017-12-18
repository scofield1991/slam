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

const int maxFeatureNumPerSubregion = 100;
const int xSubregionNum = 20;
const int ySubregionNum = 8;
const int totalSubregionNum = xSubregionNum * ySubregionNum;
const int MAXFEATURENUM = maxFeatureNumPerSubregion * totalSubregionNum;

const int xBoundary = 30;
const int yBoundary = 30;
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

//(30, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=true, k=0.04);
cv::Ptr<cv::GFTTDetector> gftt = cv::GFTTDetector::create(100, 0.01, 10, 3, true, 0.0);

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

void adaptiveNonMaximalSuppresion( std::vector<cv::KeyPoint>& keypoints,
                                       const int numToKeep )
{
      if( keypoints.size() < numToKeep ) { return; }

      //
      // Sort by response
      //
      std::sort( keypoints.begin(), keypoints.end(),
                 [&]( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs )
                 {
                   return lhs.response > rhs.response;
                 } );

      std::vector<cv::KeyPoint> anmsPts;

      std::vector<double> radii;
      radii.resize( keypoints.size() );
      std::vector<double> radiiSorted;
      radiiSorted.resize( keypoints.size() );

      const float robustCoeff = 1.11; // see paper

      for( int i = 0; i < keypoints.size(); ++i )
      {
        const float response = keypoints[i].response * robustCoeff;
        double radius = std::numeric_limits<double>::max();
        for( int j = 0; j < i && keypoints[j].response > response; ++j )
        {
          radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
        }
        radii[i]       = radius;
        radiiSorted[i] = radius;
      }

      std::sort( radiiSorted.begin(), radiiSorted.end(),
                 [&]( const double& lhs, const double& rhs )
                 {
                   return lhs > rhs;
                 } );

      const double decisionRadius = radiiSorted[numToKeep];
      for( int i = 0; i < radii.size(); ++i )
      {
        if( radii[i] >= decisionRadius )
        {
          anmsPts.push_back( keypoints[i] );
        }
      }

      //std::cout << "anmsPts: " << anmsPts.size() << "\n";

      anmsPts.swap( keypoints );
    }

float findDepth ( const cv::Point2f& pt, cv::Mat &disparity)
{
    int x = cvRound(pt.x);
    int y = cvRound(pt.y);
    ushort d = disparity.ptr<ushort>(y)[x];
    if ( d!=0 )
    {
        return  (float(d)/16);
    }
    else 
    {
        // check the nearby points 
        int dx[4] = {-1,0,1,0};
        int dy[4] = {0,-1,0,1};
        for ( int i=0; i<4; i++ )
        {
            d = disparity.ptr<ushort>( y+dy[i] )[x+dx[i]];
            if ( d!=0 )
            {
                return  (float(d)/16);   // ?? base_line / d 
            }
        }
    }
    return -1.0;
}

void stereoImageCallback( const sensor_msgs::ImageConstPtr& msg_left,
						  const sensor_msgs::ImageConstPtr& msg_right)
{
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
  std::vector<KeyPoint> keypointsLeftLast;

  pcl::PointCloud<ImagePoint>::Ptr imagePointsCur(new pcl::PointCloud<ImagePoint>());
  pcl::PointCloud<DepthPoint>::Ptr imagePointsLast(new pcl::PointCloud<DepthPoint>());


  //gftt->setMaxFeatures(MAXFEATURENUM);
  //gftt->detect(imageLastLeft, subFeatures);

  //KeyPoint::convert(subFeatures, featuresLastLeft);

  for (int i = 0; i < ySubregionNum; i++) 
  {
    for (int j = 0; j < xSubregionNum; j++) 
    {
        std::vector<KeyPoint> subFeaturesLeft;
        int subregionLeft = xBoundary + (int)(subregionWidth * j);
        int subregionTop = yBoundary + (int)(subregionHeight * i);

        cv::Rect subregionMy = cv::Rect(subregionLeft, subregionTop, (int)subregionWidth, (int)subregionHeight);
        cv::Mat image_roi_l = imageLastLeft(subregionMy);
        //cv::Mat image_roi_r = imageLastRight(subregionMy);
        //cv::Mat image_roi_cur = imageCurLeft(subregionMy);

        gftt->setMaxFeatures(maxFeatureNumPerSubregion);
        gftt->detect(image_roi_l, subFeaturesLeft);
        for(int k = 0; k < subFeaturesLeft.size(); k++) 
        {
          subFeaturesLeft[k].pt.x += subregionLeft;
          subFeaturesLeft[k].pt.y += subregionTop;
          keypointsLeftLast.push_back(subFeaturesLeft[k]);
        }
    }
  }

  std::cout << "keypointsLeftLast: " << keypointsLeftLast.size() << "\n";
  adaptiveNonMaximalSuppresion(keypointsLeftLast, 1000);
  KeyPoint::convert(keypointsLeftLast, featuresLastLeft);

  //left-right-past-current check
  std::vector<Point2f> featuresPrevRight;
  std::vector<Point2f> featuresCurLeft;
  std::vector<Point2f> featuresCurLeftRight;
  std::vector<Point2f> featuresPrevCurRight;
  std:vector<Point2f> selectedFeatures;
  std::vector<unsigned char> featuresStatus;
  std::vector<float> featuresErr;

  cv::calcOpticalFlowPyrLK(imageLastLeft, imageLastRight, featuresLastLeft, featuresPrevRight,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
                           lktPyramid,
                           cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
                                            30, 0.01), 0);

  cv::calcOpticalFlowPyrLK(imageLastLeft, imageCurLeft, featuresLastLeft, featuresCurLeft,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
                           lktPyramid,
                           cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
                                            30, 0.01), 0);
  
  cv::calcOpticalFlowPyrLK(imageLastRight, imageCurRight, featuresPrevRight, featuresPrevCurRight,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
                           lktPyramid,
                           cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
                                            30, 0.01), 0);

  cv::calcOpticalFlowPyrLK(imageCurLeft, imageCurRight, featuresCurLeft, featuresCurLeftRight,
                           featuresStatus, featuresErr, cv::Size(winSize, winSize),
                           lktPyramid,
                           cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
                                            30, 0.01), 0);

  std::cout << "featuresPrevRight: " << featuresPrevRight.size() << "\n";
  std::cout << "featuresCurLeft: " << featuresCurLeft.size() << "\n";
  std::cout << "featuresPrevCurRight: " << featuresPrevCurRight.size() << "\n";
  std::cout << "featuresCurLeftRight: " << featuresCurLeftRight.size() << "\n";

for (int i = 0; i < featuresCurLeftRight.size(); i++)
{
  //std::cout << "featuresPrevCurRight[i].x: " << featuresPrevCurRight[i].x << "featuresCurLeftRight[i].x: " << featuresCurLeftRight[i].x << "\n"; 
  if( std::abs(featuresPrevCurRight[i].x - featuresCurLeftRight[i].x < 2) &&
     std::abs(featuresPrevCurRight[i].y - featuresCurLeftRight[i].y < 2))
  {
    selectedFeatures.push_back(featuresLastLeft[i]);
  }

}

std::cout << "selectedFeatures: " << selectedFeatures.size() << "\n";
 

  //getOpticalFlow(imageLastLeft, imageCurLeft, featuresLastLeft, featuresTemp,
	//			 featuresLastLeftTemp, featuresCurLeftTemp, featuresTemp, 5);

  getOpticalFlow(imageLastLeft, imageLastRight, selectedFeatures, featuresTemp,
  				 featuresLastLeftTemp, featuresLastRightTemp, featuresTemp, 1);

  getOpticalFlow(imageLastLeft, imageCurLeft, featuresLastLeftTemp, featuresLastRightTemp,
				 featuresLastLeftSelect, featuresCurLeftSelect, featuresLastRightSelect, 5, true);

  std::cout << "featuresLastLeftSelect: " << featuresLastLeftSelect.size() << "\n";
  std::cout << "featuresCurLeftSelect: " << featuresCurLeftSelect.size() << "\n";
  std::cout << "featuresLastRightSelect: " << featuresLastRightSelect.size() << "\n";

  cv::Mat depthMap;
  cv::Mat filteredDepthMap;
  //ComputeDepthMap(imageLastLeft, imageLastRight, depthMap, filteredDepthMap);

  ImagePoint point;
  DepthPoint depth_point;
  for(int i =0; i < featuresLastRightSelect.size(); i++)
  {

  	float disparity = (featuresLastLeftSelect[i].x - featuresLastRightSelect[i].x);
    //float disparity = findDepth(cv::Point2f(featuresLastLeftSelect[i].x, featuresLastLeftSelect[i].y), depthMap);
    float depth  = bf / disparity;
    float mThDepth = bf * 80.0f / kImage[0];
  	if(depth > 0 && depth < mThDepth)
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

        cv::arrowedLine(imageTempLeft, featuresLastLeftSelect[i], featuresCurLeftSelect[i],
                 Scalar(0), 2, CV_AA);

  	}

  }

  //std::cout << "imagePointsLast: " << imagePointsLast->size() << "\n";
  //std::cout << "imagePointsCur: " << imagePointsCur->size() << "\n";
  
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