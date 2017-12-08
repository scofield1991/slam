#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>

#include <opencv2/ximgproc/disparity_filter.hpp>

#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include "cameraParameters.h"
#include "pointDefinition.h"

using namespace cv;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
                                                        sensor_msgs::PointCloud2> sync_pol;

const double PI = 3.1415926;

pcl::PointCloud<ImagePoint>::Ptr imagePointsCur(new pcl::PointCloud<ImagePoint>());
pcl::PointCloud<ImagePoint>::Ptr imagePointsLast(new pcl::PointCloud<ImagePoint>());
pcl::PointCloud<ImagePoint>::Ptr startPointsCur(new pcl::PointCloud<ImagePoint>());
pcl::PointCloud<ImagePoint>::Ptr startPointsLast(new pcl::PointCloud<ImagePoint>());
pcl::PointCloud<pcl::PointXYZHSV>::Ptr startTransCur(new pcl::PointCloud<pcl::PointXYZHSV>());
pcl::PointCloud<pcl::PointXYZHSV>::Ptr startTransLast(new pcl::PointCloud<pcl::PointXYZHSV>());
pcl::PointCloud<pcl::PointXYZHSV>::Ptr ipRelations(new pcl::PointCloud<pcl::PointXYZHSV>());
pcl::PointCloud<pcl::PointXYZHSV>::Ptr ipRelations2(new pcl::PointCloud<pcl::PointXYZHSV>());
pcl::PointCloud<pcl::PointXYZ>::Ptr imagePointsProj(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<DepthPoint>::Ptr depthPointsCur(new pcl::PointCloud<DepthPoint>());
pcl::PointCloud<DepthPoint>::Ptr depthPointsLast(new pcl::PointCloud<DepthPoint>());
pcl::PointCloud<DepthPoint>::Ptr depthPointsSend(new pcl::PointCloud<DepthPoint>());

std::vector<int> ipInd;
std::vector<float> ipy2;

std::vector<float>* ipDepthCur = new std::vector<float>();
std::vector<float>* ipDepthLast = new std::vector<float>();

double imagePointsCurTime;
double imagePointsLastTime;

int imagePointsCurNum = 0;
int imagePointsLastNum = 0;


int depthPointsCurNum = 0;
int depthPointsLastNum = 0;

pcl::PointCloud<pcl::PointXYZI>::Ptr depthCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdTree(new pcl::KdTreeFLANN<pcl::PointXYZI>());

double depthCloudTime;
int depthCloudNum = 0;

const int showDSRate = 1;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqrDis;

double transformSum[6] = {0};
double angleSum[3] = {0};

int imuPointerFront = 0;
int imuPointerLast = -1;
const int imuQueLength = 200;
bool imuInited = false;

double imuRollCur = 0, imuPitchCur = 0, imuYawCur = 0;
double imuRollLast = 0, imuPitchLast = 0, imuYawLast = 0;

double imuYawInit = 0;
double imuTime[imuQueLength] = {0};
double imuRoll[imuQueLength] = {0};
double imuPitch[imuQueLength] = {0};
double imuYaw[imuQueLength] = {0};

ros::Publisher *voDataPubPointer = NULL;
tf::TransformBroadcaster *tfBroadcasterPointer = NULL;
ros::Publisher *depthPointsPubPointer = NULL;
ros::Publisher *imagePointsProjPubPointer = NULL;
ros::Publisher *imageShowPubPointer;

cv::Mat disparity;
std::vector<cv::Point3f> mvP3Dw;
std::vector<cv::Point2f> mvP2D;

// Camera pose.
cv::Mat mTcw = cv::Mat::eye(4,4,CV_32F);

// Rotation, translation and camera center
cv::Mat mRcw = mTcw.rowRange(0,3).colRange(0,3);;
cv::Mat mtcw = mTcw.rowRange(0,3).col(3);
cv::Mat mRwc =  mRcw.t();;
cv::Mat mOw = -mRcw.t()*mtcw; //==mtwc

cv::Mat A_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // intrinsic camera parameters

//RANSAC parameters
cv::Mat inliersIdx;
int pnpMethod = cv::SOLVEPNP_ITERATIVE;
int iterationsCount = 500;
float reprojectionError = 0.1;
float confidence = 0.99;

  // For visualization

  cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);
  char text[100];
  int fontFace = cv::FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;
  cv::Point textOrg(10, 50);
  cv::Mat frameVis;
  cv::Scalar red(0, 0, 255);


void disparityHandler(const sensor_msgs::Image::ConstPtr& dispImage )
{

  cv_bridge::CvImageConstPtr imageDataCv = cv_bridge::toCvShare(dispImage, sensor_msgs::image_encodings::TYPE_16SC1);

   double vis_mult = 1.0;
   cv::Mat filtered_disp_vis;

    //cv::ximgproc::getDisparityVis(left_disp, raw_disp_vis, vis_mult);
    //cv::namedWindow("raw disparity", cv::WINDOW_AUTOSIZE);
    //cv::imshow("raw disparity", raw_disp_vis);

  

   if(!imageDataCv->image.empty())
   {
      disparity = imageDataCv->image;

      //std::cout << "imageDataCv->image.empty(): " << imageDataCv->image.empty() << "\n";
      cv::ximgproc::getDisparityVis(imageDataCv->image, filtered_disp_vis, vis_mult);

      cv::imshow("filtered disparity", filtered_disp_vis);
      cv::waitKey(1);

   }
    
}

void curImageHandler(const sensor_msgs::PointCloud2ConstPtr& imagePoints2)
{

  imagePointsCurTime = imagePoints2->header.stamp.toSec();

  imagePointsCur->clear();
  pcl::fromROSMsg(*imagePoints2, *imagePointsCur);

}

float findDepth ( const cv::Point2f& pt )
{
    int x = cvRound(pt.x);
    int y = cvRound(pt.y);
    ushort d = disparity.ptr<ushort>(y)[x];
    if ( d!=0 )
    {
        return bf / (float(d)/16);
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
                return bf / (float(d)/16);   // ?? base_line / d 
            }
        }
    }
    return -1.0;
}

void UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

cv::Mat UnprojectStereo(cv::Point3f point3f)
{
    const float z = point3f.z;
    if(z>0)
    {
        const float u = point3f.x;
        const float v = point3f.y;
        const float x = (u - kImage[2]) * z / kImage[0];
        const float y = (v - kImage[5]) * z / kImage[4];
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

void set_T_matrix(cv::Mat &T_matrix, const cv::Mat &R_matrix, const cv::Mat &t_matrix)
{
       R_matrix.copyTo(T_matrix.rowRange(0,3).colRange(0,3));
       t_matrix.copyTo(T_matrix.rowRange(0,3).col(3));
}



void estimatePoseRANSAC( const std::vector<cv::Point3f> &list_points3d, // list with model 3D coordinates
                         const std::vector<cv::Point2f> &list_points2d,     // list with scene 2D coordinates
                         int flags, cv::Mat &inliers, int iterationsCount,  // PnP method; inliers container
                         float reprojectionError, double confidence,
                         cv::Mat &mTcw)    // Ransac parameters
{
  cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);  // vector of distortion coefficients
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);          // output rotation vector
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);    // output translation vector
  cv::Mat _R_matrix;
  cv::Mat _t_matrix;


  bool useExtrinsicGuess = false;   // if true the function uses the provided rvec and tvec values as
            // initial approximations of the rotation and translation vectors

  cv::solvePnPRansac( list_points3d, list_points2d, A_matrix, distCoeffs, rvec, tvec,
                useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                inliers, flags );

  Rodrigues(rvec,_R_matrix);      // converts Rotation Vector to Matrix
  _t_matrix = tvec;       // set translation matrix

  std::cout << "_t_matrix: " << _t_matrix << "\n";
  set_T_matrix(mTcw, _R_matrix, _t_matrix); // set rotation-translation matrix

  std::cout << "mTcw: " << mTcw << "\n";
}

void imagePointsHandler(const sensor_msgs::PointCloud2ConstPtr& imagePoints2)
{
  imagePointsLastTime = imagePoints2->header.stamp.toSec();

  pcl::PointCloud<ImagePoint>::Ptr imagePointsTemp = imagePointsLast;
  imagePointsLast = imagePointsCur;
  imagePointsCur = imagePointsTemp;

  imagePointsLast->clear();
  pcl::fromROSMsg(*imagePoints2, *imagePointsLast);

     std::cout << "imagePointsLastOdom.size: " << imagePointsLast->size() << "\n";
   //cout <<  "imagePointsCur.size: " << imagePointsCur->size() << "\n";

  if(!disparity.empty())
  {
    for (int i = 0; i < imagePointsLast->size(); i++)
    {
      //cv::Mat point3dMat = UnprojectStereo(cv::Point2f(imagePointsLast->points[i].u, imagePointsLast->points[i].v));
      cv::Mat point3dMat;
      if (!point3dMat.empty())
      {

        mvP3Dw.push_back(cv::Point3f(point3dMat.at<float>(0), point3dMat.at<float>(1), point3dMat.at<float>(2)));
        mvP2D.push_back(cv::Point2f(imagePointsCur->points[i].u, imagePointsCur->points[i].v));
      }

      //std::cout << "depth: " << depth << "\n";

      //ImagePoint ip = imagePointsLast->points[i];
    }

      std::cout << "mvP3Dw: " << mvP3Dw.size() << "\n";
      std::cout << "mvP2D: " << mvP2D.size() << "\n";

      estimatePoseRANSAC(mvP3Dw, mvP2D, pnpMethod, inliersIdx,
                      iterationsCount, reprojectionError, confidence, mTcw);

      //std::cout << "mTcw: " << mTcw << "\n";
  }


/*


*/

  imagePointsLastNum = imagePointsCurNum;
  imagePointsCurNum = imagePointsCur->points.size();

  pcl::PointCloud<ImagePoint>::Ptr startPointsTemp = startPointsLast;
  startPointsLast = startPointsCur;
  startPointsCur = startPointsTemp;

  pcl::PointCloud<pcl::PointXYZHSV>::Ptr startTransTemp = startTransLast;
  startTransLast = startTransCur;
  startTransCur = startTransTemp;

  std::vector<float>* ipDepthTemp = ipDepthLast;
  ipDepthLast = ipDepthCur;
  ipDepthCur = ipDepthTemp;








  sensor_msgs::PointCloud2 depthPoints2;
  pcl::toROSMsg(*depthPointsSend, depthPoints2);
  depthPoints2.header.frame_id = "camera2";
  depthPoints2.header.stamp = ros::Time().fromSec(imagePointsLastTime);
  depthPointsPubPointer->publish(depthPoints2);

  sensor_msgs::PointCloud2 imagePointsProj2;
  pcl::toROSMsg(*imagePointsProj, imagePointsProj2);
  imagePointsProj2.header.frame_id = "camera2";
  imagePointsProj2.header.stamp = ros::Time().fromSec(imagePointsLastTime);
  imagePointsProjPubPointer->publish(imagePointsProj2);
}

void depthCloudHandler(const sensor_msgs::PointCloud2ConstPtr& depthCloud2)
{
  depthCloudTime = depthCloud2->header.stamp.toSec();

  depthCloud->clear();
  pcl::fromROSMsg(*depthCloud2, *depthCloud);
  depthCloudNum = depthCloud->points.size();

  if (depthCloudNum > 10) {
    for (int i = 0; i < depthCloudNum; i++) {
      depthCloud->points[i].intensity = depthCloud->points[i].z;
      depthCloud->points[i].x *= 10 / depthCloud->points[i].z;
      depthCloud->points[i].y *= 10 / depthCloud->points[i].z;
      depthCloud->points[i].z = 10;
    }

    kdTree->setInputCloud(depthCloud);
  }
}

void imuDataHandler(const sensor_msgs::Imu::ConstPtr& imuData)
{
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(imuData->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  imuPointerLast = (imuPointerLast + 1) % imuQueLength;

  imuTime[imuPointerLast] = imuData->header.stamp.toSec() - 0.1068;
  imuRoll[imuPointerLast] = roll;
  imuPitch[imuPointerLast] = pitch;
  imuYaw[imuPointerLast] = yaw;
}

void imageDataHandler(const sensor_msgs::Image::ConstPtr& imageData) 
{
  cv_bridge::CvImagePtr bridge = cv_bridge::toCvCopy(imageData, "mono8");

  int ipRelationsNum = ipRelations->points.size();
  
  for (int i = 0; i < ipRelationsNum; i++) {
    if (fabs(ipRelations->points[i].v) < 0.5) {
      cv::circle(bridge->image, cv::Point((kImage[2] - ipRelations->points[i].z * kImage[0]) / showDSRate,
                (kImage[5] - ipRelations->points[i].h * kImage[4]) / showDSRate), 1, CV_RGB(255, 0, 0), 2);
    } else if (fabs(ipRelations->points[i].v - 1) < 0.5) {
      cv::circle(bridge->image, cv::Point((kImage[2] - ipRelations->points[i].z * kImage[0]) / showDSRate,
                (kImage[5] - ipRelations->points[i].h * kImage[4]) / showDSRate), 1, CV_RGB(0, 255, 0), 2);
    } else if (fabs(ipRelations->points[i].v - 2) < 0.5) {
      cv::circle(bridge->image, cv::Point((kImage[2] - ipRelations->points[i].z * kImage[0]) / showDSRate,
                (kImage[5] - ipRelations->points[i].h * kImage[4]) / showDSRate), 1, CV_RGB(0, 0, 255), 2);
    } /*else {
      cv::circle(bridge->image, cv::Point((kImage[2] - ipRelations->points[i].z * kImage[0]) / showDSRate,
                (kImage[5] - ipRelations->points[i].h * kImage[4]) / showDSRate), 1, CV_RGB(0, 0, 0), 2);
    }
    */
  }
  
  sensor_msgs::Image::Ptr imagePointer = bridge->toImageMsg();
  imageShowPubPointer->publish(imagePointer);
}


void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& depthCloudLast,
                        const sensor_msgs::PointCloud2ConstPtr& pointCloudCur)
{

  pcl::PointCloud<ImagePoint>::Ptr imagePointsCur(new pcl::PointCloud<ImagePoint>());
  pcl::PointCloud<DepthPoint>::Ptr imagePointsLast(new pcl::PointCloud<DepthPoint>());
  std::vector<cv::Point3f> mvP3Dw;
  std::vector<cv::Point2f> mvP2D;
  cv::Mat mTcw_local = cv::Mat::eye(4,4,CV_32F);

  imagePointsLastTime = depthCloudLast->header.stamp.toSec();

  pcl::fromROSMsg(*depthCloudLast, *imagePointsLast);
  pcl::fromROSMsg(*pointCloudCur, *imagePointsCur);

  std::cout << "imagePointsLast.size: " << imagePointsLast->size() << "\n";
  std::cout << "imagePointsCur.size: " << imagePointsCur->size() << "\n";

  for (int i = 0; i < imagePointsLast->size(); i++)
    {
      
      cv::Mat point3D = UnprojectStereo(cv::Point3f(imagePointsLast->points[i].u,
                                                   imagePointsLast->points[i].v,
                                                   imagePointsLast->points[i].depth));

      mvP3Dw.push_back(cv::Point3f(point3D.at<float>(0), point3D.at<float>(1), point3D.at<float>(2)));

      mvP2D.push_back(cv::Point2f(imagePointsCur->points[i].u, imagePointsCur->points[i].v));
      
      //std::cout << "depth: " << depth << "\n";

      //ImagePoint ip = imagePointsLast->points[i];
    }

      std::cout << "mvP3Dw: " << mvP3Dw.size() << "\n";
      std::cout << "mvP2D: " << mvP2D.size() << "\n";

      estimatePoseRANSAC(mvP3Dw, mvP2D, pnpMethod, inliersIdx,
                      iterationsCount, reprojectionError, confidence, mTcw_local);

      mTcw = mTcw * mTcw_local;

      std::cout << "mTcw_local: " << mTcw_local << "\n";
      //std::cout << "mTcw: " << mTcw << "\n";
      //UpdatePoseMatrices();


  float x = mTcw.at<float>(0, 3) + 300.0;
  float y = mTcw.at<float>(2, 3) + 300.0;
  cv::circle(traj, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);
  cv::rectangle(traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
  std::cout << "x, y: " << x << " " << y << "\n";
  sprintf(text, "Coordinates: x=%02fm, y=%02fm, z=%02fm", mTcw.at<float>(0, 3), mTcw.at<float>(1, 3), mTcw.at<float>(2, 3));
  cv::putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
  
  cv::imshow("Trajectory", traj);
  cv::waitKey(1);

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "visualOdometry");
  ros::NodeHandle nh;

  A_matrix.at<double>(0, 0) = kImage[0];       //      [ fx   0  cx ]
  A_matrix.at<double>(1, 1) = kImage[2];       //      [  0  fy  cy ]
  A_matrix.at<double>(0, 2) = kImage[4];       //      [  0   0   1 ]
  A_matrix.at<double>(1, 2) = kImage[5];
  A_matrix.at<double>(2, 2) = 1;


  //ros::Subscriber imagePointsSub = nh.subscribe<sensor_msgs::PointCloud2>
  //                                 ("/image_points_last", 5, imagePointsHandler);

  //ros::Subscriber depthCloudSub = nh.subscribe<sensor_msgs::PointCloud2> 
  //                                ("/kitti/velo/pointcloud", 5, depthCloudHandler);

  ros::Subscriber imuDataSub = nh.subscribe<sensor_msgs::Imu> ("/kitti/oxts/imu", 5, imuDataHandler);

  ros::Subscriber dispSub = nh.subscribe<sensor_msgs::Image> ("/image/depth", 5, disparityHandler);

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> last_img_sub = 
      std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>>(
                        new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/image_points_last", 5));
  
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cur_img_sub = 
      std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>>(
                new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/image_points_cur", 5));

  std::shared_ptr<message_filters::Synchronizer<sync_pol>> sync_ = 
        sync_ = std::shared_ptr<message_filters::Synchronizer<sync_pol>>(
            new message_filters::Synchronizer<sync_pol>(sync_pol(10), *last_img_sub, *cur_img_sub));

  sync_->registerCallback(boost::bind(pointCloudCallback, _1, _2));

  cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

  ros::Publisher voDataPub = nh.advertise<nav_msgs::Odometry> ("/cam_to_init", 5);
  voDataPubPointer = &voDataPub;

  tf::TransformBroadcaster tfBroadcaster;
  tfBroadcasterPointer = &tfBroadcaster;

  ros::Publisher depthPointsPub = nh.advertise<sensor_msgs::PointCloud2> ("/depth_points_last", 5);
  depthPointsPubPointer = &depthPointsPub;

  ros::Publisher imagePointsProjPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_proj", 1);
  imagePointsProjPubPointer = &imagePointsProjPub;

  ros::Subscriber imageDataSub = nh.subscribe<sensor_msgs::Image>("/image/show", 1, imageDataHandler);

  ros::Publisher imageShowPub = nh.advertise<sensor_msgs::Image>("/image/show_2", 1);
  imageShowPubPointer = &imageShowPub;

  cv::namedWindow("filtered disparity", cv::WINDOW_AUTOSIZE);

  ros::spin();

  return 0;
}
