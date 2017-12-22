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

#include <ktl_vo/transform_msg.h>
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Float32MultiArray.h"

#include <opencv2/ximgproc/disparity_filter.hpp>

#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include "cameraParameters.h"
#include "pointDefinition.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace cv;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
                                                        sensor_msgs::PointCloud2> sync_pol;

const double PI = 3.1415926;
double timeCur, timeLast;

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

cv::Mat kMat = cv::Mat(3, 3, CV_64FC1, kImage);
// Camera pose.
cv::Mat mTcw = cv::Mat::eye(4,4,CV_32F);
cv::Mat mTcw_local;

// Rotation, translation and camera center
cv::Mat mRcw = mTcw.rowRange(0,3).colRange(0,3);;
cv::Mat mtcw = mTcw.rowRange(0,3).col(3);
cv::Mat mRwc =  mRcw.t();;
cv::Mat mOw = -mRcw.t()*mtcw; //==mtwc

//RANSAC parameters
cv::Mat inliersIdx;
int pnpMethod = cv::SOLVEPNP_P3P;
int iterationsCount = 300;
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

ros::Publisher *imagePointsLastPubPointer;
ros::Publisher *imagePointsCurPubPointer;
ros::Publisher *localTransformPubPointer = NULL;


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
  Rodrigues(mTcw.rowRange(0,3).colRange(0,3), rvec);
  cv::Mat tvec = mTcw.rowRange(0,3).col(3);
      // output translation vector
  cv::Mat _R_matrix;
  cv::Mat _t_matrix;

  std::cout << "rvec: " << rvec << "\n";
  std::cout << "tvec: " << tvec << "\n";


  cv::Mat A_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // intrinsic camera parameters

  A_matrix.at<double>(0, 0) = kImage[0];       //      [ fx   0  cx ]
  A_matrix.at<double>(0, 2) = kImage[2];       //      [  0  fy  cy ]
  A_matrix.at<double>(1, 1) = kImage[4];       //      [  0   0   1 ]
  A_matrix.at<double>(1, 2) = kImage[5];
  A_matrix.at<double>(2, 2) = 1;

  bool useExtrinsicGuess = false;   // if true the function uses the provided rvec and tvec values as
            // initial approximations of the rotation and translation vectors

  cv::solvePnPRansac( list_points3d, list_points2d, A_matrix, distCoeffs, rvec, tvec,
                useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                inliers, flags );

  Rodrigues(rvec,_R_matrix);      // converts Rotation Vector to Matrix
  _t_matrix = tvec;       // set translation matrix

  //std::cout << "_t_matrix: " << _t_matrix << "\n";
  set_T_matrix(mTcw, _R_matrix, _t_matrix); // set rotation-translation matrix

  //std::cout << "mTcw: " << mTcw << "\n";
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

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& depthCloudLast,
                        const sensor_msgs::PointCloud2ConstPtr& pointCloudCur)
{

  timeLast = timeCur;
  timeCur = depthCloudLast->header.stamp.toSec() - 0.1163;

  pcl::PointCloud<ImagePoint>::Ptr imagePointsCur(new pcl::PointCloud<ImagePoint>());
  pcl::PointCloud<DepthPoint>::Ptr imagePointsLast(new pcl::PointCloud<DepthPoint>());
  pcl::PointCloud<ImagePoint>::Ptr imagePointsCurBA(new pcl::PointCloud<ImagePoint>());
  pcl::PointCloud<DepthPoint>::Ptr depthPointsLastBA(new pcl::PointCloud<DepthPoint>());
  ktl_vo::transform_msg localTransform;
  std::vector<cv::Point3f> mvP3Dw;
  std::vector<cv::Point2f> mvP2D;
  //cv::Mat mTcw_local = cv::Mat::eye(4,4,CV_32F);

  imagePointsLastTime = depthCloudLast->header.stamp.toSec();

  pcl::fromROSMsg(*depthCloudLast, *imagePointsLast);
  pcl::fromROSMsg(*pointCloudCur, *imagePointsCur);

  //std::cout << "imagePointsLast.size: " << imagePointsLast->size() << "\n";
  //std::cout << "imagePointsCur.size: " << imagePointsCur->size() << "\n";

  ImagePoint point;
  DepthPoint depth_point;
  for (int i = 0; i < imagePointsLast->size(); i++)
  {
      
      cv::Mat point3D = UnprojectStereo(cv::Point3f(imagePointsLast->points[i].u,
                                                   imagePointsLast->points[i].v,
                                                   imagePointsLast->points[i].depth));

      mvP3Dw.push_back(cv::Point3f(point3D.at<float>(0), point3D.at<float>(1), point3D.at<float>(2)));
      depth_point.u = point3D.at<float>(0);
      depth_point.v = point3D.at<float>(1);
      depth_point.depth = point3D.at<float>(2);
      depthPointsLastBA->push_back(imagePointsLast->points[i]);

      mvP2D.push_back(cv::Point2f(imagePointsCur->points[i].u, imagePointsCur->points[i].v));
      imagePointsCurBA->push_back(imagePointsCur->points[i]);
  }

  std::cout << "mvP3Dw: " << mvP3Dw.size() << "\n";
  std::cout << "mvP2D: " << mvP2D.size() << "\n";

  estimatePoseRANSAC(mvP3Dw, mvP2D, pnpMethod, inliersIdx,
                      iterationsCount, reprojectionError, confidence, mTcw_local);

/*
      std::vector<cv::Point2f> points2DInliers;
      std::vector<cv::Point3f> points3DInliers;
      for(int inliersIndex = 0; inliersIndex < inliersIdx.rows; inliersIndex++)
      {
        int n = inliersIdx.at<int>(inliersIndex);

        cv::Point2f point2D = mvP2D[n];
        points2DInliers.push_back(point2D);
        point.u = point2D.x;
        point.v = point2D.y;
        imagePointsCurBA->push_back(point);

        cv::Point3f point3D = mvP3Dw[n];
        points3DInliers.push_back(point3D);
        depth_point.u = point3D.x;
        depth_point.v = point3D.y;
        depth_point.depth = point3D.z;
        depthPointsLastBA->push_back(depth_point);

      }
*/
      cv::Mat R_local = mTcw_local.rowRange(0,3).colRange(0,3);
      cv::Mat t_local = mTcw_local.rowRange(0,3).col(3);

      //std::cout << "mTcw_local: " << mTcw_local << "\n";

      for(int i=0;i<4;i++)
      {
        for(int j=0; j<4; j++)
        {
          localTransform.transform[i*4+j] = mTcw_local.at<float>(i, j);
        }
      }

      //std::cout << "points2DInliers: " << points2DInliers.size() << "\n";

      mTcw = mTcw * mTcw_local;


  localTransform.header.stamp = ros::Time().fromSec(timeLast);
  voDataPubPointer->publish(localTransform);

  sensor_msgs::PointCloud2 imagePointsLast2;
  pcl::toROSMsg(*depthPointsLastBA, imagePointsLast2);
  imagePointsLast2.header.stamp = ros::Time().fromSec(timeLast);
  imagePointsLastPubPointer->publish(imagePointsLast2);

  sensor_msgs::PointCloud2 imagePointsCur2;
  pcl::toROSMsg(*imagePointsCurBA, imagePointsCur2);
  imagePointsCur2.header.stamp = ros::Time().fromSec(timeLast);
  imagePointsCurPubPointer->publish(imagePointsCur2);
    
  //std::cout << "mTcw: " << mTcw << "\n";
  //UpdatePoseMatrices();

  float x = mTcw.at<float>(0, 3) + 300.0;
  float y = mTcw.at<float>(2, 3) + 300.0;
  cv::circle(traj, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);
  cv::rectangle(traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
  //std::cout << "x, y: " << x << " " << y << "\n";
  sprintf(text, "Coordinates: x=%02fm, y=%02fm, z=%02fm", mTcw.at<float>(0, 3), mTcw.at<float>(1, 3), mTcw.at<float>(2, 3));
  cv::putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
  
  cv::imshow("Trajectory", traj);
  cv::waitKey(1);

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "visualOdometry");
  ros::NodeHandle nh;

  mTcw_local = cv::Mat::eye(4,4,CV_32F);

  //ros::Subscriber imagePointsSub = nh.subscribe<sensor_msgs::PointCloud2>
  //                                 ("/image_points_last", 5, imagePointsHandler);

  //ros::Subscriber depthCloudSub = nh.subscribe<sensor_msgs::PointCloud2> 
  //                                ("/kitti/velo/pointcloud", 5, depthCloudHandler);


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

  ros::Publisher imagePointsLastPub = nh.advertise<sensor_msgs::PointCloud2> ("/depth_points_last_ba", 5);
  imagePointsLastPubPointer = &imagePointsLastPub;

  ros::Publisher imagePointsCurPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_cur_ba", 5);
  imagePointsCurPubPointer = &imagePointsCurPub;

  cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

  ros::Publisher voDataPub = nh.advertise<ktl_vo::transform_msg> ("/local_transform", 5);
  voDataPubPointer = &voDataPub;
  
  //ros::Publisher localTransformPub = nh.advertise<std_msgs::Float32MultiArray>("/local_transform", 5);
  //localTransformPubPointer = &localTransformPub;

  tf::TransformBroadcaster tfBroadcaster;
  tfBroadcasterPointer = &tfBroadcaster;

  ros::Publisher depthPointsPub = nh.advertise<sensor_msgs::PointCloud2> ("/depth_points_last", 5);
  depthPointsPubPointer = &depthPointsPub;

  ros::Publisher imagePointsProjPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_proj", 1);
  imagePointsProjPubPointer = &imagePointsProjPub;


  ros::Publisher imageShowPub = nh.advertise<sensor_msgs::Image>("/image/show_2", 1);
  imageShowPubPointer = &imageShowPub;

  cv::namedWindow("filtered disparity", cv::WINDOW_AUTOSIZE);

  ros::spin();

  return 0;
}
