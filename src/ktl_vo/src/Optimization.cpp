#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <nav_msgs/Odometry.h>
#include <ktl_vo/transform_msg.h>

#include "cameraParameters.h"
#include "pointDefinition.h"
#include "g2o_types.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>


typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
                                                        sensor_msgs::PointCloud2,
                                                        ktl_vo::transform_msg> sync_pol;

cv::Mat mTcw;
// For visualization
cv::Mat traj = cv::Mat::zeros(700, 700, CV_8UC3);
char text[100];
int fontFace = cv::FONT_HERSHEY_PLAIN;
double fontScale = 1;
int thickness = 1;
cv::Point textOrg(10, 50);
cv::Mat frameVis;
cv::Scalar red(0, 0, 255);


cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

bool newKeyFrame = true;

void bundleAdjustment (
    const std::vector< cv::Point3f > points_3d,
    const std::vector< cv::Point2f > points_2d,
    const cv::Mat& K, cv::Mat& R, cv::Mat& t )
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat <<   R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
               R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
               R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
                            R_mat,
                            Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                        ) );
    optimizer.addVertex ( pose );

    int index = 1;
    for ( const cv::Point3f p:points_3d )   // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex ( point );
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // edges
    index = 1;
    for ( const cv::Point2f p:points_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId ( index );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
        edge->setVertex ( 1, pose );
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0,0 );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        index++;
    }

    optimizer.setVerbose ( false );
    optimizer.initializeOptimization();
    optimizer.optimize ( 30 );

    //std::cout<< "after optimization: "<< std::endl;
    //std::cout<<"T= "<<Eigen::Isometry3d ( pose->estimate() ).matrix() << std::endl;
    cv::Mat T = toCvMat(Eigen::Isometry3d ( pose->estimate() ).matrix());

    if(T.at<float>(2,3) < 3 && T.at<float>(2,3) > -3)
   	{
   		T(cv::Range(0, 3), cv::Range(0, 3)).copyTo(R);
    	T(cv::Range(0, 3), cv::Range(3, 4)).copyTo(t);   			
   	}
    
          //R_local.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
      //t_local.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));
}

void poseOptimization(const std::vector< cv::Point3f > points_3d,
    const std::vector< cv::Point2f > points_2d,
    const cv::Mat& K, cv::Mat& R, cv::Mat& t)
{
	typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block ( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat <<   R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
               R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
               R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setId ( 0 );
    pose->setFixed(false);
    pose->setEstimate ( g2o::SE3Quat (
                            R_mat,
                            Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                        ) );
    optimizer.addVertex ( pose );

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );

    std::vector<EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;

    // edges
    int index = 1;
    for ( int i=0; i<points_3d.size(); i++ )
    {
        //int index = inliers.at<int> ( i,0 );
        // 3D -> 2D projection
        EdgeSE3ProjectXYZOnlyPose* edge = new EdgeSE3ProjectXYZOnlyPose();
        edge->setId ( index );
        edge->setVertex ( 0, pose );

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        edge->setRobustKernel(rk);
        rk->setDelta(deltaMono);
        
        edge->fx = kImage[0];
        edge->fy = kImage[4];
        edge->cx = kImage[2];
        edge->cy = kImage[5];
        //edge->bf = bf;
        
        //edge->camera_ = camera;
        edge->Xw = Vector3d ( points_3d[i].x, points_3d[i].y, points_3d[i].z );
        edge->setMeasurement ( Vector2d ( points_2d[i].x, points_2d[i].y) );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        
        vpEdgesMono.push_back(edge);
        optimizer.addEdge ( edge );
        index++;

    }

    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const int its[4]={10,10,10,10};

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
    	if(optimizer.edges().size()<10)
           	break;

        if(optimizer.vertices().size()<1)
           	break;


        pose->setEstimate ( g2o::SE3Quat (R_mat,
                                          Eigen::Vector3d ( t.at<double> ( 0,0 ),
                                                            t.at<double> ( 1,0 ),
                                                            t.at<double> ( 2,0 ))) );;
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            //const size_t idx = vnIndexEdgeMono[i];

            e->computeError();

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                //pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                //pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);

        }
	}

    //optimizer.initializeOptimization();
    //optimizer.optimize ( 50 );

	//std::cout<<"T= "<<Eigen::Isometry3d ( pose->estimate() ).matrix() << std::endl;
	//std::cout << "Num otliers: " << nBad << "\n";
	if(nBad < (points_3d.size() / 2.0))
	{
		cv::Mat T = toCvMat(Eigen::Isometry3d ( pose->estimate() ).matrix());
		T(cv::Range(0, 3), cv::Range(0, 3)).copyTo(R);
    	T(cv::Range(0, 3), cv::Range(3, 4)).copyTo(t);   			

	}
	
    
}

std::vector<MapPoint> localMap;
std::vector<Pose> localPoses;

void optimizationCallback(const sensor_msgs::PointCloud2ConstPtr& depthCloudLast,
                          const sensor_msgs::PointCloud2ConstPtr& pointCloudCur,
                          const ktl_vo::transform_msgConstPtr& localTransorm)
{
	std::cout << "Optimization block!!!!" << "\n";
	cv::Mat kMat = cv::Mat(3, 3, CV_64FC1, kImage);

	pcl::PointCloud<ImagePoint>::Ptr imagePointsCurBA(new pcl::PointCloud<ImagePoint>());
    pcl::PointCloud<DepthPoint>::Ptr depthPointsLastBA(new pcl::PointCloud<DepthPoint>());
	std::vector<cv::Point3f> mvP3Dw;
  	std::vector<cv::Point2f> mvP2D;	
	cv::Mat mTcw_local = cv::Mat::eye(4,4,CV_32F);
	
	cv::Mat R_local = mTcw_local.rowRange(0,3).colRange(0,3);
    cv::Mat t_local = mTcw_local.rowRange(0,3).col(3);

	for(int i=0;i<4;i++)
      {
        for(int j=0; j<4; j++)
        {
          mTcw_local.at<float>(i, j) = localTransorm->transform[i*4+j];
        }
      }

  	pcl::fromROSMsg(*depthCloudLast, *depthPointsLastBA);
  	pcl::fromROSMsg(*pointCloudCur, *imagePointsCurBA);

  	if(newKeyFrame)
  	{
  		Pose newPose;
  		newPose.mTcw = mTcw_local;
        localPoses.push_back(newPose);

  		localMap.reserve(depthPointsLastBA->size());

  		for(int i = 0; i < depthPointsLastBA->size(); i++)
  		{
  			cv::Point3f worldPoint = cv::Point3f(depthPointsLastBA->points[i].u,
                                         depthPointsLastBA->points[i].v,
                                         depthPointsLastBA->points[i].depth);
  			MapPoint newMapPoint;
  			newMapPoint.Pwd = worldPoint;
  			newMapPoint.ind = depthPointsLastBA->points[i].ind;
  			localMap[depthPointsLastBA->points[i].ind] = newMapPoint;
  		}

  		for(int i = 0; i < imagePointsCurBA->size(); i++)
  		{
  			newPose.imagePoints[imagePointsCurBA->points[i].ind] = cv::Point2f(imagePointsCurBA->points[i].u,
  																		       imagePointsCurBA->points[i].v);

  			localMap[imagePointsCurBA->points[i].ind].observedPoses.push_back(0);	
  		}

  	}
  	else
  	{
  		Pose newPose;
  		auto lastPose = localPoses.back(); 
  		newPose.mTcw = lastPose.mTcw *  mTcw_local;
        localPoses.push_back(newPose);

        for(int i = 0; i < imagePointsCurBA->size(); i++)
  		{
  			newPose.imagePoints[imagePointsCurBA->points[i].ind] = cv::Point2f(imagePointsCurBA->points[i].u,
  																		       imagePointsCurBA->points[i].v);

  			localMap[imagePointsCurBA->points[i].ind].observedPoses.push_back(localPoses.size()-1);	
  		}

  	}

    
    for (int i = 0; i < depthPointsLastBA->size(); i++)
  	{
      
      mvP3Dw.push_back(cv::Point3f(depthPointsLastBA->points[i].u,
                                                   depthPointsLastBA->points[i].v,
                                                   depthPointsLastBA->points[i].depth));

      mvP2D.push_back(cv::Point2f(imagePointsCurBA->points[i].u, imagePointsCurBA->points[i].v));
  	}

      std::cout << "mTcw_local optimization: " << mTcw_local << "\n";

      //bundleAdjustment ( mvP3Dw, mvP2D, kMat, R_local, t_local);

      poseOptimization( mvP3Dw, mvP2D, kMat, R_local, t_local);

      std::cout << "mTcw_local after optimization: " << mTcw_local << "\n";
      
      if(mTcw.empty())
      	mTcw = cv::Mat::eye(4,4,CV_32F);

      std::cout << "mTcw: " << mTcw << "\n";


      mTcw = mTcw * mTcw_local;

	
    float x = mTcw.at<float>(0, 3) + 300.0;
  	float y = mTcw.at<float>(2, 3) + 300.0;
  	cv::circle(traj, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 1);
  	cv::rectangle(traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
  	//std::cout << "x, y: " << x << " " << y << "\n";
  	sprintf(text, "Coordinates: x=%02fm, y=%02fm, z=%02fm", mTcw.at<float>(0, 3), mTcw.at<float>(1, 3), mTcw.at<float>(2, 3));
  	cv::putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
  
  	cv::imshow("Optimization trajectory", traj);
  	cv::waitKey(1);
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "Optimization");
  ros::NodeHandle nh;


  std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> last_img_sub = 
      std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>>(
                        new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/depth_points_last_ba", 100));
  
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cur_img_sub = 
      std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>>(
                new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/image_points_cur_ba", 100));

  std::shared_ptr<message_filters::Subscriber<ktl_vo::transform_msg>> local_transform_sub = 
      std::shared_ptr<message_filters::Subscriber<ktl_vo::transform_msg>>(
                new message_filters::Subscriber<ktl_vo::transform_msg>(nh, "/local_transform", 100));

//ros::Subscriber depthCloudSub = nh.subscribe<ktl_vo::transform_msg> ("/kitti/velo/pointcloud", 5, depthCloudHandler);

  std::shared_ptr<message_filters::Synchronizer<sync_pol>> sync_ = 
        sync_ = std::shared_ptr<message_filters::Synchronizer<sync_pol>>(
            new message_filters::Synchronizer<sync_pol>(sync_pol(10), *last_img_sub, *cur_img_sub, *local_transform_sub));

  sync_->registerCallback(boost::bind(optimizationCallback, _1, _2, _3));

 
  cv::namedWindow("Optimization trajectory", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("filtered disparity", cv::WINDOW_AUTOSIZE);

  ros::spin();

  return 0;
}
