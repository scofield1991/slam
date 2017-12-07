#ifndef camera_parameters_h
#define camera_parameters_h

#include <opencv/cv.h>
#include <sensor_msgs/Image.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/xfeatures2d.hpp>

#include <cv_bridge/cv_bridge.h>

const int imageWidth = 1242;
const int imageHeight = 375;

double kImage[9] = { 707.0912, 0, 601.8873,
					 0, 707.0912, 183.1104,
					 0, 0, 1
				   };


double dImage[4] = {0, 0, 0, 0};

double bf =  386.1448;

#endif