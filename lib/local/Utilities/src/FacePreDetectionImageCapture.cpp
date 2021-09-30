///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis, all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////
#include "stdafx_ut.h"

#include "FacePreDetectionImageCapture.h"
#include "ImageManipulationHelpers.h"

using namespace Utilities;

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

bool FacePreDetectionImageCapture::Init(cv::Mat imagen, std::string nombre)
{
	if (imagen.empty()) {
		return false;
	}

	// Some default values
	std::string input_root = "";
	fx = -1; fy = -1; cx = -1; cy = -1;

	image_height = imagen.size().height;
	image_width = imagen.size().width;
	this->SetCameraIntrinsics(fx, fy, cx, cy);
	ConvertToGrayscale_8bit(latest_frame, latest_gray_frame);
	this->name = nombre;

	return true;
}


void FacePreDetectionImageCapture::SetCameraIntrinsics(float fx, float fy, float cx, float cy)
{
	// If optical centers are not defined just use center of image
	if (cx == -1)
	{
		this->cx = this->image_width / 2.0f;
		this->cy = this->image_height / 2.0f;
	}
	else
	{
		this->cx = cx;
		this->cy = cy;
	}
	// Use a rough guess-timate of focal length
	if (fx == -1)
	{
		this->fx = 500.0f * (this->image_width / 640.0f);
		this->fy = 500.0f * (this->image_height / 480.0f);

		this->fx = (this->fx + this->fy) / 2.0f;
		this->fy = this->fx;
	}
	else
	{
		this->fx = fx;
		this->fy = fy;
	}
}

std::vector<cv::Rect_<float> > FacePreDetectionImageCapture::GetBoundingBoxes()
{
	if (!bounding_boxes.empty())
	{
		return bounding_boxes[frame_num - 1];
	}
	else
	{
		return std::vector<cv::Rect_<float> >();
	}
}

double FacePreDetectionImageCapture::GetProgress()
{
	return (double)frame_num / (double)image_files.size();
}

cv::Mat_<uchar> FacePreDetectionImageCapture::GetGrayFrame()
{
	return latest_gray_frame;
}
