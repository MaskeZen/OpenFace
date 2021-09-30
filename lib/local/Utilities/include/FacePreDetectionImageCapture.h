///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis all rights reserved.
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
//       Tadas Baltru�aitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltru�aitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltru�aitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltru�aitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////

#ifndef FACE_PRE_DETECTION_IMAGE_CAPTURE_H
#define FACE_PRE_DETECTION_IMAGE_CAPTURE_H

// System includes
#include <fstream>
#include <sstream>
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace Utilities
{

	//===========================================================================
	/**
	A class for capturing sequences from video, webcam, and image directories
	*/
	class FacePreDetectionImageCapture {

	public:

		// Default constructor
		FacePreDetectionImageCapture() {};

		// Opening based on command line arguments
		bool Init(cv::Mat, std::string);

		// Getting the most recent grayscale frame (need to call GetNextImage first)
		cv::Mat_<uchar> GetGrayFrame();

		// Return bounding boxes associated with the image (if defined)
		std::vector<cv::Rect_<float> > GetBoundingBoxes();

		// Parameters describing the sequence and it's progress (what's the proportion of images opened)
		double GetProgress();

		int image_width;
		int image_height;

		float fx, fy, cx, cy;

		// Name of the video file, image directory, or the webcam
		std::string name;

		bool has_bounding_boxes;

	private:

		// Blocking copy and move, as it doesn't make sense to have several readers pointed at the same source
		FacePreDetectionImageCapture & operator= (const FacePreDetectionImageCapture& other);
		FacePreDetectionImageCapture & operator= (const FacePreDetectionImageCapture&& other);
		FacePreDetectionImageCapture(const FacePreDetectionImageCapture&& other);
		FacePreDetectionImageCapture(const FacePreDetectionImageCapture& other);

		// Storing the latest captures
		cv::Mat latest_frame;
		cv::Mat_<uchar> latest_gray_frame;

		// Keeping track of how many files are read and the filenames
		size_t  frame_num;
		std::vector<std::string> image_files;

		// Could optionally read the bounding box locations from files (each image could have multiple bounding boxes)
		std::vector<std::vector<cv::Rect_<float> > > bounding_boxes;

		void SetCameraIntrinsics(float fx, float fy, float cx, float cy);

		bool image_focal_length_set;
		bool image_optical_center_set;

		bool no_input_specified;

	};
}
#endif // FACE_PRE_DETECTION_IMAGE_CAPTURE_H