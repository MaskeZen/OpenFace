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

#ifndef RECORDER_OPENFACE_H
#define RECORDER_OPENFACE_H

#include "RecorderCSV.h"
#include "RecorderHOG.h"
#include "RecorderOpenFaceParameters.h"

// System includes
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <thread>

#include <ConcurrentQueue.h>

namespace Utilities
{

	//===========================================================================
	/**
	A class for recording data processed by OpenFace (facial landmarks, head pose, facial action units, aligned face, HOG features, and tracked video
	*/
	class RecorderPreDetection {

	public:

		// The constructor for the recorder, need to specify if we are recording a sequence or not, in_filename should be just the name and not contain extensions
		RecorderPreDetection(const std::string in_filename, const RecorderOpenFaceParameters& parameters, std::vector<std::string>& arguments);
		RecorderPreDetection(const std::string in_filename, const RecorderOpenFaceParameters& parameters, std::string output_directory);

		~RecorderPreDetection();

		// Closing and cleaning up the recorder
		void Close();

		void SetObservationFaceAlign(const cv::Mat& aligned_face);
		void WriteFaceAlign();

	private:

		// Blocking copy, assignment and move operators, as it does not make sense to save to the same location
		RecorderPreDetection & operator= (const RecorderPreDetection& other);
		RecorderPreDetection & operator= (const RecorderPreDetection&& other);
		RecorderPreDetection(const RecorderPreDetection&& other);
		RecorderPreDetection(const RecorderPreDetection& other);

		void PrepareRecording(const std::string& in_filename);

		// A thread that will write image and video output (the slowest parts of output_
		void AlignedImageWritingTask();

		// Keeping track of what to output and how to output it
		const RecorderOpenFaceParameters params;

		// Keep track of the file and output root location
		std::string record_root;
		std::string default_record_directory = "processed"; // By default we are writing in the processed directory in the working directory, if no output parameters provided
		std::string out_name; // Short name, based on which other names are constructed
		std::string csv_filename;
		std::string aligned_output_directory;
		std::ofstream metadata_file;

		// The actual output file stream that will be written
		RecorderCSV csv_recorder;
		RecorderHOG hog_recorder;

		// The actual temporary storage for the observations
		
		double timestamp;
		int face_id;
		int frame_number;

		// Facial landmark related observations
		cv::Mat_<float> landmarks_2D;
		cv::Mat_<float> landmarks_3D;
		cv::Vec6f pdm_params_global;
		cv::Mat_<float> pdm_params_local;
		double landmark_detection_confidence;
		bool landmark_detection_success;

		// Head pose related observations
		cv::Vec6f head_pose;

		// Action Unit related observations
		std::vector<std::pair<std::string, double> > au_intensities;
		std::vector<std::pair<std::string, double> > au_occurences;

		// Gaze related observations
		cv::Point3f gaze_direction0;
		cv::Point3f gaze_direction1;
		cv::Vec2f gaze_angle;
		std::vector<cv::Point2f> eye_landmarks2D;
		std::vector<cv::Point3f> eye_landmarks3D;

		// For video writing
		cv::VideoWriter video_writer;
		std::string media_filename;
		
		// Do not exceed 100MB in the concurrent queue
		const int TRACKED_QUEUE_CAPACITY = 100;
		bool tracked_writing_thread_started;
		cv::Mat vis_to_out;
		ConcurrentQueue<std::pair<std::string, cv::Mat> > vis_to_out_queue;

		// For aligned face writing
		const int ALIGNED_QUEUE_CAPACITY = 100;
		bool aligned_writing_thread_started;
		cv::Mat aligned_face;
		ConcurrentQueue<std::pair<std::string, cv::Mat> > aligned_face_queue;

		std::thread video_writing_thread;
		std::thread aligned_writing_thread;

	};
}
#endif // RECORDER_OPENFACE_H