///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

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
// FacePreDetectionImgDaemon.cpp

// dlib
#include <dlib/image_processing/frontal_face_detector.h>

#include "LandmarkCoreIncludes.h"

#include <FaceAnalyser.h>
#include <GazeEstimation.h>

#include <ImageCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <RecorderOpenFace.h>
#include <RecorderOpenFaceParameters.h>

#include <RecorderPreDetection.h>

#include <chrono>
#include <iostream>
#include <thread>

#include "command-line-parser.hpp"
#include "daemon.hpp"
#include "log.hpp"

#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

std::vector<std::string> get_arguments(int argc, char **argv)
{

	std::vector<std::string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(std::string(argv[i]));
	}
	return arguments;
}

float radianToDegrees(float radian);

int processImage(std::vector<std::string>, LandmarkDetector::CLNF);

LandmarkDetector::CLNF face_model;
LandmarkDetector::FaceDetectorMTCNN face_detector_mtcnn;
Utilities::Visualizer visualizer;
LandmarkDetector::FaceModelParameters det_parameters;
FaceAnalysis::FaceAnalyser face_analyser;

void reload() {
    LOG_INFO("Reload function called.");
}

int main(int argc, char **argv)
{
	//Convert arguments to more convenient vector form
	std::vector<std::string> arguments = get_arguments(argc, argv);

	// Load the models if images found
	det_parameters = * new LandmarkDetector::FaceModelParameters(arguments);

	// The modules that are being used for tracking
	std::cout << "Loading the model" << std::endl;
	LandmarkDetector::CLNF face_model(det_parameters.model_location);

	if (!face_model.loaded_successfully)
	{
		std::cout << "ERROR: Could not load the landmark detector" << std::endl;
		return 1;
	}

	std::cout << "Model loaded" << std::endl;

	// Load facial feature extractor and AU analyser (make sure it is static)
	FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
	face_analysis_params.OptimizeForImages();
	face_analyser = * new FaceAnalysis::FaceAnalyser(face_analysis_params);

	// If bounding boxes not provided, use a face detector
	cv::CascadeClassifier classifier(det_parameters.haar_face_detector_location);
	dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();
	face_detector_mtcnn = * new LandmarkDetector::FaceDetectorMTCNN(det_parameters.mtcnn_face_detector_location);

	// If can't find MTCNN face detector, default to HOG one
	if (det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::MTCNN_DETECTOR && face_detector_mtcnn.empty())
	{
		std::cout << "INFO: defaulting to HOG-SVM face detector" << std::endl;
		det_parameters.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;
	}

	// A utility for visualizing the results
	visualizer = * new Utilities::Visualizer(arguments);
	
	Daemon& daemon = Daemon::instance();
    // señal SIGHUP
    daemon.setReloadFunction(reload);
	int count = 0;
    while (daemon.IsRunning()) {
        LOG_DEBUG("Count: ", count++);
		count++;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    LOG_INFO("FacePreDetectionImgDaemon finalizó satisfactoriamente.");
	return 0;
}

float radianToDegrees(float radian) {
	return radian * (180.0f / 3.14159265f);
}

int processImage(std::vector<std::string> arguments, LandmarkDetector::CLNF face_model) {
	// Prepare for image reading
	Utilities::ImageCapture image_reader;

	// The sequence reader chooses what to open based on command line arguments provided
	if (!image_reader.Open(arguments))
	{
		std::cout << "Could not open any images" << std::endl;
		return 1;
	}

	cv::Mat rgb_image;

	rgb_image = image_reader.GetNextImage();
	
	if (!face_model.eye_model)
	{
		std::cout << "WARNING: no eye model found" << std::endl;
	}

	// if (face_analyser.GetAUClassNames().size() == 0 && face_analyser.GetAUClassNames().size() == 0)
	// {
	// 	std::cout << "WARNING: no Action Unit models found" << std::endl;
	// }

	std::cout << "Starting tracking" << std::endl;
	while (!rgb_image.empty())
	{
	
		Utilities::RecorderOpenFaceParameters recording_params(arguments, false, false,
			image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy);

		if (!face_model.eye_model)
		{
			recording_params.setOutputGaze(false);
		}
		// Utilities::RecorderOpenFace open_face_rec(image_reader.name, recording_params, arguments);
		Utilities::RecorderPreDetection recorder_pre_detection(image_reader.name, recording_params, arguments);

		visualizer.SetImage(rgb_image, image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy);

		// Making sure the image is in uchar grayscale (some face detectors use RGB, landmark detector uses grayscale)
		cv::Mat_<uchar> grayscale_image = image_reader.GetGrayFrame();

		// Detect faces in an image
		std::vector<cv::Rect_<float> > face_detections;

		std::vector<float> confidences;
		LandmarkDetector::DetectFacesMTCNN(face_detections, rgb_image, face_detector_mtcnn, confidences);
			

		// Detect landmarks around detected faces
		int face_det = 0;
		// perform landmark detection for every face detected
		for (size_t face = 0; face < face_detections.size(); ++face)
		{

			// if there are multiple detections go through them
			bool success = LandmarkDetector::DetectLandmarksInImage(rgb_image, face_detections[face], face_model, det_parameters, grayscale_image);

			// Estimate head pose and eye gaze				
			cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy);

			// std::cout << " ================ " << image_reader.name << " ================ " << std::endl;
			std::cout << std::setprecision(3);
			// std::cout << "Confidence: " << face_model.detection_certainty << std::endl;;
			std::cout << image_reader.name << face_model.detection_certainty << "," << radianToDegrees(pose_estimate[3]) << "," << radianToDegrees(pose_estimate[4]) << "," << radianToDegrees(pose_estimate[5]) << std::endl;

			// std::cout << "SE IMPRIME POSE ESTIMATE" << std::endl;
			// std::cout << pose_estimate[0] << "|" << pose_estimate[1] << "|" << pose_estimate[2] << std::endl;
			// std::cout << "Pitch, Yaw y Roll" << std::endl;

			// int yaw = (int)(pose_estimate[4] * 180 / 3.1415926 + 0.5);
            // int roll = (int)(pose_estimate[5] * 180 / 3.1415926 + 0.5);
			// int pitch = (int)(pose_estimate[3] * 180 /3.1415926 + 0.5);
			// std::cout<<"pitch:\t"<<"yaw:\t"<<"roll:"<< std::endl;
			// std::cout<<pitch<<"\t"<<yaw<<"\t"<<roll<< std::endl;

			// std::cout << std::setprecision(3);
			// std::cout << radianToDegrees(pose_estimate[3]) << "," << radianToDegrees(pose_estimate[4]) << "," << radianToDegrees(pose_estimate[5]) << std::endl;

			cv::Mat sim_warped_img;

			face_analyser.PredictStaticAUsAndComputeFeatures(rgb_image, face_model.detected_landmarks);
			face_analyser.GetLatestAlignedFace(sim_warped_img);

			visualizer.SetObservationFaceAlign(sim_warped_img);

			recorder_pre_detection.SetObservationFaceAlign(sim_warped_img);
			recorder_pre_detection.WriteFaceAlign();
			recorder_pre_detection.Close();
		}
		if (face_detections.size() > 0)
		{
			visualizer.ShowObservation();
		}

		// open_face_rec.SetObservationVisualization(visualizer.GetVisImage());
		// open_face_rec.WriteObservationTracked();

		// open_face_rec.Close();

		// Grabbing the next frame in the sequence
		rgb_image = image_reader.GetNextImage();

	}

	return 0;
}
