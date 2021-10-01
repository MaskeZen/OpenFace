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

#include <FacePreDetectionImageCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <RecorderOpenFace.h>
#include <RecorderOpenFaceParameters.h>

#include <RecorderPreDetection.h>

// daemon -------------
#include <chrono>
#include <iostream>
#include <thread>

#include "command-line-parser.hpp"
#include "daemon.hpp"
#include "log.hpp"

// IPC -------------
#include <sys/ipc.h>
// shared memory
#include <sys/shm.h>
// queues
#include <sys/msg.h>
// IPC -------------

#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

const int IMG_WIDTH = 512;
const int IMG_HEIGHT = 512;
const int IMG_CHANNELS = 3;
const int IMG_SIZE = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS;
const int IMG_SHM_KEY = 411367;
int last_msg_id = 0;

const int IMG_OUT_WIDTH = 112;
const int IMG_OUT_HEIGHT = 112;
const int IMG_OUT_CHANNELS = 3;
const int IMG_OUT_SIZE = IMG_OUT_WIDTH * IMG_OUT_HEIGHT * IMG_OUT_CHANNELS;


struct datos_imagen
{
    int msg_id;
    int msg_reply;
    int status;
    float yaw;
    float pitch;
    float roll;
	float detection_certainty;
    unsigned char imagen[IMG_SIZE];
};

std::vector<std::string> get_arguments(int argc, char **argv);
float radianToDegrees(float radian);
int processImage(datos_imagen*, LandmarkDetector::CLNF);
LandmarkDetector::CLNF face_model;
LandmarkDetector::FaceDetectorMTCNN face_detector_mtcnn;
Utilities::Visualizer visualizer;
LandmarkDetector::FaceModelParameters det_parameters;
FaceAnalysis::FaceAnalyser face_analyser;

void reload();



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
		std::cout << "ERROR: No se pudo cargar el landmark detector" << std::endl;
		return 1;
	}

	std::cout << "Model cargado" << std::endl;

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
	int shmid = shmget(IMG_SHM_KEY, sizeof(datos_imagen),0666|IPC_CREAT);

	LOG_INFO(" --- SE INICIA ESCUCHA --- ");
    while (daemon.IsRunning()) {
		// Se lee de la memoria compartida
		datos_imagen *datos = (datos_imagen*) shmat(shmid,(void*)0,0);
		if (datos == (void*)-1) {
			std::cout << "ERROR: No se pudo obtener la memoria compartida." << std::endl;
		 	return 1;
		}

		if (datos->msg_reply == 0 && datos->msg_id > last_msg_id) {
			last_msg_id = datos->msg_id;
			try {
				int returnValue = processImage(datos, face_model);
				if (returnValue != 0) {
					LOG_ERROR("ERROR: Could not process image");
				} else {
					LOG_INFO("INFO: Image processed");
				}
				
			} catch  (const std::exception& e) {
				LOG_ERROR(e.what());
			}
		} else {
			LOG_INFO("INFO: No hay nuevos mensajes");
			LOG_INFO("INFO: Mensaje anterior: " + std::to_string(last_msg_id));
			LOG_INFO("INFO: Mensaje actual: " + std::to_string(datos->msg_id));
			LOG_INFO("INFO: Mensaje reply: " + std::to_string(datos->msg_reply));
			// if (datos->msg_reply > 0) {
			// 	cv::Mat img_reply = cv::Mat(IMG_OUT_HEIGHT, IMG_OUT_WIDTH, CV_8UC3, &(datos->imagen[0]));
			// 	cv::imshow("img_reply", img_reply);
			// 	cv::waitKey(0);
			// }
		}
		
		shmdt(datos);
        LOG_DEBUG("Count: ", count++);
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
    }
    // Se destruye la memoria compartida
    shmctl(shmid,IPC_RMID,NULL);
    LOG_INFO("FacePreDetectionImgDaemon finalizó satisfactoriamente.");
	return 0;
}

float radianToDegrees(float radian) {
	return radian * (180.0f / 3.14159265f);
}

int processImage(datos_imagen *datos, LandmarkDetector::CLNF face_model) {
	std::vector<std::string> arguments;
	// Prepare for image reading
	Utilities::FacePreDetectionImageCapture image_reader;
	std::cout << "Se intenta inicializar la imagen" << std::endl;
	cv::Mat rgb_image = cv::Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC3, &(datos->imagen[0]));
	std::string msg_id = std::to_string(datos->msg_id);
	// cv::imshow("imagen", rgb_image);
    // cv::waitKey(0);
	if (image_reader.Init(rgb_image, msg_id))
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

			datos->pitch = radianToDegrees(pose_estimate[3]);
			datos->yaw = radianToDegrees(pose_estimate[4]);
			datos->roll = radianToDegrees(pose_estimate[5]);
			datos->detection_certainty = face_model.detection_certainty;			
			
			cv::Mat sim_warped_img;
			face_analyser.PredictStaticAUsAndComputeFeatures(rgb_image, face_model.detected_landmarks);
			face_analyser.GetLatestAlignedFace(sim_warped_img);
			memcpy(datos->imagen, sim_warped_img.data, IMG_OUT_SIZE);

			int msg_reply = datos->msg_reply == 0 ? datos->msg_id : datos->msg_reply;
			const auto p1 = std::chrono::system_clock::now();
    		datos->msg_id =  std::chrono::duration_cast<std::chrono::milliseconds>(
                   p1.time_since_epoch()).count();
			datos->msg_reply = msg_reply;

			recorder_pre_detection.Close();
		}
		if (face_detections.size() > 0)
		{
			visualizer.ShowObservation();
		}
	}
	return 0;
}

std::vector<std::string> get_arguments(int argc, char **argv)
{

	std::vector<std::string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(std::string(argv[i]));
	}
	return arguments;
}

// Se ejecutará
// cuando se envie la señal HUP
void reload() {
    LOG_INFO("Se recarga el demonio.");
}
