/*
Name       : Ravi Kant
USC ID     : 7945-0425-48	
e-mail     : rkant@usc.edu	
Submission : Dec 16, 2015

Input Format: programName Data_Location ClassA_name number_of_classA_samples
				classB_name number_of_classB_samples number_of_test_samples

 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <fstream>
#include <math.h>

using namespace cv;
using namespace std;

#define PI 3.14159265
const int slider_max = 360;
const int d_slider_max = 30;
int alpha_slider;
int beta_slider;
int gamma_slider;
int d_slider;
double alpha = 5.0;
double beta = 5.0;
double Gamma = 5.0;
double d = 5.0;
Mat projectedImage;


void crossProduct(float* result, float* a, float* b){
	result[0] = a[1]*b[2] - b[1]*a[2];
	result[1] = -1 * (a[0]*b[2] - b[0]*a[2]);
	result[2] = a[0]*b[1] - b[0]*a[1];
}

float dotProduct(float* a, float* b){
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

void normalize(float* a){
	float factor = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
	a[0] = a[0]/factor;
	a[1] = a[1]/factor;
	a[2] = a[2]/factor;
}
void getProjection(){
	//---------- Parameters to set
	int dimension = 200; 		// height of the input square image
	int outDimension = dimension * 3;
	vector<float> up_vector(3); // which direction is up?
	up_vector[0] = 0.0;
	up_vector[1] = 0.0;
	up_vector[2] = 1.0;

	vector<float> viewing_vector(3);			// where is the eye?
	float dist = getTrackbarPos("D","Projected_Image");
	double alpha_angel = double(getTrackbarPos("Rz","Projected_Image"));
	double beta_angel  = double(getTrackbarPos("Rx","Projected_Image"));
	double gamma_angel = double(getTrackbarPos("Ry","Projected_Image"));


	viewing_vector[0] = cos(alpha_angel * PI/180.0) * dist;	// This is a vector pointing to the eye

	viewing_vector[1] = cos(beta_angel * PI/180.0) * dist;

	viewing_vector[2] = cos(gamma_angel * PI/180.0) * dist;


	float focal_length = sqrt(3);

	//----------- Output

	projectedImage = imread("back_400.jpg",-1);//(3, sizes, CV_32FC1, cv::Scalar(0.0));

	// Read surface images
	Mat inputImg_1 = imread("image01.jpg",-1);
	Mat inputImg_2 = imread("image02.jpg",-1);
	Mat inputImg_3 = imread("image03.jpg",-1);
	Mat inputImg_4 = imread("image04.jpg",-1);
	Mat inputImg_5 = imread("image05.jpg",-1);
	Mat inputImg_6 = imread("image06.jpg",-1);


	//----------- Compute various Matrices
	//    ---------
	//   /        / |
	//  /   3    /  |
	//  --------/ 2 |
	// |        |  /
	// |	1	| /
	// |________|/

	// ----Computing R|t matrix
	// this matrix converts the image from world cordinates to camera coordinates

	// establishing the camera basis vectors
	vector<float> z_cam(3);
	z_cam[0]= -1.0 * viewing_vector[0];
	z_cam[1]= -1.0 * viewing_vector[1];
	z_cam[2]= -1.0 * viewing_vector[2];
	normalize(z_cam.data());
	vector<float> x_cam(3);								// x_cam = up X z_cam
	crossProduct(x_cam.data(),up_vector.data(),z_cam.data());
	normalize(x_cam.data());
	vector<float> y_cam(3);								// y_cam = z_cam X x_cam
	crossProduct(y_cam.data(),z_cam.data(),x_cam.data());
	normalize(y_cam.data());

	// Thus R|t matrix will be
	Mat worldToCam_matrix(3,4,CV_32F);
	worldToCam_matrix.at<float>(0,0) = x_cam[0];
	worldToCam_matrix.at<float>(0,1) = x_cam[1];
	worldToCam_matrix.at<float>(0,2) = x_cam[2];
	worldToCam_matrix.at<float>(0,3) = -1.0 * dotProduct(viewing_vector.data(),x_cam.data());
	worldToCam_matrix.at<float>(1,0) = y_cam[0];
	worldToCam_matrix.at<float>(1,1) = y_cam[1];
	worldToCam_matrix.at<float>(1,2) = y_cam[2];
	worldToCam_matrix.at<float>(1,3) = -1.0 * dotProduct(viewing_vector.data(),y_cam.data());
	worldToCam_matrix.at<float>(2,0) = z_cam[0];
	worldToCam_matrix.at<float>(2,1) = z_cam[1];
	worldToCam_matrix.at<float>(2,2) = z_cam[2];
	worldToCam_matrix.at<float>(2,3) = -1.0 * dotProduct(viewing_vector.data(),z_cam.data());

	// The K matrix will be
	Mat camToImgPlane_matrix(3,3,CV_32F);
	vector<int> imageCenter(2);
	imageCenter[0] = dimension / 2;
	imageCenter[1] = dimension / 2;

	camToImgPlane_matrix.at<float>(0,0) = focal_length;
	camToImgPlane_matrix.at<float>(0,1) = 0.0;
	camToImgPlane_matrix.at<float>(0,2) = 0.0;//imageCenter[0];
	camToImgPlane_matrix.at<float>(1,0) = 0.0;
	camToImgPlane_matrix.at<float>(1,1) = focal_length;
	camToImgPlane_matrix.at<float>(1,2) = 0.0;//imageCenter[1];
	camToImgPlane_matrix.at<float>(2,0) = 0.0;
	camToImgPlane_matrix.at<float>(2,1) = 0.0;
	camToImgPlane_matrix.at<float>(2,2) = 1.0;

	// Image to Cartesian coordinates
	Mat imgToCart(3,3,CV_32F);
	imgToCart.at<float>(0,0) = 0.0;
	imgToCart.at<float>(0,1) = 2.0/float(dimension);
	imgToCart.at<float>(0,2) = (0.5/float(dimension)) -  1.0;
	imgToCart.at<float>(1,0) = -2.0/float(dimension);
	imgToCart.at<float>(1,1) = 0.0;
	imgToCart.at<float>(1,2) = (-0.5/float(dimension)) +  1.0;
	imgToCart.at<float>(2,0) = 0.0;
	imgToCart.at<float>(2,1) = 0.0;
	imgToCart.at<float>(2,2) = 1.0;

	// Cartesian to Image coordinates
	Mat cartToImg(3,3,CV_32F);
	cartToImg.at<float>(0,0) = 0.0;
	cartToImg.at<float>(0,1) = -0.5 * float(outDimension) ;
	cartToImg.at<float>(0,2) = 0.5 * float(dimension) -  0.25;
	cartToImg.at<float>(1,0) = 0.5 * float(outDimension);
	cartToImg.at<float>(1,1) = 0.0;
	cartToImg.at<float>(1,2) = 0.5 * float(dimension) -  0.25;
	cartToImg.at<float>(2,0) = 0.0;
	cartToImg.at<float>(2,1) = 0.0;
	cartToImg.at<float>(2,2) = 1.0;
	// Define Normals to the surfaces
	vector<float> surfaceNormal_1(3);				// positive YZ plane
	surfaceNormal_1[0] = 1.0;						// normal is positive x axis
	surfaceNormal_1[1] = 0.0;
	surfaceNormal_1[2] = 0.0;

	vector<float> surfaceNormal_2(3);				// positive XZ plane
	surfaceNormal_2[0] = 0.0;						// normal is positive y axis
	surfaceNormal_2[1] = 1.0;
	surfaceNormal_2[2] = 0.0;

	vector<float> surfaceNormal_3(3);				// negative YZ plane
	surfaceNormal_3[0] = -1.0;						// normal is negative x axis
	surfaceNormal_3[1] = 0.0;
	surfaceNormal_3[2] = 0.0;

	vector<float> surfaceNormal_4(3);				// negative XZ plane
	surfaceNormal_4[0] = 0.0;						// normal is negative y axis
	surfaceNormal_4[1] = -1.0;
	surfaceNormal_4[2] = 0.0;

	vector<float> surfaceNormal_5(3);				// positive XY plane
	surfaceNormal_5[0] = 0.0;						// normal is positive z axis
	surfaceNormal_5[1] = 0.0;
	surfaceNormal_5[2] = 1.0;

	vector<float> surfaceNormal_6(3);				// negative XY plane
	surfaceNormal_6[0] = 0.0;						// normal is negative z axis
	surfaceNormal_6[1] = 0.0;
	surfaceNormal_6[2] = -1.0;

	// Project the visible surface onto image plane

	// plane 1, positive YZ
	if(dotProduct(surfaceNormal_1.data(),viewing_vector.data())>0){  // check if plane 1 is visible from the current view point

		float x = 1.0-(0.5 / float(dimension));
		for(float row = 0.0; row < dimension; row = row+1.0) {
			for(float col= 0.0; col < dimension; col = col+1.0) {

				Mat curPoint_inputImg(3,1,CV_32F);				// coordinates of surface image in image coordinate system
				curPoint_inputImg.at<float>(0,0) = row;
				curPoint_inputImg.at<float>(1,0) = col;
				curPoint_inputImg.at<float>(2,0) = 1.0;

				Mat curPoint_temp(3,1,CV_32F);					// coordinates of surface image in Cartesian coordinate system
				curPoint_temp = imgToCart * curPoint_inputImg;

				Mat curPoint_World(4,1,CV_32F);					// coordinates of surface image in world coordinate system
				curPoint_World.at<float>(0,0) = x;
				curPoint_World.at<float>(1,0) = curPoint_temp.at<float>(0,0);
				curPoint_World.at<float>(2,0) = curPoint_temp.at<float>(0,1);
				curPoint_World.at<float>(3,0) = 1.0;

				Mat curPoint_cam(3,1,CV_32F);					// coordinates of surface image in camera coordinate system
				curPoint_cam = worldToCam_matrix * curPoint_World;

				Mat curPoint_imgPlane(3,1,CV_32F);				// dividing by z coordinate to get projections
				curPoint_cam = curPoint_cam / curPoint_cam.at<float>(2,0);
				curPoint_imgPlane = camToImgPlane_matrix * curPoint_cam;

				Mat uv_projected(3,1,CV_32F);					// coordinates of projected image in image plane coordinate system
				uv_projected = cartToImg * curPoint_imgPlane;

				int u_projected = round(uv_projected.at<float>(0,0));
				int v_projected = round(uv_projected.at<float>(1,0));
				//cout<<u_projected<<","<<v_projected<<"\n";
				if(u_projected>=0 && u_projected<800 && v_projected>=0 && v_projected <800){
					projectedImage.at<Vec3b>(u_projected,v_projected)[0] = inputImg_1.at<Vec3b>(row,col)[0];
					projectedImage.at<Vec3b>(u_projected,v_projected)[1] = inputImg_1.at<Vec3b>(row,col)[1];
					projectedImage.at<Vec3b>(u_projected,v_projected)[2] = inputImg_1.at<Vec3b>(row,col)[2];
				}

			}
		}
	}
	// plane 2, positive XZ
	if(dotProduct(surfaceNormal_2.data(),viewing_vector.data())>0){

		float y = 1.0-(0.5 / float(dimension));
		for(float row = 0.0; row < dimension; row = row+1.0) {
			for(float col= 0.0; col < dimension; col = col+1.0) {

				Mat curPoint_inputImg(3,1,CV_32F);
				curPoint_inputImg.at<float>(0,0) = row;
				curPoint_inputImg.at<float>(1,0) = col;
				curPoint_inputImg.at<float>(2,0) = 1.0;

				Mat curPoint_temp(3,1,CV_32F);
				curPoint_temp = imgToCart * curPoint_inputImg;

				Mat curPoint_World(4,1,CV_32F);
				curPoint_World.at<float>(0,0) = curPoint_temp.at<float>(0,0);
				curPoint_World.at<float>(1,0) = y;
				curPoint_World.at<float>(2,0) = curPoint_temp.at<float>(0,1);
				curPoint_World.at<float>(3,0) = 1.0;

				Mat curPoint_cam(3,1,CV_32F);
				curPoint_cam = worldToCam_matrix * curPoint_World;

				Mat curPoint_imgPlane(3,1,CV_32F);
				curPoint_cam = curPoint_cam / curPoint_cam.at<float>(2,0);
				curPoint_imgPlane = camToImgPlane_matrix * curPoint_cam;

				Mat uv_projected(3,1,CV_32F);
				uv_projected = cartToImg * curPoint_imgPlane;
				int u_projected = round(uv_projected.at<float>(0,0));
				int v_projected = round(uv_projected.at<float>(1,0));


				if(u_projected>=0 && u_projected<800 && v_projected>=0 && v_projected <800){
					projectedImage.at<Vec3b>(u_projected,v_projected)[0] = inputImg_2.at<Vec3b>(row,col)[0];
					projectedImage.at<Vec3b>(u_projected,v_projected)[1] = inputImg_2.at<Vec3b>(row,col)[1];
					projectedImage.at<Vec3b>(u_projected,v_projected)[2] = inputImg_2.at<Vec3b>(row,col)[2];
				}

			}
		}
	}

	// plane 3, negative YZ
	if(dotProduct(surfaceNormal_3.data(),viewing_vector.data())>0){

		float x = -(1.0-(0.5 / float(dimension)));
		for(float row = 0.0; row < dimension; row = row+1.0) {
			for(float col= 0.0; col < dimension; col = col+1.0) {

				Mat curPoint_inputImg(3,1,CV_32F);
				curPoint_inputImg.at<float>(0,0) = row;
				curPoint_inputImg.at<float>(1,0) = col;
				curPoint_inputImg.at<float>(2,0) = 1.0;

				Mat curPoint_temp(3,1,CV_32F);
				curPoint_temp = imgToCart * curPoint_inputImg;

				Mat curPoint_World(4,1,CV_32F);
				curPoint_World.at<float>(0,0) = x;
				curPoint_World.at<float>(1,0) = curPoint_temp.at<float>(0,0);
				curPoint_World.at<float>(2,0) = curPoint_temp.at<float>(0,1);
				curPoint_World.at<float>(3,0) = 1.0;

				Mat curPoint_cam(3,1,CV_32F);
				curPoint_cam = worldToCam_matrix * curPoint_World;

				Mat curPoint_imgPlane(3,1,CV_32F);
				curPoint_cam = curPoint_cam / curPoint_cam.at<float>(2,0);
				curPoint_imgPlane = camToImgPlane_matrix * curPoint_cam;

				Mat uv_projected(3,1,CV_32F);
				uv_projected = cartToImg * curPoint_imgPlane;
				int u_projected = round(uv_projected.at<float>(0,0));
				int v_projected = round(uv_projected.at<float>(1,0));


				if(u_projected>=0 && u_projected<800 && v_projected>=0 && v_projected <800){
					projectedImage.at<Vec3b>(u_projected,v_projected)[0] = inputImg_3.at<Vec3b>(row,col)[0];
					projectedImage.at<Vec3b>(u_projected,v_projected)[1] = inputImg_3.at<Vec3b>(row,col)[1];
					projectedImage.at<Vec3b>(u_projected,v_projected)[2] = inputImg_3.at<Vec3b>(row,col)[2];
				}

			}
		}
	}
	// plane 4, negative XZ
	if(dotProduct(surfaceNormal_4.data(),viewing_vector.data())>0){

		float y = -(1.0-(0.5 / float(dimension)));
		for(float row = 0.0; row < dimension; row = row+1.0) {
			for(float col= 0.0; col < dimension; col = col+1.0) {

				Mat curPoint_inputImg(3,1,CV_32F);
				curPoint_inputImg.at<float>(0,0) = row;
				curPoint_inputImg.at<float>(1,0) = col;
				curPoint_inputImg.at<float>(2,0) = 1.0;

				Mat curPoint_temp(3,1,CV_32F);
				curPoint_temp = imgToCart * curPoint_inputImg;

				Mat curPoint_World(4,1,CV_32F);
				curPoint_World.at<float>(0,0) = curPoint_temp.at<float>(0,0);
				curPoint_World.at<float>(1,0) = y;
				curPoint_World.at<float>(2,0) = curPoint_temp.at<float>(0,1);
				curPoint_World.at<float>(3,0) = 1.0;

				Mat curPoint_cam(3,1,CV_32F);
				curPoint_cam = worldToCam_matrix * curPoint_World;

				Mat curPoint_imgPlane(3,1,CV_32F);
				curPoint_cam = curPoint_cam / curPoint_cam.at<float>(2,0);
				curPoint_imgPlane = camToImgPlane_matrix * curPoint_cam;

				Mat uv_projected(3,1,CV_32F);
				uv_projected = cartToImg * curPoint_imgPlane;
				int u_projected = round(uv_projected.at<float>(0,0));
				int v_projected = round(uv_projected.at<float>(1,0));


				if(u_projected>=0 && u_projected<800 && v_projected>=0 && v_projected <800){
					projectedImage.at<Vec3b>(u_projected,v_projected)[0] = inputImg_4.at<Vec3b>(row,col)[0];
					projectedImage.at<Vec3b>(u_projected,v_projected)[1] = inputImg_4.at<Vec3b>(row,col)[1];
					projectedImage.at<Vec3b>(u_projected,v_projected)[2] = inputImg_4.at<Vec3b>(row,col)[2];
				}

			}
		}
	}
	// Plane 5 postive XY plane
	if(dotProduct(surfaceNormal_5.data(),viewing_vector.data())>0){

		float z = 1.0-(0.5 / float(dimension));
		for(float row = 0.0; row < dimension; row = row+1.0) {
			for(float col= 0.0; col < dimension; col = col+1.0) {

				Mat curPoint_inputImg(3,1,CV_32F);
				curPoint_inputImg.at<float>(0,0) = row;
				curPoint_inputImg.at<float>(1,0) = col;
				curPoint_inputImg.at<float>(2,0) = 1.0;

				Mat curPoint_temp(3,1,CV_32F);
				curPoint_temp = imgToCart * curPoint_inputImg;

				Mat curPoint_World(4,1,CV_32F);
				curPoint_World.at<float>(0,0) = curPoint_temp.at<float>(0,0);
				curPoint_World.at<float>(1,0) = curPoint_temp.at<float>(0,1);
				curPoint_World.at<float>(2,0) = z;
				curPoint_World.at<float>(3,0) = 1.0;

				Mat curPoint_cam(3,1,CV_32F);
				curPoint_cam = worldToCam_matrix * curPoint_World;

				Mat curPoint_imgPlane(3,1,CV_32F);
				curPoint_cam = curPoint_cam / curPoint_cam.at<float>(2,0);
				curPoint_imgPlane = camToImgPlane_matrix * curPoint_cam;

				Mat uv_projected(3,1,CV_32F);
				uv_projected = cartToImg * curPoint_imgPlane;
				int u_projected = round(uv_projected.at<float>(0,0));
				int v_projected = round(uv_projected.at<float>(1,0));

				if(u_projected>=0 && u_projected<800 && v_projected>=0 && v_projected <800){
					projectedImage.at<Vec3b>(u_projected,v_projected)[0] = inputImg_5.at<Vec3b>(row,col)[0];
					projectedImage.at<Vec3b>(u_projected,v_projected)[1] = inputImg_5.at<Vec3b>(row,col)[1];
					projectedImage.at<Vec3b>(u_projected,v_projected)[2] = inputImg_5.at<Vec3b>(row,col)[2];
				}

			}
		}
	}

	// Plane 6 negative XY plane
	if(dotProduct(surfaceNormal_6.data(),viewing_vector.data())>0){

		float z = -(1.0-(0.5 / float(dimension)));
		for(float row = 0.0; row < dimension; row = row+1.0) {
			for(float col= 0.0; col < dimension; col = col+1.0) {

				Mat curPoint_inputImg(3,1,CV_32F);
				curPoint_inputImg.at<float>(0,0) = row;
				curPoint_inputImg.at<float>(1,0) = col;
				curPoint_inputImg.at<float>(2,0) = 1.0;

				Mat curPoint_temp(3,1,CV_32F);
				curPoint_temp = imgToCart * curPoint_inputImg;

				Mat curPoint_World(4,1,CV_32F);
				curPoint_World.at<float>(0,0) = curPoint_temp.at<float>(0,0);
				curPoint_World.at<float>(1,0) = curPoint_temp.at<float>(0,1);
				curPoint_World.at<float>(2,0) = z;
				curPoint_World.at<float>(3,0) = 1.0;

				Mat curPoint_cam(3,1,CV_32F);
				curPoint_cam = worldToCam_matrix * curPoint_World;

				Mat curPoint_imgPlane(3,1,CV_32F);
				curPoint_cam = curPoint_cam / curPoint_cam.at<float>(2,0);
				curPoint_imgPlane = camToImgPlane_matrix * curPoint_cam;

				Mat uv_projected(3,1,CV_32F);
				uv_projected = cartToImg * curPoint_imgPlane;
				int u_projected = round(uv_projected.at<float>(0,0));
				int v_projected = round(uv_projected.at<float>(1,0));

				if(u_projected>=0 && u_projected<800 && v_projected>=0 && v_projected <800){
					projectedImage.at<Vec3b>(u_projected,v_projected)[0] = inputImg_6.at<Vec3b>(row,col)[0];
					projectedImage.at<Vec3b>(u_projected,v_projected)[1] = inputImg_6.at<Vec3b>(row,col)[1];
					projectedImage.at<Vec3b>(u_projected,v_projected)[2] = inputImg_6.at<Vec3b>(row,col)[2];
				}

			}
		}
	}
}

void on_trackbar_X( int, void* )
{
	alpha = (double) alpha_slider/slider_max ;

	getProjection();
	// destroyWindow("projectedImage");
	imshow( "projectedImage", projectedImage );
}
void on_trackbar_Y( int, void* )
{
	beta = (double) beta_slider/slider_max ;
	getProjection();
	// destroyWindow("projectedImage");
	imshow( "projectedImage", projectedImage );
}
void on_trackbar_Z( int, void* )
{
	Gamma = (double) gamma_slider/slider_max ;
	getProjection();
	// destroyWindow("projectedImage");
	imshow( "projectedImage", projectedImage );
}
void on_trackbar_D( int, void* )
{
	d = (double) d_slider/d_slider_max ;
	getProjection();
	// destroyWindow("projectedImage");
	imshow( "projectedImage", projectedImage );
}
int main()
{

	/// Initialize values
	alpha_slider = 5;
	beta_slider = 5;
	gamma_slider = 5;
	d_slider = 5;
	/// Create Windows
	namedWindow("Projected_Image", 1);

	/// Create Trackbars
	createTrackbar( "Rz", "Projected_Image", &alpha_slider, slider_max, on_trackbar_X );
	createTrackbar( "Rx", "Projected_Image", &beta_slider, slider_max, on_trackbar_Y );
	createTrackbar( "Ry", "Projected_Image", &gamma_slider, slider_max, on_trackbar_Z );
	createTrackbar( "D", "Projected_Image", &d_slider, d_slider_max, on_trackbar_D );
	/// Show some stuff
	on_trackbar_X( alpha_slider, 0 );
	on_trackbar_Y( beta_slider, 0 );
	on_trackbar_Z( gamma_slider, 0 );
	/// Wait until user press some key

	waitKey(0);
}// end of main
