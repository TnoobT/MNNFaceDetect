#include <iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include "facedet.hpp"

using namespace std;
using namespace cv;
using namespace MNN;

int main(){
    Mat src_img = imread("./resource/face_det.jpg");
    Mat image;
    cv::resize(src_img,image,cv::Size(640,480),0,0,INTER_LINEAR);
    Facedet facedet("./resource/facedetmodel.mnn");
    const float* score = facedet.inference(image);
    vector<Face_Array> vec_rect_predict;
    vector<Face_Array>det_result = facedet.det_result(score,vec_rect_predict,src_img.rows,src_img.cols);
    for(int i = 0; i < det_result.size(); i++){
        cv::rectangle(src_img, det_result[i].rt_f.tl(), det_result[i].rt_f.br(), (0,0,255), 3);
        std::cout<<det_result[i].fscore<<endl;
    }
    cv::imwrite("result.jpg",src_img);
    return 0;
}
