#pragma once

#include <iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<MNN/Interpreter.hpp>
#include<MNN/ImageProcess.hpp>

using namespace std;
using namespace cv;
using namespace MNN;

#ifndef YM_MIN
#define YM_MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef YM_MAX
#define YM_MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef FACE_ARRAY
#define FACE_ARRAY
struct Face_Array
{
	int class_ind;
	float overlap_roi;
	float x1;
	float y1;
	float x2;
	float y2;
	float fscore;
	float fscale;
	cv::Rect_<float> rt_f;
	Face_Array()
	{
		class_ind = 0;
		overlap_roi = 0.0;
		fscore = 0.0;
		fscale = 0.0;
		rt_f = cv::Rect_<float>(0, 0, 0, 0);
		x1 = 0.0;
		y1 = 0.0;
		x2 = 0.0;
		y2 = 0.0;
	}
};
#endif

class Facedet
{
private:
    /* data */
    std::shared_ptr<MNN::Interpreter> net = nullptr;
    const char* modelPath;
    ScheduleConfig config;
    Session *session;
    BackendConfig backendConfig;
    float biases[10] = {1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0};
    int box_num = 5;
    int cls_num = 0;
    float conf_thresh = 0.6;
    float nms_thresh = 0.3;
    int target_h = 480;
    int target_w = 640;
    int precision  = 0;
    int power      = 0;
    int memory     = 0;

public:
    Facedet();
    Facedet(const char *modelPath);
    /*图像预处理，包括：归一化、减均值、除方差、HWC格式转为CHW格式*/
    void mat2tensor(const Mat& image);
    /*MNN inference，返回模型输出的结果指针*/
    const float* inference(const Mat& image);
     /*MNN inference返回模型输出的结果指针解析成对应特征图的值，最终结果保存在vec_rect_predict中*/
    void det_result_out(const float *data_out, vector<Face_Array> &vec_rect_predict,
                       int input_orgimg_h, int input_orgimg_w);
    /*获取特征值索引*/
    int entry_index_out(int size_h, int size_w, int coords, int classes, int location, int entry);
    static inline float logistic_activate_out(float x) { return 1. / (1. + exp(-x)); }
    int max_index_out(float *a, int n);
    /*nms*/
    vector<Face_Array> nms_out(vector<Face_Array> &_detRes, bool bend);
    void quick_sort_out(float *x, int *index, int low, int high);
    /*nms后的最终结果*/
    vector<Face_Array> det_result(const float *data_out, vector<Face_Array> &vec_rect_predict,
                                        int input_orgimg_h, int input_orgimg_w);
    int Partition_out(float *x, int *index, int low, int high);
    ~Facedet();
};
