#include "facedet.hpp"

Facedet::Facedet(){
}

Facedet::~Facedet(){
}

Facedet::Facedet(const char *modelPath){
    this->net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelPath));
    this->backendConfig.precision = (MNN::BackendConfig::PrecisionMode) this->precision;
    this->backendConfig.power = (MNN::BackendConfig::PowerMode) this->power;
    this->backendConfig.memory = (MNN::BackendConfig::MemoryMode) this->memory;
    
	this->config.backendConfig = &this->backendConfig;
	this->config.type = MNN_FORWARD_AUTO;
    
	this->session = net->createSession(config);//创建session
    
	cout << "session created" << endl;
}

void Facedet::mat2tensor(const Mat& image){
	cv::Mat preImage = image.clone();
	preImage.convertTo(preImage,CV_32FC3,1/255.);
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(preImage, bgrChannels);
    std::vector<float> chwImage;
    for (auto i = 0; i < bgrChannels.size(); i++)
    {  
        bgrChannels[i] = (bgrChannels[i] - 0.5) / 0.5;
        //HWC->CHW
        std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, preImage.cols * preImage.rows));
        chwImage.insert(chwImage.end(), data.begin(), data.end());
    }
    auto inTensor = this->net->getSessionInput(session, NULL);
	auto nchw_Tensor = new Tensor(inTensor, Tensor::CAFFE);
    ::memcpy(nchw_Tensor->host<float>(), chwImage.data(), nchw_Tensor->elementSize() * 4);
    inTensor->copyFromHostTensor(nchw_Tensor);
}

const float* Facedet::inference(const Mat& image){
    mat2tensor(image);
    this->net->runSession(this->session);
    auto output= net->getSessionOutput(this->session, NULL);
    auto nchwTensor = new Tensor(output, Tensor::CAFFE);
    output->copyToHostTensor(nchwTensor);
    auto score = nchwTensor->host<float>();
    return score;
}

vector<Face_Array> Facedet::det_result(const float *data_out, vector<Face_Array> &vec_rect_predict,
                                       int input_orgimg_h, int input_orgimg_w){
    det_result_out(data_out,vec_rect_predict,input_orgimg_h,input_orgimg_w);
    vector<Face_Array> lines_result;
    if (vec_rect_predict.size() > 0)
	{
		vector<int> idx;
		vector<float> vScore;
		for (int i = 0; i < vec_rect_predict.size(); i++)
		{
			float score = vec_rect_predict[i].fscore;//1
			vScore.push_back(score);
			idx.push_back(i);
		}
		if (idx.size())
		{
			quick_sort_out(&vScore[0], &idx[0], 0, idx.size() - 1);
		}
		for (int i = 0; i < idx.size(); i++)
		{
			lines_result.push_back(vec_rect_predict[idx[i]]);
		}
	}
	vec_rect_predict = nms_out(lines_result, true);
    return vec_rect_predict;

}

void Facedet::det_result_out(const float *data_out, vector<Face_Array> &vec_rect_predict,
                            int input_orgimg_h, int input_orgimg_w)
{
    int size_h = this->target_h / 16;
    int size_w = this->target_w / 16;
    int locations = size_h * size_w;
    cv::Rect_<float> *boxes = (cv::Rect_<float> *)calloc(locations * this->box_num, sizeof(cv::Rect_<float>));
    float **probs = (float **)calloc(locations * box_num, sizeof(float *));
    for (int j = 0; j < locations * box_num; ++j) probs[j] = (float *)calloc(1, sizeof(float));
	for (int i = 0; i < locations; ++i) {
		int row = i / size_w;
		int col = i % size_w;
		for (int n = 0; n < this->box_num; ++n) {
			int index = n * locations + i;
			for (int j = 0; j < cls_num; ++j) {
				probs[index][j] = 0;
			}
			int obj_index = entry_index_out(size_h, size_w, 4, this->cls_num, n * locations + i, 4);
			int box_index = entry_index_out(size_h, size_w, 4, this->cls_num, n * locations + i, 0);
			float scale = logistic_activate_out(data_out[obj_index]);			
            if (cls_num == 0) {
				probs[index][0] = scale > this->conf_thresh ? scale : 0;
			}
            boxes[index].x = ((col + logistic_activate_out(data_out[box_index + 0 * locations])) / size_w);
			boxes[index].y = ((row + logistic_activate_out(data_out[box_index + 1 * locations])) / size_h);
			boxes[index].width = (exp(data_out[box_index + 2 * locations]) * this->biases[2 * n] / size_w);
			boxes[index].height = (exp(data_out[box_index + 3 * locations]) * this->biases[2 * n + 1] / size_h);
		}
	}

	for (int i = 0; i < size_h * size_w * box_num; i++)
	{
		int class_ind = YM_MAX(max_index_out(probs[i], cls_num), 0);
		float prob = probs[i][class_ind];
		if (prob > 0)
		{
			int x1 = YM_MIN(YM_MAX((boxes[i].x - boxes[i].width / 2.0) * input_orgimg_w, 0), input_orgimg_w - 1);
			int x2 = YM_MIN(YM_MAX((boxes[i].x + boxes[i].width / 2.0) * input_orgimg_w, 0), input_orgimg_w - 1);
			int y1 = YM_MIN(YM_MAX((boxes[i].y - boxes[i].height / 2.0) * input_orgimg_h, 0), input_orgimg_h - 1);
			int y2 = YM_MIN(YM_MAX((boxes[i].y + boxes[i].height / 2.0) * input_orgimg_h, 0), input_orgimg_h - 1);
			Face_Array windata;
			windata.rt_f = cv::Rect_<float>(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
			windata.class_ind = class_ind;
			windata.fscore = prob;
			vec_rect_predict.push_back(windata);
		}
	}
	for (int j = 0; j < locations*box_num; ++j) free(probs[j]);
	free(probs);
	free(boxes);
}

int Facedet::max_index_out(float *a, int n)
{
	if (n <= 0) return -1;
	int i, max_i = 0;
	float max = a[0];
	for (i = 1; i < n; ++i) {
		if (a[i] > max) {
			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}


vector<Face_Array> Facedet::nms_out(vector<Face_Array> &_detRes, bool bend)
{
	vector<Face_Array> &detRes = _detRes;
	for (vector<Face_Array>::iterator ite = detRes.begin(); ite != detRes.end(); ite++)
	{
		for (vector<Face_Array>::iterator ite2 = ite + 1; ite2 != detRes.end(); )
		{
			float xx1 = (float)YM_MAX(ite->rt_f.x, ite2->rt_f.x);
			float yy1 = (float)YM_MAX(ite->rt_f.y, ite2->rt_f.y);
			float xx2 = (float)YM_MIN((float)(ite->rt_f.x + ite->rt_f.width - 1), (float)(ite2->rt_f.x + ite2->rt_f.width - 1));
			float yy2 = (float)YM_MIN((float)(ite->rt_f.y + ite->rt_f.height - 1), (float)(ite2->rt_f.y + ite2->rt_f.height - 1));

			float w = (float)YM_MAX(0.0, xx2 - xx1 + 1);
			float h = (float)YM_MAX(0.0, yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (ite->rt_f.area() + ite2->rt_f.area() - inter);

			if (bend)
			{
				if (ovr > this->nms_thresh || inter / YM_MIN(ite->rt_f.area(), ite2->rt_f.area()) > 0.7)
				{
					ite2 = detRes.erase(ite2);
				}
				else
				{
					ite2++;
				}
			}
			else
			{
				if (ovr > this->nms_thresh)
				{
					ite2 = detRes.erase(ite2);
				}
				else
				{
					ite2++;
				}
			}
		}
	}
	return detRes;
}

void Facedet::quick_sort_out(float *x, int *index, int low, int high)
{
	int pivotloc;
	if (low < high)
	{
		pivotloc = Partition_out(x, index, low, high);

		quick_sort_out(x, index, low, pivotloc - 1);
		quick_sort_out(x, index, pivotloc + 1, high);
	}
}

int Facedet::entry_index_out(int size_h, int size_w, int coords, int classes, int location, int entry)
{
	int n = location / (size_h * size_w);
	int loc = location % (size_h * size_w);
	return n * size_h * size_w *(coords + classes + 1) + entry * size_h * size_w + loc;
}


int Facedet::Partition_out(float *x, int *index, int low, int high)
{
	float pivotkey, temp;
	int _temp = 0;
	temp = x[low];
	pivotkey = x[low];
	while (low < high)
	{
		while (low < high && x[high] <= pivotkey)
		{
			--high;
		}
		x[low] = x[high];
		_temp = index[high];
		index[high] = index[low];
		index[low] = _temp;

		while (low < high && x[low] >= pivotkey)
		{
			++low;
		}
		x[high] = x[low];
		_temp = index[high];
		index[high] = index[low];
		index[low] = _temp;
	}
	x[low] = temp;
	return(low);
}

