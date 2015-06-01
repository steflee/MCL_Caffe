#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MCLMultinomialLogisticLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
	this->best_pred_.Reshape(bottom[0]->num(), 1, 1, 1);
	this->pred_scales_.Reshape(bottom.size()-1, 1, 1, 1);
	top[0]->Reshape(bottom.size()-1, 1, 1, 1);
  CHECK_EQ(bottom[bottom.size()-1]->channels(), 1);
  CHECK_EQ(bottom[bottom.size()-1]->height(), 1);
  CHECK_EQ(bottom[bottom.size()-1]->width(), 1);
}

template <typename Dtype>
void MCLMultinomialLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int n_pred = bottom.size()-1;

  const Dtype* bottom_label = bottom[n_pred]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  
	Dtype* best = this->best_pred_.mutable_cpu_data();

	//top[0]->mutable_cpu_data()[0] = 0;
	int* counts = new int[n_pred];
	for(int i = 0; i < n_pred; i++){
		counts[i] = 0;
		this->pred_scales_.mutable_cpu_data()[i] = 0;
	}


	Dtype total_loss = 0;
	for (int i = 0; i < num; ++i) {
    Dtype loss = 10000000.00;
    int arg_min = 0;
    Dtype loss_pred = 0;
    int label = static_cast<int>(bottom_label[i]);
    for (int j = 0; j< n_pred; ++j) {
      const Dtype* bottom_data = bottom[j]->cpu_data();
      Dtype prob = std::max(
          bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
      loss_pred = log(prob);
      if(loss>loss_pred){
				loss=loss_pred;
				arg_min=j;
      }
    }
    top[0]->mutable_cpu_data()[arg_min] -= loss;
		counts[arg_min]++;
		this->pred_scales_.mutable_cpu_data()[arg_min]++;
		best[i] = arg_min;
		total_loss -= loss;

  }
	
	for(int i = 0; i < n_pred; i++){
		if(counts[i] > 0) 
			top[0]->mutable_cpu_data()[i] /= Dtype(counts[i]);
		//LOG(INFO) << i << " " << counts[i];
	}
	//LOG(INFO) << "F" << total_loss/num;
	top[0]->mutable_cpu_data()[0] = total_loss/num;
}


template <typename Dtype>
void MCLMultinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  int n_pred = bottom.size()-1;
  Dtype* best = this->best_pred_.mutable_cpu_data();
	
  const Dtype* bottom_label = bottom[n_pred]->cpu_data();
	const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / bottom[0]->num();
	//const Dtype scale = - top[0]->cpu_diff()[0] / Dtype(num);
	//LOG(INFO) << "B" << scale; 

  //For each predictor in the ensemble
  for (int j=0; j<n_pred; ++j){
		//if(propagate_down[j]){
			
			//Get predictions from predictor j
      const Dtype* bottom_data = bottom[j]->cpu_data();
			
			//Clear diff blob
      Dtype* bottom_diff = bottom[j]->mutable_cpu_diff();
      caffe_set(bottom[j]->count(), Dtype(0), bottom_diff);
      
			Dtype scale = -top[0]->cpu_diff()[j]/this->pred_scales_.mutable_cpu_data()[j];
			//LOG(INFO) << j << " " << scale;
	
			//Pass back gradient at gradient scale / predicted prob
      for (int i = 0; i < num; ++i){
				if(best[i] == j){
					
		      int label = static_cast<int>(bottom_label[i]);
		      Dtype prob = std::max(
		          bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
		      bottom_diff[i * dim + label] =  scale / prob;
				}
      }
    //}
  }
}

INSTANTIATE_CLASS(MCLMultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(MCLMultinomialLogisticLoss);
}  // namespace caffe
