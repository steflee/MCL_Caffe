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
  
	this->best_pred_.Reshape(bottom[0]->num(), this->layer_param_.mcl_param().hard_k(), 1, 1);
	this->assign_counts_.Reshape(bottom.size()-1, 1, 1, 1);
	
	top[0]->Reshape(bottom.size()-1, 1, 1, 1);
  CHECK_EQ(bottom[bottom.size()-1]->channels(), 1);
  CHECK_EQ(bottom[bottom.size()-1]->height(), 1);
  CHECK_EQ(bottom[bottom.size()-1]->width(), 1);
}



template <typename Dtype>
void MCLMultinomialLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
   
  int n_pred = bottom.size()-1;
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int k =  this->layer_param_.mcl_param().hard_k();

  const Dtype* bottom_label = bottom[n_pred]->cpu_data();
  Dtype* best = this->best_pred_.mutable_cpu_data();
  Dtype* counts = this->assign_counts_.mutable_cpu_data();
  Dtype* losses = top[0]->mutable_cpu_data();
  caffe_set(n_pred, Dtype(0), counts);
 
  for (int i = 0; i < num; ++i) {
	
	vector< pair<Dtype, int> > scores;
    int label = static_cast<int>(bottom_label[i]);
	
    for (int j = 0; j< n_pred; ++j) {
      const Dtype* bottom_data = bottom[j]->cpu_data();
      Dtype prob = std::max(bottom_data[i * dim + label], 
								Dtype(kLOG_THRESHOLD));
	  scores.push_back(make_pair(log(prob), j));
    }

	std::partial_sort(
          scores.begin(), scores.begin() + k,
          scores.end(), std::less<std::pair<Dtype, int> >());
		
	for(int l = 0; l < k; l++){
		losses[scores[l].second] -= scores[l].first;
		counts[scores[l].second]++;
		best[i*k+l] = scores[l].second;
	}
  }
	
  for(int i = 0; i < n_pred; i++)
	if(counts[i] > 0) 
		losses[i] /= counts[i];
}

template <typename Dtype>
void MCLMultinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  int n_pred = bottom.size()-1;
  int k =  this->layer_param_.mcl_param().hard_k();
	
  Dtype* best = this->best_pred_.mutable_cpu_data();
	
  const Dtype* bottom_label = bottom[n_pred]->cpu_data();
  const	Dtype* counts = this->assign_counts_.cpu_data();
  const Dtype* losses = top[0]->cpu_diff();

  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / bottom[0]->num();
	

  //For each predictor in the ensemble
  for (int j=0; j<n_pred; ++j){
			
	//Get predictions from predictor j
    const Dtype* bottom_data = bottom[j]->cpu_data();
			
	//Clear diff blob
    Dtype* bottom_diff = bottom[j]->mutable_cpu_diff();
    caffe_set(bottom[j]->count(), Dtype(0), bottom_diff);
      
	//Compute loss scale (adjusted for network parameters)
	Dtype scale = -losses[j]/counts[j];
			
	//Pass back gradient at gradient scale / predicted prob
    for (int i = 0; i < num; ++i){
		for(int l = 0; l < k; l++){ 				
			if(best[i*k+l] == j){	
				int label = static_cast<int>(bottom_label[i]);
				Dtype prob = std::max(bottom_data[i * dim + label],
										Dtype(kLOG_THRESHOLD));
				bottom_diff[i * dim + label] =  scale / prob;
			}
		}
    }
  }
}

INSTANTIATE_CLASS(MCLMultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(MCLMultinomialLogisticLoss);
}  // namespace caffe
