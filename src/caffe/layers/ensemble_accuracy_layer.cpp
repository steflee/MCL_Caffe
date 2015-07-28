#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EnsembleAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);  
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void EnsembleAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  Dtype accuracy = 0;
  int n_pred = bottom.size()-1;
  const Dtype* bottom_label = bottom[n_pred]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  
	Dtype max_prob = 0;
	int arg_max = 0;
	Dtype prob_pred = 0;
  //For each data point
  for (int i = 0; i < num; ++i) {
		
		int label = static_cast<int>(bottom_label[i]);

		//Find if any classifier is correct
		for (int j = 0; j< n_pred; ++j) {
			const Dtype* bottom_data = bottom[j]->cpu_data();
			max_prob = 0;
			for(int k = 0; k < dim; k++){
				prob_pred = std::max(bottom_data[i * dim + k], 
															Dtype(kLOG_THRESHOLD));
				if(prob_pred > max_prob){
					max_prob = prob_pred;
					arg_max = k;
				} 
			
			}
			if(arg_max == label){
				accuracy++;
				break;
  		}
    }
	}
    
		
	

  top[0]->mutable_cpu_data()[0] = accuracy/(double)num;

}

INSTANTIATE_CLASS(EnsembleAccuracyLayer);
REGISTER_LAYER_CLASS(EnsembleAccuracy);

}  // namespace caffe
