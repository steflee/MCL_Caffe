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
void SequentialMCLMultinomialLogisticLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);


  if(bottom.size() == 4){
    CHECK_EQ(bottom[0]->num(), bottom[2]->num());
    CHECK_EQ(bottom[2]->num(), bottom[3]->num());

    CHECK_EQ(bottom[2]->channels(), 1);
    CHECK_EQ(bottom[2]->height(), 1);
    CHECK_EQ(bottom[2]->width(), 1);

    CHECK_EQ(bottom[3]->channels(), 1);
    CHECK_EQ(bottom[3]->height(), 1);
    CHECK_EQ(bottom[3]->width(), 1);
  }

  top[1]->Reshape(bottom[1]->num(), 1,1,1);
  top[2]->Reshape(bottom[1]->num(), 1,1,1);


  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype>
void SequentialMCLMultinomialLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  

  const double sigma = this->layer_param_.seq_mcl_param().sigma();
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_weights = top[1]->mutable_cpu_data();
  Dtype* top_min_loss = top[2]->mutable_cpu_data();


  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype weight_sum = 0;

  //If there isnt an input for weights/min_loss then we are the first
  //in the sequence and 
  if(bottom.size()==2){ 

    Dtype loss = 0;
    for (int i = 0; i < num; ++i) {
      int label = static_cast<int>(bottom_label[i]);
      Dtype prob = std::max(
          bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
      top_min_loss[i] = -log(prob);
      top_weights[i] = 1-exp(-top_min_loss[i]/(sigma*sigma));
      weight_sum += top_weights[i];
      loss += top_min_loss[i];
    }
    top[0]->mutable_cpu_data()[0] = loss/num;
  
  }else{
    const Dtype* bottom_weights = bottom[2]->cpu_data();
    const Dtype* bottom_min_loss = bottom[3]->cpu_data();
  
    Dtype loss = 0;
    Dtype lprob;
    for (int i = 0; i < num; ++i) {
      int label = static_cast<int>(bottom_label[i]);
      Dtype prob = std::max(
          bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
      lprob = -log(prob);
      top_min_loss[i] = std::min(lprob, bottom_min_loss[i]);
      top_weights[i] = bottom_weights[i]*(1-exp(-top_min_loss[i]/(sigma*sigma)));
      weight_sum += top_weights[i];
      loss += bottom_weights[i]*lprob;
    }

    top[0]->mutable_cpu_data()[0] = loss;

  }

  //Normalize weights  
  for(int i = 0; i < num; i++)
    top_weights[i] /= weight_sum;


}

template <typename Dtype>
void SequentialMCLMultinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const Dtype scale = - top[0]->cpu_diff()[0];


    if(bottom.size()==2){
      for (int i = 0; i < num; ++i) {
        int label = static_cast<int>(bottom_label[i]);
        Dtype prob = std::max(
            bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
        bottom_diff[i * dim + label] = scale / prob / num;
      }
    }
    else{
      const Dtype* bottom_weights = bottom[2]->cpu_data();

      for (int i = 0; i < num; ++i) {
        int label = static_cast<int>(bottom_label[i]);
        Dtype prob = std::max(
            bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
        bottom_diff[i * dim + label] = scale / prob * bottom_weights[i];
      }
    }
  }
}

INSTANTIATE_CLASS(SequentialMCLMultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(SequentialMCLMultinomialLogisticLoss);

}  // namespace caffe
