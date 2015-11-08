use af;
use af::{MatProp};

use activations;
use params::{Input, Params};
use layer::Layer;
use tensor::Tensor;

pub struct Dense {
  pub input_size: usize,
  pub output_size: usize,
}

impl Layer for Dense
{
  fn forward(&self, params: &mut Params, inputs: &Input, train: bool)-> Input
  {
    // z_t = Wx + b [the bias is added in parallel for batch]
    let z_t = params.weights[0].matmul(&inputs.data    //activated_input
                                    , MatProp::NONE
                                    , MatProp::NONE) + params.biases[0];

    let z_t = af::add(&af::matmul( &params.weights[0]
                      , &params.biases[0], true).unwrap();
    // a_t = sigma(z_t)
    let a_t = Input{ data: activations::get_activation(&params.activations[0], &z_t).unwrap()
                     , activation: params.activations[0].clone() };

    // parameter manager keeps the output & inputs
    // these are only needed for training, so dont store otherwise
    if train {
      params.inputs = vec![inputs.clone()];
      params.outputs = vec![a_t.clone()];
    }

    a_t.clone() // clone just increases the ref count
  }

  fn backward(&self, params: &mut Params, delta: &Tensor) -> Tensor {
    // delta_t     = (transpose(W_{t+1}) * d_{l+1}) .* dActivation(z)
    // delta_{t-1} = (transpose(W_{t}) * d_{l})
    params.deltas = vec![delta.batch_mul(&activations::get_activation_derivative(&params.activations[0]
                                                                                 , &params.outputs[0].data).unwrap(), false)];
    params.weights[0].matmul(&params.deltas[0], af::MatProp::TRANS, af::MatProp::NONE)
  }
}
