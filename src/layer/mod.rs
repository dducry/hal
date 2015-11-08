pub use self::dense::Dense;
mod dense;

// pub use self::lstm::LSTM;
// mod lstm;

use tensor::Tensor;
use params::{Input, Params};

pub trait Layer {
  fn forward(&self, params: &mut Params, inputs: &Input, train: bool) -> Input;
  fn backward(&self, params: &mut Params, delta: &Tensor) -> Tensor;
}

pub trait RecurrentLayer: Layer {
  fn new(input_size: u64, output_size: u64
         , inner_activation: &str, outer_activation: &str
         , w_init: &str, b_init: &str) -> Self where Self: Sized;
}

pub trait RTRL{
  fn rtrl(&self, dW_tm1: &mut Tensor  // previous W derivatives for [I, F, Ct]
              , dU_tm1: &mut Tensor   // previous U derivatives for [I, F, Ct]
              , db_tm1: &mut Tensor   // previous b derivatives for [I, F, Ct]
              , z_t: &Tensor          // current time activation
              , inputs: &Input);     // x_t & h_{t-1}
}
