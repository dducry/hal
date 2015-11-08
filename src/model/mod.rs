//mod Optimizer;
//mod Loss;

pub use self::sequential::Sequential;
mod sequential;

use af::{AfBackend};
use std::collections::HashMap;

use tensor::Tensor;
use optimizer::Optimizer;

pub trait Model {
  fn new(optimizer: Box<Optimizer>
         , loss: &str
         , backend: AfBackend
         , device: i32) -> Self;
  fn fit(&mut self, input: &mut Tensor, target: &mut Tensor, batch_size: u64
         , shuffle: bool, verbose: bool) -> Vec<f32>;
  fn forward(&mut self, activation: &Tensor, train: bool) -> Tensor;
  fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> f32;
  fn add(&mut self, layer: &str
         , params: HashMap<&str, String>);
  fn info(&self);
}
