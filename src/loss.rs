use af;

use tensor::Tensor;
use activations;
use error::HALError;

pub fn mse(pred: &Tensor, target: &Tensor) -> f32 {
  //let diff = af::sub(pred, target, false).unwrap();
  //  (af::mean_all(&af::mul(&diff, &diff, false).unwrap()).unwrap()).0 as f32;
  let diff = pred - target;
  (af::mean_all(diff.get()).unwrap()).0 as f32
}

pub fn cross_entropy(pred: &Tensor, target: &Tensor) -> f32 {
  // y: true target, x: prediction
  // E = sum(-ylnx - [1-y]ln[1-x])
  let pos = af::mul(&af::mul(&-1.0, target.get(), false).unwrap(), &af::log(pred.get()).unwrap(), false).unwrap(); // -ylnx
  let neg = af::mul(&af::sub(&1.0, target.get(), false).unwrap(), &af::log(&(af::sub(&1.0, pred.get(), false).unwrap())).unwrap(), false).unwrap(); //[1-y]ln[1-x]
  let e = af::sub(&pos, &neg, false).unwrap();
  af::sum_all(&e).unwrap().0 as f32
}

pub fn mse_derivative(pred: &Tensor, target: &Tensor) -> Tensor {
  //af::sub(pred, target, false).unwrap();
  pred - target
}

pub fn cross_entropy_derivative(pred: &Tensor, target: &Tensor) -> Tensor {
  mse_derivative(pred, target)
}

pub fn loss_delta(prediction: &Tensor, target: &Tensor
              , loss: &str, activation_type: &str) -> Tensor
{
  // d_L = d_loss * d(z) where z = activation w/out non-linearity (& in this case the predictions)
  let activated_prediction = activations::get_activation(activation_type, prediction.get()).unwrap();
  let d_loss = get_loss_derivative(loss, &activated_prediction, target.get()).unwrap();
  let d_z = activations::get_activation_derivative(activation_type, &activated_prediction).unwrap();
  Tensor { array: af::mul(&d_loss, &d_z, false).unwrap()
           , device: prediction.device
           , manager: prediction.manager.clone() }
}

pub fn get_loss(name: &str, pred: &Tensor, target: &Tensor) -> Result<f32, HALError> {
  match name {
    "mse"           => Ok(mse(pred, target)),
    "cross_entropy" => Ok(cross_entropy(pred, target)),
    _               => Err(HALError::UNKNOWN),
  }
}

pub fn get_loss_derivative(name: &str, pred: &Tensor, target: &Tensor) -> Result<Tensor, HALError> {
  match name {
    "mse"           => Ok(mse_derivative(pred, target)),
    "cross_entropy" => Ok(cross_entropy_derivative(pred, target)),
    _               => Err(HALError::UNKNOWN),
  }
}
