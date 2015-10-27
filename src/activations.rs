use af;

use tensor::Tensor;
use error::HALError;

pub fn tanh(x: &Tensor) -> Tensor {
  Tensor{ array: af::tanh(x.get()).unwrap()
          , device: x.device
          , manager: x.manager.clone() }
}

pub fn sigmoid(x: &Tensor) -> Tensor {
  // let exponentiated = x.map(|e| 1.0/(1.0 + af::exp(-1.0 * e)));
  // exponentiated.unwrap();
  let denominator = af::add(&1.0f32, &af::exp(&af::mul(&-1.0f32, x.get(), false).unwrap()).unwrap(), false).unwrap();
  Tensor { array: af::div(&1.0f32, &denominator, false).unwrap()
          , device: x.device
          , manager: x.manager.clone() }
}

pub fn softmax(x: &Tensor) -> Tensor {
  let exponentiated = af::exp(x.get()).unwrap();
  // let exponentialted_sum = af::sum(exponentiated).unwrap();
  // let smax = exponentiated.map(|elem| af::div(elem, exponentialted_sum)).unwrap();
  // smax;
  Tensor{ array: af::div(&exponentiated, &af::sum_all(&exponentiated).unwrap().0, false).unwrap()
          , device: x.device
          , manager: x.manager.clone() }
}

pub fn tanh_derivative(x: &Tensor) -> Tensor {
  // 1 - tanh(x)*tanh(x)
  // let t = tanh(x);
  // af::sub(&1.0f32, &af::mul(&t, &t, false).unwrap(), false).unwrap();
  Tensor{ array: af::sub(&1.0f32, &af::mul(x.get(), x.get(), false).unwrap(), false).unwrap()
          , device: x.device
          , manager: x.manager.clone() }
}

pub fn sigmoid_derivative(x: &Tensor) -> Tensor {
  // x * (1 - x)
  //let s = sigmoid(x);
  //af::mul(&s, &af::sub(&1.0f32, &s, false).unwrap(), false).unwrap();
  Tensor { array: af::mul(x.get(), &af::sub(&1.0f32, x.get(), false).unwrap(), false).unwrap()
           , device: x.device
           , manager: x.manager.clone() }
}

pub fn softmax_derivative(x: &Tensor) -> Tensor {
  // x * (1 - x)
  //let s = softmax(x);
  //af::mul(&s, &af::sub(&1.0f32, &s, false).unwrap(), false).unwrap();
  Tensor{ array: sigmoid_derivative(x.get())
          , device: x.device
          , manager: x.manager.clone() }
}

pub fn ones(x: &Tensor) -> Tensor {
  Tensor{ array: x.clone()
          , device: x.device
          , manager: x.manager.clone() }
}


pub fn get_activation(name: &str, x: &Tensor) -> Result<Tensor, HALError> {
  match name {
    "softmax" => Ok(softmax(x)),
    "sigmoid" => Ok(sigmoid(x)),
    "tanh"    => Ok(tanh(x)),
    "ones"    => Ok(ones(x)),
    _         => Err(HALError::UNKNOWN),
  }
}

pub fn get_activation_derivative(name: &str, x: &Tensor) -> Result<Tensor, HALError> {
  match name {
    "softmax" => Ok(softmax_derivative(x)),
    "sigmoid" => Ok(sigmoid_derivative(x)),
    "tanh"    => Ok(tanh_derivative(x)),
    "ones"    => Ok(ones(x)),
    _         => Err(HALError::UNKNOWN),
  }
}
