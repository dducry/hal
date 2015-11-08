use af;
use af::{Dim4, Array};
use std::sync::{Arc, RwLock};

use tensor::Tensor;
use error::HALError;
use device::{Device, DeviceManager};


pub fn get_fans(dims: Dim4
                , device: Device
                , manager: &Arc<RwLock<DeviceManager>>) -> (f32, f32){
  let ndims = dims.ndims();
  let fan_in = match ndims {
    2  => dims[0],
    _  => dims.get()[1..ndims].iter().fold(1, |prod, x| prod * x) as u64,
  };
  let fan_out = match dims[1] {
    2  => dims[1],
    _  => dims[0],
  };
  (fan_in as f32, fan_out as f32)
}

pub fn normal(dims: Dim4, scale: f32
              , device: Device
              , manager: &Arc<RwLock<DeviceManager>>) -> Tensor {
  Tensor {array: af::mul(&af::randn(dims, af::Aftype::F32).unwrap(), &scale, false).unwrap()
          , device: device
          , manager: manager.clone() }
}

pub fn uniform(dims: Dim4, scale: f32
               , device: Device
               , manager: &Arc<RwLock<DeviceManager>>) -> Tensor{
  Tensor {array: af::sub(&af::mul(&af::randu(dims, af::Aftype::F32).unwrap(), &scale, false).unwrap()
                         , &(scale / 2.0f32), false).unwrap()
          , device: device
          , manager: manager.clone() }
}

pub fn zeros(dims: Dim4
             , device: Device
             , manager: &Arc<RwLock<DeviceManager>>) -> Tensor {
  Tensor{ array: af::constant(0.0 as f32, dims).unwrap()
          , device: device
          , manager: manager.clone() }
}

pub fn ones(dims: Dim4
            , device: Device
            , manager: &Arc<RwLock<DeviceManager>>) -> Tensor {
  Tensor{ array: af::constant(1.0 as f32, dims).unwrap()
          , device: device
          , manager: manager.clone() }
}

pub fn glorot_uniform(dims: Dim4
                      , device: Device
                      , manager: &Arc<RwLock<DeviceManager>>) -> Tensor {
  let (fan_in, fan_out) = get_fans(dims);
  let s = (6.0f32 / (fan_in + fan_out)).sqrt();
  uniform(dims, s)
}

pub fn glorot_normal(dims: Dim4
                     , device: Device
                     , manager: &Arc<RwLock<DeviceManager>>) -> Tensor {
  let (fan_in, fan_out) = get_fans(dims);
  let s = (2.0f32 / (fan_in + fan_out)).sqrt();
  normal(dims, s)
}

pub fn lecun_uniform(dims: Dim4
                     , device: Device
                     , manager: &Arc<RwLock<DeviceManager>>) -> Tensor {
  let (fan_in, _) = get_fans(dims);
  let s = 3.0f32 / fan_in;
  uniform(dims, s)
}

//TODO: Orthogonal

pub fn get_initialization(name: &str
                          , dims: Dim4
                          , device: Device
                          , manager: &Arc<RwLock<DeviceManager>>) -> Result<Tensor, HALError> {
  match name {
    "glorot_uniform" => Ok(glorot_uniform(dims, device, manager)),
    "glorot_normal"  => Ok(glorot_normal(dims, device, manager)),
    "lecun_uniform"  => Ok(lecun_uniform(dims, device, manager)),
    "normal"         => Ok(normal(dims, 0.05f32, device, manager)), //TODO: Parameterize
    "uniform"        => Ok(uniform(dims, 0.05f32, device, manager)), //TODO: Parameterize
    "zeros"          => Ok(zeros(dims, device, manager)),
    "ones"           => Ok(ones(dims, device, manager)),
    _                => Err(HALError::UNKNOWN),
  }
}
