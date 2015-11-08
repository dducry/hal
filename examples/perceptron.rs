#[macro_use] extern crate hal;
extern crate arrayfire as af;

use hal::Model;
use hal::optimizer::{Optimizer, SGD};
use hal::Tensor::tensor;
use hal::device::{Device, DeviceManager, Manager};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::plot::{plot_vec, plot_array};
use af::{Array, Dim4, AfBackend, Aftype};

fn build_optimizer(name: &str) -> Result<Box<Optimizer>, HALError> {
  match name{
    "SGD" => Ok(Box::new(SGD::default())),
    _     => Err(HALError::UNKNOWN),
  }
}

fn generate_sin_wave(input_dims: u64, num_rows: u64, manager: &Manager) ->  Tensor {
  let dims = Dim4::new(&[input_dims * num_rows, 1, 1, 1]);
  let x = af::div(&af::sin(&af::range(dims, 0, Aftype::F32).unwrap()).unwrap()
                  , &input_dims, false).unwrap();
  let wave = af::sin(&x).unwrap();
  Tensor::new(af::moddims(&wave, Dim4::new(&[num_rows, input_dims, 1, 1])).unwrap()
              , manager.clone()
              , AfBackend::AF_BACKEND_CPU
              , 0) //generate on device 0
}

fn main() {
  // First we need to parameterize our network
  let input_dims = 64;
  let hidden_dims = 32;
  let output_dims = 64;
  let num_train_samples = 65536;
  let batch_size = 32;
  let optimizer_type = "SGD";

  // instantiate a new device manager
  // this context is needed by all tensors
  let device_manager = DeviceManager::new();

  // Now, let's build a model with an optimizer and a loss function
  let mut model = Box::new(Sequential::new(build_optimizer(optimizer_type).unwrap() //optimizer
                                           , "mse"                                  // loss
                                           , AfBackend::AF_BACKEND_CUDA             // backend
                                           , 0));                                   // device_id

  // Let's add a few layers why don't we?
  model.add("dense", hashmap![  "activation"  => "tanh".to_string()
                              , "input_size"  => input_dims.to_string()
                              , "output_size" => hidden_dims.to_string()
                              , "w_init"      => "glorot_uniform".to_string()
                              , "b_init"      => "zeros".to_string()]);
  model.add("dense", hashmap![  "activation"  => "tanh".to_string()
                              , "input_size"  => hidden_dims.to_string()
                              , "output_size" => output_dims.to_string()
                              , "w_init"      => "glorot_uniform".to_string()
                              , "b_init"      => "zeros".to_string()]);

  // Get some nice information about our model
  model.info();

  // Temporarily set the backend to CPU so that we can load data into RAM
  // The model will automatically toggle to the desired backend during training
  //set_device(AfBackend::AF_BACKEND_CPU, 0);

  // Test with learning to predict sin wave
  let mut data = generate_sin_wave(input_dims, num_train_samples, &device_manager);
  let mut target = data.clone();

  // iterate our model in Verbose mode (printing loss)
  let loss = model.fit(&mut data, &mut target, batch_size
                       , false  // shuffle
                       , true); // verbose


  // plot our loss
  plot_vec(loss, "Loss vs. Iterations", 512, 512);

  // infer on one of our samples
  let temp = af::rows(&data, 0, batch_size - 1).unwrap();
  println!("temp shape= {:?}", temp.dims().unwrap().get().clone());
  let prediction = model.forward(&af::rows(&data, 0, batch_size - 1).unwrap(), false);
  println!("prediction shape: {:?}", prediction.dims().unwrap().get().clone());
  plot_array(&prediction, "Model Inference", 512, 512);
}
