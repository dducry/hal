use af;
use af::AfBackend;
use std::cmp::max;
use std::default::Default;
use std::collections::HashMap;

use utils;
use loss;
use tensor::Tensor;
use layer::{Layer, Dense};
use model::Model;
use optimizer::{Optimizer, SGD};
use params::{ParamManager, DenseGenerator, LSTMGenerator, Input};

pub struct Sequential {
  layers: Vec<Box<Layer>>,
  param_manager: ParamManager,
  optimizer: Box<Optimizer>,
  loss: String,
  device: i32,
  backend: AfBackend,
}

impl Default for Sequential {
  fn default() -> Sequential {
    Sequential {
      layers: Vec::new(),
      param_manager: ParamManager::default(),
      optimizer: Box::new(SGD::default()),
      loss: "mse".to_string(),
      backend: AfBackend::AF_BACKEND_CPU,
      device: 0,
    }
  }
}

impl Model for Sequential {
  fn new(optimizer: Box<Optimizer>
         , loss: &str
         , backend: AfBackend
         , device: i32) -> Sequential {
    Sequential {
      layers: Vec::new(),
      param_manager: ParamManager::default(),
      loss: loss.to_string(),
      optimizer: optimizer,
      backend: backend,
      device: device,
    }
  }

  fn add(&mut self, layer: &str, params: HashMap<&str, String>)
  {
    //TODO: Error handling for hashmap
    let input_size = params.get("input_size").unwrap().parse::<u64>().unwrap() as usize;
    let output_size = params.get("output_size").unwrap().parse::<u64>().unwrap() as usize;
    match layer {
      "dense" => {
        self.param_manager.add_dense(self.backend, self.device
                                     , input_size, output_size
                                     , params.get("activation").unwrap()
                                     , params.get("w_init").unwrap()
                                     , params.get("b_init").unwrap());
        self.layers.push(Box::new(Dense{input_size: input_size
                                        , output_size: output_size}));
      },
      // "lstm"  => {
      //   self.param_manager.add_lstm(input_size, output_size
      //                               , params.get("input_activation").unwrap()
      //                               , params.get("outer_activation").unwrap()
      //                               , params.get("w_init").unwrap()
      //                               , params.get("w_recurrent_ippnit").unwrap()
      //                               , params.get("forget_b_init").unwrap(),
      //                               , params.get("b_init").unwrap());
      //   self.layers.push(Box::new(LSTM{input_size: input_size
      //                                   , output_size: output_size}));
      // },
      _  => panic!("Error unknown layer type"),
    }
  }

  //TODO: convert to log crate [or hashmap]
  fn info(&self) {
    println!("");
    self.optimizer.info();
    println!("loss:           {}\nnum_layers:     {}", self.loss, self.layers.len());
  }

  fn forward(&mut self, activation: &Tensor, train: bool) -> Tensor {
    // if dim[3] > 1 we assume we have an RNN
    // we will need to unwind at least once for non RNNs
    let bptt_unroll = max(activation.get().dims().unwrap()[2], 1);
    let mut activate = Input {data: Tensor{ array: af::slice(activation.get(), 0).unwrap()
                                            , device: activation.device
                                            , manager: activation.manager.clone() }
                              , activation: "ones".to_string()};
    for t in 0..bptt_unroll {
      activate.data = Tensor { array: af::slice(activation, t).unwrap()
                               , device: activation.device
                               , manager: activation.manager.clone() };
      for i in 0..self.layers.len() {
        activate = self.layers[i].forward(self.param_manager.get_mut_params(i)
                                          , &activate, train);
      }
    }
    activate.data
  }


  fn fit(&mut self, input: &mut Tensor, target: &mut Tensor, batch_size: u64
         , shuffle: bool, verbose: bool) -> Vec<f32>
  {
    // some required data validity checks
    let idims = input.dims().unwrap().get().clone();
    let tdims = target.dims().unwrap().get().clone();
    let iter =  idims[0] as u64 / batch_size as u64;
    println!("\ntrain samples: {:?} | target samples: {:?} | batch size: {} | iterations: {}"
             , idims, tdims, batch_size, iter);
    assert!(tdims[0] == idims[0]);
    assert!(idims[0] >= batch_size
            && idims[0] % batch_size == 0); //ease up later

    // create the container to hold the forward pass & loss results
    let mut a_t: Tensor;
    let mut loss: f32;
    let mut lossvec = Vec::<f32>::new();

    // randomly shuffle the data
    if shuffle {
      utils::shuffle_array(&mut[input, target], idims[0]);
    }

    // normalize the data by mean and 3 std deviations
    *input = utils::normalize_array(input, 3.0f32);
    *target = utils::normalize_array(target, 3.0f32);

    // over every batch
    let mut current_iteration = 0;
    for i in (0..idims[0]).filter(|&x| x % batch_size == 0) {
      if verbose {
        print!("\n[iter: {}] ", current_iteration);
        current_iteration += 1;
      }

      // extract part of the array onto the GPU
      let cpu_batch_input  = utils::row_planes(input, i, i + batch_size - 1).unwrap();
      let cpu_batch_target = utils::row_planes(target, i, i+ batch_size - 1).unwrap();
      let batch_input  = af::transpose(&utils::array_swap_backend(&cpu_batch_input
                                                                  , AfBackend::AF_BACKEND_CPU
                                                                  , self.backend
                                                                  , 0, self.device), false).unwrap();
      let batch_target = af::transpose(&utils::array_swap_backend(&cpu_batch_target
                                                                  , AfBackend::AF_BACKEND_CPU
                                                                  , self.backend
                                                                  , 0, self.device), false).unwrap();
      self.optimizer.setup(self.param_manager.get_all_weight_dims()
                           , self.param_manager.get_all_bias_dims());

      // DEBUG:
      // println!("batched [input: {:?} | target: {:?}]"
      //          , batch_input.dims().unwrap().get().clone()
      //          , batch_target.dims().unwrap().get().clone());

      a_t = self.forward(&batch_input, true);
      loss = self.backward(&a_t, &batch_target);
      self.optimizer.update(&mut self.param_manager, batch_size as u64);
      lossvec.push(loss);
      if verbose {
        print!("{} ", loss);
      }
    }

    utils::write_csv::<f32>("loss.csv", &lossvec);
    lossvec
  }

  fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> f32 {
    let last_index = self.layers.len() - 1;
    let mut delta = loss::loss_delta(prediction
                                     , target
                                     , &self.loss
                                     , &self.param_manager.get_activation(last_index, 0));

    for i in (0..last_index + 1).rev() {
      delta = self.layers[i].backward(self.param_manager.get_mut_params(i), &delta);
    }

    loss::get_loss(&self.loss, prediction, target).unwrap()
  }
}
