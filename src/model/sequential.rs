use af;
use af::{Array, Dim4, AfBackend};
use na::{DMat, Shape, Transpose};
use std::cmp::max;
use std::default::Default;
use std::collections::HashMap;
use itertools::Zip;

use utils;
use loss;
use activations;
use initializations;
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
      backend: AfBackend::AP_BACKEND_CPU,
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

  fn add(&mut self, layer: &str
         , params: HashMap<&str, &str>)
  {
    //TODO: Error handling for hashmap
    let input_size = params.get("input_size").unwrap().parse::<u64>().unwrap() as usize;
    let output_size = params.get("output_size").unwrap().parse::<u64>().unwrap() as usize;
    match layer {
      "dense" => {
        self.param_manager.add_dense(input_size, output_size
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
      //                               , params.get("w_recurrent_init").unwrap()
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
    match af::info() {
      Ok(_)   => {},
      Err(e)  => panic!("could not get info: {:?}", e),
    };

    println!("");
    self.optimizer.info();
    println!("loss:           {}\nnum_layers:     {}", self.loss, self.layers.len());
  }

  fn set_device(&mut self, backend: AfBackend, device_id: i32) {
    match af::set_backend(backend) {
      Ok(_)  => {},
      Err(e) =>  panic!("could not set backend: {:?}", e),
     };

    match af::set_device(device_id) {
      Ok(_)  => {},
      Err(e) =>  panic!("could not set device: {:?}", e),
     };
  }

  fn forward(&mut self, activation: &Array) -> Array {
    // if dim[3] > 1 we assume we have an RNN
    // we will need to unwind at least once for non RNNs
    let bptt_unroll = max(activation.dims().unwrap()[2], 1);
    let mut activate = Input {data: af::slice(activation, 0).unwrap(), activation: "ones".to_string()};
    for t in 0..bptt_unroll {
      activate.data = af::slice(activation, t).unwrap();
      for i in 0..self.layers.len() {
        activate = self.layers[i].forward(self.param_manager.get_mut_params(i)
                                          , &activate);
      }
    }
    activate.clone()
  }


  fn fit(&mut self, input: &mut Array, target: &mut Array
         , batch_size: usize, return_predictions: bool
         , shuffle: bool, verbose: bool) -> (Vec<f32>, Option<Vec<Array>>)
  {
    // some required data validity checks
    let idims = input.dims().unwrap().get();
    let tdims = target.dims().unwrap().get();
    let iter =  idims[0] as u64 / batch_size as u64;
    println!("\ntrain samples: {:?} | target samples: {:?} | batch size: {} | iterations: {}"
             , idims, tdims, batch_size, iter);
    assert!(tdims[0] == idims[0]);
    assert!(idims[0] >= batch_size
            && idims[0] % batch_size == 0); //ease up later
    self.optimizer.setup(self.param_manager.get_all_weight_dims()
                         , self.param_manager.get_all_bias_dims());

    // create the container to hold the forward pass & loss results
    let mut forward_pass = initializations::zeros(Dim4::new(&[1, input.ncols() as u64, 1, 1]));
    let mut lossvec = Vec::<f32>::new();
    let mut fwdpassvec = Vec::<Array>::new();
    let mut loss: f32;

    // randomly shuffle the data
    if shuffle {
      let col_vec = [input.ncols(), target.ncols()];
      utils::shuffle(&mut[input.as_mut_vec(), target.as_mut_vec()], &col_vec, false);
    }

    // normalize the data by mean and 3 std deviations
    *input = utils::normalize_array(input, 3.0f32);
    *target = utils::normalize_array(target, 3.0f32);

    // over every batch
    let mut current_iteration = 0; //TODO: get this working with array chunks!!
    for (i, t) in Zip::new((input.transpose().as_vec().chunks(idims[1] * batch_size)
                            , target.transpose().as_vec().chunks(tdims[1] * batch_size as usize)))
    {
      if verbose {
        print!("\n[iter: {}] ", current_iteration);
        current_iteration += 1;
      }

      // column major order is preferred for BLAS
      let batch_input  = utils::raw_to_array(i, idims[1],  batch_size as usize);
      let batch_target = utils::raw_to_array(t, tdims[1], batch_size as usize);

      // DEBUG:
      // println!("batched [input: {:?} | target: {:?}]"
      //          , batch_input.dims().unwrap()
      //          , batch_target.dims().unwrap());
      forward_pass = self.forward(&batch_input);
      loss = self.backward(&forward_pass, &batch_target);
      self.optimizer.update(&mut self.param_manager, batch_size as u64);

      lossvec.push(loss);
      if verbose {
        print!("{} ", loss);
      }

      if return_predictions {
        self.set_device(AfBackend::AF_BACKEND_CPU, 0);
        fwdpassvec.push(loss.clone());
        self.set_device(self.backend, self.device_id);
      }
    }

    utils::write_csv::<f32>("loss.csv", &lossvec);
    match fwdpassvec.len() {
      0 => (lossvec, None),
      _ => (lossvec, fwdpassvec)
    }
  }

  fn backward(&mut self, prediction: &Array, target: &Array) -> f32 {
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
