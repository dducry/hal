use num;
use af;
use af::{Array, MatProp, Dim4, DType};

use utils;
use std::sync::{Arc, Mutex};
use activations;
use initializations;
use params::{Params};
use layer::Layer;

use num::Complex;


pub struct Unitary {
  pub input_size: usize,
  pub output_size: usize,
}

/// Compute the multiplication with the diagonal matrix of phases D
///
/// # Parameters
/// - 'param' is the vector of phases
/// - 'ar' is the current hidden state
fn h_d(param: Array, ar: Array) -> Array {
  let D = af::cplx2(&af::cos(&param)
                    , &af::sin(&param)
                    ,false);
  af::mul(&D, &ar, true)
}

/// Compute the multiplication with the diagonal matrix of phases D for the backpropagation
/// (complex conjugate)
///
/// # Parameters
/// - 'param' is the vector of phases
/// - 'ar' is the current hidden state
fn r_d(param: Array, ar: Array) -> Array {
  let D = af::cplx2(&af::cos(&param)
                    , &af::mul(&af::sin(&param), &-1, true)
                    ,false);
  af::mul(&D, &ar, true)
}
/// Wrapper to apply the fft to the current hidden state
///
/// # Parameters
/// - 'ar' is the current hidden state
fn h_fft(ar: Array) -> Array {
  let ar_size = ar.dims()[1];
  af::transpose(&af::fft(&af::transpose(&ar, false), 1.0, ar_size as i64)
                , false)
}

/// Wrapper to apply the fft inverse to the current hidden state for the backpropagation
///
/// # Parameters
/// - 'ar' is the current hidden state
fn r_fft(ar: Array) -> Array {
  let ar_size = ar.dims()[1];
  af::transpose(&af::ifft(&af::transpose(&ar, false), 1.0, ar_size as i64)
                , false)
}

/// Wrapper to apply the fft inverse to the current hidden state
///
/// # Parameters
/// - 'ar' is the current hidden state
fn h_ifft(ar: Array) -> Array {
  let ar_size = ar.dims()[1];
  af::transpose(&af::ifft(&af::transpose(&ar, false), 1.0/(ar_size as f64), ar_size as i64)
                , false)
}

/// Wrapper to apply the fft to the current hidden state for the backpropagation
///
/// # Parameters
/// - 'ar' is the current hidden state
fn r_ifft(ar: Array) -> Array {
  let ar_size = ar.dims()[1];
  af::transpose(&af::fft(&af::transpose(&ar, false), 1.0/(ar_size as f64), ar_size as i64)
                , false)
}

/// Apply the permutation pi
///
/// # Parameters
/// - 'param' is the vector defining the permutation
/// - 'ar' is the current hidden state
fn h_pi(param: Array, ar: Array) -> Array {
  af::lookup(&ar, &param, 1)
}

/// Compute the multiplication with the Householder reflexion matrix R
///
/// # Parameters
/// - 'param' is the vector defining the direction of the reflexion
/// - 'ar' is the current hidden state
fn h_r(param: Array, mut ar: Array) -> Array {
  let ar_temp = ar.clone();
  ar = af::matmul(&param
                  , &ar
                  , MatProp::NONE
                  , MatProp::TRANS);
  ar = af::matmul(&ar
                  , &af::conjg(&param)
                  , MatProp::TRANS
                  , MatProp::NONE);
  ar = af::mul(&ar, &2, true);
  ar = af::sub(&ar_temp, &ar, false);
  ar

}

// Wh = (D3 R2 F-1 D2 Pi R1 F D1) h
/// Wrapper to compute the multiplication with the whole hidden2hidden matrix
///
/// # Parameters
/// - 'p' parameters of the hidden2hidden matrix
fn wh(p1: Array, p2: Array, p3: Array, p4: Array, p5: Array, p6: Array, ar: Array) -> Array { 
  let mut current = ar;
  current = h_d(p1, current);
  current = h_fft(current);
  current = h_r(p2, current);
  current = h_pi(p3, current);
  current = h_d(p4, current);
  current = h_ifft(current);
  current = h_r(p5, current);
  current = h_d(p6, current);
  current

}

/// Convert real vectors to a complex ones using the first and second half as real part and imaginary part respectively
/// # Parameters
/// - 'ar' is the matrix of real vectors to convert
fn to_complex(ar:Array) -> Array {  
  let dim = ar.dims()[1];
  assert!(dim % 2 == 0, "The dimension of the complex split has to be even");
  af::cplx2(&af::cols(&ar, 0, dim/2-1), &af::cols(&ar, dim/2, dim-1), false)
}

/// Convert complex vectors to real ones concatenating real and imaginary parts
///
/// # Parameters
/// - 'ar' is the matrix of complex vectors to convert
fn to_real(ar: Array) -> Array {
  af::join(1, &af::real(&ar), &af::imag(&ar))
}

impl Layer for Unitary
{
  fn forward(&self, params:  Arc<Mutex<Params>>, inputs: &Array, state: Option<&Vec<Array>>) -> (Array, Option<Vec<Array>>)
  {
    let mut ltex = params.lock().unwrap();
    let t = ltex.current_unroll;


    // Transformation of complex parameters
    let mut weight0 = to_complex(ltex.weights[0].clone());
    let mut weight4 = to_complex(ltex.weights[4].clone());
    let mut weight5 = to_complex(ltex.weights[5].clone());

    // Not complex
    let mut weight1 = ltex.weights[1].clone();
    let mut weight2 = ltex.weights[2].clone();
    let mut weight3 = ltex.weights[3].clone();
    let mut weight6 = ltex.weights[6].clone();
    let mut bias0 = ltex.biases[0].clone();
    let mut bias1 = ltex.biases[1].clone();

    if t == 0 {
      // Initialize hidden state in the first iteration
      if ltex.recurrences.len() == 0 {
        let temp = ltex.weights[7].clone();
        ltex.recurrences.push(temp);
      }
      else {
        ltex.recurrences[0] = ltex.weights[7].clone();
      }
      let hidden_size = ltex.weights[7].dims()[1];
      let batch_size = inputs.dims()[0];

      // Make a copy of h0 for all batch inputs
      let zero = af::constant(0f32, Dim4::new(&[batch_size, hidden_size, 1, 1]));
      ltex.recurrences[0] = af::add(&zero, &ltex.recurrences[0], true);

      // We normalize Householder parameters;
      let sqrNorm = af::norm(&weight4, af::NormType::VECTOR_2, 1., 1.)as f32;
      weight4 = af::div(&weight4, &sqrNorm, true);
      ltex.weights[4] = to_real(weight4.clone());
      let sqrNorm = af::norm(&weight5, af::NormType::VECTOR_2, 1., 1.)as f32;
      weight5 = af::div(&weight5, &sqrNorm, true);
      ltex.weights[5] = to_real(weight5.clone());
    }
    let mut rec_t = to_complex(ltex.recurrences[t].clone());

    rec_t = match state {
      Some(init_state)    => init_state[0].clone(),
      None                => rec_t
    };

    // we compute h_t+1 = sigma1(W*h_t + V*x_t + b1) 
    let wh = wh(weight1.clone()
                , weight4.clone()
                , ltex.optional[0].clone()
                , weight2.clone()
                , weight5.clone()
                , weight3.clone()
                , rec_t.clone());

    // In order to convert inputs.data into a complex array
    let c_zeros = initializations::zeros::<Complex<f32>>(inputs.dims()); 

    let c_inputs = af::add(inputs, &c_zeros, false);
    let vx = af::matmul(&c_inputs
                        , &weight0
                        , MatProp::NONE
                        , MatProp::NONE);

    let vx_wh = af::add(&vx, &wh, false);
    let new_h = activations::mod_relu(vx_wh.clone(), bias0.clone());

    // we compute o_t = sigma2(U*h_t + b2)
    let r_h = af::real(&new_h);
    let c_h = af::imag(&new_h);
    let concat_h = af::join(1, &r_h, &c_h);
    let uh = af::matmul(&concat_h
                        , &weight6
                        , MatProp::NONE
                        , MatProp::NONE);

    let new_o = af::add(&uh, &bias1, true);


    let out = activations::get_activation(&ltex.activations[0]
                                          , &new_o).unwrap(); 


    if ltex.inputs.len() > t {
      ltex.inputs[t] = c_inputs.clone();
      ltex.recurrences[t+1] = to_real(new_h.clone());
      ltex.outputs[t] = out.clone();
      ltex.optional[t+2] = vx_wh.clone();

    }
    else{
      ltex.inputs.push(c_inputs.clone());
      ltex.recurrences.push(to_real(new_h.clone()));
      ltex.outputs.push(out.clone());
      ltex.optional.push(vx_wh.clone());
    }
    //println!("{}", &(af::norm(&ltex.recurrences[t], af::NormType::VECTOR_2, 1.,1.)as f32));
    ltex.current_unroll += 1;

    (out.clone(), None)
  }



  fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array) -> Array {
    let mut ltex = params.lock().unwrap();
    let t = ltex.current_unroll;

    // Transformation of complex parameters
    let mut weight0 = to_complex(ltex.weights[0].clone());
    let mut weight4 = to_complex(ltex.weights[4].clone());
    let mut weight5 = to_complex(ltex.weights[5].clone());
    let mut rec_t0 = to_complex(ltex.recurrences[t-1].clone());
    let mut rec_t1 = to_complex(ltex.recurrences[t].clone());

    let mut delta0 = to_complex(ltex.deltas[0].clone());
    let mut delta4 = to_complex(ltex.deltas[4].clone());
    let mut delta5 = to_complex(ltex.deltas[5].clone());
    let mut delta7 = to_complex(ltex.deltas[7].clone());

    // Not complex
    let mut weight1 = ltex.weights[1].clone();
    let mut weight2 = ltex.weights[2].clone();
    let mut weight3 = ltex.weights[3].clone();
    let mut weight6 = ltex.weights[6].clone();
    let mut bias0 = ltex.biases[0].clone();
    let mut bias1 = ltex.biases[1].clone();

    let mut delta1 = ltex.deltas[1].clone();
    let mut delta2 = ltex.deltas[2].clone();
    let mut delta3 = ltex.deltas[3].clone();
    let mut delta6 = ltex.deltas[6].clone();
    let mut delta8 = ltex.deltas[8].clone();
    let mut delta9 = ltex.deltas[9].clone();

    let p1 = weight1.clone();
    let p2 = weight4.clone();
    let p3 = ltex.optional[0].clone();
    let p3_bis = ltex.optional[1].clone();
    let p4 = weight2.clone();
    let p5 = weight5.clone();
    let p6 = weight3.clone();

    let dim_h = rec_t0.dims()[1];
    assert!(t >= 0
            , "Cannot call backward pass without at least 1 forward pass");


    // We write d_ to say dL/d_
    // do => dz2

    // Check to see if we already have a state derivative, else add one
    if ltex.state_derivatives.len() == 0 {
      let h_size = ltex.recurrences[0].dims();
      let h_type = ltex.recurrences[0].get_type();
      ltex.state_derivatives.push(utils::constant(h_size, h_type, 0f32));
    }

    let d_z2 = af::mul(delta
                       , &activations::get_derivative(&ltex.activations[0], &ltex.outputs[t-1]).unwrap()
                       , false);

    // dz2 => dh_{t}
    let prod = af::matmul(&d_z2, &weight6, MatProp::NONE, MatProp::TRANS);
    let d_h1 = af::cplx2(&af::cols(&prod, 0, dim_h-1)
                         , &af::cols(&prod, dim_h, 2*dim_h-1)
                         , false);

    // dz2 & dh_{t+1} => dh_{t}
    let d_h2 = to_complex(ltex.state_derivatives[0].clone());
    let d_rec = af::add(&d_h1, &d_h2, false);

    // dh_{t} => dz
    let d_z = activations::mod_relu_derivative_z(ltex.optional[t+1].clone()
                                                 , bias0.clone()
                                                 , d_rec.clone());

    // dz => dh_{t-1} (used in the next step)
    let new_d_h2 = r_d(p1.clone()
                       , r_fft(h_r(p2.clone()
                                   , h_pi(p3_bis.clone()
                                          , r_d(p4.clone()
                                                , r_ifft(h_r(p5.clone()
                                                             , r_d(p6.clone(), d_z.clone()))))))));

    ltex.state_derivatives[0] = to_real(new_d_h2.clone());

    //-----------------------------------------------------------------------------
    // dz => dW
    // dD

    // D1
    let dd1_left = rec_t0.clone();
    let dd1_right = r_fft(h_r(p2.clone()
                              , h_pi(p3_bis.clone()
                                     , r_d(p4.clone()
                                           , r_ifft(h_r(p5.clone()
                                                        , r_d(p6.clone(), d_z.clone())))))));
    // We add the derivatives from the real and imaginary parts
    let dd1_sin = af::sin(&weight1);
    let dd1_cos = af::cos(&weight1);
    let dd1_real = af::mul(&af::add(&af::mul(&af::real(&dd1_left), &dd1_sin, true), &af::mul(&af::imag(&dd1_left), &dd1_cos, true), false), &-1, true);
    let dd1_imag = af::sub(&af::mul(&af::real(&dd1_left), &dd1_cos, true), &af::mul(&af::imag(&dd1_left), &dd1_sin, true), false);
    let dd1_phase = af::add(&af::mul(&dd1_real, &af::real(&dd1_right), false)
                            , &af::mul(&dd1_imag, &af::imag(&dd1_right), false)
                            , false);
    ltex.deltas[1] = af::add(&delta1, &af::sum(&dd1_phase, 0), false);

    // D2
    let dd2_left = h_pi(p3.clone()
                        , h_r(p2.clone()
                              , h_fft(h_d(p1.clone(), rec_t0.clone()))));   
    let dd2_right = r_ifft(h_r(p5.clone()
                               , r_d(p6.clone(), d_z.clone())));
    let dd2_sin = af::sin(&weight2);
    let dd2_cos = af::cos(&weight2);
    let dd2_real = af::mul(&af::add(&af::mul(&af::real(&dd2_left), &dd2_sin, true), &af::mul(&af::imag(&dd2_left), &dd2_cos, true), false), &-1, true);
    let dd2_imag = af::sub(&af::mul(&af::real(&dd2_left), &dd2_cos, true), &af::mul(&af::imag(&dd2_left), &dd2_sin, true), false);
    let dd2_phase = af::add(&af::mul(&dd2_real, &af::real(&dd2_right), false)
                            , &af::mul(&dd2_imag, &af::imag(&dd2_right), false)
                            , false);
    ltex.deltas[2] = af::add(&delta2, &af::sum(&dd2_phase, 0), false);

    // D3
    let dd3_left = h_r(p5.clone()
                       , h_ifft(h_d(p4.clone()
                                    , h_pi(p3.clone()
                                           , h_r(p2.clone()
                                                 , h_fft(h_d(p1.clone(), rec_t0.clone())))))));
    let dd3_right = d_z.clone();
    let dd3_sin = af::sin(&weight3);
    let dd3_cos = af::cos(&weight3);
    let dd3_real = af::mul(&af::add(&af::mul(&af::real(&dd3_left), &dd3_sin, true), &af::mul(&af::imag(&dd3_left), &dd3_cos, true), false), &-1, true);
    let dd3_imag = af::sub(&af::mul(&af::real(&dd3_left), &dd3_cos, true), &af::mul(&af::imag(&dd3_left), &dd3_sin, true), false);
    let dd3_phase = af::add(&af::mul(&dd3_real, &af::real(&dd3_right), false)
                            , &af::mul(&dd3_imag, &af::imag(&dd3_right), false)
                            , false);
    ltex.deltas[3] = af::add(&delta3, &af::sum(&dd3_phase, 0), false);


    //------------------------------------------------------------------------------
    // dR

    // R1
    let dr1_left = h_fft(h_d(p1.clone(), rec_t0.clone()));
    let dr1_right = h_pi(p3_bis.clone()
                         , r_d(p4.clone()
                               , r_ifft(h_r(p5.clone()
                                            , r_d(p6.clone(), d_z.clone())))));

    let w = weight4.clone();
    let dh = dr1_right.clone();
    let dh2 = dh.clone();
    let h0 = dr1_left.clone();
    let h1 = af::matmul(&w, &h0, MatProp::NONE, MatProp::TRANS);
    let h2 = af::matmul(&h1, &af::conjg(&w), MatProp::TRANS, MatProp::NONE);

    let dh1 = af::transpose(&af::matmul(&w, &dh2, MatProp::NONE, MatProp::TRANS), false);
    let dr11 = af::mul(&af::conjg(&h0), &dh1, true);

    let dr12 = af::conjg(&af::mul(&af::transpose(&af::conjg(&h1), false), &dh2, true));

    let dh3 = af::sum(&af::mul(&dh, &af::conjg(&h2), false), 1);
    let dr13 = af::matmul(&dh3 , &w, MatProp::NONE, MatProp::NONE);

    let dr14 = af::conjg(&af::matmul(&dh3 , &af::conjg(&w), MatProp::NONE, MatProp::NONE));

    let dr1 = af::mul(&af::sub(&af::sub(&af::add(&dr11, &dr12, false), &dr13, false), &dr14, false), &-2, true);

    delta4 = af::add(&delta4, &af::sum(&dr1, 0), false);
    ltex.deltas[4] = to_real(delta4.clone());

    // R2

    let dr2_left = h_ifft(h_d(p4.clone()
                              , h_pi(p3.clone()
                                     , h_r(p2.clone()
                                           , h_fft(h_d(p1.clone(), rec_t0.clone()))))));
    let dr2_right = r_d(p6.clone(), d_z.clone());

    let w = weight5.clone();
    let dh = dr2_right.clone();
    let dh2 = dh.clone();
    let h0 = dr2_left.clone();
    let h1 = af::matmul(&w, &h0, MatProp::NONE, MatProp::TRANS);
    let h2 = af::matmul(&h1, &af::conjg(&w), MatProp::TRANS, MatProp::NONE);

    let dh1 = af::transpose(&af::matmul(&w, &dh2, MatProp::NONE, MatProp::TRANS), false);
    let dr21 = af::mul(&af::conjg(&h0), &dh1, true);

    let dr22 = af::conjg(&af::mul(&af::transpose(&af::conjg(&h1), false), &dh2, true));

    let dh3 = af::sum(&af::mul(&dh, &af::conjg(&h2), false), 1);
    let dr23 = af::matmul(&dh3 , &w, MatProp::NONE, MatProp::NONE);

    let dr24 = af::conjg(&af::matmul(&dh3 , &af::conjg(&w), MatProp::NONE, MatProp::NONE));

    let dr2 = af::mul(&af::sub(&af::sub(&af::add(&dr21, &dr22, false), &dr23, false), &dr24, false), &-2, true);

    delta5 = af::add(&delta5, &af::sum(&dr2, 0), false);
    ltex.deltas[5] = to_real(delta5.clone());




    // TO DO : fix the name of parameters to be coherent with the one of params.rs 
    //-----------------------------------------------------------------------------
    // dz => dU
    let d_u = af::matmul(&ltex.inputs[t-1]
                         , &d_z
                         , MatProp::CTRANS
                         , MatProp::NONE);
    delta0 = af::add(&delta0, &d_u, false);
    ltex.deltas[0] = to_real(delta0.clone());


    //-----------------------------------------------------------------------------
    // dz => db
    let mut d_b = activations::mod_relu_derivative_b(ltex.optional[t+1].clone()
                                                     , bias0.clone()
                                                     , d_rec.clone());
    //d_b = af::add(&af::real(&d_b), &af::imag(&d_b), false);
    d_b = af::real(&d_b);
    delta8 = af::add(&delta8, &af::sum(&d_b, 0), false);
    ltex.deltas[8] = delta8.clone();



    //-----------------------------------------------------------------------------
    // dz2 => db2
    ltex.deltas[9] = af::add(&delta9, &af::sum(&d_z2, 0), false);


    //-----------------------------------------------------------------------------
    // dz2 => dV
    let concat_h = af::join(1,
                            &af::real(&rec_t1)
                            , &af::imag(&rec_t1));
    let d_v = af::matmul(&concat_h
                         , &d_z2
                         , MatProp::TRANS
                         , MatProp::NONE);
    ltex.deltas[6] = af::add(&delta6, &d_v, false);


    //-----------------------------------------------------------------------------
    // dz => dx
    let new_delta = af::real(&af::matmul(&d_z, &weight0, MatProp::NONE, MatProp::TRANS));

    //------------------------------------------------------------------------------
    // dL => dh_{0}
    if t == 1 {
      delta7 = af::add(&delta7, &af::sum(&new_d_h2, 0), false);
      ltex.deltas[7] = to_real(delta7.clone());
    }
    ltex.current_unroll -= 1;
    new_delta
  }
}

