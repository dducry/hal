use af;
use std;
use csv;
use rand;
use rand::Rng;
use std::path::Path;
use std::ops::Sub;
use num::traits::Float;
use statistical::{standard_deviation, mean};
use af::{Dim4, Array, Aftype, AfBackend, Seq, AfError};
use itertools::Zip;
use rustc_serialize::Encodable;

//use error::HALError;

// allows for let a = hashmap!['key1' => value1, ...];
// http://stackoverflow.com/questions/28392008/more-concise-hashmap-initialization
#[macro_export]
macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

// Convert a vector of elements to a vector of Tensor
pub fn vec_to_array<T>(vec_values: Vec<T>, rows: usize, cols: usize) -> Tensor {
  raw_to_array(vec_values.as_ref(), rows, cols)
}

// Convert a generic vector to an Tensor
pub fn raw_to_array<T>(raw_values: &[T], rows: usize, cols: usize) -> Tensor {
  let dims = Dim4::new(&[rows as u64, cols as u64, 1, 1]);
  Tensor::new(dims, &raw_values, Aftype::F32).unwrap()
}

// convert an array into a vector of rows
pub fn array_to_rows(input: &Tensor) -> Vec<Tensor> {
  let mut rows = Vec::new();
  for r in (0..input.dims().unwrap()[0]) {
    rows.push(af::row(input, r as u64).unwrap());
  }
  rows
}

// convert a vector of rows into a single array
pub fn rows_to_array(input: Vec<&Tensor>) -> Tensor {
  // let mut arr = vec![input[0]];
  // // af_join_many supports up to 10 (9 + previous) arrays being joined at once
  // for rows in input[1..input.len()].iter().collect::<Vec<_>>().chunks(9) {
  //   arr.extend(Vec::from(rows));
  //   arr = vec![&af::join_many(0, arr).unwrap()];
  // }
  // arr[0].clone();
  if input.len() > 10 {
    panic!("cannot currently handle array merge of more than 10 items");
  }

  af::join_many(0, input).unwrap()
}

// Convert an array from one backend to the other
pub fn array_swap_backend(input: &Tensor
                          , from: af::AfBackend
                          , to: af::AfBackend
                          , from_device_id: i32
                          , to_device_id: i32) -> Tensor
{
  // swap to the old buffer
  af::set_backend(from).unwrap();
  af::set_device(from_device_id).unwrap();

  let dims = input.dims().unwrap();
  let mut buffer: Vec<f32> = vec![0.0f32; dims.elements() as usize];
  input.host(&mut buffer).unwrap();

  // swap to the new buffer
  af::set_backend(to).unwrap();
  af::set_device(to_device_id).unwrap();

  let converted = Tensor::new(dims, &buffer, Aftype::F32).unwrap();
  converted
}

// Helper to swap rows (row major order) in a generic type [non GPU]
pub fn swap_row<T>(matrix: &mut [T], row_src: usize, row_dest: usize, cols: usize){
  assert!(matrix.len() % cols == 0);
  if row_src != row_dest {
    for c in 0..cols {
      matrix.swap(cols * row_src + c, cols * row_dest + c);
    }
  }
}

// Helper to swap rows (col major order) in a generic type [non GPU]
pub fn swap_col<T>(matrix: &mut [T], row_src: usize, row_dest: usize, cols: usize){
  assert!(matrix.len() % cols == 0);
  let row_count = matrix.len() / cols;
  if row_src != row_dest {
    for c in 0..cols {
      matrix.swap(c * row_count + row_src, c * row_count + row_dest);
    }
  }
}

// Randomly shuffle a set of 2d matrices [or vectors] using knuth shuffle
pub fn shuffle_matrix<T>(v: &mut[&mut [T]], cols: &[usize], row_major: bool) {
  assert!(v.len() > 0 && cols.len() > 0);

  let total_length = v[0].len();
  assert!(total_length % cols[0] == 0);
  let row_count = total_length / cols[0];

  let mut rng = rand::thread_rng();
  for row in (0..row_count) {
    let rnd_row = rng.gen_range(0, row_count - row);
    for (mat, col) in Zip::new((v.iter_mut(), cols.iter())) { //swap all matrices similarly
      assert!(mat.len() % col == 0);
      match row_major{
        true  => swap_row(mat, rnd_row, row_count - row - 1, col.clone()),
        false => swap_col(mat, rnd_row, row_count - row - 1, col.clone()),
      };
    }
  }
}

// Randomly shuffle planes of an array
// SLOOOOOOW
pub fn shuffle_array(v: &mut[&mut Tensor], rows: u64) {
  let mut rng = rand::thread_rng();
  for row in (0..rows) {
    let rnd_row = rng.gen_range(0, rows - row);
    for mat in v.iter_mut() { //swap all tensors similarly
      let dims = mat.get().dims().unwrap();
      let rnd_plane  = row_plane(mat, rnd_row).unwrap();
      let orig_plane = row_plane(mat, dims[0] - row - 1).unwrap();
      **mat = set_row_plane(mat, &rnd_plane, dims[0] - row - 1).unwrap();
      **mat = set_row_plane(mat, &orig_plane, rnd_row).unwrap();
    }
  }
}

pub fn row_plane(input: &Tensor, slice_num: u64) -> Result<Tensor, AfError> {
  Tensor{ array: af::index(input, &[Seq::new(slice_num as f64, slice_num as f64, 1.0)
                                    , Seq::default()
                                    , Seq::default()]).unwrap()
          , device: input.device
          , manager: input.manager.clone() }
}

pub fn set_row_plane(input: &Tensor, new_plane: &Tensor, plane_num: u64) -> Result<Tensor, AfError> {
  match input.dims().unwrap().ndims() {
    4 => Tensor{ array: af::assign_seq(input, &[Seq::new(plane_num as f64, plane_num as f64, 1.0)
                                                , Seq::default()
                                                , Seq::default()
                                                , Seq::default()]
                                       , new_plane).unwrap(),
                 , device: input.device
                 , manager: input.manager.clone() },
    3 => Tensor { array: af::assign_seq(input, &[Seq::new(plane_num as f64, plane_num as f64, 1.0)
                                                 , Seq::default()
                                                 , Seq::default()]
                                        , new_plane).unwrap()
                  , device: input.device
                  , manager: input.manager.clone() },
    2 => Tensor { array: af::assign_seq(input, &[Seq::new(plane_num as f64, plane_num as f64, 1.0)
                                                 , Seq::default()]
                                        , new_plane).unwrap()
                  , device: input.device
                  , manager: input.manager.clone() },
    1 => Tensor { array: af::assign_seq(input, &[Seq::new(plane_num as f64, plane_num as f64, 1.0)]
                                        , new_plane).unwrap()
                  , device: input.device
                  , manager: input.manager.clone() },
    _ => panic!("unknown dimensions provided to set_row_planes"),
  }
}

pub fn row_planes(input: &Tensor, first: u64, last: u64) -> Result<Tensor, AfError> {
  Tensor { array : af::index(input, &[Seq::new(first as f64, last as f64, 1.0)
                                      , Seq::default()
                                      , Seq::default()]).unwrap()
           , device: input.device
           , manager: input.manager.clone() }
}

pub fn set_row_planes(input: &Tensor, new_planes: &Tensor
                      , first: u64, last: u64) -> Result<Tensor, AfError>
{
  match input.dims().unwrap().ndims() {
    4 => Tensor{ array: af::assign_seq(input, &[Seq::new(first as f64, last as f64, 1.0)
                                                , Seq::default()
                                                , Seq::default()
                                                , Seq::default()]
                                       , new_planes).unwrap()
                 , device: input.device
                 , manager: input.manager.clone() },
    3 => Tensor{ array: af::assign_seq(input, &[Seq::new(first as f64, last as f64, 1.0)
                                                , Seq::default()
                                                , Seq::default()]
                                       , new_planes).unwrap()
                 , device: input.device
                 , manager: input.manager.clone() },
    2 => Tensor {array: af::assign_seq(input, &[Seq::new(first as f64, last as f64, 1.0)
                                                , Seq::default()]
                                       , new_planes).unwrap()
                 , device: input.device
                 , manager: input.manager.clone() },
    1 => Tensor { array: af::assign_seq(input, &[Seq::new(first as f64, last as f64, 1.0)]
                                        , new_planes).unwrap(),
                  , device: input.device
                  , manager: input.manager.clone() },
    _ => panic!("unknown dimensions provided to set_row_planes"),
  }
}

// Helper to write a vector to a csv file
pub fn write_csv<T>(filename: &str, v: &Vec<T>)
  where T: Encodable
{
  let wtr = csv::Writer::from_file(Path::new(filename));
  match wtr {
    Ok(mut writer) => {
      for record in v {
        let result = writer.encode(record);
        assert!(result.is_ok());
      }
    },
    Err(e)    => panic!("error writing to csv file {} : {}", filename, e),
  };
}

// Helper to read a csv file to a vector
pub fn read_csv<T>(filename: &str) -> Vec<T>
  where T: std::str::FromStr, <T as std::str::FromStr>::Err: std::fmt::Debug
{
  let mut retval: Vec<T> = Vec::new();
  let rdr = csv::Reader::from_file(Path::new(filename));
  match rdr {
    Ok(mut reader) => {
      for row in reader.records() {
        let row = row.unwrap();
        for value in row {
          retval.push(value.parse::<T>().unwrap());
        }
      }
    },
    Err(e)     => panic!("error reader from csv file {} : {}", filename, e),
  }
  retval
}

// Generic Normalizer
pub fn normalize<T: Float + Sub>(src: &[T], num_std_dev: T) -> Vec<T> {
  let mean = mean(src);
  let std_dev = standard_deviation(src, Some(mean));
  src.iter().map(|&x| (x - mean) / (num_std_dev * std_dev)).collect()
}

// Normalize an array based on mean & num_std_dev deviations of the variance
pub fn normalize_array(src: &Tensor, num_std_dev: f32) -> Tensor {
  let mean = af::mean_all(src.get()).unwrap().0 as f32;
  let var = num_std_dev * af::var_all(src.get(), false).unwrap().0 as f32;
  if var > 0.00000001 || var < 0.00000001 {
    (src - mean) / var
    //af::div(&af::sub(src, &mean, false).unwrap(), &var, false).unwrap()
  }else{
    //af::sub(src, &mean, false).unwrap()
    src - mean
  }
}

pub fn scale(src: &Tensor, low: f32, high: f32) -> Tensor {
  let min = af::min_all(src.get()).unwrap().0 as f32;
  let max = af::max_all(src.get()).unwrap().0 as f32;

  (((high - low)*(src - min))/(max - min)) + low
  // af::add(&af::div(&af::mul(&(high - low), &af::sub(src, &min, false).unwrap(), false).unwrap()
  //                  , &(max - min), false).unwrap()
  //         , &low, false).unwrap()
}
