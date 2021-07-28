// #![no_main]

use wasm_graph::*;
use std::fs::File;

fn main(){
    unsafe {
        __wasm_call_ctors();
    }

    // let input_filename = "./cat.png";

    // println!("Reading from {}", input_filename);

    // let f = File::open(input_filename).unwrap();

    // let tensor = unsafe {utils::load_input(0, 0)};
    let rv = run(0, 0);
    println!("DEBUG: RV: {:?}", rv);

    // println!("DEBUG: TENSOR: {:?}", tensor);
}