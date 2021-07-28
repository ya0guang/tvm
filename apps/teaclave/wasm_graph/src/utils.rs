/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

use super::types::*;
use image::{imageops::FilterType, GenericImageView};
use ndarray::Array;
// use serde_json;
use std::fs::File;
use std::io::BufReader;
use std::ptr;
use std::collections::HashMap;

const IMG_HEIGHT: usize = 224;
const IMG_WIDTH: usize = 224;

pub fn load_input(in_filename: String) -> Tensor {

    println!("Reading from {}", in_filename);

    let f = File::open(in_filename).unwrap();
    let reader = BufReader::new(f);
    let img = image::load(reader, image::ImageFormat::PNG).unwrap();
    println!("load image succeed");
    data_preprocess(img)
}


pub fn output_assert(out_tensor: Tensor, label_class_file: String) {
    let output = out_tensor.to_vec::<f32>();

    // Find the maximum entry in the output and its index.
    let mut argmax = -1;
    let mut max_prob = 0.;
    for i in 0..output.len() {
        if output[i] > max_prob {
            max_prob = output[i];
            argmax = i as i32;
        }
    }

    // Create a hash map of (class id, class name)
    let mut synset: HashMap<i32, String> = HashMap::new();
    let mut rdr = csv::ReaderBuilder::new().from_reader(BufReader::new(
        File::open(label_class_file.as_str()).unwrap(),
    ));

    for result in rdr.records() {
        let record = result.unwrap();
        let id: i32 = record[0].parse().unwrap();
        let cls = record[1].to_string();
        synset.insert(id, cls);
    }

    println!(
        "input image belongs to the class `{}`",
        synset
            .get(&argmax)
            .expect("cannot find the class id for argmax")
    );
}


fn data_preprocess(img: image::DynamicImage) -> Tensor {
    println!("original image dimensions: {:?}", img.dimensions());
    let img = img
        .resize_exact(IMG_HEIGHT as u32, IMG_WIDTH as u32, FilterType::Nearest)
        .to_rgb();
    println!("resized image dimensions: {:?}", img.dimensions());
    let mut pixels: Vec<f32> = vec![];
    for pixel in img.pixels() {
        let tmp = pixel.data;
        // normalize the RGB channels using mean, std of imagenet1k
        let tmp = [
            (tmp[0] as f32 - 123.0) / 58.395, // R
            (tmp[1] as f32 - 117.0) / 57.12,  // G
            (tmp[2] as f32 - 104.0) / 57.375, // B
        ];
        for e in &tmp {
            pixels.push(*e);
        }
    }

    // (H,W,C) -> (C,H,W)
    let arr = Array::from_shape_vec((IMG_HEIGHT, IMG_WIDTH, 3), pixels).unwrap();
    let arr = arr.permuted_axes([2, 0, 1]);
    let arr = Array::from_iter(arr.into_iter().copied().map(|v| v));

    Tensor::from(arr)
}
