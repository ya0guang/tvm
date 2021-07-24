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

use anyhow::Result;
use wasmtime::*;

use super::Tensor;

pub struct GraphExecutor {
    pub(crate) wasm_addr: i32,
    pub(crate) input_size: i32,
    pub(crate) output_size: i32,
    // pub(crate) store: Option<Store<WasiCtx>>,
    pub(crate) store: Option<Store<()>>,
    pub(crate) instance: Option<Instance>,
}

#[allow(dead_code)]
impl GraphExecutor {
    pub fn new() -> Self {
        Self {
            wasm_addr: 0,
            input_size: 0,
            output_size: 0,
            store: None,
            instance: None,
        }
    }

    pub fn instantiate(&mut self, wasm_graph_file: String) -> Result<()> {
        // ---------------------------------------------- wasi
        // //  let engine = Engine::new(Config::new().wasm_simd(true)).unwrap();
        //  let engine = Engine::default();
        //  // let store = Store::new(&engine);

        //  // First set up our linker which is going to be linking modules together. We
        //  // want our linker to have wasi available, so we set that up here as well.
        //  // let mut linker = Linker::new(&store);
        //  let mut linker = Linker::new(&engine);
        //  wasmtime_wasi::add_to_linker(&mut linker, |s| s)?;
        //  // Create an instance of `Wasi` which contains a `WasiCtx`. Note that
        //  // `WasiCtx` provides a number of ways to configure what the target program
        //  // will have access to.
        //  let wasi = WasiCtxBuilder::new()
        //      .inherit_stdio()
        //      .inherit_args()?
        //      .build();
        //  // let wasi = Wasi::new(&store, WasiCtx::new(std::env::args())?);
        //  let mut store = Store::new(&engine, wasi);
        //  // wasi.add_to_linker(&mut linker)?;

        //  let module = Module::from_file(&engine, &wasm_graph_file)?;
        //  self.instance = Some(linker.instantiate(&mut store, &module)?);
        //  self.store = Some(store);

        // -------------------------------------------- /wasi

        // without wasi

        //  let engine = Engine::new(Config::new().wasm_simd(true)).unwrap();
        let mut store = Store::<()>::default();
        let module = Module::from_file(
            store.engine(),
            &wasm_graph_file,
        )?;

        let instance = Instance::new(&mut store, &module, &[])?;
    
        // First set up our linker which is going to be linking modules together. We
        // want our linker to have wasi available, so we set that up here as well.
        // let mut linker = Linker::new(&store);
        // let mut linker = Linker::new(&engine);
        // wasmtime_wasi::add_to_linker(&mut linker, |s| s)?;
        // Create an instance of `Wasi` which contains a `WasiCtx`. Note that
        // `WasiCtx` provides a number of ways to configure what the target program
        // will have access to.
        // let wasi = WasiCtxBuilder::new()
        //     .inherit_stdio()
        //     .inherit_args()?
        //     .build();
        // let wasi = Wasi::new(&store, WasiCtx::new(std::env::args())?);
        // let mut store = Store::new(&engine, wasi);
        // wasi.add_to_linker(&mut linker)?;

        self.instance = Some(instance);
        self.store = Some(store);

        Ok(())
    }

    pub fn set_input(&mut self, input_data: Tensor) -> Result<()> {
        let memory = self
            .instance
            .as_ref()
            .unwrap()
            .get_memory(self.store.as_mut().unwrap(), "memory")
            .ok_or_else(|| anyhow::format_err!("failed to find `memory` export"))?;

        // Specify the wasm address to access the wasm memory.
        let wasm_addr = memory.data_size(self.store.as_mut().unwrap());
        println!("DEBUG: data_size before grow: {:?}", wasm_addr);
        // Serialize the data into a JSON string.
        let in_data = serde_json::to_vec(&input_data)?;
        let in_size = in_data.len();
        // Grow up memory size according to in_size to avoid memory leak.
        memory.grow(self.store.as_mut().unwrap(), (in_size >> 16) as u32 + 1)?;
        println!(
            "DEBUG: data_size after grow: {:?}",
            memory.data_size(self.store.as_mut().unwrap())
        );

        // Insert the input data into wasm memory.
        // for i in 0..in_size {
        //     unsafe {
        //         memory.data_unchecked_mut()[wasm_addr + i] = *in_data.get(i).unwrap();
        //     }
        // }
        memory.write(self.store.as_mut().unwrap(), wasm_addr, &in_data)?;

        self.wasm_addr = wasm_addr as i32;
        self.input_size = in_size as i32;
        println!(
            "DEBUG, input addr: {:?}, size: {:?}",
            self.wasm_addr, self.input_size
        );

        let mem_ptr = memory.data_ptr(self.store.as_mut().unwrap());
        let bug_region = unsafe {
            std::slice::from_raw_parts(((mem_ptr as usize) + wasm_addr + 95000) as *mut u8, 10)
        };
        println!("DEBUG: after set_input {:?}", bug_region);

        Ok(())
    }

    pub fn run(&mut self) -> Result<()> {
        // Invoke `run` export.
        let run = self
            .instance
            .as_ref()
            .unwrap()
            .get_func(self.store.as_mut().unwrap(), "run")
            .ok_or_else(|| anyhow::format_err!("failed to find `run` function export!"))?;
        // .wrap2_async::<i32, i32, i32>(self.store.unwrap())?;
        // .get2::<i32, i32, i32>()?;

        // DEBUG
        let memory = self
            .instance
            .as_ref()
            .unwrap()
            .get_memory(self.store.as_mut().unwrap(), "memory")
            .ok_or_else(|| anyhow::format_err!("failed to find `memory` export"))?;
        let mem_ptr = memory.data_ptr(self.store.as_mut().unwrap());
        let bug_region = unsafe {
            std::slice::from_raw_parts(
                ((mem_ptr as usize) + (self.wasm_addr as usize) + 95000) as *mut u8,
                10,
            )
        };
        println!("DEBUG: in run {:?}", bug_region);

        let params = [Val::I32(self.wasm_addr), Val::I32(self.input_size)];
        let out_size = run.call(self.store.as_mut().unwrap(), &params[..])?;
        let out_size = (*out_size)[0].unwrap_i32();
        if out_size == 0 {
            panic!("graph run failed!");
        }

        self.output_size = out_size;
        Ok(())
    }

    pub fn get_output(&mut self) -> Result<Tensor> {
        let memory = self
            .instance
            .as_ref()
            .unwrap()
            .get_memory(self.store.as_mut().unwrap(), "memory")
            .ok_or_else(|| anyhow::format_err!("failed to find `memory` export"))?;

        let mut out_data = vec![0 as u8; self.output_size as _];
        memory.read(
            self.store.as_mut().unwrap(),
            self.wasm_addr as _,
            &mut out_data,
        )?;

        // let out_data = unsafe {
        //     &memory.data_unchecked()[self.wasm_addr as usize..][..self.output_size as usize]
        // };

        let out_vec: Tensor = serde_json::from_slice(&out_data).unwrap();
        Ok(out_vec)
    }
}

impl Default for GraphExecutor {
    fn default() -> Self {
        Self::new()
    }
}
