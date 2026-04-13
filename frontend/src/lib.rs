mod protocol;
mod render;
mod connection;

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    connection::connect();
}
