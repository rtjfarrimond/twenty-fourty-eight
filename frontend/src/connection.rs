use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{Event, KeyboardEvent, MessageEvent, WebSocket};

use crate::protocol::{ClientMessage, ModelInfo, ServerMessage};
use crate::render;

pub fn connect() {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let location = window.location();

    let host = location.host().unwrap();
    let protocol = if location.protocol().unwrap() == "https:" {
        "wss:"
    } else {
        "ws:"
    };
    let ws_url = format!("{protocol}//{host}/ws");

    let websocket = WebSocket::new(&ws_url).unwrap();

    // Shared state: available models and which one we're watching
    let models: Rc<RefCell<Vec<ModelInfo>>> = Rc::new(RefCell::new(Vec::new()));

    // On message: handle all server messages
    let doc_clone = document.clone();
    let ws_for_msg = websocket.clone();
    let models_for_msg = models.clone();
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data = event.data().as_string().unwrap();
        handle_server_message(&doc_clone, &ws_for_msg, &models_for_msg, &data);
    }) as Box<dyn FnMut(MessageEvent)>);
    websocket.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();

    // Keyboard input
    let ws_for_keys = websocket.clone();
    let on_keydown = Closure::wrap(Box::new(move |event: KeyboardEvent| {
        let direction = match event.key().as_str() {
            "ArrowUp" => Some("up"),
            "ArrowDown" => Some("down"),
            "ArrowLeft" => Some("left"),
            "ArrowRight" => Some("right"),
            _ => None,
        };

        if let Some(direction) = direction {
            event.prevent_default();
            let message = ClientMessage::make_move(direction);
            ws_for_keys.send_with_str(&message).unwrap();
        }
    }) as Box<dyn FnMut(KeyboardEvent)>);
    document
        .add_event_listener_with_callback("keydown", on_keydown.as_ref().unchecked_ref())
        .unwrap();
    on_keydown.forget();

    // Play button
    let ws_for_play = websocket.clone();
    let doc_for_play = document.clone();
    if let Some(button) = document.get_element_by_id("new-game-btn") {
        let on_click = Closure::wrap(Box::new(move |_: JsValue| {
            let message = ClientMessage::new_game();
            ws_for_play.send_with_str(&message).unwrap();
            render::render_play_hint(&doc_for_play, true);
            render::render_model_description(&doc_for_play, "");
        }) as Box<dyn FnMut(JsValue)>);
        button
            .add_event_listener_with_callback("click", on_click.as_ref().unchecked_ref())
            .unwrap();
        on_click.forget();
    }

    // Model selector dropdown
    let ws_for_select = websocket.clone();
    let doc_for_select = document.clone();
    let models_for_select = models.clone();
    if let Some(select) = document.get_element_by_id("model-select") {
        let on_change = Closure::wrap(Box::new(move |_: Event| {
            let select_el = doc_for_select
                .get_element_by_id("model-select")
                .unwrap();
            let value = select_el
                .dyn_ref::<web_sys::HtmlSelectElement>()
                .unwrap()
                .value();
            let message = ClientMessage::watch_agent(&value);
            ws_for_select.send_with_str(&message).unwrap();
            render::render_play_hint(&doc_for_select, false);

            // Show description for selected model
            let models = models_for_select.borrow();
            if let Some(model) = models.iter().find(|m| m.name == value) {
                render::render_model_description(&doc_for_select, &model.description);
            }
        }) as Box<dyn FnMut(Event)>);
        select
            .add_event_listener_with_callback("change", on_change.as_ref().unchecked_ref())
            .unwrap();
        on_change.forget();
    }
}

fn handle_server_message(
    document: &web_sys::Document,
    websocket: &WebSocket,
    models: &Rc<RefCell<Vec<ModelInfo>>>,
    data: &str,
) {
    match serde_json::from_str::<ServerMessage>(data) {
        Ok(ServerMessage::GameState {
            board,
            score,
            game_over,
            last_move,
        }) => {
            render::render_board(document, &board);
            render::render_score(document, score);
            render::render_game_over(document, game_over);
            render::render_last_move(document, &last_move);
        }
        Ok(ServerMessage::ModelList {
            models: model_list,
        }) => {
            render::render_model_list(document, &model_list);
            *models.borrow_mut() = model_list.clone();

            // Show description for the default (first) model
            if let Some(first) = model_list.first() {
                render::render_model_description(document, &first.description);
            }
        }
        Ok(ServerMessage::Error { message }) => {
            web_sys::console::error_1(&message.into());
        }
        Err(err) => {
            web_sys::console::error_1(&format!("Failed to parse message: {err}").into());
        }
    }
}
