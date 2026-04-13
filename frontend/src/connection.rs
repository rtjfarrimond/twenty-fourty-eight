use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{Event, KeyboardEvent, MessageEvent, WebSocket};

use crate::protocol::{ClientMessage, ModelInfo, ServerMessage};
use crate::render;

struct AppState {
    models: Vec<ModelInfo>,
    current_model: Option<String>,
}

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
    let state = Rc::new(RefCell::new(AppState {
        models: Vec::new(),
        current_model: None,
    }));

    // On message
    let doc_for_msg = document.clone();
    let state_for_msg = state.clone();
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data = event.data().as_string().unwrap();
        handle_server_message(&doc_for_msg, &state_for_msg, &data);
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
            ws_for_keys.send_with_str(&ClientMessage::make_move(direction)).unwrap();
        }
    }) as Box<dyn FnMut(KeyboardEvent)>);
    document
        .add_event_listener_with_callback("keydown", on_keydown.as_ref().unchecked_ref())
        .unwrap();
    on_keydown.forget();

    // Play button → switch to playing mode, resuming existing game if any
    let ws_for_play = websocket.clone();
    let doc_for_play = document.clone();
    if let Some(button) = document.get_element_by_id("play-btn") {
        let on_click = Closure::wrap(Box::new(move |_: JsValue| {
            ws_for_play.send_with_str(&ClientMessage::resume_game()).unwrap();
            render::set_mode(&doc_for_play, false);
        }) as Box<dyn FnMut(JsValue)>);
        button
            .add_event_listener_with_callback("click", on_click.as_ref().unchecked_ref())
            .unwrap();
        on_click.forget();
    }

    // Watch button → switch back to watching the last model
    let ws_for_watch = websocket.clone();
    let doc_for_watch = document.clone();
    let state_for_watch = state.clone();
    if let Some(button) = document.get_element_by_id("watch-btn") {
        let on_click = Closure::wrap(Box::new(move |_: JsValue| {
            let app_state = state_for_watch.borrow();
            if let Some(model_name) = &app_state.current_model {
                ws_for_watch
                    .send_with_str(&ClientMessage::watch_agent(model_name))
                    .unwrap();
                if let Some(model) = app_state.models.iter().find(|m| m.name == *model_name) {
                    render::render_model_description(&doc_for_watch, &model.description);
                }
            }
            render::set_mode(&doc_for_watch, true);
        }) as Box<dyn FnMut(JsValue)>);
        button
            .add_event_listener_with_callback("click", on_click.as_ref().unchecked_ref())
            .unwrap();
        on_click.forget();
    }

    // Model selector dropdown
    let ws_for_select = websocket.clone();
    let doc_for_select = document.clone();
    let state_for_select = state.clone();
    if let Some(select) = document.get_element_by_id("model-select") {
        let on_change = Closure::wrap(Box::new(move |_: Event| {
            let select_el = doc_for_select
                .get_element_by_id("model-select")
                .unwrap();
            let value = select_el
                .dyn_ref::<web_sys::HtmlSelectElement>()
                .unwrap()
                .value();

            ws_for_select
                .send_with_str(&ClientMessage::watch_agent(&value))
                .unwrap();

            let mut app_state = state_for_select.borrow_mut();
            app_state.current_model = Some(value.clone());
            if let Some(model) = app_state.models.iter().find(|m| m.name == value) {
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
    state: &Rc<RefCell<AppState>>,
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
        Ok(ServerMessage::ModelList { models }) => {
            render::render_model_list(document, &models);

            let mut app_state = state.borrow_mut();
            app_state.models = models.clone();

            // Set default model and show its description
            if let Some(first) = models.first() {
                app_state.current_model = Some(first.name.clone());
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
