use wasm_bindgen::prelude::*;
use web_sys::{KeyboardEvent, MessageEvent, WebSocket};

use crate::protocol::{ClientMessage, ServerMessage};
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

    // On open: start watching the agent
    let ws_clone = websocket.clone();
    let on_open = Closure::wrap(Box::new(move |_: JsValue| {
        let message = ClientMessage::watch_agent();
        ws_clone.send_with_str(&message).unwrap();
    }) as Box<dyn FnMut(JsValue)>);
    websocket.set_onopen(Some(on_open.as_ref().unchecked_ref()));
    on_open.forget();

    // On message: update the board
    let doc_clone = document.clone();
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data = event.data().as_string().unwrap();
        handle_server_message(&doc_clone, &data);
    }) as Box<dyn FnMut(MessageEvent)>);
    websocket.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();

    // Keyboard input
    let ws_for_keys = websocket.clone();
    let ws_for_new_game = websocket.clone();
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

    // New game button (take over from agent)
    let doc_for_play = document.clone();
    if let Some(button) = document.get_element_by_id("new-game-btn") {
        let on_click = Closure::wrap(Box::new(move |_: JsValue| {
            let message = ClientMessage::new_game();
            ws_for_new_game.send_with_str(&message).unwrap();
            render::render_play_hint(&doc_for_play, true);
        }) as Box<dyn FnMut(JsValue)>);
        button
            .add_event_listener_with_callback("click", on_click.as_ref().unchecked_ref())
            .unwrap();
        on_click.forget();
    }

    // Watch agent button
    let ws_for_watch = websocket.clone();
    let doc_for_watch = document.clone();
    if let Some(button) = document.get_element_by_id("watch-agent-btn") {
        let on_click = Closure::wrap(Box::new(move |_: JsValue| {
            let message = ClientMessage::watch_agent();
            ws_for_watch.send_with_str(&message).unwrap();
            render::render_play_hint(&doc_for_watch, false);
        }) as Box<dyn FnMut(JsValue)>);
        button
            .add_event_listener_with_callback("click", on_click.as_ref().unchecked_ref())
            .unwrap();
        on_click.forget();
    }
}

fn handle_server_message(document: &web_sys::Document, data: &str) {
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
        Ok(ServerMessage::Error { message }) => {
            web_sys::console::error_1(&message.into());
        }
        Err(err) => {
            web_sys::console::error_1(&format!("Failed to parse message: {err}").into());
        }
    }
}
