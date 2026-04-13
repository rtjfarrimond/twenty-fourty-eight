use web_sys::Document;

const TILE_COLORS: &[(u16, &str)] = &[
    (0, "#cdc1b4"),
    (2, "#eee4da"),
    (4, "#ede0c8"),
    (8, "#f2b179"),
    (16, "#f59563"),
    (32, "#f67c5f"),
    (64, "#f65e3b"),
    (128, "#edcf72"),
    (256, "#edcc61"),
    (512, "#edc850"),
    (1024, "#edc53f"),
    (2048, "#edc22e"),
];

fn tile_color(value: u16) -> &'static str {
    TILE_COLORS
        .iter()
        .find(|(tile_value, _)| *tile_value == value)
        .map(|(_, color)| *color)
        .unwrap_or("#3c3a32")
}

fn text_color(value: u16) -> &'static str {
    if value <= 4 { "#776e65" } else { "#f9f6f2" }
}

fn font_size(value: u16) -> &'static str {
    match value {
        0..=99 => "55px",
        100..=999 => "45px",
        1000..=9999 => "35px",
        _ => "30px",
    }
}

/// Renders the 4x4 board into the DOM grid container.
pub fn render_board(document: &Document, board: &[[u16; 4]; 4]) {
    let grid = document.get_element_by_id("grid").unwrap();
    grid.set_inner_html("");

    for row in board {
        for &value in row {
            let tile = document.create_element("div").unwrap();
            tile.set_class_name("tile");

            let background = tile_color(value);
            let color = text_color(value);
            let size = font_size(value);
            let display_text = if value == 0 {
                String::new()
            } else {
                value.to_string()
            };

            let style = format!(
                "background:{background};color:{color};font-size:{size};\
                 display:flex;align-items:center;justify-content:center;\
                 border-radius:6px;font-weight:bold;"
            );

            tile.set_attribute("style", &style).unwrap();
            tile.set_text_content(Some(&display_text));
            grid.append_child(&tile).unwrap();
        }
    }
}

/// Updates the score display.
pub fn render_score(document: &Document, score: u32) {
    if let Some(element) = document.get_element_by_id("score") {
        element.set_text_content(Some(&score.to_string()));
    }
}

/// Shows or hides the game over overlay.
pub fn render_game_over(document: &Document, game_over: bool) {
    if let Some(element) = document.get_element_by_id("game-over") {
        let display = if game_over { "flex" } else { "none" };
        element
            .set_attribute("style", &format!("display:{display}"))
            .unwrap();
    }
}

/// Switches the UI between watching and playing modes.
/// Watching: show left panel (dropdown + description), Play button, hide Watch button and hint.
/// Playing: hide left panel, show Watch button and hint, hide Play button.
pub fn set_mode(document: &Document, watching: bool) {
    fn set_display(document: &Document, element_id: &str, display: &str) {
        if let Some(element) = document.get_element_by_id(element_id) {
            element
                .set_attribute("style", &format!("display:{display}"))
                .unwrap();
        }
    }

    if watching {
        set_display(document, "left-panel", "block");
        set_display(document, "play-btn", "inline-block");
        set_display(document, "watch-btn", "none");
        set_display(document, "play-hint", "none");
    } else {
        set_display(document, "left-panel", "none");
        set_display(document, "play-btn", "none");
        set_display(document, "watch-btn", "inline-block");
        set_display(document, "play-hint", "block");
    }
}

/// Renders model description in the left panel.
pub fn render_model_description(document: &Document, description: &str) {
    if let Some(element) = document.get_element_by_id("model-description") {
        element.set_text_content(Some(description));
    }
}

/// Populates the model selector dropdown.
pub fn render_model_list(
    document: &Document,
    models: &[crate::protocol::ModelInfo],
) {
    if let Some(select) = document.get_element_by_id("model-select") {
        select.set_inner_html("");
        for model in models {
            let option = document.create_element("option").unwrap();
            option.set_attribute("value", &model.name).unwrap();
            option.set_text_content(Some(&model.name));
            select.append_child(&option).unwrap();
        }
    }
}

/// Displays the last move as an arrow symbol.
pub fn render_last_move(document: &Document, last_move: &Option<String>) {
    if let Some(element) = document.get_element_by_id("last-move") {
        let arrow = match last_move.as_deref() {
            Some("up") => "\u{2191}",
            Some("down") => "\u{2193}",
            Some("left") => "\u{2190}",
            Some("right") => "\u{2192}",
            _ => "",
        };
        element.set_text_content(Some(arrow));
    }
}
