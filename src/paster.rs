use core_graphics::event::{CGEvent, CGEventFlags, CGKeyCode};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use std::ffi::c_void;
use std::thread;
use std::time::Duration;

const VK_ANSI_V: CGKeyCode = 9;

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXIsProcessTrusted() -> bool;
}

pub fn paste(text: &str) -> Result<(), String> {
    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| format!("Failed to open clipboard: {e}"))?;
    clipboard
        .set_text(text)
        .map_err(|e| format!("Failed to set clipboard: {e}"))?;

    if !has_accessibility_permission() {
        return Err(
            "Copied to clipboard, but automatic paste is blocked because Accessibility access is not enabled for Screamer.".to_string(),
        );
    }

    // Minimal delay for clipboard sync
    thread::sleep(Duration::from_millis(5));

    // Simulate Cmd+V
    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|_| "Failed to create CoreGraphics event source".to_string())?;

    let key_down = CGEvent::new_keyboard_event(source.clone(), VK_ANSI_V, true)
        .map_err(|_| "Failed to create Cmd+V key-down event".to_string())?;
    key_down.set_flags(CGEventFlags::CGEventFlagCommand);

    let key_up = CGEvent::new_keyboard_event(source, VK_ANSI_V, false)
        .map_err(|_| "Failed to create Cmd+V key-up event".to_string())?;
    key_up.set_flags(CGEventFlags::CGEventFlagCommand);

    key_down.post(core_graphics::event::CGEventTapLocation::HID);
    key_up.post(core_graphics::event::CGEventTapLocation::HID);

    Ok(())
}

fn has_accessibility_permission() -> bool {
    let _ = std::ptr::null::<c_void>();
    unsafe { AXIsProcessTrusted() }
}
