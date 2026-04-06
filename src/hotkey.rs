use crate::config::{Config, HOTKEYS};
use objc2::msg_send;
use objc2_foundation::MainThreadMarker;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

pub struct Hotkey {
    is_pressed: Arc<AtomicBool>,
    selected_index: Arc<AtomicUsize>,
}

impl Hotkey {
    pub fn new(config: &Config) -> Self {
        Self {
            is_pressed: Arc::new(AtomicBool::new(false)),
            selected_index: Arc::new(AtomicUsize::new(hotkey_index_for_id(&config.hotkey))),
        }
    }

    pub fn set_hotkey(&self, hotkey_id: &str) {
        let index = hotkey_index_for_id(hotkey_id);
        let hotkey_info = HOTKEYS.get(index).unwrap_or(&HOTKEYS[0]);
        self.selected_index.store(index, Ordering::Relaxed);
        eprintln!(
            "[screamer] Updated hotkey to: {} (modifier=0x{:x}, device=0x{:x})",
            hotkey_info.label, hotkey_info.modifier_flag, hotkey_info.device_flag
        );
    }

    /// Start listening using NSEvent global monitor on the main thread.
    /// Reads the current hotkey selection from in-memory state so settings changes
    /// take effect without restarting the app.
    pub fn start_on_main_thread(
        &self,
        _mtm: MainThreadMarker,
        on_press: impl Fn() + 'static,
        on_release: impl Fn() + 'static,
    ) {
        let is_pressed = self.is_pressed.clone();
        let selected_index = self.selected_index.clone();

        let hotkey_info = HOTKEYS
            .get(selected_index.load(Ordering::Relaxed))
            .unwrap_or(&HOTKEYS[0]);

        eprintln!(
            "[screamer] Hotkey configured: {} (modifier=0x{:x}, device=0x{:x})",
            hotkey_info.label, hotkey_info.modifier_flag, hotkey_info.device_flag
        );

        // NSEventMaskFlagsChanged = 1 << 12 = 4096
        let mask: u64 = 1 << 12;

        let block = block2::RcBlock::new(move |event: *mut objc2::runtime::AnyObject| {
            if event.is_null() {
                return;
            }
            let flags: u64 = unsafe { msg_send![event, modifierFlags] };
            let hotkey_info = HOTKEYS
                .get(selected_index.load(Ordering::Relaxed))
                .unwrap_or(&HOTKEYS[0]);

            // Check if the modifier is down (device-independent)
            let modifier_down = (flags & hotkey_info.modifier_flag) != 0;

            // If we have a device-specific flag, also check that
            let key_down = if hotkey_info.device_flag != 0 {
                modifier_down && (flags & hotkey_info.device_flag) != 0
            } else {
                modifier_down
            };

            let was_pressed = is_pressed.load(Ordering::SeqCst);

            if key_down && !was_pressed {
                eprintln!(
                    "[screamer] Hotkey {} PRESSED (flags=0x{:x})",
                    hotkey_info.label, flags
                );
                is_pressed.store(true, Ordering::SeqCst);
                on_press();
            } else if !key_down && was_pressed {
                eprintln!(
                    "[screamer] Hotkey {} RELEASED (flags=0x{:x})",
                    hotkey_info.label, flags
                );
                is_pressed.store(false, Ordering::SeqCst);
                on_release();
            }
        });

        unsafe {
            let _monitor: *mut objc2::runtime::AnyObject = msg_send![
                objc2::class!(NSEvent),
                addGlobalMonitorForEventsMatchingMask: mask,
                handler: &*block
            ];

            if _monitor.is_null() {
                eprintln!("[screamer] Failed to create NSEvent global monitor");
            } else {
                eprintln!("[screamer] NSEvent global monitor installed for FlagsChanged");
            }
        }
    }
}

fn hotkey_index_for_id(hotkey_id: &str) -> usize {
    HOTKEYS
        .iter()
        .position(|hotkey| hotkey.id == hotkey_id)
        .unwrap_or(0)
}
