use crate::branding;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::sel;
use objc2_app_kit::{
    NSBackingStoreType, NSButton, NSButtonType, NSColor, NSFont, NSImageScaling, NSImageView,
    NSLineBreakMode, NSTextAlignment, NSTextField, NSWindow, NSWindowStyleMask,
};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_foundation::{MainThreadMarker, NSString};
use std::rc::Rc;

const WINDOW_WIDTH: f64 = 560.0;
const WINDOW_HEIGHT: f64 = 310.0;

pub struct PermissionWindow {
    window: Retained<NSWindow>,
}

impl PermissionWindow {
    pub fn new(mtm: MainThreadMarker, handler: *const AnyObject) -> Rc<Self> {
        let style = NSWindowStyleMask::Titled;
        let frame = CGRect::new(
            CGPoint::new(0.0, 0.0),
            CGSize::new(WINDOW_WIDTH, WINDOW_HEIGHT),
        );

        let window = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                mtm.alloc::<NSWindow>(),
                frame,
                style,
                NSBackingStoreType::Buffered,
                false,
            )
        };
        window.setTitle(&NSString::from_str("Allow Accessibility for Screamer"));
        window.center();
        window.setMinSize(CGSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));
        window.setMovableByWindowBackground(false);
        window.setBackgroundColor(Some(&NSColor::windowBackgroundColor()));
        window.setHidesOnDeactivate(false);
        unsafe {
            window.setReleasedWhenClosed(false);
        }

        let content = window
            .contentView()
            .expect("permission window should have content view");

        if let Some(logo) = branding::load_logo(mtm) {
            let logo_view = NSImageView::imageViewWithImage(&logo, mtm);
            logo_view.setFrame(CGRect::new(
                CGPoint::new((WINDOW_WIDTH - 64.0) / 2.0, WINDOW_HEIGHT - 92.0),
                CGSize::new(64.0, 64.0),
            ));
            logo_view.setImageScaling(NSImageScaling::ScaleProportionallyUpOrDown);
            content.addSubview(&logo_view);
        }

        let title = make_label(
            mtm,
            "Let Screamer work across your Mac",
            CGRect::new(CGPoint::new(28.0, 178.0), CGSize::new(504.0, 28.0)),
            23.0,
            true,
        );
        title.setAlignment(NSTextAlignment::Center);
        content.addSubview(&title);

        let reason = make_wrapped_label(
            mtm,
            "Accessibility lets the global hotkey work and lets Screamer paste text back where you are.",
            CGRect::new(CGPoint::new(36.0, 126.0), CGSize::new(488.0, 36.0)),
            14.0,
        );
        reason.setAlignment(NSTextAlignment::Center);
        content.addSubview(&reason);

        let steps = make_wrapped_label(
            mtm,
            "Open System Settings, enable Screamer in Accessibility, then come back here.",
            CGRect::new(CGPoint::new(56.0, 88.0), CGSize::new(448.0, 28.0)),
            12.0,
        );
        steps.setAlignment(NSTextAlignment::Center);
        content.addSubview(&steps);

        let button = unsafe {
            NSButton::buttonWithTitle_target_action(
                &NSString::from_str("Open Accessibility"),
                Some(&*handler),
                Some(sel!(openAccessibilitySettings:)),
                mtm,
            )
        };
        button.setFrame(CGRect::new(
            CGPoint::new(108.0, 34.0),
            CGSize::new(164.0, 34.0),
        ));
        button.setButtonType(NSButtonType::MomentaryPushIn);
        content.addSubview(&button);

        let dismiss = unsafe {
            NSButton::buttonWithTitle_target_action(
                &NSString::from_str("Not now"),
                Some(&*handler),
                Some(sel!(dismissAccessibilityHelper:)),
                mtm,
            )
        };
        dismiss.setFrame(CGRect::new(
            CGPoint::new(288.0, 34.0),
            CGSize::new(164.0, 34.0),
        ));
        dismiss.setButtonType(NSButtonType::MomentaryPushIn);
        content.addSubview(&dismiss);

        Rc::new(Self { window })
    }

    pub fn show(&self) {
        self.window.makeKeyAndOrderFront(None);
        self.window.orderFrontRegardless();
    }

    pub fn hide(&self) {
        self.window.orderOut(None);
    }
}

fn make_label(
    mtm: MainThreadMarker,
    text: &str,
    frame: CGRect,
    size: f64,
    bold: bool,
) -> Retained<NSTextField> {
    let label = NSTextField::labelWithString(&NSString::from_str(text), mtm);
    label.setFrame(frame);
    label.setEditable(false);
    label.setSelectable(false);
    label.setBordered(false);
    label.setDrawsBackground(false);
    label.setAlignment(NSTextAlignment::Left);
    let font = if bold {
        NSFont::boldSystemFontOfSize(size)
    } else {
        NSFont::systemFontOfSize(size)
    };
    label.setFont(Some(&font));
    label
}

fn make_wrapped_label(
    mtm: MainThreadMarker,
    text: &str,
    frame: CGRect,
    size: f64,
) -> Retained<NSTextField> {
    let label = make_label(mtm, text, frame, size, false);
    label.setMaximumNumberOfLines(0);
    label.setLineBreakMode(NSLineBreakMode::ByWordWrapping);
    label
}
