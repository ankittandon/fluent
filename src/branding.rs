use objc2::rc::Retained;
use objc2_app_kit::NSImage;
use objc2_foundation::{MainThreadMarker, NSString};
use std::path::PathBuf;

pub fn load_logo(mtm: MainThreadMarker) -> Option<Retained<NSImage>> {
    let path = find_logo_path()?;
    let path = path.to_str()?;
    NSImage::initWithContentsOfFile(mtm.alloc::<NSImage>(), &NSString::from_str(path))
}

fn find_logo_path() -> Option<PathBuf> {
    let bundled_base = std::env::current_exe().ok().and_then(|exe| {
        exe.parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("Resources"))
    });

    if let Some(base) = bundled_base {
        for name in ["logo.png", "image.png"] {
            let path = base.join(name);
            if path.exists() {
                return Some(path);
            }
        }
    }

    for name in ["resources/logo.png", "resources/image.png"] {
        let local = PathBuf::from(name);
        if local.exists() {
            return Some(local);
        }
    }

    None
}
