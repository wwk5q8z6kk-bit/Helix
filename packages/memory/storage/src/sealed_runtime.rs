use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock};

const ROOT_KEY_LEN: usize = 32;

fn root_key_slot() -> &'static RwLock<Option<[u8; ROOT_KEY_LEN]>> {
    static ROOT_KEY: OnceLock<RwLock<Option<[u8; ROOT_KEY_LEN]>>> = OnceLock::new();
    ROOT_KEY.get_or_init(|| RwLock::new(None))
}

fn degraded_slot() -> &'static AtomicBool {
    static DEGRADED: OnceLock<AtomicBool> = OnceLock::new();
    DEGRADED.get_or_init(|| AtomicBool::new(false))
}

fn sealed_mode_slot() -> &'static AtomicBool {
    static SEALED_MODE: OnceLock<AtomicBool> = OnceLock::new();
    SEALED_MODE.get_or_init(|| AtomicBool::new(false))
}

pub fn set_runtime_root_key(key: [u8; ROOT_KEY_LEN], degraded_security: bool) {
    if let Ok(mut guard) = root_key_slot().write() {
        *guard = Some(key);
    }
    degraded_slot().store(degraded_security, Ordering::SeqCst);
}

pub fn clear_runtime_root_key() {
    if let Ok(mut guard) = root_key_slot().write() {
        *guard = None;
    }
    degraded_slot().store(false, Ordering::SeqCst);
}

pub fn runtime_root_key() -> Option<[u8; ROOT_KEY_LEN]> {
    root_key_slot().read().ok().and_then(|guard| *guard)
}

pub fn runtime_has_key() -> bool {
    root_key_slot()
        .read()
        .map(|guard| guard.is_some())
        .unwrap_or(false)
}

pub fn runtime_is_degraded_security() -> bool {
    degraded_slot().load(Ordering::SeqCst)
}

pub fn set_sealed_mode_enabled(enabled: bool) {
    sealed_mode_slot().store(enabled, Ordering::SeqCst);
}

pub fn sealed_mode_enabled() -> bool {
    sealed_mode_slot().load(Ordering::SeqCst)
}
