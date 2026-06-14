use crate::ffi::*;
use anyhow::{bail, Result};
use std::ffi::CString;
use std::sync::mpsc;

pub struct GenieContext {
    dialog_handle: GenieDialog_Handle_t,
    config_handle: GenieDialogConfig_Handle_t,
}

unsafe impl Send for GenieContext {}
unsafe impl Sync for GenieContext {}

impl GenieContext {
    pub fn new(config_json: &str) -> Result<Self> {
        let config_cstr = CString::new(config_json)?;
        let mut config_handle: GenieDialogConfig_Handle_t = std::ptr::null();

        let status = unsafe { GenieDialogConfig_createFromJson(config_cstr.as_ptr(), &mut config_handle) };
        if status != 0 {
            bail!("GenieDialogConfig_createFromJson failed: status={}", status);
        }

        let mut dialog_handle: GenieDialog_Handle_t = std::ptr::null();
        let status = unsafe { GenieDialog_create(config_handle, &mut dialog_handle) };
        if status != 0 {
            unsafe { GenieDialogConfig_free(config_handle) };
            bail!("GenieDialog_create failed: status={}", status);
        }

        Ok(Self { dialog_handle, config_handle })
    }

    pub fn run_query(&self, prompt: &str, tx: mpsc::Sender<String>) -> Result<()> {
        let prompt_cstr = CString::new(prompt)?;

        extern "C" fn query_callback(
            response: *const std::os::raw::c_char,
            _sentence_code: GenieDialog_SentenceCode_t,
            user_data: *const std::os::raw::c_void,
        ) {
            let tx = unsafe { &*(user_data as *const mpsc::Sender<String>) };
            if response.is_null() {
                return;
            }
            let s = unsafe { std::ffi::CStr::from_ptr(response) }
                .to_str()
                .unwrap_or("")
                .to_string();
            let _ = tx.send(s);
        }

        let cb: GenieDialog_QueryCallback_t = Some(query_callback);
        let user_data = &tx as *const mpsc::Sender<String> as *const std::os::raw::c_void;

        let status = unsafe {
            GenieDialog_query(
                self.dialog_handle,
                prompt_cstr.as_ptr(),
                GenieDialog_SentenceCode_t_GENIE_DIALOG_SENTENCE_COMPLETE,
                cb,
                user_data,
            )
        };

        if status != 0 && status != 1 && status != 4 {
            bail!("GenieDialog_query failed: status={}", status);
        }
        Ok(())
    }

    pub fn stop(&self) -> Result<()> {
        let status = unsafe {
            GenieDialog_signal(self.dialog_handle, GenieDialog_Action_t_GENIE_DIALOG_ACTION_ABORT)
        };
        if status != 0 {
            bail!("GenieDialog_signal failed: status={}", status);
        }
        Ok(())
    }

    pub fn reset(&self) -> Result<()> {
        let status = unsafe { GenieDialog_reset(self.dialog_handle) };
        if status != 0 {
            bail!("GenieDialog_reset failed: status={}", status);
        }
        Ok(())
    }

    pub fn token_length(&self, text: &str) -> Result<u32> {
        let text_cstr = CString::new(text)?;
        let mut tokenizer_handle: GenieTokenizer_Handle_t = std::ptr::null();

        let status = unsafe { GenieDialog_getTokenizer(self.dialog_handle, &mut tokenizer_handle) };
        if status != 0 {
            return Ok(0);
        }

        extern "C" fn alloc_callback(size: usize, allocated_data: *mut *const std::os::raw::c_char) {
            let ptr = unsafe { libc::malloc(size) };
            unsafe { *allocated_data = ptr as *const std::os::raw::c_char };
        }

        let mut token_ids: *const i32 = std::ptr::null();
        let mut num_ids: u32 = 0;

        let status = unsafe {
            GenieTokenizer_encode(
                tokenizer_handle,
                text_cstr.as_ptr(),
                Some(alloc_callback),
                &mut token_ids,
                &mut num_ids,
            )
        };

        if !token_ids.is_null() {
            unsafe { libc::free(token_ids as *mut libc::c_void) };
        }

        if status != 0 {
            return Ok(0);
        }

        Ok(num_ids)
    }
}

impl Drop for GenieContext {
    fn drop(&mut self) {
        if !self.dialog_handle.is_null() {
            unsafe { GenieDialog_free(self.dialog_handle) };
        }
        if !self.config_handle.is_null() {
            unsafe { GenieDialogConfig_free(self.config_handle) };
        }
    }
}
