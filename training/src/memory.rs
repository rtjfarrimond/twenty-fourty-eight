//! Platform-specific memory hints for large allocations.

/// Hints the kernel that a large allocation should be backed by 2 MiB
/// transparent huge pages. A 268 MiB weight table covers ~65k 4 KiB pages —
/// enough to blow past the ~2k-entry L2 dTLB. 2 MiB pages cut that to ~134,
/// which fits comfortably in TLB. Best-effort — if the kernel can't allocate
/// huge pages, madvise returns success and we just stay on 4 KiB.
#[cfg(target_os = "linux")]
pub(crate) fn hint_huge_pages(ptr: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    unsafe {
        libc::madvise(
            ptr as *mut libc::c_void,
            len,
            libc::MADV_HUGEPAGE,
        );
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn hint_huge_pages(_ptr: *const u8, _len: usize) {}
