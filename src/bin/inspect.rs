use std::collections::HashMap;
use std::fs::File;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: inspect <file.safetensors>")?;

    let file = File::open(&path).with_context(|| format!("opening {path}"))?;
    // SAFETY: we only read the mapping, and nothing modifies the file while we run.
    let mmap = unsafe { Mmap::map(&file)? };
    let bytes: &[u8] = &mmap;

    if bytes.len() < 8 {
        bail!("file is {} bytes, too small for a header", bytes.len());
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
    let data_start = 8 + header_len;
    if data_start > bytes.len() {
        bail!("header length {header_len} runs past end of file");
    }

    let mut raw: HashMap<String, serde_json::Value> =
    serde_json::from_slice(&bytes[8..data_start]).context("parsing JSON header")?;
raw.remove("__metadata__");

    let mut tensors: Vec<(String, TensorInfo)> = Vec::with_capacity(raw.len());
    for (name, value) in raw {
        let info: TensorInfo =
            serde_json::from_value(value).with_context(|| format!("bad entry {name}"))?;
        tensors.push((name, info));
    }
    tensors.sort_by_key(|(_, info)| info.data_offsets[0]);

    let data_len = bytes.len() - data_start;
    let mut expected_start = 0;
    let mut total_params = 0;

    for (name, info) in &tensors {
        let [start, end] = info.data_offsets;
        let numel: usize = info.shape.iter().product();
        let nbytes = numel * dtype_size(&info.dtype)?;

        if start != expected_start {
            bail!("{name}: starts at {start}, expected {expected_start} (gap or overlap)");
        }
        if end - start != nbytes {
            bail!("{name}: {} bytes on disk, shape {:?} needs {nbytes}", end - start, info.shape);
        }
        expected_start = end;
        total_params += numel;

        println!("{name:<55} {:<5} {:?}", info.dtype, info.shape);
    }

    if expected_start != data_len {
        bail!("tensors cover {expected_start} bytes, data section is {data_len}");
    }
    println!(
        "\n{} tensors, {:.1}M params, header {} bytes, data {:.1} MB",
        tensors.len(),
        total_params as f64 / 1e6,
        header_len,
        data_len as f64 / 1e6
    );
    Ok(())
}

fn dtype_size(dtype: &str) -> Result<usize> {
    match dtype {
        "BF16" | "F16" => Ok(2),
        "F32" => Ok(4),
        other => bail!("unsupported dtype {other}"),
    }
}