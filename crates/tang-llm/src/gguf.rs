//! GGUF v3 files, memory-mapped, single-file or sharded, and dequantisation to f32.
//!
//! This is the loader for models that only ship as GGUF (Qwen3.8-Flash-Next: ISTA-DASLab's Q2_0 file
//! and unsloth's UD quants). It reads the header, the metadata key/values and the tensor directory;
//! tensor data stays in the page cache until something asks for it.
//!
//! Layout facts the rest of tang relies on:
//!
//! - **Dimension order.** `dims[0]` varies fastest, exactly as ggml's `ne[0]`. A matrix stored for
//!   `y = W x` has `dims = [in, out]`, so row `o` is `in` contiguous elements. A fused expert tensor
//!   has `dims = [in, out, n_expert]` and expert `e` is the contiguous run `e * in * out`.
//! - **Rows are block-aligned.** Every quantised type here packs whole blocks along `dims[0]`, so a
//!   row (or an expert) is a byte range and can be dequantised alone ([`Gguf::rows`],
//!   [`Gguf::expert`]).
//! - **Shards.** `name-00001-of-00003.gguf` finds its siblings. Metadata comes from the first shard;
//!   every shard contributes tensors, and a name present twice is an error.
//!
//! Dequantisers cover every type that appears in the Flash-Next files (F32, F16, BF16, Q8_0, Q4_0,
//! Q4_1, Q5_0, Q5_1, Q2_K..Q6_K, IQ4_NL, IQ4_XS and ISTA's Q2_0, ggml type 42). Each one is a
//! straight transcription of `ggml-quants.c`'s `dequantize_row_*`, checked in the tests against a
//! hand-built block.

use anyhow::{bail, ensure, Context, Result};

#[path = "gguf_grids.rs"]
mod grids;
use memmap2::Mmap;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

/// A ggml tensor type. Only the ones with a dequantiser are named; the rest keep their id so an
/// inventory can still print them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    Bf16,
    Iq1M,
    /// ISTA-DASLab's GSQ-RCO 2-bit: 64 weights in 18 bytes, `(code - 1) * d`.
    Q2_0,
    Other(u32),
}

impl GgmlType {
    pub fn from_id(id: u32) -> Self {
        use GgmlType::*;
        match id {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            16 => Iq2Xxs,
            17 => Iq2Xs,
            18 => Iq3Xxs,
            19 => Iq1S,
            20 => Iq4Nl,
            21 => Iq3S,
            22 => Iq2S,
            23 => Iq4Xs,
            24 => I8,
            25 => I16,
            26 => I32,
            29 => Iq1M,
            30 => Bf16,
            42 => Q2_0,
            other => Other(other),
        }
    }

    /// (elements per block, bytes per block), or `None` for a type whose geometry we don't know.
    pub fn geometry(self) -> Option<(usize, usize)> {
        use GgmlType::*;
        Some(match self {
            F32 | I32 => (1, 4),
            F16 | Bf16 | I16 => (1, 2),
            I8 => (1, 1),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            Q2K => (256, 84),
            Q3K => (256, 110),
            Q4K => (256, 144),
            Q5K => (256, 176),
            Q6K => (256, 210),
            Iq2Xxs => (256, 66),
            Iq2Xs => (256, 74),
            Iq3Xxs => (256, 98),
            Iq1S => (256, 50),
            Iq4Nl => (32, 18),
            Iq3S => (256, 110),
            Iq2S => (256, 82),
            Iq4Xs => (256, 136),
            Iq1M => (256, 56),
            Q2_0 => (64, 18),
            Other(_) => return None,
        })
    }

    pub fn name(self) -> String {
        use GgmlType::*;
        match self {
            F32 => "F32".into(),
            F16 => "F16".into(),
            Q4_0 => "Q4_0".into(),
            Q4_1 => "Q4_1".into(),
            Q5_0 => "Q5_0".into(),
            Q5_1 => "Q5_1".into(),
            Q8_0 => "Q8_0".into(),
            Q2K => "Q2_K".into(),
            Q3K => "Q3_K".into(),
            Q4K => "Q4_K".into(),
            Q5K => "Q5_K".into(),
            Q6K => "Q6_K".into(),
            Iq2Xxs => "IQ2_XXS".into(),
            Iq2Xs => "IQ2_XS".into(),
            Iq3Xxs => "IQ3_XXS".into(),
            Iq1S => "IQ1_S".into(),
            Iq4Nl => "IQ4_NL".into(),
            Iq3S => "IQ3_S".into(),
            Iq2S => "IQ2_S".into(),
            Iq4Xs => "IQ4_XS".into(),
            I8 => "I8".into(),
            I16 => "I16".into(),
            I32 => "I32".into(),
            Bf16 => "BF16".into(),
            Iq1M => "IQ1_M".into(),
            Q2_0 => "Q2_0".into(),
            Other(id) => format!("type{id}"),
        }
    }

    /// Whether [`dequantize`] handles this type.
    pub fn can_dequantize(self) -> bool {
        use GgmlType::*;
        matches!(
            self,
            F32 | F16
                | Bf16
                | Q4_0
                | Q4_1
                | Q5_0
                | Q5_1
                | Q8_0
                | Q2K
                | Q3K
                | Q4K
                | Q5K
                | Q6K
                | Iq4Nl
                | Iq4Xs
                | Iq2Xs
                | Iq3Xxs
                | Q2_0
        )
    }

    /// Bytes for `n` elements, which must be a whole number of blocks.
    pub fn bytes_for(self, n: usize) -> Result<usize> {
        let (el, by) = self
            .geometry()
            .with_context(|| format!("no block geometry for {}", self.name()))?;
        ensure!(
            n.is_multiple_of(el),
            "{n} elements is not a whole number of {} blocks of {el}",
            self.name()
        );
        Ok(n / el * by)
    }
}

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    /// Any integer that fits in u64 (negative values don't).
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            Value::Bool(b) => Some(b as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// A short rendering for inventories: arrays longer than 8 show their length and head.
    pub fn summary(&self) -> String {
        match self {
            Value::Str(s) if s.len() > 120 => format!(
                "{:?}… ({} bytes)",
                s.chars().take(80).collect::<String>(),
                s.len()
            ),
            Value::Str(s) => format!("{s:?}"),
            Value::Array(a) if a.len() > 8 => format!(
                "[{} items: {}, …]",
                a.len(),
                a[..4]
                    .iter()
                    .map(Value::summary)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Value::Array(a) => format!(
                "[{}]",
                a.iter().map(Value::summary).collect::<Vec<_>>().join(", ")
            ),
            Value::F32(v) => format!("{v}"),
            Value::F64(v) => format!("{v}"),
            Value::Bool(v) => format!("{v}"),
            other => format!(
                "{}",
                other
                    .as_u64()
                    .map(|v| v as i128)
                    .unwrap_or_else(|| match other {
                        Value::I8(v) => *v as i128,
                        Value::I16(v) => *v as i128,
                        Value::I32(v) => *v as i128,
                        Value::I64(v) => *v as i128,
                        _ => 0,
                    })
            ),
        }
    }
}

/// One tensor's directory entry.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// `dims[0]` varies fastest (ggml `ne`).
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Which shard holds the data.
    pub shard: usize,
    /// Absolute byte offset of the data in its shard file.
    pub offset: u64,
    /// Bytes of data (0 when the type's geometry is unknown).
    pub nbytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Elements in one row (`dims[0]`).
    pub fn row_len(&self) -> usize {
        self.dims[0] as usize
    }

    /// Number of rows (product of every dim after the first).
    pub fn n_rows(&self) -> usize {
        self.dims[1..].iter().product::<u64>() as usize
    }

    pub fn row_bytes(&self) -> Result<usize> {
        self.ty.bytes_for(self.row_len())
    }
}

struct Shard {
    path: PathBuf,
    file: File,
    map: Mmap,
}

/// An open GGUF model (all its shards).
pub struct Gguf {
    shards: Vec<Shard>,
    /// Metadata from the first shard.
    pub meta: BTreeMap<String, Value>,
    /// Every tensor of every shard, in file order.
    pub tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
}

/// `foo-00001-of-00003.gguf` -> all three paths; anything else -> itself.
pub fn discover_shards(path: &Path) -> Result<Vec<PathBuf>> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .with_context(|| format!("{}: not a file name", path.display()))?;
    let Some(stem) = name.strip_suffix(".gguf") else {
        return Ok(vec![path.to_path_buf()]);
    };
    // ...-NNNNN-of-MMMMM
    let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
    if parts.len() == 4 && parts[1] == "of" && parts[0].len() == 5 && parts[2].len() == 5 {
        if let (Ok(total), Ok(_)) = (parts[0].parse::<usize>(), parts[2].parse::<usize>()) {
            let base = parts[3];
            let dir = path.parent().unwrap_or(Path::new("."));
            let out: Vec<PathBuf> = (1..=total)
                .map(|i| dir.join(format!("{base}-{i:05}-of-{total:05}.gguf")))
                .collect();
            for p in &out {
                ensure!(
                    p.exists(),
                    "{}: shard {} is missing",
                    path.display(),
                    p.display()
                );
            }
            return Ok(out);
        }
    }
    Ok(vec![path.to_path_buf()])
}

/// A little-endian cursor over the header bytes.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("header length overflow")?;
        ensure!(
            end <= self.buf.len(),
            "truncated header at byte {}",
            self.pos
        );
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        let b = self.take(n)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<Value> {
        ensure!(depth < 4, "metadata arrays nested too deep");
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let ety = self.u32()?;
                let n = self.u64()? as usize;
                ensure!(
                    n <= self.buf.len(),
                    "array of {n} items is longer than the file"
                );
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(ety, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => bail!("unknown metadata value type {other}"),
        })
    }
}

impl Gguf {
    /// Open a GGUF file; a `-00001-of-0000N` name pulls in its siblings.
    pub fn open(path: &Path) -> Result<Self> {
        let paths = discover_shards(path)?;
        let mut shards = Vec::new();
        let mut meta = BTreeMap::new();
        let mut tensors = Vec::new();
        let mut index = HashMap::new();
        for (si, p) in paths.iter().enumerate() {
            let file = File::open(p).with_context(|| format!("opening {}", p.display()))?;
            // Safety: model files aren't modified while we run.
            let map = unsafe { Mmap::map(&file)? };
            let (kv, infos) =
                parse_header(&map, si).with_context(|| format!("reading {}", p.display()))?;
            if si == 0 {
                meta = kv;
            }
            for t in infos {
                ensure!(
                    t.offset + t.nbytes <= map.len() as u64,
                    "{}: tensor {} runs past the end of the file",
                    p.display(),
                    t.name
                );
                if index.insert(t.name.clone(), tensors.len()).is_some() {
                    bail!("tensor {} appears in more than one shard", t.name);
                }
                tensors.push(t);
            }
            shards.push(Shard {
                path: p.clone(),
                file,
                map,
            });
        }
        Ok(Self {
            shards,
            meta,
            tensors,
            index,
        })
    }

    pub fn shard_paths(&self) -> Vec<&Path> {
        self.shards.iter().map(|s| s.path.as_path()).collect()
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.get(name)
            .with_context(|| format!("missing tensor {name}"))
    }

    pub fn meta(&self, key: &str) -> Result<&Value> {
        self.meta
            .get(key)
            .with_context(|| format!("missing metadata key {key}"))
    }

    pub fn meta_u64(&self, key: &str) -> Result<u64> {
        self.meta(key)?
            .as_u64()
            .with_context(|| format!("metadata {key} is not an unsigned integer"))
    }

    pub fn meta_f64(&self, key: &str) -> Result<f64> {
        self.meta(key)?
            .as_f64()
            .with_context(|| format!("metadata {key} is not a number"))
    }

    pub fn meta_str(&self, key: &str) -> Result<&str> {
        self.meta(key)?
            .as_str()
            .with_context(|| format!("metadata {key} is not a string"))
    }

    /// An integer array (or a scalar, broadcast to `n` when `n` is given).
    pub fn meta_u64s(&self, key: &str) -> Result<Vec<u64>> {
        let v = self.meta(key)?;
        match v {
            Value::Array(a) => a
                .iter()
                .map(|x| {
                    x.as_u64()
                        .with_context(|| format!("metadata {key}: not an unsigned integer"))
                })
                .collect(),
            other => Ok(vec![other
                .as_u64()
                .with_context(|| format!("metadata {key} is not an integer"))?]),
        }
    }

    /// The raw bytes of a tensor, borrowed from the mapping.
    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        let map = &self.shards[t.shard].map;
        &map[t.offset as usize..(t.offset + t.nbytes) as usize]
    }

    /// The whole tensor as f32, `dims[0]` fastest.
    pub fn dequantize(&self, t: &TensorInfo) -> Result<Vec<f32>> {
        let mut out = vec![0f32; t.n_elements() as usize];
        dequantize(t.ty, self.bytes(t), &mut out)
            .with_context(|| format!("dequantizing {}", t.name))?;
        Ok(out)
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.dequantize(self.info(name)?)
    }

    /// Rows `start..start + n` of a tensor (a row is `dims[0]` elements), from the mapping.
    pub fn rows(&self, t: &TensorInfo, start: usize, n: usize) -> Result<Vec<f32>> {
        ensure!(
            start + n <= t.n_rows(),
            "{}: rows {start}..{} of {}",
            t.name,
            start + n,
            t.n_rows()
        );
        let rb = t.row_bytes()?;
        let all = self.bytes(t);
        let mut out = vec![0f32; n * t.row_len()];
        dequantize(t.ty, &all[start * rb..(start + n) * rb], &mut out)?;
        Ok(out)
    }

    /// One row read with `pread` instead of through the mapping, so a 26 GiB table never has to be
    /// faulted in as a whole: the n-gram table is read this way, 16 rows a token.
    pub fn read_row(&self, t: &TensorInfo, row: usize, out: &mut [f32]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        ensure!(
            row < t.n_rows(),
            "{}: row {row} out of range ({})",
            t.name,
            t.n_rows()
        );
        ensure!(
            out.len() == t.row_len(),
            "{}: row is {} wide, buffer {}",
            t.name,
            t.row_len(),
            out.len()
        );
        let rb = t.row_bytes()?;
        let mut buf = vec![0u8; rb];
        self.shards[t.shard]
            .file
            .read_exact_at(&mut buf, t.offset + (row * rb) as u64)
            .with_context(|| format!("{}: reading row {row}", t.name))?;
        dequantize(t.ty, &buf, out)
    }

    /// Expert `e` of a fused `[in, out, n_expert]` tensor, as `out` rows of `in`.
    pub fn expert(&self, t: &TensorInfo, e: usize) -> Result<Vec<f32>> {
        ensure!(
            t.dims.len() == 3,
            "{}: not a fused expert tensor ({:?})",
            t.name,
            t.dims
        );
        let per = (t.dims[0] * t.dims[1]) as usize;
        ensure!(
            e < t.dims[2] as usize,
            "{}: expert {e} out of range",
            t.name
        );
        let rows = t.dims[1] as usize;
        let v = self.rows(t, e * rows, rows)?;
        debug_assert_eq!(v.len(), per);
        Ok(v)
    }
}

fn parse_header(map: &[u8], shard: usize) -> Result<(BTreeMap<String, Value>, Vec<TensorInfo>)> {
    let mut c = Cursor { buf: map, pos: 0 };
    ensure!(c.take(4)? == MAGIC, "not a GGUF file");
    let version = c.u32()?;
    ensure!(
        version == 3 || version == 2,
        "GGUF version {version} (want 2 or 3)"
    );
    let n_tensors = c.u64()? as usize;
    let n_kv = c.u64()? as usize;
    let mut kv = BTreeMap::new();
    for _ in 0..n_kv {
        let k = c.string()?;
        let ty = c.u32()?;
        let v = c.value(ty, 0).with_context(|| format!("metadata {k}"))?;
        kv.insert(k, v);
    }
    let mut infos = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = c.string()?;
        let nd = c.u32()? as usize;
        ensure!(nd <= 4, "{name}: {nd} dims");
        let dims = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
        let ty = GgmlType::from_id(c.u32()?);
        let offset = c.u64()?;
        let n: u64 = dims.iter().product();
        let nbytes = match ty.geometry() {
            Some(_) => ty.bytes_for(n as usize).with_context(|| name.clone())? as u64,
            None => 0,
        };
        infos.push(TensorInfo {
            name,
            dims,
            ty,
            shard,
            offset,
            nbytes,
        });
    }
    let align = kv
        .get("general.alignment")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_ALIGNMENT);
    ensure!(align > 0 && align.is_power_of_two(), "alignment {align}");
    let data_start = (c.pos as u64).div_ceil(align) * align;
    for t in &mut infos {
        t.offset += data_start;
    }
    Ok((kv, infos))
}

// ---------------------------------------------------------------------------------------------
// Dequantisation
// ---------------------------------------------------------------------------------------------

/// IEEE half -> f32, exact (subnormals included).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: renormalise
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> IEEE half, round to nearest even (for tests and quantisers).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rest > half || (rest == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

#[inline]
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
fn rd16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

#[inline]
fn half(b: &[u8], at: usize) -> f32 {
    f16_to_f32(rd16(b, at))
}

/// `kvalues_iq4nl` from ggml-common.h.
pub const IQ4NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Dequantise `src` (whole blocks of `ty`) into `out`, which must be exactly the element count.
pub fn dequantize(ty: GgmlType, src: &[u8], out: &mut [f32]) -> Result<()> {
    let need = ty.bytes_for(out.len())?;
    ensure!(
        src.len() == need,
        "{}: {} bytes for {} elements (want {need})",
        ty.name(),
        src.len(),
        out.len()
    );
    use GgmlType::*;
    match ty {
        F32 => {
            for (o, b) in out.iter_mut().zip(src.as_chunks::<4>().0) {
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        F16 => {
            for (o, b) in out.iter_mut().zip(src.as_chunks::<2>().0) {
                *o = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        Bf16 => {
            for (o, b) in out.iter_mut().zip(src.as_chunks::<2>().0) {
                *o = bf16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        Q8_0 => {
            for (y, b) in out
                .as_chunks_mut::<32>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<34>().0)
            {
                let d = half(b, 0);
                for j in 0..32 {
                    y[j] = d * (b[2 + j] as i8) as f32;
                }
            }
        }
        Q4_0 => {
            for (y, b) in out
                .as_chunks_mut::<32>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<18>().0)
            {
                let d = half(b, 0);
                for j in 0..16 {
                    y[j] = ((b[2 + j] & 0xf) as i32 - 8) as f32 * d;
                    y[j + 16] = ((b[2 + j] >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        Q4_1 => {
            for (y, b) in out
                .as_chunks_mut::<32>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<20>().0)
            {
                let (d, m) = (half(b, 0), half(b, 2));
                for j in 0..16 {
                    y[j] = (b[4 + j] & 0xf) as f32 * d + m;
                    y[j + 16] = (b[4 + j] >> 4) as f32 * d + m;
                }
            }
        }
        Q5_0 => {
            for (y, b) in out
                .as_chunks_mut::<32>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<22>().0)
            {
                let d = half(b, 0);
                let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                for j in 0..16 {
                    let h0 = (((qh >> j) << 4) & 0x10) as u8;
                    let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                    y[j] = (((b[6 + j] & 0xf) | h0) as i32 - 16) as f32 * d;
                    y[j + 16] = (((b[6 + j] >> 4) | h1) as i32 - 16) as f32 * d;
                }
            }
        }
        Q5_1 => {
            for (y, b) in out
                .as_chunks_mut::<32>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<24>().0)
            {
                let (d, m) = (half(b, 0), half(b, 2));
                let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                for j in 0..16 {
                    let h0 = (((qh >> j) << 4) & 0x10) as u8;
                    let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                    y[j] = ((b[8 + j] & 0xf) | h0) as f32 * d + m;
                    y[j + 16] = ((b[8 + j] >> 4) | h1) as f32 * d + m;
                }
            }
        }
        Q2_0 => {
            for (y, b) in out
                .as_chunks_mut::<64>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<18>().0)
            {
                let d = half(b, 0);
                for j in 0..64 {
                    let q = (b[2 + j / 4] >> ((j % 4) * 2)) & 3;
                    y[j] = (q as i32 - 1) as f32 * d;
                }
            }
        }
        Iq4Nl => {
            for (y, b) in out
                .as_chunks_mut::<32>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<18>().0)
            {
                let d = half(b, 0);
                for j in 0..16 {
                    y[j] = d * IQ4NL_VALUES[(b[2 + j] & 0xf) as usize] as f32;
                    y[j + 16] = d * IQ4NL_VALUES[(b[2 + j] >> 4) as usize] as f32;
                }
            }
        }
        Iq4Xs => {
            // d f16 | scales_h u16 | scales_l[4] | qs[128]
            for (y, b) in out
                .as_chunks_mut::<256>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<136>().0)
            {
                let d = half(b, 0);
                let sh = rd16(b, 2) as u32;
                let sl = &b[4..8];
                let qs = &b[8..136];
                for ib in 0..8 {
                    let ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32
                        | ((((sh >> (2 * ib)) & 3) << 4) as i32);
                    let dl = d * (ls - 32) as f32;
                    let q = &qs[ib * 16..ib * 16 + 16];
                    let yy = &mut y[ib * 32..ib * 32 + 32];
                    for j in 0..16 {
                        yy[j] = dl * IQ4NL_VALUES[(q[j] & 0xf) as usize] as f32;
                        yy[j + 16] = dl * IQ4NL_VALUES[(q[j] >> 4) as usize] as f32;
                    }
                }
            }
        }
        Q2K => {
            // scales[16] | qs[64] | d | dmin
            for (y, b) in out
                .as_chunks_mut::<256>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<84>().0)
            {
                let sc = &b[0..16];
                let d = half(b, 80);
                let min = half(b, 82);
                let mut yi = 0;
                let mut is = 0;
                for n in 0..2 {
                    let q = &b[16 + n * 32..16 + n * 32 + 32];
                    for shift in [0u32, 2, 4, 6] {
                        for half_ in 0..2 {
                            let s = sc[is];
                            is += 1;
                            let dl = d * (s & 0xf) as f32;
                            let ml = min * (s >> 4) as f32;
                            for l in 0..16 {
                                y[yi] = dl * ((q[l + 16 * half_] >> shift) & 3) as f32 - ml;
                                yi += 1;
                            }
                        }
                    }
                }
            }
        }
        Q3K => {
            // hmask[32] | qs[64] | scales[12] | d
            for (y, b) in out
                .as_chunks_mut::<256>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<110>().0)
            {
                let hm = &b[0..32];
                let d_all = half(b, 108);
                let scales = q3k_scales(&b[96..108]);
                let mut yi = 0;
                let mut is = 0;
                let mut m = 1u8;
                for n in 0..2 {
                    let q = &b[32 + n * 32..32 + n * 32 + 32];
                    for shift in [0u32, 2, 4, 6] {
                        for half_ in 0..2 {
                            let dl = d_all * (scales[is] as i32 - 32) as f32;
                            is += 1;
                            for l in 0..16 {
                                let li = l + 16 * half_;
                                let v = ((q[li] >> shift) & 3) as i32
                                    - if hm[li] & m != 0 { 0 } else { 4 };
                                y[yi] = dl * v as f32;
                                yi += 1;
                            }
                        }
                        m <<= 1;
                    }
                }
            }
        }
        Q4K => {
            // d | dmin | scales[12] | qs[128]
            for (y, b) in out
                .as_chunks_mut::<256>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<144>().0)
            {
                let d = half(b, 0);
                let min = half(b, 2);
                let sc = &b[4..16];
                let mut yi = 0;
                for j in 0..4 {
                    let q = &b[16 + j * 32..16 + j * 32 + 32];
                    let (s1, m1) = scale_min_k4(2 * j, sc);
                    let (s2, m2) = scale_min_k4(2 * j + 1, sc);
                    let (d1, mm1) = (d * s1 as f32, min * m1 as f32);
                    let (d2, mm2) = (d * s2 as f32, min * m2 as f32);
                    for l in 0..32 {
                        y[yi + l] = d1 * (q[l] & 0xf) as f32 - mm1;
                        y[yi + 32 + l] = d2 * (q[l] >> 4) as f32 - mm2;
                    }
                    yi += 64;
                }
            }
        }
        Q5K => {
            // d | dmin | scales[12] | qh[32] | qs[128]
            for (y, b) in out
                .as_chunks_mut::<256>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<176>().0)
            {
                let d = half(b, 0);
                let min = half(b, 2);
                let sc = &b[4..16];
                let qh = &b[16..48];
                let mut yi = 0;
                let (mut u1, mut u2) = (1u8, 2u8);
                for j in 0..4 {
                    let ql = &b[48 + j * 32..48 + j * 32 + 32];
                    let (s1, m1) = scale_min_k4(2 * j, sc);
                    let (s2, m2) = scale_min_k4(2 * j + 1, sc);
                    let (d1, mm1) = (d * s1 as f32, min * m1 as f32);
                    let (d2, mm2) = (d * s2 as f32, min * m2 as f32);
                    for l in 0..32 {
                        let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                        let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                        y[yi + l] = d1 * ((ql[l] & 0xf) + h1) as f32 - mm1;
                        y[yi + 32 + l] = d2 * ((ql[l] >> 4) + h2) as f32 - mm2;
                    }
                    yi += 64;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        Q6K => {
            // ql[128] | qh[64] | scales[16] (i8) | d
            for (y, b) in out
                .as_chunks_mut::<256>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<210>().0)
            {
                let d = half(b, 208);
                for n in 0..2 {
                    let ql = &b[n * 64..n * 64 + 64];
                    let qh = &b[128 + n * 32..128 + n * 32 + 32];
                    let sc = &b[192 + n * 8..192 + n * 8 + 8];
                    let yy = &mut y[n * 128..n * 128 + 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        yy[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        yy[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        yy[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        yy[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
        Iq2Xs => {
            // d | qs u16[32] | scales[8]: each u16 is a 9-bit grid index and a 7-bit sign index
            for (y, b) in out
                .as_chunks_mut::<256>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<74>().0)
            {
                let d = half(b, 0);
                for ib in 0..8 {
                    let s = b[66 + ib];
                    let db = [
                        d * (0.5 + (s & 0xf) as f32) * 0.25,
                        d * (0.5 + (s >> 4) as f32) * 0.25,
                    ];
                    for l in 0..4 {
                        let q = rd16(b, 2 + 2 * (4 * ib + l));
                        let grid = grids::IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
                        let signs = grids::KSIGNS_IQ2XS[(q >> 9) as usize];
                        let yy = &mut y[ib * 32 + l * 8..ib * 32 + l * 8 + 8];
                        for j in 0..8 {
                            let sg = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                            yy[j] = db[l / 2] * grid[j] as f32 * sg;
                        }
                    }
                }
            }
        }
        Iq3Xxs => {
            // d | qs[64] grid indices | 8 × u32 (7-bit sign indices ×4, 4-bit scale)
            for (y, b) in out
                .as_chunks_mut::<256>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<98>().0)
            {
                let d = half(b, 0);
                let qs = &b[2..66];
                for ib in 0..8 {
                    let at = 66 + 4 * ib;
                    let aux = u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
                    let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                    for l in 0..4 {
                        let signs = grids::KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                        let g1 = grids::IQ3XXS_GRID[qs[8 * ib + 2 * l] as usize].to_le_bytes();
                        let g2 = grids::IQ3XXS_GRID[qs[8 * ib + 2 * l + 1] as usize].to_le_bytes();
                        let yy = &mut y[ib * 32 + l * 8..ib * 32 + l * 8 + 8];
                        for j in 0..4 {
                            let s1 = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                            let s2 = if signs & (1 << (j + 4)) != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            yy[j] = db * g1[j] as f32 * s1;
                            yy[j + 4] = db * g2[j] as f32 * s2;
                        }
                    }
                }
            }
        }
        other => bail!("no dequantizer for {}", other.name()),
    }
    Ok(())
}

/// `get_scale_min_k4` from ggml-quants.c: the 6-bit scale and min of sub-block `j` (0..8).
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Q3_K's sixteen 6-bit scales, unpacked from 12 bytes exactly as `dequantize_row_q3_K` does.
fn q3k_scales(s: &[u8]) -> [u8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let rd = |i: usize| u32::from_le_bytes([s[i], s[i + 1], s[i + 2], s[i + 3]]);
    let (a0, a1, tmp) = (rd(0), rd(4), rd(8));
    let aux = [
        (a0 & KMASK2) | ((tmp & KMASK1) << 4),
        (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
        ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
        ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
    ];
    let mut out = [0u8; 16];
    for (i, w) in aux.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic generator so the tests need no rand crate.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn byte(&mut self) -> u8 {
            self.next() as u8
        }
        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.byte()).collect()
        }
    }

    fn put_half(b: &mut [u8], at: usize, x: f32) {
        b[at..at + 2].copy_from_slice(&f32_to_f16(x).to_le_bytes());
    }

    #[test]
    fn half_round_trips() {
        for &x in &[
            0.0f32,
            -0.0,
            1.0,
            -2.5,
            65504.0,
            6.103_515_6e-5,
            5.960_464_5e-8,
            0.333_251_95,
        ] {
            assert_eq!(f16_to_f32(f32_to_f16(x)), x, "{x}");
        }
        // every half value survives f16 -> f32 -> f16
        for h in 0..=u16::MAX {
            let x = f16_to_f32(h);
            if x.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(x), h, "{h:#06x}");
        }
    }

    #[test]
    fn q2_0_is_code_minus_one() {
        // 64 codes 0,1,2,3 repeating, d = 0.5: values -0.5, 0, 0.5, 1.0
        let mut b = vec![0u8; 18];
        put_half(&mut b, 0, 0.5);
        for i in 0..16 {
            b[2 + i] = 0b11_10_01_00;
        }
        let mut y = vec![0f32; 64];
        dequantize(GgmlType::Q2_0, &b, &mut y).unwrap();
        for (j, v) in y.iter().enumerate() {
            assert_eq!(*v, ((j % 4) as f32 - 1.0) * 0.5);
        }
    }

    #[test]
    fn q8_0_scalar() {
        let mut r = Lcg(1);
        let mut b = r.bytes(34 * 3);
        for blk in 0..3 {
            put_half(&mut b, blk * 34, 0.25 + blk as f32);
        }
        let mut y = vec![0f32; 96];
        dequantize(GgmlType::Q8_0, &b, &mut y).unwrap();
        for i in 0..96 {
            let d = 0.25 + (i / 32) as f32;
            assert_eq!(y[i], d * (b[(i / 32) * 34 + 2 + i % 32] as i8) as f32);
        }
    }

    #[test]
    fn iq4_nl_split_halves() {
        let mut b = vec![0u8; 18];
        put_half(&mut b, 0, 1.0);
        for j in 0..16 {
            b[2 + j] = (j as u8) | (((15 - j) as u8) << 4);
        }
        let mut y = vec![0f32; 32];
        dequantize(GgmlType::Iq4Nl, &b, &mut y).unwrap();
        for j in 0..16 {
            assert_eq!(y[j], IQ4NL_VALUES[j] as f32);
            assert_eq!(y[j + 16], IQ4NL_VALUES[15 - j] as f32);
        }
    }

    /// The K-quants and IQ4_XS against an independent element-at-a-time reading of the block
    /// layout (`ggml-common.h`), so a loop-structure slip in the fast path shows up.
    fn element_q4k(b: &[u8], i: usize) -> f32 {
        let d = half(b, 0);
        let min = half(b, 2);
        let sub = i / 32; // 8 sub-blocks of 32
        let (sc, m) = scale_min_k4(sub, &b[4..16]);
        let chunk = sub / 2; // 64-element chunk shares 32 bytes
        let byte = b[16 + chunk * 32 + i % 32];
        let q = if sub.is_multiple_of(2) {
            byte & 0xf
        } else {
            byte >> 4
        };
        d * sc as f32 * q as f32 - min * m as f32
    }

    fn element_q6k(b: &[u8], i: usize) -> f32 {
        let d = half(b, 208);
        let n = i / 128;
        let r = i % 128;
        let quarter = r / 32; // which of q1..q4
        let l = r % 32;
        let ql = &b[n * 64..];
        let qh = b[128 + n * 32 + l];
        let lo = match quarter {
            0 => ql[l] & 0xf,
            1 => ql[l + 32] & 0xf,
            2 => ql[l] >> 4,
            _ => ql[l + 32] >> 4,
        };
        let hi = (qh >> (2 * quarter)) & 3;
        let q = (lo | (hi << 4)) as i32 - 32;
        let sc = b[192 + n * 8 + l / 16 + 2 * quarter] as i8;
        d * sc as f32 * q as f32
    }

    fn element_q2k(b: &[u8], i: usize) -> f32 {
        let d = half(b, 80);
        let min = half(b, 82);
        let n = i / 128;
        let r = i % 128;
        let shift = (r / 32) * 2;
        let l = r % 32;
        let s = b[n * 8 + r / 16];
        let q = (b[16 + n * 32 + l] >> shift) & 3;
        d * (s & 0xf) as f32 * q as f32 - min * (s >> 4) as f32
    }

    fn element_q3k(b: &[u8], i: usize) -> f32 {
        let d = half(b, 108);
        // scales: 6-bit, low 4 bits from bytes 0..8 (nibbles), high 2 bits from bytes 8..12
        let sidx = i / 16;
        let low = if sidx < 8 {
            b[96 + sidx] & 0xf
        } else {
            b[96 + sidx - 8] >> 4
        };
        let high = (b[96 + 8 + sidx % 4] >> (2 * (sidx / 4))) & 3;
        let s = (low | (high << 4)) as i32 - 32;
        let n = i / 128;
        let r = i % 128;
        let shift = (r / 32) * 2;
        let l = r % 32;
        let q = ((b[32 + n * 32 + l] >> shift) & 3) as i32;
        let hbit = (b[l] >> (n * 4 + r / 32)) & 1;
        d * s as f32 * (q - if hbit == 1 { 0 } else { 4 }) as f32
    }

    fn element_q5k(b: &[u8], i: usize) -> f32 {
        let d = half(b, 0);
        let min = half(b, 2);
        let sub = i / 32;
        let (sc, m) = scale_min_k4(sub, &b[4..16]);
        let l = i % 32;
        let byte = b[48 + (sub / 2) * 32 + l];
        let lo = if sub.is_multiple_of(2) {
            byte & 0xf
        } else {
            byte >> 4
        };
        let hi = (b[16 + l] >> sub) & 1;
        d * sc as f32 * (lo + 16 * hi) as f32 - min * m as f32
    }

    fn element_iq4xs(b: &[u8], i: usize) -> f32 {
        let d = half(b, 0);
        let ib = i / 32;
        let sh = rd16(b, 2);
        let ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) as i32
            | ((((sh >> (2 * ib)) & 3) << 4) as i32);
        let j = i % 32;
        let byte = b[8 + ib * 16 + j % 16];
        let q = if j < 16 { byte & 0xf } else { byte >> 4 };
        d * (ls - 32) as f32 * IQ4NL_VALUES[q as usize] as f32
    }

    fn check(ty: GgmlType, by: usize, d_at: &[usize], elem: fn(&[u8], usize) -> f32) {
        let mut r = Lcg(7 + by as u64);
        for _ in 0..20 {
            let mut b = r.bytes(by);
            for &at in d_at {
                put_half(&mut b, at, (r.next() % 1000) as f32 / 4096.0 - 0.1);
            }
            let mut y = vec![0f32; 256];
            dequantize(ty, &b, &mut y).unwrap();
            for (i, &got) in y.iter().enumerate() {
                let want = elem(&b, i);
                assert!(
                    (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                    "{} elem {i}: {got} vs {want}",
                    ty.name()
                );
            }
        }
    }

    #[test]
    fn k_quants_match_element_readers() {
        check(GgmlType::Q2K, 84, &[80, 82], element_q2k);
        check(GgmlType::Q3K, 110, &[108], element_q3k);
        check(GgmlType::Q4K, 144, &[0, 2], element_q4k);
        check(GgmlType::Q5K, 176, &[0, 2], element_q5k);
        check(GgmlType::Q6K, 210, &[208], element_q6k);
        check(GgmlType::Iq4Xs, 136, &[0], element_iq4xs);
    }

    #[test]
    fn header_round_trip() {
        // A hand-written two-tensor GGUF: one f32 vector, one Q8_0 matrix.
        let mut h = Vec::new();
        h.extend_from_slice(b"GGUF");
        h.extend_from_slice(&3u32.to_le_bytes());
        h.extend_from_slice(&2u64.to_le_bytes()); // tensors
        h.extend_from_slice(&2u64.to_le_bytes()); // kv
        let s = |h: &mut Vec<u8>, x: &str| {
            h.extend_from_slice(&(x.len() as u64).to_le_bytes());
            h.extend_from_slice(x.as_bytes());
        };
        s(&mut h, "general.architecture");
        h.extend_from_slice(&8u32.to_le_bytes());
        s(&mut h, "qwen4exp");
        s(&mut h, "x.list");
        h.extend_from_slice(&9u32.to_le_bytes());
        h.extend_from_slice(&4u32.to_le_bytes());
        h.extend_from_slice(&3u64.to_le_bytes());
        for v in [7u32, 8, 9] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        s(&mut h, "a");
        h.extend_from_slice(&1u32.to_le_bytes());
        h.extend_from_slice(&4u64.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes());
        h.extend_from_slice(&0u64.to_le_bytes());
        s(&mut h, "b");
        h.extend_from_slice(&2u32.to_le_bytes());
        h.extend_from_slice(&32u64.to_le_bytes());
        h.extend_from_slice(&2u64.to_le_bytes());
        h.extend_from_slice(&8u32.to_le_bytes());
        h.extend_from_slice(&32u64.to_le_bytes()); // after 16 B of `a`, aligned to 32
        while h.len() % 32 != 0 {
            h.push(0);
        }
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        h.resize(h.len() + 16, 0);
        for row in 0..2 {
            let mut blk = [0u8; 34];
            put_half(&mut blk, 0, 0.5);
            for j in 0..32 {
                blk[2 + j] = (j as i8 - row * 3) as u8;
            }
            h.extend_from_slice(&blk);
        }
        let dir = std::env::temp_dir().join(format!("tang-gguf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.gguf");
        std::fs::write(&p, &h).unwrap();
        let g = Gguf::open(&p).unwrap();
        assert_eq!(g.meta_str("general.architecture").unwrap(), "qwen4exp");
        assert_eq!(g.meta_u64s("x.list").unwrap(), vec![7, 8, 9]);
        assert_eq!(g.tensor_f32("a").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        let b = g.info("b").unwrap().clone();
        let row1 = g.rows(&b, 1, 1).unwrap();
        assert_eq!(row1[5], 0.5 * (5 - 3) as f32);
        let mut via_pread = vec![0f32; 32];
        g.read_row(&b, 1, &mut via_pread).unwrap();
        assert_eq!(row1, via_pread);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shard_names() {
        // single file that doesn't exist: returns itself without error
        let p = discover_shards(Path::new("/tmp/nope-not-sharded.gguf")).unwrap();
        assert_eq!(p.len(), 1);
        // sharded naming on a nonexistent dir fails with "shard missing" — proves the pattern matched
        let e = discover_shards(Path::new("/tmp/nope-00001-of-00006.gguf")).unwrap_err();
        assert!(format!("{e}").contains("shard"), "{e}");
    }
}
