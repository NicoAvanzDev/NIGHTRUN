//! Minimal GGUF v2/v3 reader: metadata KVs + tensor infos + data section.

use std::collections::HashMap;

#[derive(Debug, Clone)]
#[allow(dead_code)] // parsed for completeness; only some variants are read
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            Value::U8(v) => Some(v as u32),
            Value::U16(v) => Some(v as u32),
            Value::U32(v) => Some(v),
            Value::I32(v) => u32::try_from(v).ok(),
            Value::U64(v) => u32::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(v) => Some(v),
            _ => None,
        }
    }
}

// GGML tensor dtypes we understand.
pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q1_0: u32 = 41;
pub const GGML_Q8_0: u32 = 8;
pub const GGML_Q4_K: u32 = 12;
pub const GGML_Q5_K: u32 = 13;
pub const GGML_Q6_K: u32 = 14;

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>, // dims[0] = fastest (cols)
    pub dtype: u32,
    /// Absolute offset of the tensor within the file.
    pub file_offset: u64,
    pub byte_size: u64,
}

pub struct Gguf {
    pub kv: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub data: Vec<u8>,
}

pub fn tensor_byte_size(dtype: u32, nelem: u64) -> u64 {
    match dtype {
        GGML_Q1_0 => {
            assert!(
                nelem.is_multiple_of(128),
                "Q1_0 element count must be divisible by 128"
            );
            nelem / 128 * 18
        }
        GGML_F32 => nelem * 4,
        GGML_F16 => nelem * 2,
        GGML_Q8_0 => nelem / 32 * 34,
        GGML_Q4_K => nelem / 256 * 144,
        GGML_Q5_K => nelem / 256 * 176,
        GGML_Q6_K => nelem / 256 * 210,
        _ => panic!("unsupported ggml dtype {dtype}"),
    }
}

struct Cur<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn str(&mut self, version: u32) -> String {
        let len = if version >= 2 {
            self.u64() as usize
        } else {
            self.u32() as usize
        };
        String::from_utf8_lossy(self.take(len)).into_owned()
    }
    fn value(&mut self, ty: u32, version: u32) -> Value {
        match ty {
            0 => Value::U8(self.take(1)[0]),
            1 => Value::I8(self.take(1)[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2).try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.take(2).try_into().unwrap())),
            4 => Value::U32(self.u32()),
            5 => Value::I32(self.u32() as i32),
            6 => Value::F32(f32::from_bits(self.u32())),
            7 => Value::Bool(self.take(1)[0] != 0),
            8 => Value::Str(self.str(version)),
            9 => {
                let elem_ty = self.u32();
                let count = if version >= 2 {
                    self.u64() as usize
                } else {
                    self.u32() as usize
                };
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.value(elem_ty, version));
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()),
            11 => Value::I64(self.u64() as i64),
            12 => Value::F64(f64::from_bits(self.u64())),
            _ => panic!("unknown gguf value type {ty}"),
        }
    }
}

pub fn parse(path: &str) -> Gguf {
    let data = std::fs::read(path).expect("read gguf file");
    let mut c = Cur { b: &data, pos: 0 };
    assert_eq!(c.take(4), b"GGUF", "not a GGUF file");
    let version = c.u32();
    assert!(
        (2..=3).contains(&version),
        "unsupported GGUF version {version}"
    );
    let n_tensors = c.u64();
    let n_kv = c.u64();

    let mut kv = HashMap::new();
    for _ in 0..n_kv {
        let key = c.str(version);
        let ty = c.u32();
        let val = c.value(ty, version);
        kv.insert(key, val);
    }

    let alignment = kv
        .get("general.alignment")
        .and_then(Value::as_u32)
        .unwrap_or(32) as u64;

    struct RawInfo {
        name: String,
        dims: Vec<u64>,
        dtype: u32,
        rel_offset: u64,
    }
    let mut raw = Vec::new();
    for _ in 0..n_tensors {
        let name = c.str(version);
        let n_dims = c.u32();
        let dims: Vec<u64> = (0..n_dims).map(|_| c.u64()).collect();
        let dtype = c.u32();
        let rel_offset = c.u64();
        raw.push(RawInfo {
            name,
            dims,
            dtype,
            rel_offset,
        });
    }
    let data_start = (c.pos as u64).div_ceil(alignment) * alignment;

    let tensors = raw
        .into_iter()
        .map(|r| {
            let nelem: u64 = r.dims.iter().product();
            TensorInfo {
                file_offset: data_start + r.rel_offset,
                byte_size: tensor_byte_size(r.dtype, nelem),
                name: r.name,
                dims: r.dims,
                dtype: r.dtype,
            }
        })
        .collect();

    Gguf { kv, tensors, data }
}
