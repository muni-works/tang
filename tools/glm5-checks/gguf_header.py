#!/usr/bin/env python3
"""Parse the GGUF v3 header + tensor table from a partial shard file, exactly as tang's reader does."""
import struct, sys

def parse(path):
    data = open(path, 'rb').read()
    off = [0]
    def take(n):
        end = off[0] + n
        if end > len(data):
            raise EOFError(f"truncated at {off[0]} of {len(data)}")
        s = data[off[0]:end]; off[0] = end
        return s
    def u8(): return take(1)[0]
    def u32(): return struct.unpack('<I', take(4))[0]
    def u64(): return struct.unpack('<Q', take(8))[0]
    def string(): return take(u64()).decode('utf-8', 'replace')
    def value(ty):
        if ty == 0: return u8()
        if ty == 1: return struct.unpack('<b', take(1))[0]
        if ty == 2: return struct.unpack('<H', take(2))[0]
        if ty == 3: return struct.unpack('<h', take(2))[0]
        if ty == 4: return u32()
        if ty == 5: return struct.unpack('<i', take(4))[0]
        if ty == 6: return struct.unpack('<f', take(4))[0]
        if ty == 7: return u8() != 0
        if ty == 8: return string()
        if ty == 9:
            ety = u32(); n = u64()
            return [value(ety) for _ in range(n)]
        if ty == 10: return u64()
        if ty == 11: return struct.unpack('<q', take(8))[0]
        if ty == 12: return struct.unpack('<d', take(8))[0]
        raise ValueError(f'type {ty}')
    magic = struct.unpack('<I', take(4))[0]
    assert magic == 0x46554747, hex(magic)
    ver = u32(); assert ver == 3
    n_tensors = u64(); n_kv = u64()
    meta = {}
    for _ in range(n_kv):
        k = string(); ty = u32()
        meta[k] = value(ty)
    tensors = []
    try:
        for _ in range(n_tensors):
            name = string()
            nd = u32()
            dims = struct.unpack(f'<{nd}Q', take(8 * nd))
            ty = u32()
            offset = u64()
            tensors.append((name, list(dims), ty, offset))
    except EOFError:
        pass
    return meta, tensors, off[0], len(data)

meta, tensors, off, sz = parse(sys.argv[1])
print(f'# kv={len(meta)} tensors={len(tensors)} hdr_end={off} file_bytes={sz}')
for k, v in meta.items():
    if isinstance(v, list) and len(v) > 8:
        print(f'{k} = [{len(v)} items: {v[:4]}...]')
    else:
        print(f'{k} = {v}')
for name, dims, ty, offset in tensors:
    print(f'T {name} {dims} {ty} @{offset}')
