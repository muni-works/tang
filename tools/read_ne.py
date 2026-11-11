import struct
f=open("/home/cam/tang-scratch/glm53/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00002-of-00006.gguf","rb")
def u32(): return struct.unpack("<I", f.read(4))[0]
def u64(): return struct.unpack("<Q", f.read(8))[0]
def st(): n=u64(); return f.read(n).decode()
f.read(4)
n_tensors=u64(); n_kv=u64()
for i in range(n_kv):
    k=st(); vt=u32()
    if vt==8: v=st()
    elif vt in (0,1,2,3): f.read({0:1,1:1,2:2,3:2}[vt])
    elif vt in (4,5,6,7,10,11,12): f.read(8)
    elif vt==9:
        et=u32(); cnt=u64(); f.read(cnt*{0:1,1:1,2:2,3:2,4:4,5:8,10:4,12:8}.get(et,1))
found=0
i=0
while found<3:
    klen=u64(); name=f.read(klen).decode()
    n_dims=u32(); dims=[u64() for _ in range(n_dims)]
    ty=u32(); off=u64()
    i+=1
    if name.endswith(".weight") and ("attn_q" in name or "attn_output" in name or "ssm_f_a" in name) and ".blk.3." in name:
        print(name, "ne=", dims, "ty=", ty); found+=1
    if i>1000: print("stopped at", i); break
