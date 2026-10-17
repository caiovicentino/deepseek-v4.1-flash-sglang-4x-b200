"""Extract lmsysorg/sglang:<tag> (amd64) into a directory without Docker, applying OCI whiteouts.

Usage: python3 extract_image.py nightly-dev-cu13-20261016-a93f16f0 [/root/dsv41img]
Skips the CUDA toolkit and docs (the host needs CUDA 13 already). After extraction, relocate the image's Python:
  I=/root/dsv41img; ln -sfn $I/opt/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12 $I/opt/sglang/bin/python
  ln -sfn cpython-3.12.15-linux-x86_64-gnu $I/opt/python/cpython-3.12-linux-x86_64-gnu
  sed -i "s|^home = .*|home = $I/opt/python/cpython-3.12.15-linux-x86_64-gnu/bin|" $I/opt/sglang/pyvenv.cfg
  sed -i "s|/sgl-workspace|$I/sgl-workspace|g" $I/opt/sglang/lib/python3.12/site-packages/__editable___sglang_*_finder.py
The production SGLang install of the host is never touched."""
import io, json, os, shutil, sys, tarfile, time, urllib.request
TAG = sys.argv[1]; ROOT = sys.argv[2] if len(sys.argv) > 2 else "/root/dsv41img"; REPO = "lmsysorg/sglang"; R = f"https://registry-1.docker.io/v2/{REPO}"
SKIP = ("usr/local/cuda", "usr/share/doc", "usr/share/man", "var/cache", "var/lib/apt", "root/.cache", "tmp/")
def token():
    u = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPO}:pull"
    return json.load(urllib.request.urlopen(u))["token"]
def get(url, accept, tok):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "Accept": accept})
    return urllib.request.urlopen(req)
tok = token()
idx = json.load(get(f"{R}/manifests/{TAG}", "application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json", tok))
dig = [m["digest"] for m in idx["manifests"] if m.get("platform", {}).get("architecture") == "amd64"][0]
man = json.load(get(f"{R}/manifests/{dig}", "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json", tok))
json.dump(man, open(f"{ROOT}.manifest.json", "w"))
layers = man["layers"]; print(time.strftime("%H:%M:%S"), TAG, len(layers), "layers", round(sum(l["size"] for l in layers) / 1e9, 1), "GB", flush=True)
for i, l in enumerate(layers):
    tok = token()
    with get(f"{R}/blobs/{l['digest']}", "*/*", tok) as resp:
        with tarfile.open(fileobj=resp, mode="r|*") as tf:
            for m in tf:
                name = m.name.lstrip("./")
                if not name or name.startswith(SKIP): continue
                d, b = os.path.split(name)
                if b == ".wh..wh..opq":
                    p = os.path.join(ROOT, d)
                    if os.path.isdir(p):
                        for x in os.listdir(p):
                            q = os.path.join(p, x); shutil.rmtree(q) if os.path.isdir(q) and not os.path.islink(q) else os.remove(q)
                    continue
                if b.startswith(".wh."):
                    p = os.path.join(ROOT, d, b[4:])
                    if os.path.islink(p) or os.path.isfile(p): os.remove(p)
                    elif os.path.isdir(p): shutil.rmtree(p)
                    continue
                if m.isdev(): continue
                p = os.path.join(ROOT, name)
                if os.path.lexists(p) and not (m.isdir() and os.path.isdir(p)):
                    shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
                try: tf.extract(m, ROOT, set_attrs=not m.issym(), filter="fully_trusted")
                except Exception as e: print("  skip", name, type(e).__name__, flush=True)
    print(time.strftime("%H:%M:%S"), f"layer {i+1}/{len(layers)} done", flush=True)
print("EXTRACT_DONE", flush=True)
