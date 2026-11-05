"""Interactive (rotatable) 3D view of a raw skeleton over the SLAP2 image, as one self-contained HTML file.

Image: green channel averaged to 1 um voxels; the brightest voxels (above --pct) are drawn as additive points
with brightness from intensity. Skeleton: p >= --thr thinned (skimage), no cleanup, 26-neighbour segments.
Plain WebGL, no external libraries. Drag to rotate, wheel to zoom, shift-drag to pan.

Usage:
  uv run --with scikit-image python scripts/slap2_skeleton_html.py <REFERENCE.tif> <trailmap npz> <grid key> \
      <run index> <out.html> [--thr 0.5] [--pct 98.5]
"""

import argparse
import base64
import itertools
import json

import numpy as np
import tifffile
from skimage.morphology import skeletonize

p = argparse.ArgumentParser()
for k in ("tif", "npz", "key", "run", "out"):
    p.add_argument(k)
p.add_argument("--thr", type=float, default=0.5)
p.add_argument("--pct", type=float, default=98.5)
a = p.parse_args()
d = np.load(a.npz)
res = json.loads(str(d["results"]))[int(a.run)]
g = np.array([float(v) for v in a.key.split("x")])
sk = skeletonize(d[f"prob_{a.key}_{a.run}"].astype(np.float32) >= a.thr)
vox = np.argwhere(sk)
lut = {tuple(v): i for i, v in enumerate(vox)}
pairs = [(i, j) for i, v in enumerate(vox) for o in itertools.product((-1, 0, 1), repeat=3) if o > (0, 0, 0)
         for j in [lut.get((v[0] + o[0], v[1] + o[1], v[2] + o[2]))] if j is not None]  # fmt: skip
zyx = (vox + 0.5) * g
seg = np.stack([zyx[[i for i, _ in pairs]], zyx[[j for _, j in pairs]]], 1)  # (S, 2, zyx)
length_mm = float(np.linalg.norm(seg[:, 0] - seg[:, 1], axis=1).sum() / 1e3)

n = len(tifffile.TiffFile(a.tif).pages)
img = tifffile.imread(a.tif, key=range(0, n, 2))
img = img.reshape(img.shape[0], img.shape[1] // 4, 4, img.shape[2] // 4, 4).mean((2, 4))  # 1 um voxels (z, y, x)
lo = np.percentile(img, a.pct)
hi = np.percentile(img, 99.97)
pts = np.argwhere(img > lo)
val = np.clip((img[tuple(pts.T)] - lo) / (hi - lo), 0, 1)
pts_um = pts + np.array([0.5, 0.5, 0.5])
shape_um = np.array(img.shape, float)
del img
print("image points", len(pts), "skeleton segments", len(seg), "length mm", round(length_mm, 2))


def world(zyx_um):
    """(z, y, x) um -> centred world (x, -z, y): depth points down."""
    c = shape_um / 2
    q = zyx_um - c
    return np.stack([q[..., 2], -q[..., 0], q[..., 1]], -1).astype(np.float32)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


data = {
    "pts": b64(world(pts_um)),
    "val": b64((val * 255).astype(np.uint8)),
    "seg": b64(world(seg).reshape(-1, 3)),
    "box": (shape_um[[2, 0, 1]] / 2).tolist(),
}
title = f"raw skeleton of {res['weights'].split('/')[0]} on slap2 878030 dmd1"
note = (f"orange: p &ge; {a.thr:g} on the {a.key.replace('x', ' &times; ')} &micro;m grid (z, y, x), thinned, no "
        f"cleanup ({length_mm:.1f} mm). gray: green channel at 1 &micro;m, voxels above the {a.pct:g}th percentile. "
        "Depth points down. Drag to rotate, wheel to zoom, shift-drag to pan.")  # fmt: skip
html = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SLAP2 skeleton 3D</title><style>
:root{--bg:#000;--fg:#eee;--muted:#999;--panel:rgba(20,20,20,.8)}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;overflow:hidden}
canvas{display:block;width:100vw;height:100vh;touch-action:none}
#hud{position:fixed;left:12px;top:10px;right:12px;pointer-events:none}
#hud h1{font-size:15px;margin:0 0 4px}#hud p{margin:0;color:var(--muted);max-width:760px}
#ctl{position:fixed;left:12px;bottom:12px;background:var(--panel);padding:8px 10px;border-radius:8px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
label{cursor:pointer}input[type=range]{width:110px;vertical-align:middle}button{background:#333;color:var(--fg);border:1px solid #555;border-radius:5px;padding:3px 8px;cursor:pointer}
</style></head><body><canvas id="c"></canvas>
<div id="hud"><h1>__TITLE__</h1><p>__NOTE__</p></div>
<div id="ctl"><label><input type="checkbox" id="si" checked> image</label><label><input type="checkbox" id="ss" checked> skeleton</label>
<label>image brightness <input type="range" id="ib" min="0.05" max="1.5" step="0.05" value="0.25"></label>
<label>point size <input type="range" id="ps" min="1" max="6" step="0.5" value="2"></label>
<label><input type="checkbox" id="box" checked> box</label>
<button id="v1">oblique</button><button id="v2">top (xy)</button><button id="v3">side (xz)</button></div>
<script>
const D=__DATA__;
function dec(s,T){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer)}
const P=dec(D.pts,Float32Array),V=dec(D.val,Uint8Array),S=dec(D.seg,Float32Array),B=D.box;
const cv=document.getElementById('c'),gl=cv.getContext('webgl',{antialias:true,alpha:false});
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(o);return o}
function prog(vs,fs){const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,vs));gl.attachShader(p,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);return p}
const pp=prog(`attribute vec3 p;attribute float v;uniform mat4 m;uniform float s,b;varying float w;void main(){gl_Position=m*vec4(p,1.);gl_PointSize=s;w=v/255.*b;}`,
`precision mediump float;varying float w;void main(){vec2 d=gl_PointCoord-.5;if(dot(d,d)>.25)discard;gl_FragColor=vec4(vec3(w),1.);}`);
const lp=prog(`attribute vec3 p;uniform mat4 m;void main(){gl_Position=m*vec4(p,1.);}`,
`precision mediump float;uniform vec4 c;void main(){gl_FragColor=c;}`);
function buf(a){const o=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,o);gl.bufferData(gl.ARRAY_BUFFER,a,gl.STATIC_DRAW);return o}
const bp=buf(P),bv=buf(V),bs=buf(S);
const bx=[],[X,Y,Z]=B;for(const a of[-1,1])for(const b of[-1,1]){bx.push(-X,a*Y,b*Z,X,a*Y,b*Z,a*X,-Y,b*Z,a*X,Y,b*Z,a*X,b*Y,-Z,a*X,b*Y,Z)}
const bb=buf(new Float32Array(bx));
let yaw=-0.6,pitch=0.45,dist=2.4*Math.hypot(X,Y,Z),pan=[0,0];
function views(k){[yaw,pitch]=[[-0.6,0.45],[0,Math.PI/2-1e-3],[0,0]][k];pan=[0,0];draw()}
function mul(a,b){const o=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s}return o}
function mat(){const w=cv.width,h=cv.height,f=1/Math.tan(0.35),n=1,fa=dist*4;
const pr=new Float32Array([f*h/w,0,0,0,0,f,0,0,0,0,(fa+n)/(n-fa),-1,0,0,2*fa*n/(n-fa),0]);
const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
const ry=new Float32Array([cy,0,-sy,0,0,1,0,0,sy,0,cy,0,0,0,0,1]),rx=new Float32Array([1,0,0,0,0,cp,sp,0,0,-sp,cp,0,0,0,0,1]);
const t=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,pan[0],pan[1],-dist,1]);return mul(pr,mul(t,mul(rx,ry)))}
const $=id=>document.getElementById(id);
function draw(){const r=devicePixelRatio||1;cv.width=innerWidth*r;cv.height=innerHeight*r;gl.viewport(0,0,cv.width,cv.height);
gl.clearColor(0,0,0,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);const m=mat();
if($('si').checked){gl.useProgram(pp);gl.disable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE);
gl.uniformMatrix4fv(gl.getUniformLocation(pp,'m'),false,m);gl.uniform1f(gl.getUniformLocation(pp,'s'),+$('ps').value*r);
gl.uniform1f(gl.getUniformLocation(pp,'b'),+$('ib').value);
let l=gl.getAttribLocation(pp,'p');gl.bindBuffer(gl.ARRAY_BUFFER,bp);gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,3,gl.FLOAT,false,0,0);
const lv=gl.getAttribLocation(pp,'v');gl.bindBuffer(gl.ARRAY_BUFFER,bv);gl.enableVertexAttribArray(lv);gl.vertexAttribPointer(lv,1,gl.UNSIGNED_BYTE,false,0,0);
gl.drawArrays(gl.POINTS,0,P.length/3);gl.disableVertexAttribArray(lv)}
gl.useProgram(lp);gl.disable(gl.BLEND);gl.uniformMatrix4fv(gl.getUniformLocation(lp,'m'),false,m);const l2=gl.getAttribLocation(lp,'p');
function lines(b,n,c){gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.enableVertexAttribArray(l2);gl.vertexAttribPointer(l2,3,gl.FLOAT,false,0,0);gl.uniform4fv(gl.getUniformLocation(lp,'c'),c);gl.drawArrays(gl.LINES,0,n)}
if($('box').checked)lines(bb,bx.length/3,[.35,.35,.35,1]);if($('ss').checked)lines(bs,S.length/3,[1,.35,.21,1])}
let drag=null;cv.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,s:e.shiftKey};cv.setPointerCapture(e.pointerId)};
cv.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
if(drag.s){const k=dist/innerHeight*0.75;pan[0]+=dx*k;pan[1]-=dy*k}else{yaw+=dx*0.008;pitch=Math.max(-1.57,Math.min(1.57,pitch+dy*0.008))}draw()};
cv.onpointerup=()=>drag=null;cv.onwheel=e=>{e.preventDefault();dist*=Math.exp(e.deltaY*0.001);draw()};
for(const id of['si','ss','ib','ps','box'])$(id).oninput=draw;$('v1').onclick=()=>views(0);$('v2').onclick=()=>views(1);$('v3').onclick=()=>views(2);
onresize=draw;draw();
</script></body></html>"""
html = html.replace("__TITLE__", title).replace("__NOTE__", note).replace("__DATA__", json.dumps(data))
open(a.out, "w").write(html)
print("saved", a.out, round(len(html) / 1e6, 1), "MB")
