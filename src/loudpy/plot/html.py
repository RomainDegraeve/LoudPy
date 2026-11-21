"""
Interactive viewer in one self-contained HTML file (lp.plot.html): no server, no internet, no extra package.

    lp.plot.html(res.field("p", f=500), "p_500.html")              a field (2D section or 3D surface)
    lp.plot.html([res.field("p", f=f) for f in (200, 500, 1000)], "p.html")   frames with a slider
    lp.plot.html(modes.mode(0), "mode0.html", show="z")            a component; show= as in Field.plot
    res.field("p", f=500).export("p_500.html")                     the same through Field.export

What is drawn: 3D meshes — the outer surface of the tetrahedra and the surface groups (shells, walls);
2D axisymmetric meshes — the r–z section (mirror=True: −r … r). The values (SPL, |u|, a component…) are
coloured with the quantity's colormap on a common range over the frames. Drag to rotate, wheel to zoom,
shift-drag (or right-drag) to pan, double-click to reset; the slider changes the frame.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from loudpy.errors import ResultsError


def _surface(mesh):
    """Corner triangles (rows) to draw and the 3D coordinates of the mesh nodes."""
    coords = np.asarray(mesh.coords, float)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])
    tris = []
    for g in mesh.groups.values():
        cells = mesh.rows_of(np.asarray(g.cells).ravel()).reshape(np.shape(g.cells))
        if g.dim == 3:                                    # outer faces of the tetrahedra (corners)
            c = cells[:, :4]
            f = np.concatenate([c[:, [0, 2, 1]], c[:, [0, 1, 3]], c[:, [0, 3, 2]], c[:, [1, 2, 3]]])
            key = np.sort(f, axis=1)
            _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
            tris.append(f[idx[cnt == 1]])
        elif g.dim == 2 and mesh.dim == 3:                # surface groups of a 3D mesh (shells, walls)
            tris.append(cells[:, :3] if cells.shape[1] in (3, 6) else
                        np.vstack([cells[:, [0, 1, 2]], cells[:, [0, 2, 3]]]))
    if mesh.dim == 2 or not tris:
        tris = [mesh.triangles(refine=True)]
    return coords, np.vstack(tris).astype(np.int64)


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def html(fields, path, show: str | None = None, mirror: bool = False, title: str | None = None,
         cmap=None, vmin=None, vmax=None, range_db: float | None = 50.0) -> Path:
    """
    Write an interactive viewer of a Field (or a list of Fields on the same mesh: frames) to `path` (.html).
    show: what to colour, as in Field.plot ("spl", "abs", "real", "imag", "phase", a component); mirror: 2D
    sections on −r … r; range_db: SPL range below the top. Returns the path.
    """
    import matplotlib as mpl
    from loudpy.plot.fields import scalar_values, field_title
    frames = list(fields) if isinstance(fields, (list, tuple)) else [fields]
    if not frames:
        raise ResultsError("lp.plot.html: no field")
    mesh = frames[0].mesh
    if any(f.mesh is not mesh and f.mesh.n_nodes != mesh.n_nodes for f in frames):
        raise ResultsError("lp.plot.html: the frames must share one mesh")
    coords, tris = _surface(mesh)
    vals, labels = [], []
    q: Any = None
    cm: Any = None
    symmetric = False
    for f in frames:
        v, q, cm, symmetric = scalar_values(f, show)
        vals.append(np.asarray(v, float))
        labels.append(field_title(f))
    V = np.vstack(vals)
    if mirror and mesh.dim == 2:
        n = len(coords)
        coords = np.vstack([coords, coords * [-1, 1, 1]])
        tris = np.vstack([tris, tris[:, [0, 2, 1]] + n])
        V = np.hstack([V, V])
    lo = float(np.nanmin(V)) if vmin is None else vmin
    hi = float(np.nanmax(V)) if vmax is None else vmax
    if q.name == "spl" and vmin is None and range_db is not None:
        hi = float(np.nanpercentile(V, 99.5)) if vmax is None else vmax
        lo = hi - range_db
    elif symmetric and vmin is None and vmax is None:
        hi = float(np.nanmax(np.abs(V))) or 1.0
        lo = -hi
    if hi <= lo:
        hi = lo + 1.0
    lut = (np.asarray(mpl.colormaps[cmap or cm](np.linspace(0, 1, 256)))[:, :3] * 255).astype(np.uint8)
    used = np.unique(tris)
    remap = -np.ones(len(coords), np.int64)
    remap[used] = np.arange(len(used))
    P = coords[used]
    centre = 0.5 * (P.min(axis=0) + P.max(axis=0))
    scale = float(np.ptp(P, axis=0).max()) or 1.0
    data = dict(
        pos=_b64((P - centre) / scale, np.float32), tri=_b64(remap[tris], np.uint32), n_tri=int(len(tris)),
        n=int(len(P)), frames=[_b64(np.nan_to_num(v[used], nan=lo), np.float32) for v in V],
        labels=labels, lut=_b64(lut, np.uint8), lo=lo, hi=hi, unit=q.axis_label(), flat=bool(mesh.dim == 2),
        title=title or (labels[0] if len(labels) == 1 else q.label))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(_PAGE.replace("__TITLE__", _esc(data["title"])).replace("__DATA__", json.dumps(data)))
    tmp.replace(out)
    return out


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fafafa;--fg:#222;--muted:#666}
@media (prefers-color-scheme:dark){:root{--bg:#1d1d1f;--fg:#eee;--muted:#aaa}}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px/1.4 Georgia,serif}
#wrap{display:flex;flex-direction:column;height:100%}
header{padding:8px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:normal}
#gl{flex:1;width:100%;min-height:0;touch-action:none;cursor:grab}
footer{padding:6px 16px 10px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
#bar{width:min(360px,60vw);height:12px;border:1px solid var(--muted)}
.small{color:var(--muted);font-size:12px}
input[type=range]{width:min(320px,50vw)}
</style></head><body><div id="wrap">
<header><h1 id="t"></h1><span id="frame" class="small"></span>
<span id="sl"><input id="slider" type="range" min="0" value="0"></span></header>
<canvas id="gl"></canvas>
<footer><span id="lo" class="small"></span><canvas id="bar" width="256" height="1"></canvas><span id="hi" class="small"></span>
<span id="unit" class="small"></span><span class="small">drag: rotate · wheel: zoom · shift-drag: pan · double-click: reset</span></footer>
</div>
<script>
const D=__DATA__;
function dec(s,T){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer)}
const pos=dec(D.pos,Float32Array),tri=dec(D.tri,Uint32Array),lut=dec(D.lut,Uint8Array),frames=D.frames.map(f=>dec(f,Float32Array));
const fmt=v=>Math.abs(v)>=1e4||(Math.abs(v)<1e-2&&v!==0)?v.toExponential(2):v.toFixed(2);
document.getElementById('t').textContent=D.title;document.getElementById('lo').textContent=fmt(D.lo);
document.getElementById('hi').textContent=fmt(D.hi);document.getElementById('unit').textContent=D.unit;
const bar=document.getElementById('bar').getContext('2d'),img=bar.createImageData(256,1);
for(let i=0;i<256;i++){img.data.set([lut[3*i],lut[3*i+1],lut[3*i+2],255],4*i)}bar.putImageData(img,0,0);
// unindexed triangles: flat normals
const nt=D.n_tri,P=new Float32Array(9*nt),N=new Float32Array(9*nt),C=new Float32Array(9*nt);
for(let t=0;t<nt;t++){const a=tri[3*t],b=tri[3*t+1],c=tri[3*t+2];
 const ax=pos[3*a],ay=pos[3*a+1],az=pos[3*a+2],bx=pos[3*b]-ax,by=pos[3*b+1]-ay,bz=pos[3*b+2]-az,cx=pos[3*c]-ax,cy=pos[3*c+1]-ay,cz=pos[3*c+2]-az;
 let nx=by*cz-bz*cy,ny=bz*cx-bx*cz,nz=bx*cy-by*cx;const l=Math.hypot(nx,ny,nz)||1;nx/=l;ny/=l;nz/=l;
 [a,b,c].forEach((v,k)=>{P.set([pos[3*v],pos[3*v+1],pos[3*v+2]],9*t+3*k);N.set([nx,ny,nz],9*t+3*k)})}
const cv=document.getElementById('gl'),gl=cv.getContext('webgl',{antialias:true});
function sh(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);return s}
const pr=gl.createProgram();
gl.attachShader(pr,sh(gl.VERTEX_SHADER,`attribute vec3 p,n,c;uniform mat4 M,Pj;varying vec3 vc;varying float li;
void main(){vec4 q=M*vec4(p,1.);gl_Position=Pj*q;vec3 nn=normalize((M*vec4(n,0.)).xyz);li=0.35+0.65*abs(nn.z);vc=c;}`));
gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,`precision mediump float;varying vec3 vc;varying float li;uniform float flat_;
void main(){gl_FragColor=vec4(vc*mix(li,1.,flat_),1.);}`));
gl.linkProgram(pr);gl.useProgram(pr);
function buf(a,name){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,a,gl.STATIC_DRAW);
 const l=gl.getAttribLocation(pr,name);gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,3,gl.FLOAT,false,0,0);return b}
buf(P,'p');buf(N,'n');const cb=buf(C,'c');
gl.uniform1f(gl.getUniformLocation(pr,'flat_'),D.flat?1:0);
function colour(k){const f=frames[k];for(let t=0;t<nt;t++)for(let j=0;j<3;j++){const v=f[tri[3*t+j]];
 let i=Math.round(255*(v-D.lo)/(D.hi-D.lo));i=i<0?0:i>255?255:i;C.set([lut[3*i]/255,lut[3*i+1]/255,lut[3*i+2]/255],9*t+3*j)}
 gl.bindBuffer(gl.ARRAY_BUFFER,cb);gl.bufferData(gl.ARRAY_BUFFER,C,gl.DYNAMIC_DRAW);
 document.getElementById('frame').textContent=D.labels.length>1?D.labels[k]:'';draw()}
let yaw=D.flat?0:-0.6,pitch=D.flat?0:0.45,zoom=1,px=0,py=0;
function mat(){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 return new Float32Array([cy,sy*sp,-sy*cp,0, 0,cp,sp,0, sy,-cy*sp,cy*cp,0, px,py,-3,1])}
function draw(){const w=cv.clientWidth*devicePixelRatio,h=cv.clientHeight*devicePixelRatio;if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h}
 gl.viewport(0,0,w,h);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);
 const a=w/h,s=0.9/zoom,Pj=new Float32Array([1/(s*Math.max(a,1)),0,0,0, 0,1/(s*Math.max(1/a,1)),0,0, 0,0,-0.2,0, 0,0,0,1]);
 gl.uniformMatrix4fv(gl.getUniformLocation(pr,'M'),false,mat());gl.uniformMatrix4fv(gl.getUniformLocation(pr,'Pj'),false,Pj);
 gl.drawArrays(gl.TRIANGLES,0,3*nt)}
let drag=null;cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2};cv.setPointerCapture(e.pointerId)});
cv.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
 if(drag.pan){px+=2*dx/cv.clientHeight/zoom;py-=2*dy/cv.clientHeight/zoom}else{yaw+=dx*0.008;pitch+=dy*0.008}draw()});
cv.addEventListener('pointerup',()=>drag=null);cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('wheel',e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*0.001);draw()},{passive:false});
cv.addEventListener('dblclick',()=>{yaw=D.flat?0:-0.6;pitch=D.flat?0:0.45;zoom=1;px=py=0;draw()});
const sl=document.getElementById('slider');if(frames.length>1){sl.max=frames.length-1;sl.oninput=()=>colour(+sl.value)}else document.getElementById('sl').style.display='none';
addEventListener('resize',draw);colour(0);
</script></body></html>
"""
