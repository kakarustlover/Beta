// Velora – pure Rust, GPU-rendered (wgpu + winit). No HTML / WebView / Tauri / Dioxus.
// Glass is analytic: the animated background is a closed-form shader, panels are SDF shapes with
// tint + rim light + shadow, so there is NO live backdrop blur pass at all.
use fontdue::{Font, FontSettings};
use std::{collections::HashMap, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    event::{TouchPhase, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

const ATL: u32 = 1024;
const MAXI: usize = 6000;
type C = [f32; 4];
fn hx(h: u32, a: f32) -> C { [(h >> 16 & 255) as f32 / 255., (h >> 8 & 255) as f32 / 255., (h & 255) as f32 / 255., a] }
fn op(c: C, a: f32) -> C { [c[0], c[1], c[2], a] }
fn mmss(s: f32) -> String { let s = s as u32; format!("{}:{:02}", s / 60, s % 60) }

const PALS: [(u32, u32); 3] = [(0x7c5cff, 0xff5fa2), (0x2bb6ff, 0x2bffc6), (0xff8a3d, 0xff3d6e)];
const SONGS: [(&str, &str, u32, u32, u32); 12] = [
    ("Midnight Glass", "Velora Sessions", 214, 0x7c5cff, 0xff5fa2), ("Aurora Drive", "Neon Harbor", 187, 0x2bb6ff, 0x2bffc6),
    ("Golden Hour", "Lumen", 242, 0xff8a3d, 0xff3d6e), ("Slow Tide", "Paper Lanterns", 199, 0x3d7bff, 0x9d5cff),
    ("Silk Road", "Atlas Bloom", 225, 0xff5fa2, 0xffb86b), ("Ember", "Hollow Pines", 176, 0xff5a3d, 0xffc23d),
    ("Crystal Rain", "Mira Vale", 203, 0x5cf0ff, 0x5c7cff), ("Afterglow", "Static Garden", 231, 0xb35cff, 0xff5f8f),
    ("Horizon", "Velora Sessions", 258, 0x2bffc6, 0x2b8bff), ("Velvet Static", "Neon Harbor", 194, 0x7c5cff, 0x2bb6ff),
    ("Low Light", "Lumen", 210, 0xff7a5c, 0x7c5cff), ("Drift", "Atlas Bloom", 221, 0x5cffb0, 0x5c9cff),
];

#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct I { r: C, c0: C, c1: C, p: C, uv: C }
#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U { s: C, a1: C, a2: C, b1: C, b2: C }
const ATTRS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![0=>Float32x4,1=>Float32x4,2=>Float32x4,3=>Float32x4,4=>Float32x4];

const SHADER: &str = r#"
struct U{s:vec4<f32>,a1:vec4<f32>,a2:vec4<f32>,b1:vec4<f32>,b2:vec4<f32>}
@group(0) @binding(0) var<uniform> u:U;
@group(0) @binding(1) var tex:texture_2d<f32>;
@group(0) @binding(2) var smp:sampler;
struct B{@builtin(position) p:vec4<f32>,@location(0) uv:vec2<f32>}
@vertex fn vbg(@builtin(vertex_index) i:u32)->B{
  let x=f32((i<<1u)&2u); let y=f32(i&2u);
  var o:B; o.p=vec4<f32>(x*2.-1.,1.-y*2.,0.,1.); o.uv=vec2<f32>(x,y); return o;}
@fragment fn fbg(i:B)->@location(0) vec4<f32>{
  let t=u.s.z; let asp=u.s.x/u.s.y;
  var c=mix(u.b1.rgb,u.b2.rgb,i.uv.y);
  let p=vec2<f32>(i.uv.x*asp,i.uv.y);
  let c1=vec2<f32>(.2*asp+.10*asp*sin(t*.11),.18+.06*cos(t*.13));
  let c2=vec2<f32>(.85*asp+.08*asp*cos(t*.09),.55+.08*sin(t*.12));
  let c3=vec2<f32>(.45*asp+.10*asp*sin(t*.07+2.),.95+.05*cos(t*.10));
  c=mix(c,u.a1.rgb,.42*exp(-dot(p-c1,p-c1)/.09));
  c=mix(c,u.a2.rgb,.38*exp(-dot(p-c2,p-c2)/.10));
  c=mix(c,u.a1.rgb,.30*exp(-dot(p-c3,p-c3)/.12));
  return vec4<f32>(c,1.);}
struct I{@builtin(position) p:vec4<f32>,@location(0) lp:vec2<f32>,@location(1) hs:vec2<f32>,@location(2) c0:vec4<f32>,@location(3) c1:vec4<f32>,@location(4) q:vec4<f32>,@location(5) uv:vec4<f32>}
@vertex fn vi(@builtin(vertex_index) k:u32,@location(0) r:vec4<f32>,@location(1) c0:vec4<f32>,@location(2) c1:vec4<f32>,@location(3) q:vec4<f32>,@location(4) uv:vec4<f32>)->I{
  var cs=array<vec2<f32>,6>(vec2<f32>(-1.,-1.),vec2<f32>(1.,-1.),vec2<f32>(-1.,1.),vec2<f32>(-1.,1.),vec2<f32>(1.,-1.),vec2<f32>(1.,1.));
  let hs=r.zw*.5; var m=2.; if(q.z>.5){m=34.;} if(q.y>6.5){m=0.;}
  let c=cs[k]*(hs+vec2<f32>(m)); let w=r.xy+hs+c;
  var o:I; o.p=vec4<f32>(w.x/u.s.x*2.-1.,1.-w.y/u.s.y*2.,0.,1.); o.lp=c; o.hs=hs; o.c0=c0; o.c1=c1; o.q=q; o.uv=uv; return o;}
fn sdr(p:vec2<f32>,b:vec2<f32>,r:f32)->f32{let q=abs(p)-b+vec2<f32>(r);return length(max(q,vec2<f32>(0.)))+min(max(q.x,q.y),0.)-r;}
fn d2(v:vec2<f32>)->f32{return dot(v,v);}
fn heart(pp:vec2<f32>)->f32{var p=pp;p.x=abs(p.x);
  if(p.y+p.x>1.){return sqrt(d2(p-vec2<f32>(.25,.75)))-sqrt(2.)/4.;}
  return sqrt(min(d2(p-vec2<f32>(0.,1.)),d2(p-.5*max(p.x+p.y,0.))))*sign(p.x-p.y);}
@fragment fn fi(i:I)->@location(0) vec4<f32>{
  let k=i.q.y;
  if(k>6.5){let t=(i.lp/i.hs+vec2<f32>(1.))*.5; let a=textureSampleLevel(tex,smp,mix(i.uv.xy,i.uv.zw,t),0.).r; return vec4<f32>(i.c0.rgb,i.c0.a*a);}
  if(k<.5){
    let d=sdr(i.lp,i.hs,i.q.x); let aa=clamp(.5-d,0.,1.); let g=i.q.z>.5;
    let t=clamp(dot(i.lp/i.hs,vec2<f32>(.55,.83))*.5+.5,0.,1.);
    var f=mix(i.c0,i.c1,t); var sh=0.;
    if(g){
      let ds=sdr(i.lp-vec2<f32>(0.,10.),i.hs,i.q.x); sh=exp(-max(ds,0.)/15.)*.30*smoothstep(-.5,.5,d);
      let e=max(-d,0.); let rim=exp(-e/1.2)*step(0.,-d); let top=clamp(.5-i.lp.y/i.hs.y*.5,0.,1.);
      f=vec4<f32>(f.rgb+vec3<f32>(rim*top*.6)-vec3<f32>(rim*(1.-top)*.22),max(f.a,rim*.35));
    }
    let a=f.a*aa; let oa=a+sh*(1.-a);
    return vec4<f32>(f.rgb*a/max(oa,.0001),oa);
  }
  var p=i.lp/i.hs; var dd=1.;
  if(k>2.5&&k<3.5){p.x=-p.x;}
  let ky=floor(k+.5);
  if(ky==1.){dd=max(-(p.x+.42),(abs(p.y)-.58*(.62-p.x)/1.04)*.85);}
  else if(ky==2.){dd=min(sdr(p-vec2<f32>(-.3,0.),vec2<f32>(.15,.52),.07),sdr(p-vec2<f32>(.3,0.),vec2<f32>(.15,.52),.07));}
  else if(ky==3.||ky==4.){let tri=max(-(p.x+.5),(abs(p.y)-.55*(.42-p.x)/.92)*.85);let bar=sdr(p-vec2<f32>(.58,0.),vec2<f32>(.07,.55),.05);dd=min(tri,bar);}
  else{var h=heart(vec2<f32>(p.x*.55,.5-p.y*.5))*2.; if(i.q.w<.5){h=abs(h+.1)-.1;} dd=h;}
  return vec4<f32>(i.c0.rgb,i.c0.a*clamp(.5-dd*i.hs.x,0.,1.));
}
"#;

#[derive(Clone, Copy, Default)]
struct G { u0: f32, v0: f32, u1: f32, v1: f32, w: f32, h: f32, xmin: f32, ymin: f32, adv: f32 }
struct Atlas { tex: wgpu::Texture, m: HashMap<(char, u32), G>, x: u32, y: u32, rh: u32 }
impl Atlas {
    fn get(&mut self, f: &Font, q: &wgpu::Queue, ch: char, px: u32) -> G {
        if let Some(g) = self.m.get(&(ch, px)) { return *g; }
        let (m, bm) = f.rasterize(ch, px as f32);
        let (w, h) = (m.width as u32, m.height as u32);
        if self.x + w + 1 > ATL { self.x = 0; self.y += self.rh + 1; self.rh = 0; }
        if self.y + h > ATL { return G::default(); }
        if w > 0 && h > 0 {
            q.write_texture(
                wgpu::ImageCopyTexture { texture: &self.tex, mip_level: 0, origin: wgpu::Origin3d { x: self.x, y: self.y, z: 0 }, aspect: wgpu::TextureAspect::All },
                &bm,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(w), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
        let a = ATL as f32;
        let g = G { u0: self.x as f32 / a, v0: self.y as f32 / a, u1: (self.x + w) as f32 / a, v1: (self.y + h) as f32 / a, w: w as f32, h: h as f32, xmin: m.xmin as f32, ymin: m.ymin as f32, adv: m.advance_width };
        self.rh = self.rh.max(h); self.x += w + 1; self.m.insert((ch, px), g); g
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Act { None, Row(usize), Open, Toggle, Next, Prev, Pal, Dark, Like, Seek, Close }

struct Ui<'a> { v: Vec<I>, hits: Vec<(f32, f32, f32, f32, Act)>, at: &'a mut Atlas, q: &'a wgpu::Queue, f: &'a Option<Font>, la: u32, lb: u32 }
impl Ui<'_> {
    fn rr(&mut self, x: f32, y: f32, w: f32, h: f32, r: f32, c0: C, c1: C, glass: bool) {
        self.v.push(I { r: [x, y, w, h], c0, c1, p: [r.min(w.min(h) / 2.), 0., glass as u32 as f32, 0.], uv: [0.; 4] });
    }
    fn ic(&mut self, k: f32, cx: f32, cy: f32, s: f32, c: C, fill: bool) {
        self.v.push(I { r: [cx - s / 2., cy - s / 2., s, s], c0: c, c1: c, p: [0., k, 0., fill as u32 as f32], uv: [0.; 4] });
    }
    fn hit(&mut self, x: f32, y: f32, w: f32, h: f32, a: Act) { self.hits.push((x, y, w, h, a)); }
    fn tw(&self, t: &str, px: f32) -> f32 {
        match self.f { Some(f) => t.chars().map(|c| f.metrics(c, px.round()).advance_width).sum(), None => 0. }
    }
    fn txt(&mut self, t: &str, mut x: f32, base: f32, px: f32, c: C) {
        let Some(f) = self.f else { return };
        let p = px.round().max(1.) as u32;
        for ch in t.chars() {
            let g = self.at.get(f, self.q, ch, p);
            if g.w > 0. {
                self.v.push(I { r: [x + g.xmin, base - g.ymin - g.h, g.w, g.h], c0: c, c1: c, p: [0., 7., 0., 0.], uv: [g.u0, g.v0, g.u1, g.v1] });
            }
            x += g.adv;
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum M { None, Seek, Sheet, Scroll, Open }
struct Tch { x0: f32, y0: f32, x: f32, y: f32, moved: bool, mode: M, np0: f32, sc0: f32 }

#[cfg(feature = "audio")]
mod snd {
    use rodio::{OutputStream, OutputStreamHandle, Sink, Source};
    use std::time::Duration;
    struct Pad { f: [f32; 3], n: u64 }
    impl Iterator for Pad {
        type Item = f32;
        fn next(&mut self) -> Option<f32> {
            let t = self.n as f32 / 44100.; self.n += 1;
            let m = 0.6 + 0.4 * (t * 1.3).sin();
            Some(self.f.iter().map(|f| (t * f * 6.2831).sin()).sum::<f32>() * 0.07 * m)
        }
    }
    impl Source for Pad {
        fn current_frame_len(&self) -> Option<usize> { None }
        fn channels(&self) -> u16 { 1 }
        fn sample_rate(&self) -> u32 { 44100 }
        fn total_duration(&self) -> Option<Duration> { None }
    }
    pub struct Snd { _s: OutputStream, h: OutputStreamHandle, k: Option<Sink>, c: usize }
    impl Snd {
        pub fn new() -> Option<Snd> { let (s, h) = OutputStream::try_default().ok()?; Some(Snd { _s: s, h, k: None, c: usize::MAX }) }
        pub fn sync(&mut self, cur: usize, play: bool) {
            if self.c != cur || self.k.is_none() {
                if let Some(k) = self.k.take() { k.stop(); }
                let f = 110. * 2f32.powf(cur as f32 / 12.);
                if let Ok(k) = Sink::try_new(&self.h) { k.append(Pad { f: [f, f * 1.25, f * 1.5], n: 0 }); self.k = Some(k); }
                self.c = cur;
            }
            if let Some(k) = &self.k { if play { k.play() } else { k.pause() } }
        }
    }
}

struct St {
    cur: usize, play: bool, pos: f32, scr: f32, vel: f32, maxs: f32, np: f32, npt: f32, pal: usize, dark: bool,
    like: [bool; 12], t: f32, ev: f32, w: f32, h: f32, lp: (f32, f32, f32, f32), sx: (f32, f32),
    hits: Vec<(f32, f32, f32, f32, Act)>, down: Option<Tch>,
    #[cfg(feature = "audio")] snd: Option<snd::Snd>,
}
impl St {
    fn new() -> St {
        St { cur: 0, play: false, pos: 0., scr: 0., vel: 0., maxs: 0., np: 0., npt: 0., pal: 0, dark: true, like: [false; 12], t: 0., ev: 0.,
             w: 390., h: 800., lp: (0., 0., 0., 0.), sx: (0., 1.), hits: vec![], down: None,
             #[cfg(feature = "audio")] snd: snd::Snd::new() }
    }
    fn base(&self) -> (C, C) { if self.dark { (hx(0x0a0b14, 1.), hx(0x171b38, 1.)) } else { (hx(0xf1f2ff, 1.), hx(0xdfe4ff, 1.)) } }
    fn uni(&self) -> U {
        let (a1, a2) = PALS[self.pal]; let b = self.base();
        U { s: [self.w, self.h, self.t, self.dark as u32 as f32], a1: hx(a1, 1.), a2: hx(a2, 1.), b1: b.0, b2: b.1 }
    }
    fn next(&mut self) { self.cur = (self.cur + 1) % 12; self.pos = 0.; }
    fn hit(&self, x: f32, y: f32) -> Act {
        self.hits.iter().rev().find(|h| x >= h.0 && x <= h.0 + h.2 && y >= h.1 && y <= h.1 + h.3).map(|h| h.4).unwrap_or(Act::None)
    }
    fn seek(&mut self, x: f32) {
        let (a, b) = self.sx; if b <= a { return; }
        self.pos = ((x - a) / (b - a)).clamp(0., 1.) * SONGS[self.cur].2 as f32;
    }
    fn update(&mut self, dt: f32) {
        self.t += dt;
        self.ev += ((if self.play { 1. } else { 0. }) - self.ev) * (1. - (-dt * 8.).exp());
        if self.play { self.pos += dt; if self.pos >= SONGS[self.cur].2 as f32 { self.next(); } }
        let drag = matches!(&self.down, Some(t) if matches!(t.mode, M::Sheet | M::Open));
        if !drag { self.np += (self.npt - self.np) * (1. - (-dt * 14.).exp()); }
        let sc = matches!(&self.down, Some(t) if t.mode == M::Scroll);
        if !sc {
            self.scr += self.vel * dt; self.vel *= (-dt * 3.5).exp();
            if self.scr < 0. { self.scr = 0.; self.vel = 0.; }
            if self.scr > self.maxs { self.scr = self.maxs; self.vel = 0.; }
        }
        #[cfg(feature = "audio")]
        { if let Some(s) = self.snd.as_mut() { s.sync(self.cur, self.play); } }
    }
    fn down(&mut self, x: f32, y: f32) {
        let a = self.hit(x, y); let (lx, ly, lw, lh) = self.lp;
        let mode = if a == Act::Seek { M::Seek } else if self.np > 0.5 { M::Sheet } else if a == Act::Open { M::Open }
            else if x >= lx && x <= lx + lw && y >= ly && y <= ly + lh && self.np < 0.05 { M::Scroll } else { M::None };
        self.down = Some(Tch { x0: x, y0: y, x, y, moved: false, mode, np0: self.np, sc0: self.scr });
        self.vel = 0.;
        if mode == M::Seek { self.seek(x); }
    }
    fn mv(&mut self, x: f32, y: f32) {
        let h = self.h;
        let Some(t) = self.down.as_mut() else { return };
        let pv = t.y; t.x = x; t.y = y;
        if (x - t.x0).abs() + (y - t.y0).abs() > 14. { t.moved = true; }
        let (mode, y0, np0, sc0, mvd) = (t.mode, t.y0, t.np0, t.sc0, t.moved);
        let dy = y - pv;
        match mode {
            M::Seek => self.seek(x),
            M::Sheet if mvd => { self.np = (np0 - (y - y0) / (h * 0.75)).clamp(0., 1.); }
            M::Open if mvd => { self.np = ((y0 - y) / (h * 0.55)).clamp(0., 1.); }
            M::Scroll if mvd => { self.scr = (sc0 - (y - y0)).clamp(-60., self.maxs + 60.); self.vel = self.vel * 0.5 - dy * 27.; }
            _ => {}
        }
    }
    fn up(&mut self) {
        let Some(t) = self.down.take() else { return };
        match t.mode {
            M::Sheet if t.moved => { self.npt = if self.np > 0.62 { 1. } else { 0. }; }
            M::Open if t.moved => { self.npt = if self.np > 0.3 { 1. } else { 0. }; }
            M::Scroll if t.moved => {}
            M::Seek => {}
            _ => { if !t.moved { self.tap(t.x, t.y); } }
        }
    }
    fn tap(&mut self, x: f32, y: f32) {
        match self.hit(x, y) {
            Act::Row(i) => { self.cur = i; self.pos = 0.; self.play = true; }
            Act::Open => self.npt = 1.,
            Act::Close => self.npt = 0.,
            Act::Toggle => self.play = !self.play,
            Act::Next => self.next(),
            Act::Prev => { if self.pos > 3. { self.pos = 0.; } else { self.cur = (self.cur + 11) % 12; self.pos = 0.; } }
            Act::Pal => self.pal = (self.pal + 1) % 3,
            Act::Dark => self.dark = !self.dark,
            Act::Like => { let c = self.cur; self.like[c] = !self.like[c]; }
            _ => {}
        }
    }

    fn build(&mut self, u: &mut Ui) {
        let (w, h) = (self.w, self.h); let s = w / 390.; let top = 28. * s; let bot = 16. * s;
        let (a1, a2) = PALS[self.pal]; let (ca, cb) = (hx(a1, 1.), hx(a2, 1.));
        let d = self.dark; let bs = self.base(); let sg = SONGS[self.cur];
        let tx: C = if d { [1., 1., 1., 0.96] } else { [0.09, 0.09, 0.2, 0.95] };
        let t2 = op(tx, 0.6);
        let (g0, g1): (C, C) = if d { ([1., 1., 1., 0.17], [1., 1., 1., 0.06]) } else { ([1., 1., 1., 0.62], [1., 1., 1., 0.30]) };

        // header
        u.txt("Velora", 22. * s, top + 36. * s, 30. * s, tx);
        u.txt("Your library  ·  12 songs", 22. * s, top + 58. * s, 13. * s, t2);
        let bx = w - 20. * s - 44.* s; let by = top + 10. * s;
        u.rr(bx, by, 44. * s, 44. * s, 22. * s, g0, g1, true);
        u.rr(bx + 13. * s, by + 13. * s, 18. * s, 18. * s, 9. * s, ca, cb, false); u.hit(bx, by, 44. * s, 44. * s, Act::Pal);
        let bx2 = bx - 54. * s;
        u.rr(bx2, by, 44. * s, 44. * s, 22. * s, g0, g1, true);
        let r = if d { 16. * s } else { 10. * s };
        u.rr(bx2 + 22. * s - r / 2., by + 22. * s - r / 2., r, r, r / 2., tx, tx, false); u.hit(bx2, by, 44. * s, 44. * s, Act::Dark);
        u.rr(16. * s, top + 76. * s, w - 32. * s, 46. * s, 23. * s, g0, g1, true);
        u.txt("Search songs", 40. * s, top + 105. * s, 14. * s, t2);

        // list panel (scissored by la..lb)
        let (px, py, pw) = (10. * s, top + 134. * s, w - 20. * s);
        let mini_h = 72. * s; let my0 = h - bot - mini_h - 6. * s; let ph = my0 - 10. * s - py; let rh = 64. * s;
        u.rr(px, py, pw, ph, 30. * s, g0, g1, true);
        self.lp = (px, py, pw, ph); self.maxs = ((12. * rh + 16. * s) - ph).max(0.);
        u.la = u.v.len() as u32;
        for i in 0..12 {
            let y = py + 8. * s + i as f32 * rh - self.scr;
            if y + rh < py || y > py + ph { continue; }
            let so = SONGS[i];
            if i == self.cur { u.rr(px + 8. * s, y + 2. * s, pw - 16. * s, rh - 4. * s, 22. * s, hx(a1, 0.26), hx(a2, 0.12), false); }
            u.rr(px + 18. * s, y + 9. * s, 46. * s, 46. * s, 15. * s, hx(so.3, 1.), hx(so.4, 1.), false);
            u.txt(so.0, px + 78. * s, y + 29. * s, 16. * s, tx); u.txt(so.1, px + 78. * s, y + 49. * s, 13. * s, t2);
            let ds = mmss(so.2 as f32); let wd = u.tw(&ds, 13. * s);
            u.txt(&ds, px + pw - 20. * s - wd, y + 37. * s, 13. * s, t2);
            if i == self.cur {
                for k in 0..3 {
                    let hh = (3. + 13. * (self.t * 5. + k as f32 * 1.7).sin().abs() * self.ev) * s;
                    u.rr(px + pw - 54. * s - wd + k as f32 * 6. * s, y + 38. * s - hh, 3. * s, hh, 1.5 * s, ca, ca, false);
                }
            }
            let (y0, y1) = (y.max(py), (y + rh).min(py + ph));
            if y1 > y0 { u.hit(px, y0, pw, y1 - y0, Act::Row(i)); }
        }
        u.lb = u.v.len() as u32;

        // mini player
        let my = my0 + self.np * (mini_h + 40. * s); let (mx, mw) = (12. * s, w - 24. * s);
        if self.np < 0.995 {
            u.rr(mx, my, mw, mini_h, 30. * s, op(bs.0, 0.8), op(bs.1, 0.8), true); u.rr(mx, my, mw, mini_h, 30. * s, g0, g1, false);
            u.hit(mx, my, mw, mini_h, Act::Open);
            u.rr(mx + 12. * s, my + 12. * s, 48. * s, 48. * s, 16. * s, hx(sg.3, 1.), hx(sg.4, 1.), false);
            u.txt(sg.0, mx + 72. * s, my + 32. * s, 16. * s, tx); u.txt(sg.1, mx + 72. * s, my + 52. * s, 13. * s, t2);
            let pbx = mx + mw - 60. * s;
            u.rr(pbx, my + 12. * s, 48. * s, 48. * s, 24. * s, ca, cb, true);
            u.ic(if self.play { 2. } else { 1. }, pbx + 24. * s, my + 36. * s, 22. * s, [1.; 4], true); u.hit(pbx, my + 12. * s, 48. * s, 48. * s, Act::Toggle);
            let nx = pbx - 46. * s; u.ic(4., nx + 22. * s, my + 36. * s, 24. * s, tx, true); u.hit(nx, my + 12. * s, 46. * s, 48. * s, Act::Next);
            let f = (self.pos / sg.2 as f32).clamp(0., 1.);
            u.rr(mx + 26. * s, my + mini_h - 7. * s, (mw - 52. * s) * f, 2.5 * s, 1.25 * s, ca, cb, false);
        }

        // now playing sheet
        if self.np > 0.002 {
            let oy = (1. - self.np) * h; let yy = |v: f32| oy + v * s;
            u.rr(0., 0., w, h, 0., [0., 0., 0., 0.42 * self.np], [0., 0., 0., 0.42 * self.np], false);
            u.rr(0., oy, w, h + 40. * s, 36. * s, op(bs.0, 0.95), op(bs.1, 0.95), true);
            u.hit(0., oy, w, h, Act::None); u.hit(0., oy, w, 46. * s, Act::Close);
            u.rr(w / 2. - 20. * s, yy(10.), 40. * s, 4. * s, 2. * s, t2, t2, false);
            let l = "NOW PLAYING"; let lw = u.tw(l, 12. * s); u.txt(l, (w - lw) / 2., yy(46.), 12. * s, t2);
            let cs = (w - 76. * s).min(h * 0.38); let c2 = cs * (1. + 0.012 * (self.t * 3.2).sin() * self.ev);
            let (cx, cy) = ((w - c2) / 2., yy(64.) + (cs - c2) / 2.);
            u.rr(cx, cy, c2, c2, 34. * s, hx(sg.3, 1.), hx(sg.4, 1.), true);
            u.rr(cx + c2 * 0.29, cy + c2 * 0.29, c2 * 0.42, c2 * 0.42, c2 * 0.21, [1., 1., 1., 0.22], [1., 1., 1., 0.08], false);
            u.rr(cx + c2 * 0.44, cy + c2 * 0.44, c2 * 0.12, c2 * 0.12, c2 * 0.06, op(bs.0, 0.9), op(bs.1, 0.9), false);
            let ty = yy(64.) + cs + 44. * s;
            let a = u.tw(sg.0, 26. * s); u.txt(sg.0, (w - a) / 2., ty, 26. * s, tx);
            let b = u.tw(sg.1, 16. * s); u.txt(sg.1, (w - b) / 2., ty + 26. * s, 16. * s, t2);
            let liked = self.like[self.cur];
            u.ic(5., w - 44. * s, ty - 8. * s, 28. * s, if liked { [1., 0.3, 0.45, 1.] } else { tx }, liked); u.hit(w - 70. * s, ty - 34. * s, 52. * s, 52. * s, Act::Like);
            let sy = ty + 88. * s; let n = 28; let bw = (w - 64. * s) / n as f32;
            for i in 0..n {
                let f = i as f32;
                let amp = 0.18 + 0.82 * ((self.t * 2.9 + f * 0.55).sin() * (self.t * 1.7 + f * 0.31).cos()).abs();
                let hh = 4. * s + 44. * s * amp * self.ev;
                u.rr(32. * s + f * bw + bw * 0.15, sy - hh, bw * 0.7, hh, bw * 0.35, ca, cb, false);
            }
            let (x0, x1) = (32. * s, w - 32. * s); self.sx = (x0, x1);
            let ly = sy + 34. * s; let f = (self.pos / sg.2 as f32).clamp(0., 1.);
            u.rr(x0, ly, x1 - x0, 5. * s, 2.5 * s, op(tx, 0.2), op(tx, 0.2), false);
            u.rr(x0, ly, (x1 - x0) * f, 5. * s, 2.5 * s, ca, cb, false);
            u.rr(x0 + (x1 - x0) * f - 9. * s, ly + 2.5 * s - 9. * s, 18. * s, 18. * s, 9. * s, [1.; 4], [0.92, 0.92, 1., 1.], true);
            u.hit(x0 - 12. * s, ly - 24. * s, x1 - x0 + 24. * s, 54. * s, Act::Seek);
            u.txt(&mmss(self.pos), x0, ly + 28. * s, 12. * s, t2);
            let e = mmss(sg.2 as f32); let ew = u.tw(&e, 12. * s); u.txt(&e, x1 - ew, ly + 28. * s, 12. * s, t2);
            let cyc = ly + 88. * s;
            u.ic(3., w / 2. - 96. * s, cyc, 36. * s, tx, true); u.hit(w / 2. - 124. * s, cyc - 30. * s, 56. * s, 60. * s, Act::Prev);
            u.rr(w / 2. - 38. * s, cyc - 38. * s, 76. * s, 76. * s, 38. * s, ca, cb, true);
            u.ic(if self.play { 2. } else { 1. }, w / 2., cyc, 32. * s, [1.; 4], true); u.hit(w / 2. - 38. * s, cyc - 38. * s, 76. * s, 76. * s, Act::Toggle);
            u.ic(4., w / 2. + 96. * s, cyc, 36. * s, tx, true); u.hit(w / 2. + 68. * s, cyc - 30. * s, 56. * s, 60. * s, Act::Next);
        }
    }
}

struct Gfx {
    surf: wgpu::Surface<'static>, dev: wgpu::Device, q: wgpu::Queue, cfg: wgpu::SurfaceConfiguration,
    bgp: wgpu::RenderPipeline, ip: wgpu::RenderPipeline, bind: wgpu::BindGroup, ub: wgpu::Buffer, vb: wgpu::Buffer,
    at: Atlas, font: Option<Font>,
}
fn load_font() -> Option<Font> {
    for p in ["/system/fonts/Roboto-Regular.ttf", "/system/fonts/RobotoStatic-Regular.ttf", "/system/fonts/NotoSans-Regular.ttf", "/system/fonts/DroidSans.ttf", "/system/fonts/SourceSansPro-Regular.ttf"] {
        if let Ok(b) = std::fs::read(p) { if let Ok(f) = Font::from_bytes(b, FontSettings::default()) { return Some(f); } }
    }
    log::warn!("no system font found: text disabled");
    None
}
impl Gfx {
    fn new(win: Arc<Window>) -> Gfx {
        let sz = win.inner_size();
        let ins = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surf = ins.create_surface(win).unwrap();
        let ad = pollster::block_on(ins.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surf), force_fallback_adapter: false })).expect("adapter");
        let (dev, q) = pollster::block_on(ad.request_device(&wgpu::DeviceDescriptor {
            label: None, required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(ad.limits()), memory_hints: wgpu::MemoryHints::default(),
        }, None)).expect("device");
        let caps = surf.get_capabilities(&ad);
        let fmt = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let cfg = wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt, width: sz.width.max(1), height: sz.height.max(1),
            present_mode: wgpu::PresentMode::Fifo, desired_maximum_frame_latency: 2, alpha_mode: caps.alpha_modes[0], view_formats: vec![] };
        surf.configure(&dev, &cfg);

        let tex = dev.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: ATL, height: ATL, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[] });
        let tv = tex.create_view(&Default::default());
        let smp = dev.create_sampler(&wgpu::SamplerDescriptor { mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        let ub = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: 80, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let vb = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: (MAXI * 80) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
        ] });
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&tv) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&smp) },
        ] });
        let sm = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let mk = |vs: &str, fs: &str, inst: bool, blend: Option<wgpu::BlendState>| {
            let bufs = [wgpu::VertexBufferLayout { array_stride: 80, step_mode: wgpu::VertexStepMode::Instance, attributes: &ATTRS }];
            dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None, layout: Some(&pl),
                vertex: wgpu::VertexState { module: &sm, entry_point: vs, compilation_options: Default::default(), buffers: if inst { &bufs } else { &[] } },
                fragment: Some(wgpu::FragmentState { module: &sm, entry_point: fs, compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: fmt, blend, write_mask: wgpu::ColorWrites::ALL })] }),
                primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
            })
        };
        let bgp = mk("vbg", "fbg", false, None);
        let ip = mk("vi", "fi", true, Some(wgpu::BlendState::ALPHA_BLENDING));
        Gfx { surf, dev, q, cfg, bgp, ip, bind, ub, vb, at: Atlas { tex, m: HashMap::new(), x: 0, y: 0, rh: 0 }, font: load_font() }
    }
    fn resize(&mut self, w: u32, h: u32) { self.cfg.width = w.max(1); self.cfg.height = h.max(1); self.surf.configure(&self.dev, &self.cfg); }
    fn render(&mut self, v: &[I], la: u32, lb: u32, sc: (f32, f32, f32, f32), un: &U) {
        let n = v.len().min(MAXI) as u32; let (la, lb) = (la.min(n), lb.min(n));
        self.q.write_buffer(&self.ub, 0, bytemuck::bytes_of(un));
        self.q.write_buffer(&self.vb, 0, bytemuck::cast_slice(&v[..n as usize]));
        let fr = match self.surf.get_current_texture() { Ok(f) => f, Err(_) => { self.surf.configure(&self.dev, &self.cfg); return; } };
        let view = fr.texture.create_view(&Default::default());
        let mut enc = self.dev.create_command_encoder(&Default::default());
        let (w, h) = (self.cfg.width, self.cfg.height);
        let sx = (sc.0.max(0.) as u32).min(w - 1); let sy = (sc.1.max(0.) as u32).min(h - 1);
        let sw = (sc.2.max(1.) as u32).min(w - sx).max(1); let sh = (sc.3.max(1.) as u32).min(h - sy).max(1);
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None });
            rp.set_bind_group(0, &self.bind, &[]);
            rp.set_pipeline(&self.bgp); rp.draw(0..3, 0..1);
            rp.set_pipeline(&self.ip); rp.set_vertex_buffer(0, self.vb.slice(..));
            rp.draw(0..6, 0..la);
            rp.set_scissor_rect(sx, sy, sw, sh); rp.draw(0..6, la..lb);
            rp.set_scissor_rect(0, 0, w, h); rp.draw(0..6, lb..n);
        }
        self.q.submit(Some(enc.finish())); fr.present();
    }
}

struct App { win: Option<Arc<Window>>, g: Option<Gfx>, st: St, last: Instant }
impl App { fn new() -> App { App { win: None, g: None, st: St::new(), last: Instant::now() } } }
impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(el.create_window(Window::default_attributes()).unwrap());
        let sz = w.inner_size(); self.st.w = sz.width as f32; self.st.h = sz.height as f32;
        self.g = Some(Gfx::new(w.clone())); self.last = Instant::now(); w.request_redraw(); self.win = Some(w);
    }
    fn suspended(&mut self, _: &ActiveEventLoop) { self.g = None; self.win = None; }
    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        match ev {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => { if let Some(g) = self.g.as_mut() { g.resize(s.width, s.height); } self.st.w = s.width as f32; self.st.h = s.height as f32; }
            WindowEvent::Touch(t) => {
                let (x, y) = (t.location.x as f32, t.location.y as f32);
                match t.phase { TouchPhase::Started => self.st.down(x, y), TouchPhase::Moved => self.st.mv(x, y), _ => self.st.up() }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now(); let dt = (now - self.last).as_secs_f32().min(0.05); self.last = now;
                self.st.update(dt);
                if let Some(g) = self.g.as_mut() {
                    let (v, hits, la, lb) = {
                        let mut ui = Ui { v: Vec::with_capacity(2048), hits: Vec::new(), at: &mut g.at, q: &g.q, f: &g.font, la: 0, lb: 0 };
                        self.st.build(&mut ui); (ui.v, ui.hits, ui.la, ui.lb)
                    };
                    self.st.hits = hits; g.render(&v, la, lb, self.st.lp, &self.st.uni());
                }
                if let Some(w) = &self.win { w.request_redraw(); }
            }
            _ => {}
        }
    }
}

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;
    android_logger::init_once(android_logger::Config::default().with_max_level(log::LevelFilter::Info));
    let el = EventLoop::builder().with_android_app(app).build().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
