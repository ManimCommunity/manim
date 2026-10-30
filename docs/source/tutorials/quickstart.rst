<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Penyebab Gempa Bumi (30 detik)</title>
<style>
:root{--bg:#f3efe8;--fg:#1d1a17;--muted:#6b645b;--btn:#1d1a17;--btnfg:#f3efe8;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px);color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14120f;--fg:#efe9df;--muted:#a39b8f;--btn:#efe9df;--btnfg:#14120f;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#14120f;--fg:#efe9df;--muted:#a39b8f;--btn:#efe9df;--btnfg:#14120f;color-scheme:dark}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:"Trebuchet MS",Verdana,system-ui,sans-serif;display:flex;flex-direction:column;align-items:center;padding:16px}
.box{width:100%;max-width:960px}
h1{font-size:1.15rem;margin:0 0 10px;font-weight:700}
canvas{width:100%;height:auto;aspect-ratio:16/9;display:block;border-radius:8px;background:#0b1020}
.ctl{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
button{background:var(--btn);color:var(--btnfg);border:0;border-radius:6px;padding:8px 14px;font:inherit;cursor:pointer}
button:focus-visible,input:focus-visible{outline:3px solid #e0732a;outline-offset:2px}
input[type=range]{flex:1;min-width:140px}
label{color:var(--muted);font-size:.9rem;display:flex;gap:6px;align-items:center}
</style>
</head>
<body>
<div class="box">
<h1>Penyebab Gempa Bumi: animasi 30 detik</h1>
<canvas id="c" width="1920" height="1080" aria-label="Animasi penyebab gempa bumi"></canvas>
<div class="ctl">
<button id="pp">Jeda</button>
<button id="rs">Ulang</button>
<input id="sk" type="range" min="0" max="30" step="0.05" value="0" aria-label="Posisi video">
<label><input id="vo" type="checkbox"> Suara narasi</label>
</div>
</div>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const W=960,H=540,DUR=30;
const clamp=(x,a=0,b=1)=>Math.min(b,Math.max(a,x));
const ease=x=>{x=clamp(x);return x*x*(3-2*x)};
const FONT='"Trebuchet MS",Verdana,sans-serif';

const CAPS=[
 [0,6,"Bumi tersusun dari kerak, mantel, dan inti. Kerak tidak utuh, tetapi terpecah menjadi lempeng-lempeng."],
 [6,13,"Panas dari inti menggerakkan arus di dalam mantel, sehingga lempeng bergerak beberapa sentimeter per tahun."],
 [13,20.4,"Lempeng bertumbukan, saling menjauh, atau bergesekan. Batuan tertahan dan energi terus menumpuk."],
 [20.4,26,"Saat batuan tak kuat lagi menahan tekanan, patahan terjadi tiba-tiba. Energi dilepas sebagai gelombang seismik: gempa!"],
 [26,30,"Selain gerakan lempeng, gempa juga bisa dipicu hal lain. Kenali risikonya, siapkan mitigasi."]
];

function txt(s,x,y,size,color,align='left',weight='normal',alpha=1){
 ctx.globalAlpha=alpha;ctx.fillStyle=color;ctx.font=`${weight} ${size}px ${FONT}`;ctx.textAlign=align;ctx.fillText(s,x,y);ctx.globalAlpha=1;
}
function wrap(s,x,y,maxW,lh,size,color){
 ctx.font=`${size}px ${FONT}`;ctx.textAlign='left';ctx.fillStyle=color;
 let line='',yy=y;
 for(const w of s.split(' ')){
  const t=line?line+' '+w:w;
  if(ctx.measureText(t).width>maxW&&line){ctx.fillText(line,x,yy);line=w;yy+=lh}else line=t;
 }
 ctx.fillText(line,x,yy);
}

/* ---------- Adegan 1: struktur Bumi (0-6 dtk) ---------- */
function sceneEarth(t){
 const cx=330,cy=235,R=170*ease(t/1);
 const layer=(r,col,start)=>{ctx.globalAlpha=clamp((t-start)/0.6);ctx.fillStyle=col;ctx.beginPath();ctx.arc(cx,cy,R*r,0,7);ctx.fill();ctx.globalAlpha=1};
 ctx.fillStyle='#2f7d5b';ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.fill();
 layer(0.93,'#e0732a',1);layer(0.62,'#f2b134',2);layer(0.32,'#fff1a8',2.8);
 if(t>3.8){
  ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.globalAlpha=clamp((t-3.8)/0.8);
  [0.3,1.2,2.1,3.3,4.4,5.4].forEach(a=>{ctx.beginPath();ctx.moveTo(cx+R*0.93*Math.cos(a),cy+R*0.93*Math.sin(a));ctx.lineTo(cx+R*Math.cos(a),cy+R*Math.sin(a));ctx.stroke()});
  ctx.globalAlpha=1;
 }
 const L=[[0.8,0.97,-0.7,'Kerak bumi',140,'#7ed6a8','Terpecah menjadi lempeng'],[1.8,0.78,-0.2,'Mantel',215,'#ffa45c','Batuan panas, kental'],[2.6,0.47,0.3,'Inti luar',290,'#ffd36b','Logam cair'],[3.2,0.16,0.9,'Inti dalam',365,'#fff1a8','Logam padat, sangat panas']];
 L.forEach(([s,r,a,n,y,c,d])=>{
  const al=clamp((t-s)/0.5);if(al<=0)return;
  const px=cx+R*r*Math.cos(a),py=cy+R*r*Math.sin(a);
  ctx.globalAlpha=al;ctx.strokeStyle=c;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(600,y-8);ctx.lineTo(620,y-8);ctx.stroke();
  ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(px,py,4,0,7);ctx.fill();ctx.globalAlpha=1;
  txt(n,630,y-2,24,c,'left','bold',al);txt(d,630,y+24,17,'#c9d1e6','left','normal',al);
 });
}

/* ---------- Adegan 2: arus konveksi (6-13 dtk) ---------- */
function sceneConv(t){
 const lt=t-6,mv=lt*6;
 const g=ctx.createLinearGradient(0,250,0,470);g.addColorStop(0,'#c4561f');g.addColorStop(1,'#ffb347');
 ctx.fillStyle=g;ctx.fillRect(60,250,840,220);
 ctx.fillStyle='#7a6a58';ctx.fillRect(80+mv,195,390,55);ctx.fillRect(490-mv,195,390,55);
 ctx.fillStyle='#4a3f33';ctx.fillRect(80+mv,195,390,10);ctx.fillRect(490-mv,195,390,10);
 [[270,1],[690,-1]].forEach(([cx,dir])=>{
  ctx.strokeStyle='rgba(255,255,255,.35)';ctx.lineWidth=2;ctx.beginPath();ctx.ellipse(cx,360,170,70,0,0,7);ctx.stroke();
  for(let i=0;i<12;i++){
   const a=dir*(lt*1.1)+i*Math.PI/6;
   const x=cx+170*Math.cos(a),y=360+70*Math.sin(a);
   ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(x,y,5,0,7);ctx.fill();
  }
 });
 const al=ease(lt/1);
 txt('Lempeng tektonik',480,180,22,'#e8dccb','center','bold',al);
 txt('Arus konveksi mantel',270,375,18,'#fff','center','normal',al);
 txt('Arus konveksi mantel',690,375,18,'#fff','center','normal',al);
 txt('Panas dari inti Bumi',480,458,18,'#3a1a08','center','bold',al);
 ctx.globalAlpha=al;ctx.fillStyle='#fff';
 [[450,150,1],[510,150,-1]].forEach(([x,y,d])=>{ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+d*30,y);ctx.moveTo(x+d*30,y);ctx.lineTo(x+d*22,y-7);ctx.lineTo(x+d*22,y+7);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+d*24,y);ctx.stroke()});
 ctx.globalAlpha=1;
}

/* ---------- Adegan 3-4: tekanan lalu patahan (13-26 dtk) ---------- */
const RUP=20.4,FX=480,FY=285;
function sceneFault(t){
 const u=clamp((t-13)/(RUP-13)),r=ease((t-RUP)/0.5);
 const A=48*(t<RUP?0.15+0.85*u:1);
 const f=x=>{const d=x-FX,s=Math.sign(d);return -A*s*(1-(1-r)*Math.exp(-Math.abs(d)/110))};
 const al=ease((t-13)/0.6);
 ctx.fillStyle='#3a3226';ctx.fillRect(60,110,840,330);
 const lines=[[150,'#5a4d3c',2],[215,'#5a4d3c',2],[FY,'#ffd23f',6],[355,'#5a4d3c',2],[410,'#5a4d3c',2]];
 lines.forEach(([y0,c,w])=>{
  ctx.strokeStyle=c;ctx.lineWidth=w;
  [[60,FX-1],[FX+1,900]].forEach(([a,b])=>{ctx.beginPath();for(let x=a;x<=b;x+=6){const y=y0+f(x);x===a?ctx.moveTo(x,y):ctx.lineTo(x,y)}ctx.stroke()});
 });
 ctx.setLineDash([10,8]);ctx.strokeStyle=r>0?'#ff5a4d':'#9aa0ad';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(FX,110);ctx.lineTo(FX,440);ctx.stroke();ctx.setLineDash([]);
 txt('Patahan',FX,102,18,'#ff9b91','center','bold',al);
 txt('Lempeng A',200,132,18,'#e8dccb','center','normal',al);
 txt('Lempeng B',760,430,18,'#e8dccb','center','normal',al);
 const arrow=(x,y,d)=>{const p=(t<RUP?1:0)*Math.sin(t*6)*3;ctx.globalAlpha=al*(t<RUP?1:0.3);ctx.fillStyle='#fff';ctx.strokeStyle='#fff';ctx.lineWidth=6;ctx.beginPath();ctx.moveTo(x,y-d*26+p);ctx.lineTo(x,y+d*20+p);ctx.stroke();ctx.beginPath();ctx.moveTo(x,y+d*38+p);ctx.lineTo(x-12,y+d*18+p);ctx.lineTo(x+12,y+d*18+p);ctx.fill();ctx.globalAlpha=1};
 arrow(240,175,1);arrow(720,395,-1);
 /* meter tekanan */
 const mv=t<RUP?u:1-r;
 txt('Tekanan batuan',700,60,16,'#c9d1e6','left','normal',al);
 ctx.globalAlpha=al;ctx.fillStyle='#232a40';ctx.fillRect(700,68,200,14);
 ctx.fillStyle=`hsl(${120-120*mv},80%,50%)`;ctx.fillRect(700,68,200*mv,14);ctx.globalAlpha=1;
 /* gelombang seismik */
 if(t>RUP){
  const tt=t-RUP;
  if(tt<0.5){const g=ctx.createRadialGradient(FX,FY,0,FX,FY,140);g.addColorStop(0,`rgba(255,255,255,${1-tt/0.5})`);g.addColorStop(1,'rgba(255,255,255,0)');ctx.fillStyle=g;ctx.fillRect(0,0,W,H)}
  for(let k=0;k<4;k++){
   const rr=(tt-k*0.5)*230;if(rr<=0||rr>720)continue;
   ctx.strokeStyle=k%2?'#ffb347':'#ff5a4d';ctx.globalAlpha=1-rr/720;ctx.lineWidth=5;ctx.beginPath();ctx.arc(FX,FY,rr,0,7);ctx.stroke();ctx.globalAlpha=1;
  }
  ctx.fillStyle='#ff3b30';ctx.beginPath();ctx.arc(FX,FY,9,0,7);ctx.fill();
  txt('Pusat gempa',FX+16,FY-14,18,'#fff','left','bold',ease(tt/0.5));
 }
}

/* ---------- Adegan 5: ringkasan (26-30 dtk) ---------- */
function sceneSum(t){
 const items=[['Tektonik','Sekitar 90% gempa: gerakan dan patahan lempeng','#ff7a59'],['Vulkanik','Magma naik dan letusan gunung api','#ffb347'],['Runtuhan','Longsor besar atau ambruknya gua dan tambang','#7ed6a8'],['Aktivitas manusia','Waduk besar, injeksi limbah, ledakan','#7aa7ff']];
 txt('Apa saja penyebab gempa?',480,100,30,'#fff','center','bold',ease((t-26)/0.5));
 items.forEach(([n,d,c],i)=>{
  const a=ease((t-26.3-i*0.6)/0.5);if(a<=0)return;
  const x=90+(i%2)*400,y=130+Math.floor(i/2)*140+(1-a)*20;
  ctx.globalAlpha=a;ctx.fillStyle='#1a2240';ctx.beginPath();ctx.roundRect(x,y,380,120,10);ctx.fill();
  ctx.fillStyle=c;ctx.fillRect(x,y+14,6,92);ctx.globalAlpha=1;
  txt(n,x+24,y+44,26,c,'left','bold',a);wrap(d,x+24,y+78,330,22,18,`rgba(220,226,240,${a})`);
 });
}

/* ---------- Bingkai utama ---------- */
function draw(t){
 ctx.setTransform(2,0,0,2,0,0);
 const bg=ctx.createLinearGradient(0,0,0,H);bg.addColorStop(0,'#0b1020');bg.addColorStop(1,'#1b1f3a');
 ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);
 ctx.save();
 if(t>RUP&&t<24.5){const k=1-(t-RUP)/4.1;ctx.translate((Math.random()-.5)*12*k,(Math.random()-.5)*12*k)}
 if(t<6)sceneEarth(t);else if(t<13)sceneConv(t);else if(t<26)sceneFault(t);else sceneSum(t);
 ctx.restore();
 [6,13,26].forEach(b=>{const a=clamp(1-Math.abs(t-b)/0.35);if(a>0){ctx.fillStyle=`rgba(11,16,32,${a})`;ctx.fillRect(0,0,W,H)}});
 txt('Penyebab Gempa Bumi',28,44,26,'#fff','left','bold');
 const s=Math.floor(t);txt(`00:${String(s).padStart(2,'0')} / 00:30`,932,44,18,'#c9d1e6','right');
 const cap=CAPS.find(c=>t>=c[0]&&t<c[1])||CAPS[4];
 ctx.fillStyle='rgba(0,0,0,.62)';ctx.fillRect(0,448,W,92);
 wrap(cap[2],32,484,896,28,22,'#fff');
 ctx.fillStyle='#ff7a59';ctx.fillRect(0,H-5,W*t/DUR,5);
}

/* ---------- Pemutar ---------- */
let t=0,playing=true,last=performance.now(),lastCap=-1;
const pp=document.getElementById('pp'),sk=document.getElementById('sk'),vo=document.getElementById('vo');
function speak(){
 if(!vo.checked||!('speechSynthesis' in window))return;
 const i=CAPS.findIndex(c=>t>=c[0]&&t<c[1]);if(i===lastCap||i<0)return;lastCap=i;
 speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(CAPS[i][2]);u.lang='id-ID';u.rate=1.08;speechSynthesis.speak(u);
}
function setPlaying(p){playing=p;pp.textContent=p?'Jeda':(t>=DUR?'Putar lagi':'Putar');last=performance.now();if(!p&&'speechSynthesis' in window)speechSynthesis.cancel()}
function loop(now){
 if(playing){t+=(now-last)/1000;if(t>=DUR){t=DUR;setPlaying(false)}}
 last=now;sk.value=t;draw(Math.min(t,DUR-0.001));if(playing)speak();
 requestAnimationFrame(loop);
}
pp.onclick=()=>{if(t>=DUR)t=0;lastCap=-1;setPlaying(!playing)};
document.getElementById('rs').onclick=()=>{t=0;lastCap=-1;setPlaying(true)};
sk.oninput=()=>{t=parseFloat(sk.value);lastCap=-1};
vo.onchange=()=>{lastCap=-1;if(!vo.checked&&'speechSynthesis' in window)speechSynthesis.cancel()};
requestAnimationFrame(loop);
</script>
</body>
</html>
