"""Read-only localhost preview in a separate process; never opens a camera."""
import base64
import json
import multiprocessing as mp
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from ..policy.joint_target import outline_target_images
from ..mask_codec import decode_mask

PAGE = '''<!doctype html><meta charset="utf-8"><title>AUBO 目标预测预览</title>
<style>body{background:#15191e;color:#eee;font:18px sans-serif;margin:24px}img{max-width:100%;border:1px solid #555}.old{color:#ffb45a}small{color:#bbb}</style>
<h2>模型输入与目标轮廓</h2><p id="status">等待预测</p><img id="frame"><p><small>左：该次预测的全局图像，紫色为模型输入轮廓，黄色框仅用于显示。右：同次请求腕部图像。<br>这是最近一次预测的观测，不是无延迟摄像头画面。搬运阶段模型使用原图，不保留接近阶段的框。此页面不发送运动命令。</small></p>
<script>
async function refresh(){let controller=new AbortController();let timer=setTimeout(()=>controller.abort(),1500);try{let r=await fetch('/snapshot',{cache:'no-store',signal:controller.signal});if(!r.ok)throw Error(r.status);let d=await r.json();let s=document.getElementById('status');s.className=(d.age_ms>1500?'old':'');s.textContent=`目标 ${d.target_id||'-'} · ${d.phase||'-'} · ${d.status} · 预测 ${d.index??'-'} · 图像年龄 ${d.age_ms==null?'-':Math.round(d.age_ms)} ms · 目标 ${d.pixels??'-'} 像素`;let im=document.getElementById('frame');if(d.jpeg){im.src='data:image/jpeg;base64,'+d.jpeg;im.hidden=false}else{im.hidden=true}}catch(e){let s=document.getElementById('status');s.className='old';s.textContent='预览已断开，画面为旧记录；请查看运行终端'}finally{clearTimeout(timer);setTimeout(refresh,250)}}refresh();
</script>'''


def render_packet(images, evidence, *, index, sensor_times, status='prediction accepted'):
    """Render only the RGB belonging to this result, never newer camera pixels."""
    phase = evidence['phase']
    marked = images['global_rgb'].copy()
    pixels = 0
    if phase == 'approach' and evidence.get('mask_png'):
        mask = decode_mask(evidence['mask_png'], allow_empty=True)
        pixels = int(mask.sum())
        if pixels:
            marked = outline_target_images(images, mask)['target_global_rgb']
            y, x = np.nonzero(mask)
            cv2.rectangle(marked, (int(x.min()), int(y.min())),
                          (int(x.max()), int(y.max())), (255, 220, 0), 1)
    pair = np.hstack([marked, images['grasp_rgb']])
    ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(pair, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError('preview encoding failed')
    return dict(target_id=evidence['target_id'], phase=phase, index=index,
        source_time=min(sensor_times[:2]), status=status, pixels=pixels,
        jpeg=base64.b64encode(jpeg).decode('ascii'))


def _serve(inbox, ready):
    current = dict(status='waiting for prediction', jpeg=None)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            if self.path == '/':
                data, kind = PAGE.encode(), 'text/html; charset=utf-8'
            elif self.path == '/snapshot':
                row = dict(current)
                row['age_ms'] = (time.perf_counter()-row['source_time'])*1000 if row.get('source_time') else None
                data, kind = json.dumps(row).encode(), 'application/json'
            else:
                self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ready.send(f'http://127.0.0.1:{server.server_port}/'); ready.close()
    try:
        while True:
            packet = inbox.get()
            if packet is None:break
            try:
                current = render_packet(**packet) if 'images' in packet else packet
            except Exception as exc:
                current = dict(status=f'preview error: {exc}', jpeg=None)
    finally:
        server.shutdown(); server.server_close()


class LiveTargetPreview:
    def __init__(self):
        context = mp.get_context('spawn')
        self.inbox = context.Queue(maxsize=2)
        receiver, sender = context.Pipe(duplex=False)
        self.process = context.Process(target=_serve, args=(self.inbox, sender), daemon=True)
        self.process.start(); sender.close()
        try:
            if not receiver.poll(10):raise RuntimeError('preview startup timed out')
            self.url = receiver.recv()
        except BaseException:
            self.close(); raise
        finally:receiver.close()

    def publish(self, **packet):
        # Never make the servo wait for a browser, JPEG encoder or slow client.
        try:self.inbox.put_nowait(packet)
        except queue.Full:pass

    def close(self):
        try:self.inbox.put_nowait(None)
        except queue.Full:pass
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate(); self.process.join(timeout=1)
        self.inbox.cancel_join_thread(); self.inbox.close()
