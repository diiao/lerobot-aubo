"""Non-blocking recording of existing camera streams; never opens a camera."""
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time


class TrialVideoRecorder:
    """25-fps wall-clock video, with source timestamps and explicit repeats.

    One sampler reads the cameras' cached RGB frames; bounded per-camera queues
    feed independent encoder threads. No encoding or pipe writes on servo thread.
    """
    fps=25

    def __init__(self,cameras,output):
        self.cameras=dict(cameras);self.output=Path(output)/'videos'
        self.stop_event=threading.Event();self.capture_done=threading.Event()
        self.jobs={name:queue.Queue(maxsize=8) for name in cameras}
        self.errors=[];self.stats={name:{'encoded_frames':0,'repeated_frames':0,'queue_drops':0,'read_misses':0} for name in cameras}
        self.processes={};self.logs={};self.workers=[];self.sampler=None;self.started=False
        self.final_tick=-1

    def start(self):
        if not self.cameras:raise ValueError('no cameras to record')
        binary=shutil.which('ffmpeg')
        if binary is None:raise RuntimeError('ffmpeg is required for recording')
        self.output.mkdir(exist_ok=False)
        try:
            for name in self.cameras:
                log=(self.output/f'{name}_encoder.log').open('w');self.logs[name]=log
                self.processes[name]=subprocess.Popen([binary,'-nostdin','-hide_banner','-loglevel','error','-n',
                    '-f','rawvideo','-pix_fmt','rgb24','-video_size','640x480','-framerate',str(self.fps),
                    '-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23','-threads','1',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(self.output/f'{name}.mp4')],
                    stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
            self.started_at=time.perf_counter();self.started=True
            for name in self.cameras:
                t=threading.Thread(target=self._encode,args=(name,),daemon=True,name=f'video-{name}')
                self.workers.append(t);t.start()
            self.sampler=threading.Thread(target=self._sample,daemon=True,name='video-sampler');self.sampler.start()
        except BaseException:
            self.close();raise

    def _sample(self):
        try:
            while not self.stop_event.is_set():
                now=time.perf_counter();tick=int((now-self.started_at)*self.fps)
                for name,camera in self.cameras.items():
                    try:
                        frame,stamp=camera.read_latest_with_timestamp(max_age_ms=100)
                        if frame.shape!=(480,640,3) or str(frame.dtype)!='uint8':raise ValueError('unexpected RGB frame schema')
                    except TimeoutError:
                        self.stats[name]['read_misses']+=1;continue
                    # Cached frames are replaced, not modified, by OpenCVCamera.
                    # A reference is sufficient; no copy under its frame lock.
                    try:self.jobs[name].put_nowait((tick,frame,stamp))
                    except queue.Full:self.stats[name]['queue_drops']+=1
                delay=self.started_at+(tick+1)/self.fps-time.perf_counter()
                self.stop_event.wait(max(0.,delay))
        except BaseException as exc:
            self.errors.append(f'capture: {type(exc).__name__}: {exc}')
        finally:
            self.final_tick=max(0,math.ceil((time.perf_counter()-self.started_at)*self.fps)-1)
            self.capture_done.set()

    def _encode(self,name):
        process=self.processes[name];last=None;last_stamp=None;next_index=0
        stats=self.stats[name]
        try:
            with (self.output/f'{name}_frames.jsonl').open('w') as timestamps:
                def emit(frame,stamp,filled):
                    nonlocal next_index,last_stamp
                    process.stdin.write(frame.tobytes())
                    repeated=stamp==last_stamp
                    timestamps.write(json.dumps({'frame_index':next_index,'video_time_s':next_index/self.fps,
                        'scheduled_time':self.started_at+next_index/self.fps,'source_timestamp':stamp,
                        'repeated_source':repeated,'filled_missing_tick':filled})+'\n')
                    stats['encoded_frames']+=1;stats['repeated_frames']+=int(repeated)
                    last_stamp=stamp;next_index+=1
                while not self.capture_done.is_set() or not self.jobs[name].empty():
                    try:tick,frame,stamp=self.jobs[name].get(timeout=.1)
                    except queue.Empty:continue
                    if tick<next_index:continue
                    filler=last if last is not None else (frame,stamp)
                    while next_index<tick:emit(*filler,True)
                    emit(frame,stamp,False);last=(frame,stamp)
                if last is not None:
                    while next_index<=self.final_tick:emit(*last,True)
                else:raise RuntimeError('no frames captured')
            process.stdin.close()
            if process.wait(timeout=10)!=0:raise RuntimeError('ffmpeg failed; inspect encoder log')
        except BaseException as exc:
            self.errors.append(f'{name}: {type(exc).__name__}: {exc}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:process.kill();process.wait()

    def close(self):
        self.stop_event.set()
        if self.sampler is not None:self.sampler.join(timeout=2)
        else:self.capture_done.set()
        if self.sampler is not None and self.sampler.is_alive():self.errors.append('capture thread shutdown timed out')
        for worker in self.workers:worker.join(timeout=15)
        for name,process in self.processes.items():
            if process.poll() is None:
                self.errors.append(f'{name}: encoder shutdown timed out')
                process.terminate()
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            if process.stdin and not process.stdin.closed:process.stdin.close()
        for worker in self.workers:
            if worker.is_alive():worker.join(timeout=2)
        for log in self.logs.values():log.close()
        result={'fps':self.fps,'resolution':[640,480],'cameras':self.stats,'errors':self.errors,
            'complete':self.started and not self.errors,'started_at':getattr(self,'started_at',None),
            'timing':'constant 25 fps; repeated/missed source frames recorded in sidecars'}
        if self.output.exists():(self.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
