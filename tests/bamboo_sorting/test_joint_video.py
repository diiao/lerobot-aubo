import io
import json
import shutil
import subprocess
import time

import numpy as np
import pytest
from lerobot.bamboo_sorting.joint_video import TrialVideoRecorder


@pytest.mark.skipif(shutil.which('ffmpeg') is None,reason='ffmpeg unavailable')
def test_two_camera_video_real_encoding_and_timestamps(tmp_path):
    class Camera:
        def __init__(self,color):self.frame=np.full((480,640,3),color,dtype=np.uint8)
        def read_latest_with_timestamp(self,**kwargs):return self.frame,time.perf_counter()
    recorder=TrialVideoRecorder({'global_rgb':Camera([255,0,0]),'grasp_rgb':Camera([0,0,255])},tmp_path)
    recorder.start()
    try:time.sleep(.35)
    finally:result=recorder.close()
    assert result['complete'] and not result['errors']
    for name in ('global_rgb','grasp_rgb'):
        path=tmp_path/'videos'/f'{name}.mp4'
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
        frame=np.frombuffer(raw,dtype=np.uint8).reshape(480,640,3)
        assert frame[:,:,0 if name=='global_rgb' else 2].mean()>240
        records=[json.loads(x) for x in (tmp_path/'videos'/f'{name}_frames.jsonl').read_text().splitlines()]
        assert len(records)>=5
        assert [r['frame_index'] for r in records]==list(range(len(records)))
        assert records[-1]['video_time_s']==(len(records)-1)/25


def test_dropped_queue_ticks_preserve_video_duration_and_are_marked(tmp_path):
    class Pipe(io.BytesIO):
        def close(self):pass
    class Process:
        stdin=Pipe()
        def poll(self):return 0
        def wait(self,**kwargs):return 0
    recorder=TrialVideoRecorder({'global_rgb':None},tmp_path)
    recorder.output.mkdir();recorder.started_at=10.;recorder.final_tick=4
    recorder.processes['global_rgb']=Process()
    image=np.zeros((480,640,3),dtype=np.uint8)
    recorder.jobs['global_rgb'].put((0,image,10.))
    recorder.jobs['global_rgb'].put((3,image,10.12))
    recorder.capture_done.set();recorder._encode('global_rgb')
    frames=[json.loads(x) for x in (recorder.output/'global_rgb_frames.jsonl').read_text().splitlines()]
    assert len(frames)==5
    assert [r['filled_missing_tick'] for r in frames]==[False,True,True,False,True]
    assert [r['source_timestamp'] for r in frames]==[10.,10.,10.,10.12,10.12]
    assert recorder.stats['global_rgb']['repeated_frames']==3
    assert not recorder.errors


@pytest.mark.skipif(shutil.which('ffmpeg') is None,reason='ffmpeg unavailable')
def test_no_frames_reports_incomplete_and_cleans_up(tmp_path):
    class Camera:
        def read_latest_with_timestamp(self,**kwargs):raise TimeoutError('stale')
    recorder=TrialVideoRecorder({'global_rgb':Camera()},tmp_path)
    recorder.start()
    try:time.sleep(.06)
    finally:result=recorder.close()
    assert not result['complete'] and result['errors']
    assert recorder.processes['global_rgb'].poll() is not None
    assert result['cameras']['global_rgb']['read_misses']>0
