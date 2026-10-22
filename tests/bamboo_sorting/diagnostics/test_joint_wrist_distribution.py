from tests.bamboo_sorting.paths import REPO_ROOT
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

path=REPO_ROOT/'examples/phone_to_auboi10/review_joint_wrist_distribution.py'
spec=importlib.util.spec_from_file_location('wrist_distribution',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def rows_fixture(tmp_path):
    rows=[]
    for group in ('expert','model'):
        for i in range(12):
            p=tmp_path/f'{group}-{i}.png';Image.new('RGB',(640,480),'white').save(p)
            rows.append(dict(id=f'{group}-{i}',episode_id=f'{group}-run-{i%3}',group=group,image=str(p),
                measurement=dict(valid=i!=0,angle_deg=(i-6)*3,axis_span_px=180+i,reason='missing_or_occluded'),
                gap_line=[[320,80],[320,400]],wrist_source_monotonic_s=100+i,sequence_start_monotonic_s=100,
                accepted_close_monotonic_s=None if group=='model' else 113.))
    return rows


def test_distribution_retains_invalid_frames_and_counts_independent_runs(tmp_path):
    rows=rows_fixture(tmp_path);unique=m.unique_rows(dict(samples=rows+[rows[0]]))
    assert len(unique)==24
    result=m.distribution_summary(unique)
    assert result['groups']['expert']['independent_sources']==3
    assert result['groups']['expert']['valid_frames']==11
    assert any(not t['valid'] for e in result['episodes'] for t in e['timeline'])
    assert result['measurement_error_deg'] is None and not result['regression_run']
    selection=m.calibration_samples(unique)
    assert len(selection)==20 and len({r['id'] for r in selection})==20
    assert any(not r['measurement']['valid'] for r in selection)


def test_two_point_reference_direction_and_error_report(tmp_path):
    rows=rows_fixture(tmp_path);source=tmp_path/'measurements.json';source.write_text(json.dumps(dict(samples=rows)))
    out=tmp_path/'review';m.prepare(source,out)
    selection=json.loads((out/'calibration_samples.json').read_text())
    assert 'angle_deg' not in (out/'manual_axes.html').read_text()  # do not show the automatic angle while annotating
    annotations=[]
    for row in selection['samples']:
        # Synthetic human reference is exactly 2 degrees below the automatic result.
        angle=row['measurement']['angle_deg']-2
        delta=np.array([-np.sin(np.radians(angle)),np.cos(np.radians(angle))])*120
        points=[(np.array([320.,240.])-delta).tolist(),(np.array([320.,240.])+delta).tolist()]
        annotations.append(dict(id=row['id'],usable=True,points=points,occlusion='partial',phase='descent',reason='synthetic'))
    review=dict(schema='joint_wrist_manual_axes',source_report=str(source.resolve()),human_annotated=False,
                annotated_at='synthetic',samples=annotations)
    p=tmp_path/'manual.json';p.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='explicit matching human'):
        m.import_manual_axes(out,p)
    review['human_annotated']=True;p.write_text(json.dumps(review))
    result=m.import_manual_axes(out,p)
    assert result['median_absolute_error_deg']==pytest.approx(2)
    assert result['max_absolute_error_deg']==pytest.approx(2)
    assert result['program_misses_on_manual_usable_frames']==2
    assert not result['regression_run']
    assert m.manual_angle([[320,400],[320,80]],[[320,80],[320,400]],[640,480])==pytest.approx(0)
    with pytest.raises(ValueError,match='80 pixels'):
        m.manual_angle([[320,200],[320,220]],[[320,80],[320,400]],[640,480])
    with pytest.raises(FileExistsError):m.import_manual_axes(out,p)


def test_missing_human_cases_cannot_silently_disappear(tmp_path):
    source=tmp_path/'measurements.json';source.write_text(json.dumps(dict(samples=rows_fixture(tmp_path))))
    out=tmp_path/'review';m.prepare(source,out)
    review=dict(schema='joint_wrist_manual_axes',source_report=str(source.resolve()),human_annotated=True,
                annotated_at='synthetic',samples=[])
    p=tmp_path/'manual.json';p.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='all selected samples'):
        m.import_manual_axes(out,p)
