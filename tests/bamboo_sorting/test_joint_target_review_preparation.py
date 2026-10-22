import copy,importlib.util,json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
MODULE = Path(__file__).resolve().parents[2] / 'examples/phone_to_auboi10/prepare_joint_target_review.py'
spec=importlib.util.spec_from_file_location('review_prepare',MODULE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def setup(tmp_path,n=1):
 for name in ['masks','images','outlines']:(tmp_path/name).mkdir()
 frames=[dict(sample_id=f's{i}',frame_id=f'0/global_rgb/{i}',frame_index=i,target_id='strip_001',schema='aubo_joint_target_frame',image=f'images/s{i}.png',mask_path=f'masks/s{i}.png',reviewed=False,status='pending') for i in range(n)]
 return {'schema':'aubo_joint_target_annotations','source_dataset_root':'/example','episodes':[{'episode_index':0,'scene_id':'001','target_id':'strip_001','reviewed':False,'frames':frames}]}

def prediction(tmp_path,spec,arrays,segments):
 rows=[]
 for f,a,s in zip(spec['episodes'][0]['frames'],arrays,segments):
  Image.fromarray(a).save(tmp_path/(f['sample_id']+'.png'))
  rows.append(dict(id=f['sample_id'],frame_index=f['frame_index'],frame_id=f['frame_id'],image=f['image'],instance_map=f['sample_id']+'.png',segments=s))
 p=tmp_path/'predictions.json';m.write(p,{'complete':True,'plan':{'source_run':str(tmp_path.parent.resolve())},'samples':rows});return p

def test_single_id_keeps_both_visible_components_without_gap_fill(tmp_path):
 s=setup(tmp_path);a=np.zeros((480,640),np.uint16);a[20:30,20:60]=7;a[20:30,90:130]=7
 p=prediction(tmp_path,s,[a],[[dict(mask_id=7,class_id=0,pixels=800,score=.9)]])
 m.import_predictions(tmp_path,s,p,single_strip_scenes=True);result=np.array(Image.open(tmp_path/'masks/s0.png'))
 assert np.array_equal(result!=0,a==7)
 assert not s['episodes'][0]['frames'][0]['reviewed']
 with pytest.raises(ValueError,match='overwritten'):m.import_predictions(tmp_path,s,p)

def test_predictions_accept_relocated_source_alias_and_reject_other_run(tmp_path):
 old=tmp_path/'old_run';new=tmp_path/'classified'/'run';folder=new/'target_annotations'
 folder.mkdir(parents=True);old.symlink_to(new,target_is_directory=True)
 s=setup(folder);a=np.zeros((480,640),np.uint16);a[10:30,10:30]=7
 p=prediction(folder,s,[a],[[dict(mask_id=7,class_id=0,pixels=400,score=.9)]])
 report=m.read(p);report['plan']['source_run']=str(old);m.write(p,report)
 m.import_predictions(folder,s,p,single_strip_scenes=True)
 assert np.array_equal(np.array(Image.open(folder/'masks/s0.png'))!=0,a==7)
 other=tmp_path/'other_run';other.mkdir()
 report['plan']['source_run']=str(other);m.write(p,report)
 with pytest.raises(ValueError,match='source run mismatch'):m.import_predictions(folder,s,p)

def test_verify_entry_accepts_dataset_alias_without_changing_identity(tmp_path,monkeypatch):
 import sys
 dataset=tmp_path/'classified'/'data';dataset.mkdir(parents=True)
 old=tmp_path/'old_data';old.symlink_to(dataset,target_is_directory=True)
 evidence=tmp_path/'evidence';folder=evidence/'target_annotations';folder.mkdir(parents=True)
 spec={'source_dataset_root':str(old),'episodes':[]};m.write(folder/'annotations.json',spec)
 calls=[]
 monkeypatch.setattr(m,'verify_dataset',lambda data,ev,f,s:calls.append((data,s['source_dataset_root'])))
 monkeypatch.setattr(sys,'argv',['review','--stage','verify','--dataset-root',str(dataset),'--evidence-root',str(evidence)])
 m.main();assert calls==[(dataset,str(old))]
 other=tmp_path/'other_data';other.mkdir();spec['source_dataset_root']=str(other)
 m.write(folder/'annotations.json',spec)
 with pytest.raises(ValueError,match='source dataset mismatch'):m.main()

def test_missing_and_ambiguous_candidates_remain_missing_not_union(tmp_path):
 s=setup(tmp_path,2);a=np.zeros((480,640),np.uint16);b=a.copy();b[10:20,10:30]=1;b[10:20,40:60]=2
 p=prediction(tmp_path,s,[a,b],[[],[dict(mask_id=1,class_id=0,pixels=200,score=.8),dict(mask_id=2,class_id=0,pixels=200,score=.8)]])
 m.import_predictions(tmp_path,s,p)
 assert [f['status'] for f in s['episodes'][0]['frames']]==['missing','ambiguous']
 assert not list((tmp_path/'masks').glob('*.png'))

def test_sole_detection_is_not_selected_without_single_strip_context(tmp_path):
 # In an A/B scene, the sole detection could be B while the requested target is A.
 s=setup(tmp_path);a=np.zeros((480,640),np.uint16);a[10:30,10:30]=7
 p=prediction(tmp_path,s,[a],[[dict(mask_id=7,class_id=0,pixels=400,score=.9)]])
 m.import_predictions(tmp_path,s,p)
 row=s['episodes'][0]['frames'][0]
 assert row['target_id']=='strip_001' and row['status']=='ambiguous'
 assert row['draft_selection']=='human_selection_required' and not row['reviewed']
 assert not (tmp_path/row['mask_path']).exists()

def test_review_correction_preserves_prior_mask_and_requires_second_review(tmp_path):
 s=setup(tmp_path);row=s['episodes'][0]['frames'][0];row['status']='visible'
 original=np.zeros((480,640),np.uint8);original[10:30,10:30]=255;Image.fromarray(original).save(tmp_path/row['mask_path'])
 d={'schema':'aubo_joint_target_review_decisions','source_dataset_root':'/example','frames':{row['frame_id']:dict(frame_id=row['frame_id'],frame_index=0,episode_index=0,annotation_revision=0,reviewed=False,polygons=[[[10,10],[25,10],[25,20]],[[50,10],[60,10],[60,20]]])}}
 p=tmp_path/'decisions.json';m.write(p,d);m.import_decisions(tmp_path,s,p)
 assert row['previous_masks'] and row['annotation_revision']==1 and not row['reviewed']
 a=np.array(Image.open(tmp_path/row['mask_path']));assert a[12,20] and a[12,55] and not a[12,40]
 with pytest.raises(ValueError,match='revision'):m.import_decisions(tmp_path,s,p)
 d['frames'][row['frame_id']].update(annotation_revision=1,polygons=[],reviewed=True,confirmation='explicit_user_range_review',reviewed_at='synthetic-test')
 m.write(p,d);m.import_decisions(tmp_path,s,p)
 assert row['reviewed'] and s['episodes'][0]['reviewed'] and not s['target_training_ready']

def test_prediction_identity_failure_writes_no_masks(tmp_path):
 s=setup(tmp_path);a=np.zeros((480,640),np.uint16);a[10:30,10:30]=1
 p=prediction(tmp_path,s,[a],[[dict(mask_id=1,class_id=0,pixels=400,score=.9)]])
 d=m.read(p);d['samples'][0]['frame_id']='1/global_rgb/0';m.write(p,d)
 with pytest.raises(ValueError,match='identity'):m.import_predictions(tmp_path,s,p)
 assert not list((tmp_path/'masks').glob('*.png'))


def test_saved_issue_cannot_be_bypassed_by_omitting_issue_from_review(tmp_path):
 s=setup(tmp_path);row=s['episodes'][0]['frames'][0]
 row.update(status='visible',review_issue='far visible segment missing')
 mask=np.zeros((480,640),np.uint8);mask[10:30,10:30]=255
 Image.fromarray(mask).save(tmp_path/row['mask_path'])
 decision=dict(frame_id=row['frame_id'],frame_index=0,episode_index=0,
               annotation_revision=0,reviewed=True,confirmation='explicit_user_range_review',
               reviewed_at='synthetic-test')
 path=tmp_path/'decisions.json'
 m.write(path,dict(schema='aubo_joint_target_review_decisions',source_dataset_root='/example',
                  frames={row['frame_id']:decision}))
 with pytest.raises(ValueError,match='explicit review'):
  m.import_decisions(tmp_path,s,path)
 assert not row['reviewed']


def test_multi_seed_import_separation_review_and_runtime_share_pixels(tmp_path):
 from tests.bamboo_sorting.perception_fakes import crossing_seeds, instance_map
 from lerobot.bamboo_sorting.runtime.joint_target_runtime import FreshTargetInputs, encode_mask, decode_mask
 masks=crossing_seeds();instances,segments=instance_map(masks)
 for segment in segments:segment['pixels']=int((instances==segment['mask_id']).sum())
 s=setup(tmp_path,2);ep=s['episodes'][0];ep.update(target_id='s-002',pick_step=1,available_target_count=2)
 for row in ep['frames']:row['target_id']='s-002'
 p=prediction(tmp_path,s,[instances,instances],[segments,segments]);m.import_predictions(tmp_path,s,p)
 assert ep['frames'][0]['instance_ids_base64']
 selection=dict(frame_id=ep['frames'][0]['frame_id'],human_reviewed=True,reviewed_at='synthetic',
                targets=[dict(target_id=t,instance_ids=[i*7]) for i,t in enumerate(masks,1)])
 export=tmp_path/'seed-decisions.json'
 m.write(export,dict(schema='aubo_joint_target_review_decisions',source_dataset_root='/example',seeds={'0':selection}))
 m.import_seeds(tmp_path,s,export);m.separate_predictions(tmp_path,s)
 assert not ep['reviewed'] and all(not f['reviewed'] for f in ep['frames'])
 seed=dict(schema='joint_target_live_seed',snapshot_id='synthetic',human_reviewed=True,target_id='s-002',
           target_selection_mode='multi_foreground',mask_png=encode_mask(masks['s-002']),
           excluded_mask_png=encode_mask(masks['s-001']|masks['s-003']),
           visible_targets=[dict(target_id=t,mask_png=encode_mask(mask)) for t,mask in masks.items()])
 runtime=FreshTargetInputs(seed,lambda rgb:(instances,segments))
 images={'observation.images.'+name:np.zeros((480,640,3),np.uint8) for name in ('global_rgb','grasp_rgb')}
 for i,row in enumerate(ep['frames']):
  _,evidence=runtime.prepare(dict(**images,task='test',**{'observation.state':np.zeros(7)}),
                             dict(frame_id=str(i),target_id='s-002',phase='approach'))
  np.testing.assert_array_equal(np.array(Image.open(tmp_path/row['mask_path']))>0,decode_mask(evidence['mask_png']))
 with pytest.raises(ValueError,match='overwrite'):m.separate_predictions(tmp_path,s)


def test_seed_selection_bound_to_first_frame_and_includes_selected_target(tmp_path):
 s=setup(tmp_path);p=tmp_path/'decisions.json'
 selection=dict(frame_id='wrong',human_reviewed=True,reviewed_at='synthetic',
                targets=[dict(target_id='strip_001',polygons=[[[10,10],[20,10],[20,30]]])])
 d=dict(schema='aubo_joint_target_review_decisions',source_dataset_root='/example',seeds={'0':selection})
 m.write(p,d)
 with pytest.raises(ValueError,match='first approach'):m.import_seeds(tmp_path,s,p)
 selection['frame_id']='0/global_rgb/0';selection['targets'][0]['target_id']='other';m.write(p,d)
 with pytest.raises(ValueError,match='selected target'):m.import_seeds(tmp_path,s,p)


def test_automatic_seed_draft_cannot_be_imported_as_human_confirmation(tmp_path):
 s=setup(tmp_path)
 selection={'frame_id':'0/global_rgb/0','human_reviewed':False,
            'annotation_source':'assistant_assisted_current_image_seed_draft',
            'targets':[{'target_id':'strip_001','polygons':[[[10,10],[30,10],[30,20],[10,20]]]}]}
 s['episodes'][0]['draft_visible_seeds']=copy.deepcopy(selection)
 p=tmp_path/'draft.json'
 m.write(p,{'schema':'aubo_joint_target_review_decisions','source_dataset_root':'/example',
            'seeds':{'0':selection},'frames':{}})
 with pytest.raises(ValueError,match='explicit seed review'):
  m.import_seeds(tmp_path,s,p)
 assert 'visible_seeds' not in s['episodes'][0]
 assert not list((tmp_path/'masks').glob('*.png'))


def test_accepted_crossing_limitation_keeps_mask_and_still_requires_review(tmp_path):
 s=setup(tmp_path);row=s['episodes'][0]['frames'][0]
 row.update(status='visible',review_issue='crossing boundary uncertain')
 mask=np.zeros((480,640),np.uint8);mask[10:30,10:30]=255
 Image.fromarray(mask).save(tmp_path/row['mask_path']);before=(tmp_path/row['mask_path']).read_bytes()
 decision=dict(frame_id=row['frame_id'],episode_index=0,frame_index=0,annotation_revision=0,reviewed=False,
               issue='',accepted_limitation='small crossing boundary error accepted',reviewed_at='synthetic-test')
 p=tmp_path/'decisions.json'
 m.write(p,dict(schema='aubo_joint_target_review_decisions',source_dataset_root='/example',frames={row['frame_id']:decision}))
 m.import_decisions(tmp_path,s,p)
 assert (tmp_path/row['mask_path']).read_bytes()==before
 assert not row['reviewed'] and row['accepted_annotation_limitations'][0]['mask_modified'] is False
