#!/usr/bin/env python
"""Offline wrist distributions and independent two-point measurement review.

prepare consumes measurements.json, retains invalid frames, and creates a
20-frame manual-axis page. plots renders standalone PNG/PDF figures using
matplotlib. errors imports the user's exported points; it never fits a response
slope. No hardware, model, or raw-data writes.
"""
import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def unique_rows(report):
    rows = {}
    for row in report['samples']:
        key = (row['episode_id'], row['image'])
        if key in rows and rows[key]['measurement'] != row['measurement']:
            raise ValueError('inconsistent measurements of the same image')
        rows[key] = row
    return list(rows.values())


def describe(rows):
    valid = [r for r in rows if r['measurement']['valid']]
    angles = [r['measurement']['angle_deg'] for r in valid]
    result = dict(frames=len(rows), valid_frames=len(valid), independent_sources=len({r['episode_id'] for r in rows}),
                  sources_with_valid_angles=len({r['episode_id'] for r in valid}),
                  angle_range_deg=[min(angles), max(angles)] if angles else None,
                  median_angle_deg=float(np.median(angles)) if angles else None)
    if valid:
        maximum = max(valid, key=lambda r: abs(r['measurement']['angle_deg']))
        result['max_absolute_angle_sample'] = dict(id=maximum['id'], episode_id=maximum['episode_id'],
            angle_deg=maximum['measurement']['angle_deg'], image=maximum['image'],
            source_monotonic_s=maximum['wrist_source_monotonic_s'])
    return result


def distribution_summary(rows):
    episodes = []
    groups_by_episode = {r['episode_id']:r['group'] for r in rows}
    for episode in sorted(groups_by_episode, key=lambda ep:(groups_by_episode[ep]!='expert',ep)):
        part = sorted([r for r in rows if r['episode_id'] == episode], key=lambda r:r['wrist_source_monotonic_s'])
        first = part[0]
        close = first.get('accepted_close_monotonic_s')
        start = first.get('sequence_start_monotonic_s', first['wrist_source_monotonic_s'])
        episodes.append(dict(episode_id=episode, group=first['group'], configuration=first.get('configuration'),
            trial_outcome=first.get('trial_outcome', 'unknown'), trial_end_reason=first.get('trial_end_reason'),
            initial_angle_deg=first.get('initial_angle_deg'), **describe(part),
            late_preclose=describe([r for r in part if close is not None and close-3 <= r['wrist_source_monotonic_s'] < close]),
            timeline=[dict(id=r['id'], elapsed_s=r['wrist_source_monotonic_s']-start,
                seconds_before_close=close-r['wrist_source_monotonic_s'] if close is not None else None,
                valid=r['measurement']['valid'], angle_deg=r['measurement'].get('angle_deg'),
                invalid_reason=r['measurement'].get('reason'), image=r['image']) for r in part]))
    groups = {g:describe([r for r in rows if r['group']==g]) for g in ('expert','model')}
    # Numerical intersections are descriptive only; pooled ranges can hide holes.
    er = groups['expert']['angle_range_deg']
    comparisons = []
    for episode in episodes:
        if episode['group'] != 'model':
            continue
        mr = episode['angle_range_deg']
        lo, hi = (max(er[0],mr[0]),min(er[1],mr[1])) if er and mr else (None,None)
        comparisons.append(dict(episode_id=episode['episode_id'],configuration=episode['configuration'],
            numerical_range_intersection_deg=[lo,hi] if lo is not None and lo <= hi else None,
            signed_boundary_gap_deg=lo-hi if lo is not None else None,
            interpretation='unreviewed numeric ranges, not proof of shared state support or distribution shift'))
    support = []
    for lower in range(-90,90,5):
        part = [r for r in rows if r['measurement']['valid'] and lower <= r['measurement']['angle_deg'] < lower+5]
        if part:
            support.append(dict(angle_bin_deg=[lower,lower+5],
                expert_sources=sorted({r['episode_id'] for r in part if r['group']=='expert'}),
                model_sources=sorted({r['episode_id'] for r in part if r['group']=='model'})))
    return dict(groups=groups, episodes=episodes, model_run_comparisons=comparisons, angle_bin_support=support,
        human_reviewed=False, measurement_error_deg=None, regression_run=False,
        conclusion='draft state coverage only; visibility, descent stage and manual measurement error are pending',
        limitations=['all preclose recording time is included; first detection is not first visibility or descent onset',
                     'invalid and partially visible frames remain; extrema cover only program-measurable frames',
                     'model runs have different configurations; inspect per-run plots and configuration metadata',
                     'consecutive frames are dependent; independent sample units are episodes/runs',
                     'wrist color proposals do not establish selected-target identity in early or multiple-strip views',
                     'initial global-image angle and wrist-relative angle use different views and sign conventions'])


def calibration_samples(rows, count=20):
    """Angle quantiles plus invalid/short-span cases, before human judgments."""
    selected = []
    for group in ('expert','model'):
        pool = [r for r in rows if r['group']==group]
        invalid = sorted([r for r in pool if not r['measurement']['valid']], key=lambda r:(r['episode_id'],r['wrist_source_monotonic_s']))
        chosen = []
        # Include invalid images too: a human may see a strip that the tool missed.
        if invalid:
            chosen.extend(invalid[i] for i in np.linspace(0,len(invalid)-1,min(2,len(invalid)),dtype=int))
        valid = sorted([r for r in pool if r['measurement']['valid']], key=lambda r:r['measurement']['angle_deg'])
        if valid:
            shortest = min(valid,key=lambda r:r['measurement'].get('axis_span_px',float('inf')))
            chosen.append(shortest)
            valid = [r for r in valid if r['id'] != shortest['id']]
            n = min(count//2-len(chosen),len(valid))
            if n:
                chosen.extend(valid[i] for i in np.linspace(0,len(valid)-1,n,dtype=int))
        selected.extend(chosen)
    selected_ids = {r['id'] for r in selected}
    selected.extend(r for r in rows if r['id'] not in selected_ids and len(selected)<count)
    return selected[:count]


def prepare(report_path, output, count=20):
    report = read(report_path)
    rows = unique_rows(report)
    if not rows:
        raise ValueError('nonempty measurement report required')
    output.mkdir(parents=True,exist_ok=False)
    summary = distribution_summary(rows)
    summary['source_report'] = str(report_path.resolve())
    write_new(output/'distribution.json',summary)
    (output/'calibration_images').mkdir()
    samples = []
    for index,row in enumerate(calibration_samples(rows,count)):
        name=f'calibration_images/frame-{index:02d}.png'
        shutil.copyfile(row['image'],output/name)
        with Image.open(output/name) as image:
            size=list(image.size)
        samples.append(dict(id=row['id'],image=name,image_size=size,gap_line=row['gap_line'],
            group=row['group'],episode_id=row['episode_id'],target_id=row.get('target_id'),
            measurement=row['measurement'],source_image=row['image'],
            video=row.get('video'),video_time_s=row.get('video_time_s'),
            elapsed_s=row['wrist_source_monotonic_s']-row.get('sequence_start_monotonic_s',row['wrist_source_monotonic_s'])))
    calibration = dict(schema='joint_wrist_manual_axis_selection',source_report=str(report_path.resolve()),
        samples=samples,selection='per group: up to two invalid frames, shortest valid axis, then angle quantiles',
        scope='deliberate coverage sample, not a random accuracy estimate; actual occlusion must be labeled by the user')
    write_new(output/'calibration_samples.json',calibration)
    render_manual_page(calibration,output)
    render_distribution_review(summary,output)
    return summary


def render_distribution_review(summary,output):
    rows_html=''.join('<tr><td>'+html.escape(e['episode_id'])+'</td><td>'+e['group']+'</td><td>'+
        str(e['valid_frames'])+'/'+str(e['frames'])+'</td><td>'+html.escape(str(e['angle_range_deg']))+'</td><td>'+
        html.escape(json.dumps({k:v for k,v in (e['configuration'] or {}).items() if k in
            ('target_selection_mode','action_selection','transport_age_aligned')},ensure_ascii=False))+'</td></tr>' for e in summary['episodes'])
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>腕部状态分布草稿</title>'
        '<style>body{max-width:1100px;margin:24px auto;font:16px sans-serif}img{max-width:100%}td{padding:6px}</style>'
        '<h1>腕部状态分布：未确认下探阶段，未量化误差</h1><p>全部曲线是程序测量草稿。'
        '不把首次检测当成首次可见，不把闭合前都当作下探，不用帧数代替独立轮次。</p>'
        '<p><a href="manual_axes.html">标注约20帧竹条方向（先看原图，不显示拟合线）</a> · '
        '<a href="distribution.json">逐条最大角度、时间序列、配置及区间支持</a></p>'
        '<p>图中灰色叉号为无效帧，曲线在无效处断开；浅灰背景是闭合前最后3秒。'
        '模型各轮分别展示；宽范围交集不表示其中每个角度都有足够样本。早期画面还须确认测到的是本轮目标竹条。</p>'
        '<img src="trajectories.png" alt="各示教与实机轮次的时间曲线"><img src="angle_ranges.png" alt="每条记录的角度范围">'
        '<table border="1" style="word-break:break-word"><tr><th>来源</th><th>组</th><th>可测/抽样帧</th><th>角度范围</th><th>运行配置</th></tr>'+rows_html+'</table>'
        '<p>人工误差报告生成前不比较纠偏斜率；暂不输出因果结论。</p>')


def render_manual_page(calibration,output):
    # Keep automatic angles out of the annotation page to reduce anchoring.
    public = {**calibration,'samples':[{k:v for k,v in r.items() if k not in ('measurement','gap_line')}
                                     for r in calibration['samples']]}
    payload=json.dumps(public,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>腕部两点人工参考</title>
<style>body{font:16px sans-serif;max-width:900px;margin:24px auto}canvas{max-width:100%;border:1px solid #aaa;touch-action:none}article{margin:32px 0}select,input,button{font:inherit;margin:6px}button{padding:10px}.status{color:#963}</style>
<h1>人工标竹条方向：原图两点</h1>
<p>在目标竹条的中心轴线上点两点，尽量拉开（至少80像素）。不点夹指，不沿单侧边缘；第二点后再次点击会重画。无需沿绿色参考线标注，页面有意不显示工具拟合线。标不清或无法确认本轮目标身份时选“无法标注”，说明原因，不能勉强连线。</p>
<p>同时记录遮挡程度与动作阶段。这约20帧是覆盖不同画面的抽查，并非总体准确率随机抽样。一个人的两点标注也有误差，最大差值不是全数据的误差上界。</p>
<div id="cards"></div><button id="export">导出人工标注</button><p id="message" class="status"></p><script>
const spec=PAYLOAD; const answers=new Map();
function choice(options){let el=document.createElement('select');el.add(new Option('请选择',''));for(let [value,label] of options)el.add(new Option(label,value));return el;}
for(const row of spec.samples){
 const card=document.createElement('article'),title=document.createElement('h2');title.textContent=row.id+' / '+row.group+' / '+(row.target_id||'目标身份待确认');card.append(title);
 const context=document.createElement('p');context.textContent=row.episode_id+' / 记录开始后 '+row.elapsed_s.toFixed(2)+' s';card.append(context);
 if(row.video){const link=document.createElement('a');link.href=row.video+'#t='+row.video_time_s;link.textContent='查看该帧对应的原录像（确认目标和动作阶段）';link.target='_blank';card.append(link);}
 const canvas=document.createElement('canvas');canvas.width=row.image_size[0];canvas.height=row.image_size[1];card.append(canvas);
 const usable=choice([['yes','可标注目标方向'],['no','无法标注']]);
 const occlusion=choice([['none','无遮挡'],['partial','部分遮挡'],['heavy','严重遮挡'],['not_visible','目标不可见'],['uncertain','遮挡不确定']]);
 const phase=choice([['early_approach','早期接近'],['descent','下探'],['contact','已接触'],['uncertain','阶段不确定']]);
 const reason=document.createElement('input');reason.placeholder='无法标注的原因 / 备注';reason.style.width='70%';
 card.append(document.createElement('br'),usable,occlusion,phase,document.createElement('br'),reason);document.getElementById('cards').append(card);
 const item={row,points:[],usable,occlusion,phase,reason};answers.set(row.id,item);
 const image=new Image(),ctx=canvas.getContext('2d');function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);ctx.strokeStyle='#00eaff';ctx.fillStyle='#00eaff';ctx.lineWidth=2;
 for(const p of item.points){ctx.beginPath();ctx.arc(p[0],p[1],5,0,2*Math.PI);ctx.fill();}if(item.points.length===2){ctx.beginPath();ctx.moveTo(...item.points[0]);ctx.lineTo(...item.points[1]);ctx.stroke();}}
 image.onload=draw;image.src=row.image;
 canvas.addEventListener('pointerdown',event=>{event.preventDefault();const b=canvas.getBoundingClientRect();const p=[Math.max(0,Math.min(canvas.width-1,(event.clientX-b.left)*canvas.width/b.width)),Math.max(0,Math.min(canvas.height-1,(event.clientY-b.top)*canvas.height/b.height))];if(item.points.length===2)item.points=[];item.points.push(p);draw();});
}
document.getElementById('export').onclick=()=>{const samples=[];for(const [id,a] of answers){const fail=(!a.usable.value||!a.occlusion.value||!a.phase.value||(a.usable.value==='yes'&&(a.points.length!==2||Math.hypot(a.points[1][0]-a.points[0][0],a.points[1][1]-a.points[0][1])<80))||(a.usable.value==='no'&&!a.reason.value.trim()));if(fail){document.getElementById('message').textContent=id+' 尚未完成：请选择状态、遮挡、阶段；可标注需要两个相距至少80px的点，无法标注需要原因。';return;}
 samples.push({id,usable:a.usable.value==='yes',points:a.usable.value==='yes'?a.points:[],occlusion:a.occlusion.value,phase:a.phase.value,reason:a.reason.value});}
 const result={schema:'joint_wrist_manual_axes',source_report:spec.source_report,human_annotated:true,annotated_at:new Date().toISOString(),samples};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));a.download='wrist_manual_axes.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);document.getElementById('message').textContent='已导出；尚未导入计算误差。';};
</script>'''.replace('PAYLOAD',payload)
    (output/'manual_axes.html').write_text(page)


def manual_angle(points,gap_line,size):
    points=np.asarray(points,float);gap=np.asarray(gap_line,float)
    if points.shape!=(2,2) or not np.isfinite(points).all() or np.any(points<0) or np.any(points>=np.asarray(size)):
        raise ValueError('two finite points inside image required')
    vector=points[1]-points[0];axis=gap[1]-gap[0]
    if np.linalg.norm(vector)<80 or np.linalg.norm(axis)<1:
        raise ValueError('manual points must be at least 80 pixels apart')
    if vector@axis<0:vector=-vector
    return float(np.degrees(np.arctan2(axis[0]*vector[1]-axis[1]*vector[0],axis@vector)))


def error_metrics(rows):
    values=[r['absolute_error_deg'] for r in rows if r.get('absolute_error_deg') is not None]
    return dict(paired_frames=len(values),median_absolute_error_deg=float(np.median(values)) if values else None,
                max_absolute_error_deg=max(values) if values else None)


def import_manual_axes(output,annotation_path):
    selection=read(output/'calibration_samples.json');review=read(annotation_path)
    if (review.get('schema')!='joint_wrist_manual_axes' or review.get('source_report')!=selection['source_report']
            or review.get('human_annotated') is not True or not review.get('annotated_at')):
        raise ValueError('explicit matching human axis annotation required')
    expected={r['id']:r for r in selection['samples']};ids=[r['id'] for r in review['samples']]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
        raise ValueError('all selected samples must be annotated once, including unusable images')
    rows=[]
    for item in review['samples']:
        row=expected[item['id']]
        if item.get('occlusion') not in ('none','partial','heavy','not_visible','uncertain') or item.get('phase') not in ('early_approach','descent','contact','uncertain'):
            raise ValueError('occlusion and phase labels required')
        result=dict(id=item['id'],group=row['group'],episode_id=row['episode_id'],occlusion=item['occlusion'],phase=item['phase'],
                    human_usable=item['usable'],program_valid=row['measurement']['valid'],absolute_error_deg=None)
        if item['usable'] is True:
            if item['occlusion']=='not_visible':
                raise ValueError('a not-visible target cannot have a usable direction annotation')
            angle=manual_angle(item['points'],row['gap_line'],row['image_size'])
            result.update(manual_angle_deg=angle,points=item['points'])
            if row['measurement']['valid']:
                error=(row['measurement']['angle_deg']-angle+90)%180-90
                result.update(signed_error_deg=error,absolute_error_deg=abs(error))
        elif item['usable'] is False and item.get('reason','').strip():
            result['reason']=item['reason']
        else:
            raise ValueError('explicit unusable reason or valid points required')
        rows.append(result)
    metrics=error_metrics(rows)
    report=dict(schema='joint_wrist_manual_axis_errors',source_report=selection['source_report'],
        annotation_source=str(annotation_path.resolve()),samples=rows,**metrics,
        by_group={g:error_metrics([r for r in rows if r['group']==g]) for g in ('expert','model')},
        by_occlusion={g:error_metrics([r for r in rows if r['occlusion']==g]) for g in sorted({r['occlusion'] for r in rows})},
        manual_unusable_frames=sum(not r['human_usable'] for r in rows),
        program_misses_on_manual_usable_frames=sum(r['human_usable'] and not r['program_valid'] for r in rows),
        program_valid_on_manual_unusable_frames=sum(not r['human_usable'] and r['program_valid'] for r in rows),
        regression_run=False,scope='disagreement with one human reference, not calibrated ground truth; sampled maximum is not a universal error bound')
    maximum=metrics['max_absolute_error_deg'];distribution=read(output/'distribution.json')
    report['range_comparisons_with_error_scale']=[dict(r,sampled_max_error_deg=maximum,
        boundary_gap_within_twice_sample_max=(abs(r['signed_boundary_gap_deg'])<=2*maximum
            if maximum is not None and r['signed_boundary_gap_deg'] is not None else None),
        decision='no slope or state-coverage conclusion; stage review and independent-source support still required')
        for r in distribution['model_run_comparisons']]
    write_new(output/'measurement_error.json',report)
    return report


def plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=read(output/'distribution.json');episodes=report['episodes']
    n=len(episodes);fig,axes=plt.subplots((n+2)//3,3,figsize=(15,3*((n+2)//3)),squeeze=False)
    colors={'expert':'#147c94','model':'#b65d20'}
    magnitudes=[abs(t['angle_deg']) for e in episodes for t in e['timeline'] if t['valid']]
    limit=max(10,10*np.ceil(max(magnitudes,default=10)/10))
    for ax,e in zip(axes.flat,episodes):
        t=[r['elapsed_s'] for r in e['timeline']];a=[r['angle_deg'] if r['valid'] else np.nan for r in e['timeline']]
        ax.plot(t,a,'.-',color=colors[e['group']],linewidth=1,markersize=3)
        bad=[r['elapsed_s'] for r in e['timeline'] if not r['valid']]
        ax.scatter(bad,[-.04]*len(bad),transform=ax.get_xaxis_transform(),marker='x',color='gray',s=12,clip_on=False)
        plot_end=max(.5,max(t,default=0))
        if e['timeline'] and e['timeline'][0]['seconds_before_close'] is not None:
            close=e['timeline'][0]['elapsed_s']+e['timeline'][0]['seconds_before_close']
            ax.axvspan(max(0,close-3),close,color='gray',alpha=.1)
            plot_end=max(plot_end,close)
        name=Path(e['episode_id']).name
        if e['group']=='expert':name=Path(e['episode_id']).parent.name.replace('joint_target_pair_','')+'/'+name
        else:name=name.replace('joint_target_','')
        ax.set_title(name,fontsize=8);ax.set_xlabel('Seconds from recording start');ax.set_ylabel('Draft wrist angle (deg)');ax.grid(alpha=.2)
        ax.set_ylim(-limit,limit)
        # NaN angle rows do not affect matplotlib's data limits. Include their
        # timestamps explicitly, otherwise early invalid frames are clipped.
        ax.set_xlim(0,plot_end)
    for ax in list(axes.flat)[n:]:ax.axis('off')
    fig.suptitle('Draft states; gray x = invalid; shading = last 3 s before closure; configurations differ',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.985));fig.savefig(output/'trajectories.png',dpi=140);fig.savefig(output/'trajectories.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,max(5,n*.35)))
    for i,e in enumerate(episodes):
        if e['angle_range_deg'] is not None:
            ax.plot(e['angle_range_deg'],[i,i],color=colors[e['group']],linewidth=3)
            ax.plot(e['median_angle_deg'],i,'o',color=colors[e['group']])
    ax.set_yticks(range(n));ax.set_yticklabels([e['group']+' / '+(Path(e['episode_id']).parent.name.replace('joint_target_pair_','')+'/' if e['group']=='expert' else '')+Path(e['episode_id']).name.replace('joint_target_','') for e in episodes],fontsize=8)
    ax.set_xlabel('Program-measurable wrist angle range / median (deg)');ax.set_title('Independent episodes/runs; extrema are not common-support evidence');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(output/'angle_ranges.png',dpi=140);fig.savefig(output/'angle_ranges.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','plots','errors'),default='prepare')
    parser.add_argument('--measurements',type=Path);parser.add_argument('--annotations',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.stage=='prepare':
        if args.measurements is None:parser.error('--measurements required')
        prepare(args.measurements.resolve(),args.output.resolve())
    elif args.stage=='errors':
        if args.annotations is None:parser.error('--annotations required')
        import_manual_axes(args.output.resolve(),args.annotations.resolve())
    else:plots(args.output.resolve())
    print(json.dumps(dict(output=str(args.output),stage=args.stage,regression_run=False)))


if __name__=='__main__':
    main()
