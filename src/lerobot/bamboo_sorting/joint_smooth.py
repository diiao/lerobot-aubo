"""Bounded online joint trajectory, independent of model and hardware imports."""
import math
from .aubo_joint_contract import finite_vector, require_joint_target

DT=.04
MAX_SPEED=5.
MAX_ACCEL=20.
MAX_TCP_SPEED=.05
MAX_TRACKING_ERROR_DEG=3.
MAX_PREDICTION_AGE=.4
MAX_WITHOUT_PREDICTION=.6
MAX_CAMERA_SKEW=.1


def validated_action_chunk(result):
    """Check the optional full-chunk wire response, including discrete IO rows."""
    decoded=result.get('action_chunk');raw=result.get('raw_action_chunk')
    if (result.get('shape')!=[50,7] or not isinstance(decoded,list) or len(decoded)!=50
            or not isinstance(raw,list) or len(raw)!=50):
        raise ValueError('full prediction must contain 50 x 7 actions')
    checked=[]
    for action,unrounded in zip(decoded,raw):
        action=finite_vector(action,7,'chunk action')
        unrounded=finite_vector(unrounded,7,'raw chunk action')
        if action[:6]!=unrounded[:6] or action[-1]!=(100. if unrounded[-1]>=50. else 0.):
            raise ValueError('inconsistent decoded action chunk')
        checked.append(list(action))
    if (list(finite_vector(result['action'],7,'first action'))!=checked[0]
            or finite_vector(result['raw_action'],7,'first raw action')!=finite_vector(raw[0],7,'raw chunk start')):
        raise ValueError('first action does not match chunk')
    return checked


def select_approach_action(chunk,timestamps,now):
    """Age-align one coherent 7D row; never skip a predicted closure boundary.

    Input chunk must pass validated_action_chunk. The caller still validates
    age, FK/workspace and limits, and freezes the selected closure row.
    """
    ts=finite_vector(timestamps,3,'sensor timestamps')
    now=finite_vector([now],1,'clock')[0]
    if min(ts)<0 or max(ts)>now:raise ValueError('invalid selection clock')
    age=now-min(ts);nominal_index=math.floor(age/DT)
    if nominal_index>=len(chunk):raise ValueError('action chunk exhausted')
    index=next((i for i in range(nominal_index+1) if chunk[i][-1]==100.),nominal_index)
    return list(chunk[index]),{'nominal_index':nominal_index,'selected_index':index,
        'observation_age_s':age,'closure_boundary':chunk[index][-1]==100.}


def workspace(tcp):
    tcp=finite_vector(tcp,3,'TCP')
    if any(not lo<=x<=hi for x,lo,hi in zip(tcp,(-.8,-1.2,0.),(1.,0.,.8))):
        raise ValueError('outside capture workspace')


def validate_prediction(action,state,tcp,target_tcp,lower,upper,timestamps,now,*,
                        max_prediction_age=MAX_PREDICTION_AGE):
    action=finite_vector(action,7,'prediction')
    state=finite_vector(state,7,'state')
    ts=finite_vector(timestamps,3,'sensor timestamps')
    age=now-min(ts)
    if not 0<=age<=max_prediction_age or max(ts)>now:
        raise ValueError(f'expired prediction: age={age*1000:.1f} ms, limit={max_prediction_age*1000:.0f} ms')
    if abs(ts[0]-ts[1])>MAX_CAMERA_SKEW:raise ValueError('camera skew')
    if state[-1] not in (0.,100.) or action[-1] not in (0.,100.):raise ValueError('invalid suction')
    lower=finite_vector(lower,6,'joint lower limits')
    upper=finite_vector(upper,6,'joint upper limits')
    # Raw model goals may be farther away; the trajectory generator determines
    # how long reaching them takes. Never send these goals directly to servo.
    for index,(q,current,lo,hi) in enumerate(zip(action[:6],state[:6],lower,upper)):
        if lo>=hi or not lo<=current<=hi or not lo<=q<=hi:
            raise ValueError(f'J{index+1} outside controller joint limits')
    workspace(tcp);workspace(target_tcp)
    return list(action)


class SmoothTrajectory:
    """Critically damped position tracking, with bounded velocity/acceleration.

    Model targets remain in the log. Only intermediate servo references are
    generated here. All position and derivative units are degrees and seconds.
    """
    def __init__(self,q):
        self.q=list(finite_vector(q,6,'initial joints'));self.v=[0.]*6

    def advance(self,target,fk):
        target=finite_vector(target,6,'target')
        old=self.q;previous_v=self.v
        velocity=[]
        for q,v,g in zip(old,previous_v,target):
            accel=max(-MAX_ACCEL,min(MAX_ACCEL,64*(g-q)-16*v))
            velocity.append(max(-MAX_SPEED,min(MAX_SPEED,v+accel*DT)))
        candidate=[q+v*DT for q,v in zip(old,velocity)]
        old_tcp=fk(old);new_tcp=fk(candidate)
        distance=math.dist(old_tcp,new_tcp)
        if distance>MAX_TCP_SPEED*DT:
            # Scaling the desired velocity toward zero can over-decelerate an
            # axis that has already used its acceleration allowance. Instead,
            # form a braking velocity reachable from previous_v in ONE tick.
            # Every convex blend with the desired velocity preserves the joint
            # speed and acceleration bounds (both endpoints satisfy them).
            peak=max(abs(v) for v in previous_v)
            braking_scale=max(0.,1.-MAX_ACCEL*DT/peak) if peak else 0.
            braking=[v*braking_scale for v in previous_v]
            candidate=[q+v*DT for q,v in zip(old,braking)]
            new_tcp=fk(candidate)
            if math.dist(old_tcp,new_tcp)>MAX_TCP_SPEED*DT:
                raise ValueError('no TCP-safe braking candidate within acceleration bound')
            desired=velocity
            velocity=braking
            low,high=0.,1.
            # At most nine FK calls per tick, including the fast-path calls.
            # Nonlinear FK is checked for each candidate; the search need not
            # find the global optimum, but never returns an unchecked point.
            for _ in range(6):
                blend=(low+high)/2.
                trial_v=[b+blend*(d-b) for b,d in zip(braking,desired)]
                trial_q=[q+v*DT for q,v in zip(old,trial_v)]
                trial_tcp=fk(trial_q)
                if math.dist(old_tcp,trial_tcp)<=MAX_TCP_SPEED*DT:
                    low=blend
                    velocity,candidate,new_tcp=trial_v,trial_q,trial_tcp
                else:
                    high=blend
        if max(abs(a-b) for a,b in zip(velocity,previous_v))>MAX_ACCEL*DT+1e-8:
            raise ValueError('TCP limiting cannot preserve acceleration bound')
        if math.dist(old_tcp,new_tcp)>MAX_TCP_SPEED*DT+1e-7:
            raise ValueError('TCP command speed exceeds limit')
        workspace(new_tcp)
        self.q,self.v=candidate,velocity
        return candidate,new_tcp
