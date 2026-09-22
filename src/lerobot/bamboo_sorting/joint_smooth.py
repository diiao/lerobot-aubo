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


def workspace(tcp):
    tcp=finite_vector(tcp,3,'TCP')
    if any(not lo<=x<=hi for x,lo,hi in zip(tcp,(-.8,-1.2,0.),(1.,0.,.8))):
        raise ValueError('outside capture workspace')


def validate_prediction(action,state,tcp,target_tcp,lower,upper,timestamps,now):
    action=finite_vector(action,7,'prediction')
    state=finite_vector(state,7,'state')
    ts=finite_vector(timestamps,3,'sensor timestamps')
    if not 0<=now-min(ts)<=MAX_PREDICTION_AGE or max(ts)>now:
        raise ValueError('expired prediction')
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
