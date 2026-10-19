# SPDX-License-Identifier: MIT
"""REM Sheet Metal 3.5.0 -- adjustable flange and return bend angles."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

KEY='rem_sheet_metal_angles'
VERSION='3.5.0'
INCH=.0254
SCALE=1000.
SIDES=('left','right','front','back')
ADJACENT={'left':('front','back'),'right':('front','back'),
          'front':('left','right'),'back':('left','right')}
CORNER_PAIRS=(('left','front'),('right','front'),('right','back'),('left','back'))
CORNER_OPTIONS=(('open','Open — square relief'),
    ('cab_lr','Butt seam — left/right through (3.2)'),
    ('cab_fb','Butt seam — front/back through (3.2)'),
    ('full_front','Full width — Front flange'),('full_back','Full width — Back flange'),
    ('full_left','Full width — Left flange'),('full_right','Full width — Right flange'),
    ('full_fb','Full width — Front + Back'),('full_lr','Full width — Left + Right'),
    ('custom','Independent corners — panel / mitre / end tab'))
CORNER_NAMES=('Front-left','Front-right','Back-right','Back-left')
PRESETS=[('20 ga stainless — shop standard','#4 stainless steel',.038,.079,.099),
         ('16 ga cold rolled steel','Cold rolled steel',.0598,.067,.111),
         ('14 ga cold rolled steel','Cold rolled steel',.0747,.075,.149),
         ('10 ga cold rolled steel','Cold rolled steel',.1345,.150,.275)]

class PanelError(ValueError):pass

LEGACY_KEYS=('rem_sheet_metal_unified','rem_sheet_metal','rem_cab_panels')

def part_data(group):
    data=getattr(group,'ext',None) or {}
    for key in (KEY,)+LEGACY_KEYS:
        if key in data:
            if not isinstance(data[key],dict) or not isinstance(data[key].get('params'),dict):raise PanelError('This part has invalid sheet-metal definitions. Recreate it from a verified base.')
            return key,data[key]
    raise PanelError('This component has no sheet-metal bend definitions. Use New Base or Base from Face to create a sheet-metal part. Existing cab solids need a separate conversion step.')

def part_parameters(group):
    key,data=part_data(group);old=data['params']
    if key!='rem_sheet_metal':return validate(old)
    q=defaults()
    for k in ('name','material','units','width','length','thickness','radius','deduction','segments','flat_view'):
        if k in old:q[k]=copy.deepcopy(old[k])
    q['gap']=max(q['gap'],q['thickness']);q['holes']=copy.deepcopy(data.get('holes',[]))
    for side,legacy in (('left','a'),('right','b')):
        for new,previous in (('direction','direction'),('height','height'),('return','return_direction'),('lip','return_length')):
            q[side+'_'+new]=old.get(legacy+'_'+previous,q[side+'_'+new])
    return validate(q)

def _area(loop):
    q=np.asarray(loop);return float(np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1)))/2

def _contains(loop,point):
    x,y=point;inside=False
    for a,b in zip(loop,loop[1:]+loop[:1]):
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return inside

def _boundary_crosses(first,second=None):
    q=np.asarray(first);s=q if second is None else np.asarray(second)
    a=q[:,None,:];b=np.roll(q,-1,axis=0)[:,None,:];c=s[None,:,:];d=np.roll(s,-1,axis=0)[None,:,:]
    ab=b-a;cd=d-c;tol=2e-7
    def cross(u,v):return u[...,0]*v[...,1]-u[...,1]*v[...,0]
    ea=tol*np.linalg.norm(ab,axis=-1);ec=tol*np.linalg.norm(cd,axis=-1)
    u,v=cross(ab,c-a),cross(ab,d-a);w,z=cross(cd,a-c),cross(cd,b-c)
    hit=(((u<=ea)&(v>=-ea))|((v<=ea)&(u>=-ea)))&(((w<=ec)&(z>=-ec))|((z<=ec)&(w>=-ec)))
    hit &= np.all(np.maximum(np.minimum(a,b),np.minimum(c,d))<=np.minimum(np.maximum(a,b),np.maximum(c,d))+tol,axis=-1)
    if second is None:
        hit=np.triu(hit,2);hit[0,-1]=False
    return bool(np.any(hit))

def openings(p):
    loops=copy.deepcopy(p.get('holes',[]));centre=opening(p)
    if centre:loops.append(centre)
    loops.extend(cut_loops(p,'base'))
    return loops

def defaults():
    p=dict(name='Sheet metal part',material='#4 stainless steel',units='in',
           width=6*INCH,length=12*INCH,thickness=.038*INCH,radius=.079*INCH,
           deduction=.099*INCH,gap=.0625*INCH,opening_width=0.,opening_length=0.,
           segments=24,flat_view=False,holes=[],cuts=[],corner_style='open',seam_gap=.020*INCH,corner_features={})
    for side in SIDES:p.update({side+'_direction':0,side+'_height':2.4375*INCH,
                               side+'_return':0,side+'_lip':.75*INCH,side+'_angle':90.,side+'_return_angle':90.})
    return p

def template_index(p):
    if not any(p[s+'_direction'] for s in SIDES):return 3
    if all(p[s+'_direction']==1 for s in SIDES):
        for index,ret in ((0,1),(1,0),(2,-1)):
            if all(p[s+'_return']==ret for s in SIDES):return index
    return 4

def full_width_sides(p):
    mode=p.get('corner_style','open')
    if not mode.startswith('full_'):return ()
    name=mode[5:]
    return ('front','back') if name=='fb' else ('left','right') if name=='lr' else (name,)


def closed_corner(p,a,b):
    if p.get('corner_style')=='custom':
        i=next(i for i,pair in enumerate(CORNER_PAIRS) if set(pair)=={a,b})
        c=corner_feature(p,i)
        return c['through'] if c['kind']!='open' and p[a+'_direction'] and p[b+'_direction'] else None
    full=full_width_sides(p)
    if full:
        if not p[a+'_direction'] or not p[b+'_direction']:return None
        return a if a in full else b if b in full else None
    if p.get('corner_style','open')=='open' or not p[a+'_direction'] or p[a+'_direction']!=p[b+'_direction']:return None
    if p.get('corner_style') not in ('cab_lr','cab_fb'):return None
    through=('left','right') if p['corner_style']=='cab_lr' else ('front','back')
    return a if a in through else b


def has_closed_corners(p):return any(closed_corner(p,a,b) for a,b in CORNER_PAIRS)


def corners(p):
    if p.get('corner_style')=='custom':
        return [corner_feature(p,i)['relief'] if all(p[s+'_direction'] for s in pair) and corner_feature(p,i)['kind']=='open' else 0. for i,pair in enumerate(CORNER_PAIRS)]
    a,b,c,d=(bool(p[s+'_direction']) for s in SIDES)
    g=p['gap']
    return [g if a and c and not closed_corner(p,'left','front') else 0.,g if b and c and not closed_corner(p,'right','front') else 0.,
            g if b and d and not closed_corner(p,'right','back') else 0.,g if a and d and not closed_corner(p,'left','back') else 0.]

def web_bounds(p):
    ro=p['radius']+p['thickness']
    a,b,c,d=(bool(p[s+'_direction']) for s in SIDES)
    return ro*a,p['width']-ro*b,ro*c,p['length']-ro*d

def side_layout(p):
    if p.get('corner_style')=='custom':return custom_layout(p)
    x0,x1,y0,y1=web_bounds(p);bl,br,tr,tl=corners(p)
    spans={'left':(y0+bl,y1-tl),'right':(y0+br,y1-tr),
           'front':(x0+bl,x1-br),'back':(x0+tl,x1-tr)}
    adjacent=ADJACENT;full=full_width_sides(p)
    ro=p['radius']+p['thickness'];ba=2*ro-p['deduction'];result={}
    for side in SIDES:
        if not p[side+'_direction']:continue
        attach_lo,attach_hi=spans[side];lo,hi=attach_lo,attach_hi;ret=p[side+'_return']
        near,far=adjacent[side];overall=p['length'] if side in ('left','right') else p['width']
        for other,is_near in ((near,True),(far,False)):
            through=closed_corner(p,side,other)
            if through:
                if full:
                    # A through bend is rectangular over the full finished span.
                    # The neighbouring ENTIRE flange starts behind its bend and
                    # inward lip. The additional web tongue ends one relief
                    # width before that bend; a relief also extends into the web.
                    reach=p[through+'_lip'] if p[through+'_return']==1 else ro
                    offset=0. if through==side else max(ro+2*p['gap'],reach+p['seam_gap'])
                else:offset=0. if through==side else p['thickness']+p['seam_gap']
                if is_near:lo=offset
                else:hi=overall-offset
                if full:
                    if is_near:attach_lo=lo
                    else:attach_hi=hi
        # Return ends stay back at the tangent-region ends. The extended wall
        # ends close the visible seam without colliding return lips.
        rlo,rhi=attach_lo,attach_hi
        if ret:
            near,far=adjacent[side]
            base_lo,base_hi=(y0,y1) if side in ('left','right') else (x0,x1)
            # Square end reliefs keep inward returns clear of adjacent returns.
            # Outward returns occupy different quadrants and need no extra trim.
            for other,is_near in ((near,True),(far,False)):
                if full and closed_corner(p,side,other):continue
                if ret==1 and p[other+'_direction'] and p[other+'_return']==1:
                    reach=max(0.,p[other+'_lip']-ro)+p['gap']
                    if is_near:rlo=max(rlo,base_lo+reach)
                    else:rhi=min(rhi,base_hi-reach)
        wall=p[side+'_height']-(2*ro if ret else ro)+ba
        lip=ba+p[side+'_lip']-ro if ret else 0.
        result[side]=dict(lo=lo,hi=hi,attach_lo=attach_lo,attach_hi=attach_hi,
                         return_lo=rlo,return_hi=rhi,wall_flat=wall,return_flat=lip)
    return result


END_FIELDS=('wall_setback','return_setback','notch_length','notch_bottom','notch_height',
            'tab_projection','tab_bottom','tab_height','tab_radius')


def corner_feature(p,index):
    pair=CORNER_PAIRS[index]
    c=dict(kind='open',through=pair[1],return_joint='butt',relief=p['gap'],clearance=p['seam_gap'],
           ends={s:dict.fromkeys(END_FIELDS,0.) for s in pair})
    raw=p.get('corner_features',{}).get(str(index),{})
    c.update({k:v for k,v in raw.items() if k!='ends'})
    if c['kind']=='miter' and 'return_joint' not in raw:c['return_joint']='miter'
    for s in pair:c['ends'][s].update(raw.get('ends',{}).get(s,{}))
    return c


def normalize_corner_features(p):
    if not isinstance(p.get('corner_features',{}),dict):raise PanelError('Corner definitions must be a dictionary.')
    result={}
    for i,pair in enumerate(CORNER_PAIRS):
        c=corner_feature(p,i)
        if c['kind'] not in ('open','closed','miter','tab') or c['through'] not in pair or c['return_joint'] not in ('butt','miter','open'):
            raise PanelError(CORNER_NAMES[i]+': choose listed corner and return treatments.')
        for k in ('relief','clearance'):
            if not isinstance(c[k],(int,float)) or not math.isfinite(c[k]) or c[k]<=2e-6:raise PanelError(CORNER_NAMES[i]+': relief and clearance must be positive.')
        if c['relief']<p['thickness']:raise PanelError(CORNER_NAMES[i]+': bend relief must be at least the sheet thickness.')
        for s in pair:
            e=c['ends'][s]
            for k in END_FIELDS:
                if not isinstance(e[k],(int,float)) or not math.isfinite(e[k]) or e[k]<0:raise PanelError(CORNER_NAMES[i]+': end dimensions must be finite and non-negative.')
            if bool(e['notch_length'])!=bool(e['notch_height']):raise PanelError('Enter both end-notch length and height, or zero both.')
            if bool(e['tab_projection'])!=bool(e['tab_height']):raise PanelError('Enter both tab projection and height, or zero both.')
            if e['tab_projection']>e['wall_setback']+1e-10:raise PanelError('A retained tab cannot extend past the nominal untrimmed wall end. Increase wall setback or reduce tab projection.')
            if e['tab_radius']>min(e['tab_projection'],e['tab_height']/2)+1e-10:raise PanelError('Tab radius must fit the projection and half the tab height.')
            if not p[s+'_direction']:continue
            top=p[s+'_height']-(p['radius']+p['thickness'] if p[s+'_return'] else 0.)
            for prefix in ('notch','tab'):
                if not e[prefix+'_height']:continue
                if e[prefix+'_bottom']<p['radius']+p['thickness']+c['relief']-1e-10 or e[prefix+'_bottom']+e[prefix+'_height']>top+1e-10:
                    raise PanelError(s.capitalize()+': local end profiles must stay in the straight wall, clear of the base bend. Use outside-height dimensions.')
        if all(p[s+'_direction'] for s in pair) and c['return_joint']=='miter':
            a,b=pair
            if p[a+'_return']!=1 or p[b+'_return']!=1 or p[a+'_direction']!=p[b+'_direction'] or abs(p[a+'_height']-p[b+'_height'])>1e-8:
                raise PanelError(CORNER_NAMES[i]+': mitred returns need two inward returns at the same finished height and bend direction.')
        result[str(i)]=c
    return result


def corner_at(side,near):
    other=ADJACENT[side][0 if near else 1]
    return next(i for i,pair in enumerate(CORNER_PAIRS) if set(pair)=={side,other}),other


def return_z_bounds(p,side):
    plus,minus,split=edge_profile(p,side)
    return min(plus[split:,1].min(),minus[split:,1].min()),max(plus[split:,1].max(),minus[split:,1].max())


_CUSTOM_LAYOUT_KEY=None
_CUSTOM_LAYOUT_VALUE=None


def custom_layout(p):
    global _CUSTOM_LAYOUT_KEY,_CUSTOM_LAYOUT_VALUE
    keys=('width','length','thickness','radius','deduction','gap','seam_gap','segments','corner_features')+tuple(s+'_'+k for s in SIDES for k in ('direction','height','return','lip'))
    key=json.dumps({k:p.get(k) for k in keys},sort_keys=True,separators=(',',':'))
    if key==_CUSTOM_LAYOUT_KEY:return _CUSTOM_LAYOUT_VALUE
    value=_custom_layout_uncached(p);_CUSTOM_LAYOUT_KEY=key;_CUSTOM_LAYOUT_VALUE=value
    return value


def _custom_layout_uncached(p):
    r,t=p['radius'],p['thickness'];ro=r+t;ba=2*ro-p['deduction'];x0,x1,y0,y1=web_bounds(p);result={}
    for side in SIDES:
        if not p[side+'_direction']:continue
        overall=p['length'] if side in ('left','right') else p['width'];ends=[]
        for near in (True,False):
            i,other=corner_at(side,near);c=corner_feature(p,i);e=c['ends'][side];g=c['relief'];active=bool(p[other+'_direction'])
            closed=active and c['kind']!='open';through=closed and c['through']==side
            root=(0. if through else ro+max(2*g,c['clearance'])) if closed else (ro+g if active else 0.)
            nominal=(0. if through else t+c['clearance']) if closed else root
            h0=ro;h1=p[side+'_height']-(ro if p[side+'_return'] else 0.)
            root_top=ro+max(0.,ro+g-ba) if closed and not through else ro
            if e['wall_setback']:root_top=max(root_top,ro+g)
            bands=[]
            if closed and not through and c['return_joint']=='butt' and p[other+'_return']==1:
                zlo,zhi=return_z_bounds(p,other);zlo-=c['clearance'];zhi+=c['clearance']
                hlo,hhi=(zlo,zhi) if p[side+'_direction']>0 else (t-zhi,t-zlo)
                if hlo<h1-1e-10 and hhi>h0+1e-10:
                    if hlo<=h0+1e-8 and p[other+'_lip']+c['clearance']>root+1e-10:raise PanelError('Return interference reaches a base bend. Increase flange height or use an open corner.')
                    extra=max(0.,g-c['clearance']);bands.append((hlo-extra,hhi+extra,p[other+'_lip']+c['clearance']))
            if e['notch_height']:bands.append((e['notch_bottom'],e['notch_bottom']+e['notch_height'],nominal+e['wall_setback']+e['notch_length']))
            if root_top>=h1-2e-7 and closed and not through:raise PanelError('The flange is too shallow for a local root relief and its return. Increase height or use an open corner.')
            end=dict(corner=i,other=other,near=near,config=c,edit=e,root=root,nominal=nominal,
                     body=nominal+e['wall_setback'],root_top=root_top,bands=bands,h0=h0,h1=h1)
            end['path']=custom_end_path(p,end);ends.append(end)
        low,high=ends;lo0=low['path'][0][1];hi0=overall-high['path'][0][1]
        if abs(lo0-low['root'])>1e-9 or abs(overall-hi0-high['root'])>1e-9:raise PanelError('An end cut enters the base-bend attachment. Raise its lower height.')
        info=dict(lo=min(d for h,d in low['path']),hi=overall-min(d for h,d in high['path']),
                  attach_lo=low['root'],attach_hi=overall-high['root'],wall_base_lo=lo0,wall_base_hi=hi0,
                  wall_top_lo=low['path'][-1][1],wall_top_hi=overall-high['path'][-1][1],ends=ends,overall=overall,
                  wall_flat=p[side+'_height']-(2*ro if p[side+'_return'] else ro)+ba,
                  return_flat=ba+p[side+'_lip']-ro if p[side+'_return'] else 0.)
        low_path=[(d,h) for h,d in low['path']];high_path=[(overall-d,h) for h,d in high['path']]
        info['wall_polygon']=low_path+high_path[::-1]
        if min(info['attach_hi']-info['attach_lo'],info['wall_top_hi']-info['wall_top_lo'])<=2e-6 or wall_profile_invalid(info):
            raise PanelError(side.capitalize()+': corner end cuts consume or cross the flange. Reduce the setbacks/notches.')
        info['return_stations']=[]
        if p[side+'_return']:
            plus,minus,split=edge_profile(p,side)
            for j in range(len(plus)-split):
                distances=[]
                for end in ends:
                    c=end['config'];other=end['other'];root=end['path'][-1][1];d=root
                    if p[other+'_direction']:
                        if c['return_joint']=='miter':
                            ratio=p[other+'_lip']/p[side+'_lip']
                            reach=ro-min(plus[split+j,0],minus[split+j,0])
                            d=max(d,ratio*reach+.5*c['clearance']*math.sqrt(1+ratio*ratio))
                        elif p[side+'_return']==1:
                            a,b=return_z_bounds(p,side)
                            if p[other+'_return']==1:
                                u,v=return_z_bounds(p,other)
                                if min(b,v)+c['clearance']>=max(a,u) and (c['kind']=='open' or c['return_joint']=='open' or c['through']!=side):
                                    d=max(d,p[other+'_lip']+c['clearance'])
                            if c['return_joint']=='open' or c['kind']=='open':d=max(d,t+c['clearance'])
                    distances.append(d+end['edit']['return_setback'])
                lo,hi=distances[0],overall-distances[1]
                if hi-lo<=2e-6:raise PanelError(side.capitalize()+': return end cuts meet. Reduce the returns or end setbacks.')
                info['return_stations'].append((lo,hi))
            info['return_lo'],info['return_hi']=info['return_stations'][0]
            if info['return_lo']<info['wall_top_lo']-1e-10 or info['return_hi']>info['wall_top_hi']+1e-10:raise PanelError('Return attachment leaves its wall.')
            for end,offset in zip(ends,(info['return_lo'],overall-info['return_hi'])):
                if offset>end['path'][-1][1]+1e-10:
                    bottom=end['h1']-end['config']['relief']
                    if bottom<=end['h0']+2e-7:raise PanelError('The wall is too shallow for relief at a shortened return. Increase flange height.')
                    end['bands'].append((bottom,end['h1']+1e-8,offset));end['path']=custom_end_path(p,end)
            info['wall_top_lo']=low['path'][-1][1];info['wall_top_hi']=overall-high['path'][-1][1]
            info['lo']=min(d for h,d in low['path']);info['hi']=overall-min(d for h,d in high['path'])
            info['wall_polygon']=[(d,h) for h,d in low['path']]+[(overall-d,h) for h,d in high['path'][::-1]]
            if wall_profile_invalid(info):raise PanelError('Return reliefs consume the wall. Reduce setbacks or increase height.')
        else:info['return_lo'],info['return_hi']=info['wall_top_lo'],info['wall_top_hi']
        result[side]=info
    return result


def custom_end_path(p,end):
    e=end['edit'];h0,h1=end['h0'],end['h1'];body=end['body'];events={h0,h1,end['root_top']}
    for a,b,d in end['bands']:events.update((a,b))
    bottom=e['tab_bottom'];top=bottom+e['tab_height'];rad=e['tab_radius']
    if e['tab_height']:
        events.update((bottom,top))
        if rad:
            for a in np.linspace(0,math.pi/2,max(8,p['segments'])+1):events.update((bottom+rad-rad*math.cos(a),top-rad+rad*math.sin(a)))
    def band(h,a,b,after):return (h>a+1e-12 or abs(h-a)<1e-12 and after) and (h<b-1e-12 or abs(h-b)<1e-12 and not after)
    def distance(h,after):
        d=body
        if e['tab_height'] and band(h,bottom,top,after):
            d-=e['tab_projection']
            if rad:
                if h<bottom+rad:d+=rad-math.sqrt(max(0.,rad*rad-(h-bottom-rad)**2))
                elif h>top-rad:d+=rad-math.sqrt(max(0.,rad*rad-(h-top+rad)**2))
        # The root stays at the actual bend end, even where the main wall
        # is set farther in. This leaves a local retained root tab.
        if h<end['root_top']-1e-12 or abs(h-end['root_top'])<1e-12 and not after:d=end['root']
        for a,b,value in end['bands']:
            if band(h,a,b,after):d=max(d,value)
        return d
    path=[]
    for h in sorted({round(x,12) for x in events if h0-1e-12<=x<=h1+1e-12}):
        if abs(h-h0)<1e-12:h=h0
        if abs(h-h1)<1e-12:h=h1
        for after in ([True] if abs(h-h0)<1e-12 else [False] if abs(h-h1)<1e-12 else [False,True]):
            q=(h,distance(h,after))
            if not path or math.dist(q,path[-1])>1e-10:path.append(q)
    return path


def wall_profile_invalid(info):
    q=info['wall_polygon']
    return len(q)<4 or abs(_area(q))<1e-12 or _boundary_crosses(q)


def custom_web(p):
    x0,x1,y0,y1=web_bounds(p);layout=custom_layout(p);positive=[(x0,x1,y0,y1)];negative=[]
    for i,(a,b) in enumerate(CORNER_PAIRS):
        if not p[a+'_direction'] or not p[b+'_direction']:continue
        c=corner_feature(p,i);g=c['relief']
        if c['kind']=='open':
            x=x0 if a=='left' else x1;y=y0 if b=='front' else y1
            negative.append((x-g,x+g,y-g,y+g));continue
        through=c['through'];butt=b if through==a else a;near=ADJACENT[butt][0]==through
        end=layout[butt]['attach_lo' if near else 'attach_hi']
        if butt in ('left','right'):
            va,vb=(y0,end-g) if near else (end+g,y1)
            if butt=='left':positive.append((0.,x0,va,vb));negative.append((x0,x0+g,end-g if near else end,end if near else end+g))
            else:positive.append((x1,p['width'],va,vb));negative.append((x1-g,x1,end-g if near else end,end if near else end+g))
        else:
            va,vb=(x0,end-g) if near else (end+g,x1)
            if butt=='front':positive.append((va,vb,0.,y0));negative.append((end-g if near else end,end if near else end+g,y0,y0+g))
            else:positive.append((va,vb,y1,p['length']));negative.append((end-g if near else end,end if near else end+g,y1-g,y1))
    xs=sorted({round(v,12) for r in positive+negative for v in r[:2]});ys=sorted({round(v,12) for r in positive+negative for v in r[2:]});patches=[]
    for a,b in zip(xs,xs[1:]):
        for c,d in zip(ys,ys[1:]):
            x,y=(a+b)/2,(c+d)/2
            if any(u<x<v and s<y<t for u,v,s,t in positive) and not any(u<x<v and s<y<t for u,v,s,t in negative):patches.append([(a,c),(b,c),(b,d),(a,d)])
    return trace_net(patches)


def custom_pattern(p):
    ro=p['radius']+p['thickness'];ba=2*ro-p['deduction'];layout=custom_layout(p);patches=[custom_web(p)];lines=[]
    for s,info in layout.items():
        lo,hi=info['attach_lo'],info['attach_hi']
        patches.append([flat_point(p,s,u,v) for u,v in ((0,lo),(0,hi),(ba,hi),(ba,lo))])
        patches.append([flat_point(p,s,ba+h-ro,v) for v,h in info['wall_polygon']])
        lines.append(dict(a=flat_point(p,s,ba/2,lo),b=flat_point(p,s,ba/2,hi),label=s.capitalize(),direction=p[s+'_direction']))
        if p[s+'_return']:
            stations=info['return_stations'];end=info['wall_flat'];n=p['segments']
            us=[end+ba*j/n for j in range(n+1)]+[end+info['return_flat']]
            for j in range(len(us)-1):
                a,b=stations[j];c,d=stations[j+1]
                patches.append([flat_point(p,s,u,v) for u,v in ((us[j],a),(us[j],b),(us[j+1],d),(us[j+1],c))])
            lo,hi=stations[n//2] if n%2==0 else tuple((stations[n//2][k]+stations[n//2+1][k])/2 for k in (0,1))
            lines.append(dict(a=flat_point(p,s,end+ba/2,lo),b=flat_point(p,s,end+ba/2,hi),label=s.capitalize()+' return',direction=p[s+'_direction']*p[s+'_return']))
    holes=openings(p)
    for c,centre in cut_instances(p):
        if c['face']!='base':
            loop=[cut_flat_point(p,c['face'],xy).tolist() for xy in cut_loop(c,centre)]
            holes.append(loop if _area(loop)>0 else loop[::-1])
    return dict(outline=trace_net(patches),holes=holes,bend_lines=lines,layout=layout)

def validate(values):
    p=defaults();p.update(copy.deepcopy(values))
    p['segments']=max(8,min(48,int(p['segments'])))
    validate_angles(p)
    for key in ('width','length','thickness','radius','gap','opening_width','opening_length','deduction','seam_gap'):
        if not isinstance(p[key],(int,float)) or not math.isfinite(p[key]):raise PanelError(f'{key.replace("_"," ").capitalize()} must be finite.')
        if p[key]<0 or (key not in ('opening_width','opening_length','deduction') and p[key]<=0):raise PanelError(f'{key.replace("_"," ").capitalize()} must be positive.')
    if p['units'] not in ('in','mm'):raise PanelError('Choose inches or millimetres.')
    if p['corner_style'] not in dict(CORNER_OPTIONS):raise PanelError('Choose a listed corner treatment.')
    if p['corner_style']=='custom':p['corner_features']=normalize_corner_features(p)
    if p['corner_style'] in ('cab_lr','cab_fb') and not 2e-6<p['seam_gap']<p['radius']:
        raise PanelError('Cab corner seam gap must be positive and smaller than the inside bend radius. It is separate from open-corner relief.')
    if p['corner_style'] in ('cab_lr','cab_fb') and p['seam_gap']<=p['radius']*(1-math.cos(math.pi/(2*p['segments'])))+2e-7:
        factor=INCH if p['units']=='in' else .001
        minimum=(p['radius']*(1-math.cos(math.pi/(2*p['segments'])))+2e-7)/factor
        raise PanelError(f'Increase the cab seam gap above {minimum:.6f} {p["units"]} for the bend resolution, or use more bend segments.')
    if p['gap']<p['thickness']:raise PanelError('Corner relief must be at least the sheet thickness.')
    full=full_width_sides(p)
    if full and p['seam_gap']<=2e-6:raise PanelError('Full-width flange clearance must be positive.')
    if full and not any(p[s+'_direction'] for s in full):raise PanelError('Add a flange on the chosen full-width edge first, or choose another corner treatment.')
    ro=p['radius']+p['thickness'];ba=2*ro-p['deduction']
    active=any(p[s+'_direction'] for s in SIDES)
    if active and not math.pi*p['radius']/2-1e-10<=ba<=math.pi*ro/2+1e-10:
        factor=INCH if p['units']=='in' else .001
        raise PanelError(f'Bend deduction must be {(2*ro-math.pi*ro/2)/factor:.6f}–{(2*ro-math.pi*p["radius"]/2)/factor:.6f} {p["units"]} for this thickness/radius. Reapply the material preset.')
    for side in SIDES:
        if p[side+'_direction'] not in (-1,0,1) or p[side+'_return'] not in (-1,0,1):raise PanelError('Choose a listed bend and return direction.')
        for suffix in ('height','lip'):
            value=p[side+'_'+suffix]
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise PanelError(f'{side.capitalize()} {suffix} must be positive.')
        if not p[side+'_direction']:continue
        if p[side+'_height']<=((bend_data(p,side)[0]+(bend_data(p,side,True)[0] if p[side+'_return'] else 0.)) if angled(p) else (2*ro if p[side+'_return'] else ro))+2e-6:raise PanelError(f'{side.capitalize()} height is too small for its bend radius and return.')
        if p[side+'_return'] and p[side+'_lip']<=(bend_data(p,side,True)[0] if angled(p) else ro)+2e-6:raise PanelError(f'{side.capitalize()} return length must exceed radius + thickness.')
    x0,x1,y0,y1=web_bounds(p)
    if min(x1-x0,y1-y0)<=2*p['gap']+2e-6:raise PanelError('Panel is too small for its bends and corner reliefs.')
    for side,info in side_layout(p).items():
        if p[side+'_return'] and (info['return_lo']<info['lo']-1e-10 or info['return_hi']>info['hi']+1e-10):raise PanelError('The corner gap leaves a return beyond its flange. Reduce the seam gap.')
        if info['hi']-info['lo']<=2e-6:raise PanelError(f'{side.capitalize()} corner cutouts consume the entire flange.')
        if p[side+'_return'] and info['return_hi']-info['return_lo']<=2e-6:raise PanelError(f'{side.capitalize()} return end cutouts meet. Increase the panel size or shorten adjacent returns.')
        across=p['width'] if side in ('left','right') else p['length']
        if p[side+'_return']==1 and p[side+'_lip']>=across-ro-p['gap']:raise PanelError(f'{side.capitalize()} inward return reaches the opposite edge. Shorten it.')
    for a,b,across in (('left','right',p['width']),('front','back',p['length'])):
        if all(p[s+'_direction'] and p[s+'_return']==1 for s in (a,b)):
            if p[a+'_lip']+p[b+'_lip']+p['gap']>=across:raise PanelError('Opposite inward returns overlap. Shorten them or increase the panel size.')
    ow,ol=p['opening_width'],p['opening_length']
    if bool(ow)!=bool(ol):raise PanelError('Enter both opening dimensions, or set both to zero for no opening.')
    if ow and (ow>=x1-x0-2*p['gap'] or ol>=y1-y0-2*p['gap']):raise PanelError('The centred opening must clear the straight web edges and corner reliefs.')
    p['cuts']=normalize_cuts(p)
    normalized=[]
    for hole in p.get('holes',[]):
        q=np.asarray(hole,dtype=float)
        if q.ndim!=2 or q.shape[1]!=2 or len(q)<3 or not np.isfinite(q).all():raise PanelError('An opening has an invalid boundary.')
        loop=clean_loop(q)
        if len(loop)<3 or abs(_area(loop))<1e-14 or _boundary_crosses(loop):raise PanelError('An opening is degenerate or crosses itself.')
        normalized.append(loop if _area(loop)>0 else loop[::-1])
    p['holes']=normalized
    loops=openings(p)
    web=web_outline(p) if full or p['corner_style']=='custom' else None
    for i,hole in enumerate(loops):
        a=np.asarray(hole);low=a.min(0);high=a.max(0)
        if low[0]<=x0+2e-7 or high[0]>=x1-2e-7 or low[1]<=y0+2e-7 or high[1]>=y1-2e-7:raise PanelError('An opening enters a bend or touches the straight-web boundary. Resize the base or revise the opening.')
        for size,cx,cy in zip(corners(p),(x0,x1,x1,x0),(y0,y0,y1,y1)):
            if size and low[0]<cx+size and high[0]>cx-size and low[1]<cy+size and high[1]>cy-size:raise PanelError('An opening enters a corner relief. Increase the base or revise the opening.')
        if web is not None and (_boundary_crosses(hole,web) or not all(_contains(web,xy) for xy in hole)):
            raise PanelError('A base opening enters a full-width corner bend relief. Move the opening or revise the part.')
        for other in loops[:i]:
            b=np.asarray(other)
            if np.any(high<b.min(0)-2e-7) or np.any(low>b.max(0)+2e-7):continue
            if _boundary_crosses(hole,other) or _contains(other,hole[0]) or _contains(hole,other[0]):raise PanelError('Two openings overlap or touch. Revise their sizes or positions.')
    validate_cuts(p)
    if angled(p):validate_angle_clearance(p)
    p['segments']=max(8,min(48,int(p['segments'])))
    p['name']=str(p['name']).strip() or 'Cab panel';p['material']=str(p['material']).strip() or 'Unspecified'
    return p

def clean_loop(points):
    out=[]
    for point in points:
        point=tuple(float(x) for x in point)
        if not out or np.linalg.norm(np.asarray(point)-out[-1])>1e-10:out.append(point)
    if len(out)>1 and np.linalg.norm(np.asarray(out[0])-out[-1])<1e-10:out.pop()
    return out

def web_outline(p):
    x0,x1,y0,y1=web_bounds(p);bl,br,tr,tl=corners(p)
    if p.get('corner_style')=='custom':return custom_web(p)
    if full_width_sides(p):return full_width_web(p)
    return clean_loop([(x0+bl,y0),(x1-br,y0),(x1-br,y0+br),(x1,y0+br),
        (x1,y1-tr),(x1-tr,y1-tr),(x1-tr,y1),(x0+tl,y1),
        (x0+tl,y1-tl),(x0,y1-tl),(x0,y0+bl),(x0+bl,y0+bl)])


def full_width_web(p):
    """Developable planar web, tongues and open rectangular bend reliefs.

    Each shortened side bend has a gap-wide relief extending gap beyond its
    web tangent. A tongue beside it supports the full-width through bend.
    No two bend strips intersect, and no uncut stock ties a bend end to a tongue.
    """
    x0,x1,y0,y1=web_bounds(p);g=p['gap'];layout=side_layout(p)
    positive=[(x0,x1,y0,y1)];negative=[]
    for size,x,y in zip(corners(p),(x0,x1,x1,x0),(y0,y0,y1,y1)):
        if size:negative.append((x-size,x+size,y-size,y+size))
    for a,b in CORNER_PAIRS:
        through=closed_corner(p,a,b)
        if not through:continue
        butt=b if through==a else a
        near=ADJACENT[butt][0]==through
        end=layout[butt]['attach_lo' if near else 'attach_hi']
        if butt in ('left','right'):
            va,vb=(y0,end-g) if near else (end+g,y1)
            if butt=='left':positive.append((0.,x0,va,vb));negative.append((x0,x0+g,end-g if near else end,end if near else end+g))
            else:positive.append((x1,p['width'],va,vb));negative.append((x1-g,x1,end-g if near else end,end if near else end+g))
        else:
            va,vb=(x0,end-g) if near else (end+g,x1)
            if butt=='front':positive.append((va,vb,0.,y0));negative.append((end-g if near else end,end if near else end+g,y0,y0+g))
            else:positive.append((va,vb,y1,p['length']));negative.append((end-g if near else end,end if near else end+g,y1-g,y1))
    xs=sorted({round(v,12) for r in positive+negative for v in r[:2]})
    ys=sorted({round(v,12) for r in positive+negative for v in r[2:]})
    patches=[]
    for a,b in zip(xs,xs[1:]):
        for c,d in zip(ys,ys[1:]):
            x,y=(a+b)/2,(c+d)/2
            if any(u<x<v and s<y<t for u,v,s,t in positive) and not any(u<x<v and s<y<t for u,v,s,t in negative):
                patches.append([(a,c),(b,c),(b,d),(a,d)])
    return trace_net(patches)

def opening(p):
    if not p['opening_width']:return []
    x0,x1,y0,y1=web_bounds(p);cx=(x0+x1)/2;cy=(y0+y1)/2
    a,b=p['opening_width']/2,p['opening_length']/2
    return [(cx-a,cy-b),(cx+a,cy-b),(cx+a,cy+b),(cx-a,cy+b)]

def edge_profile(p,side):
    t,r=p['thickness'],p['radius'];ro=r+t;rm=r+t/2
    d=p[side+'_direction'];ret=p[side+'_return'];h=p[side+'_height']
    pts=[(0.,t/2)];tans=[(1.,0.)];split=None
    cz=ro if d>0 else -r
    for j in range(1,p['segments']+1):
        a=-d*math.pi/2+d*math.pi*j/(2*p['segments'])
        pts.append((rm*math.cos(a),cz+rm*math.sin(a)));tans.append((-d*math.sin(a),d*math.cos(a)))
    pts.append((rm,(h-ro if d>0 else t-h+ro) if ret else (h if d>0 else t-h)))
    tans.append((0.,d));split=len(pts)-1
    if ret:
        v=-ret;rz=pts[-1][1];cx=rm+v*rm;start=math.pi if v>0 else 0.
        for j in range(1,p['segments']+1):
            a=start-v*d*math.pi*j/(2*p['segments'])
            pts.append((cx+rm*math.cos(a),rz+rm*math.sin(a)));tans.append((v*d*math.sin(a),-v*d*math.cos(a)))
        pts.append((rm-v*t/2+v*p[side+'_lip'],rz+d*rm));tans.append((v,0.))
    q=np.asarray(pts);tan=np.asarray(tans);normal=np.column_stack((-tan[:,1],tan[:,0]))
    return q+t/2*normal,q-t/2*normal,split

def point_map(p,side,q,v):
    x0,x1,y0,y1=web_bounds(p);u,z=q
    if side=='left':return (x0-u,v,z)
    if side=='right':return (x1+u,v,z)
    if side=='front':return (v,y0-u,z)
    return (v,y1+u,z)

def flat_point(p,side,u,v):return point_map(p,side,(u,0.),v)[:2]


# Face coordinates are geometric datums, not transient mesh/face indices.
# A wall's x runs front-to-back (left/right) or left-to-right (front/back).
# Its y is outside height from the base, even when its bend points DOWN.
def cut_face_info(p,face):
    if face=='base':
        x0,x1,y0,y1=web_bounds(p)
        return dict(label='Base',bounds=(x0,x1,y0,y1),xrefs=(0.,p['width']),
                    yrefs=(0.,p['length']),xlabels=('Left outside edge','Right outside edge'),
                    ylabels=('Front outside edge','Back outside edge'))
    if face not in SIDES or not p[face+'_direction']:
        raise PanelError('A cut needs its original flange. Delete its cut features before removing that flange.')
    info=side_layout(p)[face];ro=p['radius']+p['thickness'];h=p[face+'_height']
    if p.get('corner_style')=='custom':
        return dict(label=face.capitalize()+' flange',bounds=(info['lo'],info['hi'],ro,h-ro if p[face+'_return'] else h),
                    xrefs=(info['ends'][0]['body'],info['overall']-info['ends'][1]['body']),yrefs=(0.,h),
                    xlabels=('Main wall front end','Main wall back end') if face in ('left','right') else ('Main wall left end','Main wall right end'),
                    ylabels=('Base outside plane','Top outside sharp'))
    return dict(label=face.capitalize()+' flange',bounds=(info['lo'],info['hi'],ro,h-ro if p[face+'_return'] else h),
                xrefs=(info['lo'],info['hi']),yrefs=(0.,h),
                xlabels=('Front end','Back end') if face in ('left','right') else ('Left end','Right end'),
                ylabels=('Base outside plane','Top outside sharp'))


def normalize_cuts(p):
    raw=p.get('cuts',[])
    if not isinstance(raw,list) or len(raw)>128:raise PanelError('Use at most 128 cut features per part.')
    result=[];ids=set();instances=0
    for index,source in enumerate(raw):
        if not isinstance(source,dict):raise PanelError('A cut feature has invalid data.')
        c=dict(id=str(source.get('id','cut_'+str(index+1))),name=str(source.get('name','Cut '+str(index+1))).strip() or 'Cut',
               face=source.get('face','base'),kind=source.get('kind','hole'),
               x_ref=source.get('x_ref','start'),y_ref=source.get('y_ref','end'))
        if c['id'] in ids:raise PanelError('Cut features have duplicate identifiers.')
        ids.add(c['id'])
        if c['kind'] not in ('hole','slot','rectangle'):raise PanelError('Choose a round hole, obround slot or rectangular cutout.')
        if c['x_ref'] not in ('start','end') or c['y_ref'] not in ('start','end'):raise PanelError('Choose the cut position reference edges.')
        cut_face_info(p,c['face'])
        for key,default in (('x',INCH),('y',INCH),('diameter',.375*INCH),('width',.375*INCH),('length',.75*INCH),('angle',0.),('pitch',INCH)):
            value=source.get(key,default)
            if not isinstance(value,(int,float)) or not math.isfinite(value):raise PanelError('Cut dimensions must be finite numbers.')
            if key!='angle' and (value<0 or (key in ('diameter','width','length') and value<=2e-6)):
                raise PanelError('Cut sizes must be positive and centre offsets cannot be negative.')
            c[key]=float(value)
        count=source.get('count',1)
        if not isinstance(count,(int,float)) or not math.isfinite(count) or int(count)!=count or not 1<=count<=100:
            raise PanelError('A cut row needs a quantity from 1 to 100.')
        c['count']=int(count);instances+=c['count']
        if instances>256:raise PanelError('Use at most 256 cut instances per part.')
        if c['count']>1 and c['pitch']<=0:raise PanelError('Repeated cuts need a positive centre-to-centre pitch.')
        if c['kind']=='slot' and c['length']<c['width']:raise PanelError('Slot overall length must be at least its width.')
        result.append(c)
    return result


def cut_instances(p,face=None):
    for c in p.get('cuts',[]):
        if face is not None and c['face']!=face:continue
        f=cut_face_info(p,c['face']);xr,yr=f['xrefs'],f['yrefs']
        for i in range(c.get('count',1)):
            dx=c['x']+i*c['pitch'];x=xr[0]+dx if c['x_ref']=='start' else xr[1]-dx
            y=yr[0]+c['y'] if c['y_ref']=='start' else yr[1]-c['y']
            yield c,np.array([x,y],dtype=float)


def cut_core(c,centre):
    a=math.radians(c['angle']);u=np.array([math.cos(a),math.sin(a)]);v=np.array([-u[1],u[0]])
    if c['kind']=='hole':return np.array([centre]),c['diameter']/2
    if c['kind']=='slot':
        h=(c['length']-c['width'])/2
        return np.array([centre-h*u,centre+h*u]),c['width']/2
    h,w=c['length']/2,c['width']/2
    return np.array([centre-h*u-w*v,centre+h*u-w*v,centre+h*u+w*v,centre-h*u+w*v]),0.


def cut_loop(c,centre):
    core,r=cut_core(c,centre)
    if not r:return core.tolist()
    angles=np.arange(64)*2*math.pi/64
    if len(core)==1 or np.linalg.norm(core[-1]-core[0])<1e-12:
        return (centre+r*np.column_stack((np.cos(angles),np.sin(angles)))).tolist()
    a=math.radians(c['angle']);right=np.linspace(a-math.pi/2,a+math.pi/2,33)
    left=np.linspace(a+math.pi/2,a+3*math.pi/2,33)
    return np.concatenate((core[1]+r*np.column_stack((np.cos(right),np.sin(right))),
                           core[0]+r*np.column_stack((np.cos(left),np.sin(left))))).tolist()


def cut_loops(p,face):return [cut_loop(c,centre) for c,centre in cut_instances(p,face)]


def _core_distance(a,b):
    # Exact distance between convex point/segment/rectangle cores; adding the
    # radii below also catches touching circles and rotated capsule slots.
    def segments(q):return [(q[0],q[0])] if len(q)==1 else list(zip(q,np.roll(q,-1,axis=0)))
    def point_segment(p,u,v):
        d=v-u;length=float(d@d);k=0. if length<1e-24 else max(0.,min(1.,float((p-u)@d)/length))
        return float(np.linalg.norm(p-u-k*d))
    if len(a)>2 and _contains(a.tolist(),b[0]):return 0.
    if len(b)>2 and _contains(b.tolist(),a[0]):return 0.
    best=float('inf')
    for u,v in segments(a):
        for w,z in segments(b):
            d=v-u;e=z-w;den=d[0]*e[1]-d[1]*e[0]
            if abs(den)>1e-24:
                q=w-u;s=(q[0]*e[1]-q[1]*e[0])/den;t=(q[0]*d[1]-q[1]*d[0])/den
                if 0<=s<=1 and 0<=t<=1:return 0.
            best=min(best,point_segment(u,w,z),point_segment(v,w,z),point_segment(w,u,v),point_segment(z,u,v))
    return best


def validate_cuts(p):
    previous=[]
    for c,centre in cut_instances(p):
        f=cut_face_info(p,c['face']);x0,x1,y0,y1=f['bounds'];core,r=cut_core(c,centre)
        low,high=core.min(0)-r,core.max(0)+r;tol=2e-7
        if low[0]<=x0+tol or high[0]>=x1-tol or low[1]<=y0+tol or high[1]>=y1-tol:
            raise PanelError(c['name']+' enters a bend, relief or face edge. Keep the entire cut inside the straight face.')
        if p.get('corner_style')=='custom' and c['face']!='base':
            wall=custom_layout(p)[c['face']]['wall_polygon'];loop=cut_loop(c,centre)
            distance=min(_core_distance(core,np.array([u,v])) for u,v in zip(wall,wall[1:]+wall[:1]))
            if distance<=r+tol or not _contains(wall,centre):raise PanelError(c['name']+' enters a local corner end cut or tab boundary.')
        for other,other_core,other_r,other_low,other_high in previous:
            if c['face']!=other['face'] or high[0]<other_low[0]-tol or low[0]>other_high[0]+tol or high[1]<other_low[1]-tol or low[1]>other_high[1]+tol:continue
            if _core_distance(core,other_core)<=r+other_r+tol:
                raise PanelError(c['name']+' overlaps or touches another cut. Revise the sizes, positions or row pitch.')
        previous.append((c,core,r,low,high))


def cut_face_point(p,face,xy,skin=0):
    if face=='base':return (xy[0],xy[1],skin*p['thickness'])
    plus,minus,_=edge_profile(p,face);chain=plus if skin==0 else minus
    z=xy[1] if p[face+'_direction']>0 else p['thickness']-xy[1]
    return point_map(p,face,(chain[p['segments']][0],z),xy[0])


def cut_flat_point(p,face,xy):
    if face=='base':return np.asarray(xy,dtype=float)
    ro=p['radius']+p['thickness'];ba=2*ro-p['deduction']
    return np.asarray(flat_point(p,face,ba+xy[1]-ro,xy[0]))


def match_cut_face(values,points,tolerance=2e-7):
    p=validate(values)
    if p['flat_view']:return None
    q=np.asarray(points)
    if len(q)<3:return None
    for face in ('base',)+tuple(s for s in SIDES if p[s+'_direction']):
        f=cut_face_info(p,face);x0,x1,y0,y1=f['bounds']
        if face=='base' and (full_width_sides(p) or p.get('corner_style')=='custom'):x0,x1,y0,y1=0.,p['width'],0.,p['length']
        for skin in (0,1):
            origin=np.asarray(cut_face_point(p,face,(0.,0.),skin))
            u=np.asarray(cut_face_point(p,face,(1.,0.),skin))-origin
            v=np.asarray(cut_face_point(p,face,(0.,1.),skin))-origin;n=np.cross(u,v)
            if np.max(np.abs((q-origin)@n))>tolerance:continue
            xy=np.column_stack(((q-origin)@u,(q-origin)@v))
            if np.all(xy.min(0)>=np.array([x0,y0])-tolerance) and np.all(xy.max(0)<=np.array([x1,y1])+tolerance):return face
    return None

def strip_span(p,info,index):
    n=p['segments'];index=min(float(index),n)
    if index<n and abs(index-round(index))>1e-10:
        i=math.floor(index);a=strip_span(p,info,i);b=strip_span(p,info,i+1);f=index-i
        return tuple(a[k]+(b[k]-a[k])*f for k in (0,1))
    f=min(1.,index/n);r=p['radius'];ro=r+p['thickness']
    def extension(amount):
        if abs(abs(amount)-ro)>1e-10:return amount*f
        # Until the through flange has cleared the neighbouring bend, keep
        # its projected end behind its inner-radius projection. This prevents
        # intersecting bent stock, including with R approximately thickness.
        step=math.pi/(2*n);threshold=math.asin((r-p['seam_gap'])/r)
        start=min(n-1,int(math.floor(threshold/step))+1);angle=f*math.pi/2
        if index<=start:value=r*math.sin(angle)
        else:
            initial=r*math.sin(start*step);value=initial+(ro-initial)*(index-start)/(n-start)
        return math.copysign(value,amount)
    return (info['attach_lo']+extension(info['lo']-info['attach_lo']),
            info['attach_hi']+extension(info['hi']-info['attach_hi']))


def trace_net(polygons):
    # Adjacent non-overlapping blank patches share partial edges. Split every
    # shared edge at its actual endpoints, then remove only internal edges.
    polygons=[clean_loop(q if _area(q)>0 else q[::-1]) for q in polygons]
    points={tuple(round(float(v),12) for v in q) for poly in polygons for q in poly};points=np.array(sorted(points))
    boundary={}
    for poly in polygons:
        for a,b in zip(poly,poly[1:]+poly[:1]):
            a,b=np.array(a),np.array(b);d=b-a;length=float(d@d)
            if length<1e-22:continue
            fraction=(points-a)@d/length;distance=np.abs(d[0]*(points[:,1]-a[1])-d[1]*(points[:,0]-a[0]))/math.sqrt(length)
            margin=2e-11/math.sqrt(length)
            ids=np.flatnonzero((fraction>=-margin)&(fraction<=1+margin)&(distance<2e-11))
            ids=ids[np.argsort(fraction[ids])];chain=[tuple(points[i]) for i in ids]
            for u,v in zip(chain,chain[1:]):
                if (v,u) in boundary:boundary.pop((v,u))
                elif (u,v) in boundary:raise PanelError('Corner blank patches overlap.')
                else:boundary[u,v]=True
    next_point={}
    for a,b in boundary:
        if a in next_point:raise PanelError('A developed corner touches itself. Increase the seam gap.')
        next_point[a]=b
    loops=[]
    while next_point:
        start=next(iter(next_point));point=start;loop=[]
        while True:
            loop.append(point)
            if point not in next_point:raise PanelError('The developed corner has an open cut outline.')
            point=next_point.pop(point)
            if point==start:break
        loops.append(loop)
    if len(loops)!=1 or _boundary_crosses(loops[0]):raise PanelError('The developed corner blank is disconnected or overlaps.')
    loop=loops[0];changed=True
    while changed and len(loop)>4:
        changed=False
        for i in range(len(loop)):
            a,b,c=np.asarray(loop[i-1]),np.asarray(loop[i]),np.asarray(loop[(i+1)%len(loop)])
            u,v=b-a,c-b
            if abs(u[0]*v[1]-u[1]*v[0])<2e-15:loop.pop(i);changed=True;break
    return loop


def closed_flat_pattern(p):
    ba=2*(p['radius']+p['thickness'])-p['deduction'];polygons=[web_outline(p)];lines=[]
    def patch(side,points):polygons.append([flat_point(p,side,u,s) for u,s in points])
    for side,info in side_layout(p).items():
        lo,hi,alo,ahi=info['lo'],info['hi'],info['attach_lo'],info['attach_hi'];end=info['wall_flat']
        for i in range(p['segments']):
            l0,h0=strip_span(p,info,i);l1,h1=strip_span(p,info,i+1)
            patch(side,[(ba*i/p['segments'],l0),(ba*i/p['segments'],h0),
                        (ba*(i+1)/p['segments'],h1),(ba*(i+1)/p['segments'],l1)])
        patch(side,[(ba,lo),(ba,hi),(end,hi),(end,lo)])
        ml,mh=strip_span(p,info,p['segments']/2)
        lines.append(dict(a=flat_point(p,side,ba/2,ml),b=flat_point(p,side,ba/2,mh),label=side.capitalize(),direction=p[side+'_direction']))
        if p[side+'_return']:
            rlo,rhi=info['return_lo'],info['return_hi'];rend=end+info['return_flat']
            patch(side,[(end,rlo),(end,rhi),(rend,rhi),(rend,rlo)])
            lines.append(dict(a=flat_point(p,side,end+ba/2,rlo),b=flat_point(p,side,end+ba/2,rhi),label=side.capitalize()+' return',direction=p[side+'_direction']*p[side+'_return']))
    holes=openings(p)
    for c,centre in cut_instances(p):
        if c['face']!='base':
            ring=[cut_flat_point(p,c['face'],xy).tolist() for xy in cut_loop(c,centre)]
            holes.append(ring if _area(ring)>0 else ring[::-1])
    return dict(outline=trace_net(polygons),holes=holes,bend_lines=lines,layout=side_layout(p))


def flat_pattern(values):
    p=validate(values);x0,x1,y0,y1=web_bounds(p);g=p['gap'];cuts=corners(p)
    if p.get('corner_style')=='custom':return custom_pattern(p)
    if has_closed_corners(p):return closed_flat_pattern(p)
    rects=[];lines=[]
    for side,info in side_layout(p).items():
        for lo,hi,u0,u1 in [(info['lo'],info['hi'],0.,info['wall_flat'])]+(
                [(info['return_lo'],info['return_hi'],info['wall_flat'],info['wall_flat']+info['return_flat'])] if p[side+'_return'] else []):
            a,b=flat_point(p,side,u0,lo),flat_point(p,side,u1,hi)
            rects.append((min(a[0],b[0]),max(a[0],b[0]),min(a[1],b[1]),max(a[1],b[1])))
        ba=2*(p['radius']+p['thickness'])-p['deduction']
        lines.append(dict(a=flat_point(p,side,ba/2,info['lo']),b=flat_point(p,side,ba/2,info['hi']),label=side.capitalize(),direction=p[side+'_direction']))
        if p[side+'_return']:
            u=info['wall_flat']+ba/2
            lines.append(dict(a=flat_point(p,side,u,info['return_lo']),b=flat_point(p,side,u,info['return_hi']),label=side.capitalize()+' return',direction=p[side+'_direction']*p[side+'_return']))
    # Trace the union of the axis-aligned blank regions. This preserves the
    # square reliefs without inserting grid seams in the physical faces.
    xs={x0,x1,x0+cuts[0],x1-cuts[1],x1-cuts[2],x0+cuts[3]}
    ys={y0,y1,y0+cuts[0],y0+cuts[1],y1-cuts[2],y1-cuts[3]}
    for a,b,c,d in rects:xs.update((a,b));ys.update((c,d))
    xs=sorted(set(round(x,12) for x in xs));ys=sorted(set(round(y,12) for y in ys));filled=set()
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            x=(xs[i]+xs[i+1])/2;y=(ys[j]+ys[j+1])/2
            base=x0<x<x1 and y0<y<y1
            for size,cx,cy in zip(cuts,(x0,x1,x1,x0),(y0,y0,y1,y1)):
                if size and abs(x-cx)<size and abs(y-cy)<size:base=False
            if base or any(a<x<b and c<y<d for a,b,c,d in rects):filled.add((i,j))
    boundary=[]
    for i,j in filled:
        for other,a,b in [((i,j-1),(i,j),(i+1,j)),((i+1,j),(i+1,j),(i+1,j+1)),
                          ((i,j+1),(i+1,j+1),(i,j+1)),((i-1,j),(i,j+1),(i,j))]:
            if other not in filled:boundary.append((a,b))
    next_point={}
    for a,b in boundary:
        if a in next_point:raise PanelError('Blank has an ambiguous corner. Increase the corner relief.')
        next_point[a]=b
    loops=[]
    while next_point:
        start=next(iter(next_point));point=start;loop=[]
        while True:
            loop.append((xs[point[0]],ys[point[1]]));point=next_point.pop(point)
            if point==start:break
        loops.append(loop)
    if len(loops)!=1:raise PanelError('The developed blank is disconnected.')
    loop=loops[0];changed=True
    while changed and len(loop)>4:
        changed=False
        for i in range(len(loop)):
            a,b,c=np.asarray(loop[i-1]),np.asarray(loop[i]),np.asarray(loop[(i+1)%len(loop)])
            u,v=b-a,c-b
            if abs(u[0]*v[1]-u[1]*v[0])<1e-18:loop.pop(i);changed=True;break
    holes=openings(p)
    for c,centre in cut_instances(p):
        if c['face']!='base':
            ring=[cut_flat_point(p,c['face'],xy).tolist() for xy in cut_loop(c,centre)]
            holes.append(ring if _area(ring)>0 else ring[::-1])
    return dict(outline=loop,holes=holes,bend_lines=lines,layout=side_layout(p))

def orient_and_check(faces):
    edges=defaultdict(list)
    for fi,rings in enumerate(faces):
        for ring in rings:
            keys=[tuple(round(x,10) for x in q) for q in ring]
            if len(set(keys))<3:raise PanelError('A corner feature is too small.')
            for a,b in zip(keys,keys[1:]+keys[:1]):
                if a==b:raise PanelError('A corner contains a zero length edge.')
                edges[tuple(sorted((a,b)))].append((fi,a,b))
    graph=defaultdict(list)
    for key,entries in edges.items():
        if len(entries)!=2:raise PanelError(f'Closed sheet boundary failed ({len(entries)} faces at {key}).')
        (a,u,v),(b,s,t)=entries;same=(u==s)
        graph[a].append((b,same));graph[b].append((a,same))
    flips={0:False};todo=deque([0])
    while todo:
        a=todo.popleft()
        for b,same in graph[a]:
            value=flips[a]^same
            if b in flips:
                if flips[b]!=value:raise PanelError('The sheet surface is not orientable.')
            else:flips[b]=value;todo.append(b)
    if len(flips)!=len(faces):raise PanelError('The sheet contains disconnected pieces.')
    for i,flip in flips.items():
        if flip:faces[i]=[ring[::-1] for ring in faces[i]]
    volume=0.
    for rings in faces:
        for ring in rings:
            q=np.asarray(ring)
            for i in range(1,len(q)-1):volume+=float(q[0]@np.cross(q[i],q[i+1]))/6
    if abs(volume)<1e-16:raise PanelError('The sheet has no solid volume.')
    if volume<0:
        for i in range(len(faces)):faces[i]=[ring[::-1] for ring in faces[i]]
    return abs(volume)

def build_geometry(values):
    p=validate(values);t=p['thickness'];faces=[];side_ranges={}
    if p.get('corner_style')=='custom':return build_custom_geometry(p)
    def add(outer,holes=()):faces.append([clean_loop(outer)]+[clean_loop(h) for h in holes])
    flat=flat_pattern(p)
    if p['flat_view']:
        loops=[flat['outline']]+flat['holes']
        for z in (0.,t):add([(x,y,z) for x,y in loops[0]], [[(x,y,z) for x,y in h[::-1]] for h in loops[1:]])
        for ring in loops:
            for a,b in zip(ring,ring[1:]+ring[:1]):add([(a[0],a[1],0.),(b[0],b[1],0.),(b[0],b[1],t),(a[0],a[1],t)])
    else:
        outline=web_outline(p);holes=openings(p)
        for z in (0.,t):add([(x,y,z) for x,y in outline], [[(x,y,z) for x,y in hole[::-1]] for hole in holes])
        interfaces={}
        for side,info in side_layout(p).items():
            a=flat_point(p,side,0.,info['attach_lo']);b=flat_point(p,side,0.,info['attach_hi'])
            interfaces[frozenset((tuple(round(v,10) for v in a),tuple(round(v,10) for v in b)))]=side
        for ring,is_hole in [(outline,False)]+[(hole,True) for hole in holes]:
            for a,b in zip(ring,ring[1:]+ring[:1]):
                key=frozenset((tuple(round(v,10) for v in a),tuple(round(v,10) for v in b)))
                if not is_hole and key in interfaces:continue
                add([(a[0],a[1],0.),(b[0],b[1],0.),(b[0],b[1],t),(a[0],a[1],t)])
        for side,info in side_layout(p).items():
            first_face=len(faces)
            plus,minus,split=edge_profile(p,side);lo,hi=info['lo'],info['hi']
            ret=p[side+'_return'];rlo,rhi=info['return_lo'],info['return_hi']
            def sweep(start,end,v0,v1):
                tapered=start==0 and (abs(info['attach_lo']-v0)>1e-12 or abs(info['attach_hi']-v1)>1e-12)
                for skin,chain in enumerate((plus,minus)):
                    for i in range(start,end):
                        # Split the last wall-skin edge where the shorter return
                        # attaches; the adjacent free end caps share these edges.
                        l0,h0=strip_span(p,info,i) if tapered else (v0,v1)
                        l1,h1=strip_span(p,info,i+1) if tapered else (v0,v1)
                        breaks=sorted(set([l1,h1]+([rlo,rhi] if ret and end==split and i==end-1 else [])))
                        ring=[point_map(p,side,chain[i],l0),point_map(p,side,chain[i],h0)]
                        ring += [point_map(p,side,chain[i+1],v) for v in breaks[::-1]]
                        holes=([cut_face_point(p,side,xy,skin) for xy in loop[::-1]] for loop in cut_loops(p,side)) if i==p['segments'] else ()
                        add(ring,holes)
                if tapered:
                    for which in (0,1):
                        for i in range(start,end):
                            s0=strip_span(p,info,i)[which];s1=strip_span(p,info,i+1)[which]
                            ring=[point_map(p,side,plus[i],s0),point_map(p,side,plus[i+1],s1),point_map(p,side,minus[i+1],s1),point_map(p,side,minus[i],s0)]
                            # A tapered bend cut is a ruled surface; triangles
                            # keep its small end faces planar in the native mesh.
                            add([ring[0],ring[1],ring[2]]);add([ring[0],ring[2],ring[3]])
                else:
                    section=list(plus[start:end+1])+list(minus[start:end+1][::-1])
                    for v in (v0,v1):add([point_map(p,side,q,v) for q in section])
            sweep(0,split,lo,hi)
            if ret:
                sweep(split,len(plus)-1,rlo,rhi)
                free=[(lo,rlo),(rhi,hi)]
            else:free=[(lo,hi)]
            for a,b in free:
                if b-a>1e-10:add([point_map(p,side,plus[split],a),point_map(p,side,plus[split],b),point_map(p,side,minus[split],b),point_map(p,side,minus[split],a)])
            if ret:add([point_map(p,side,plus[-1],rlo),point_map(p,side,plus[-1],rhi),point_map(p,side,minus[-1],rhi),point_map(p,side,minus[-1],rlo)])
            for loop in cut_loops(p,side):
                for a,b in zip(loop,loop[1:]+loop[:1]):
                    add([cut_face_point(p,side,a,0),cut_face_point(p,side,b,0),cut_face_point(p,side,b,1),cut_face_point(p,side,a,1)])
            side_ranges[side]=(first_face,len(faces))
    volume=orient_and_check(faces)
    return dict(params=p,faces=faces,flat=flat,volume=volume,side_face_ranges=side_ranges)

def build_custom_geometry(p):
    t=p['thickness'];flat=custom_pattern(p);faces=[];side_ranges={}
    def add(loop,holes=()):faces.append([clean_loop(loop)]+[clean_loop(h) for h in holes])
    if p['flat_view']:
        loops=[flat['outline']]+flat['holes']
        for z in (0.,t):add([(x,y,z) for x,y in loops[0]],[[(x,y,z) for x,y in h[::-1]] for h in loops[1:]])
        for ring in loops:
            for a,b in zip(ring,ring[1:]+ring[:1]):add([(a[0],a[1],0.),(b[0],b[1],0.),(b[0],b[1],t),(a[0],a[1],t)])
    else:
        web=custom_web(p);holes=openings(p);layout=flat['layout'];n=p['segments']
        for z in (0.,t):add([(x,y,z) for x,y in web],[[(x,y,z) for x,y in h[::-1]] for h in holes])
        interfaces={frozenset(tuple(round(v,10) for v in flat_point(p,s,0.,x)) for x in (info['attach_lo'],info['attach_hi'])) for s,info in layout.items()}
        for ring,is_hole in [(web,False)]+[(h,True) for h in holes]:
            for a,b in zip(ring,ring[1:]+ring[:1]):
                key=frozenset(tuple(round(v,10) for v in x) for x in (a,b))
                if not is_hole and key in interfaces:continue
                add([(a[0],a[1],0.),(b[0],b[1],0.),(b[0],b[1],t),(a[0],a[1],t)])
        for s,info in layout.items():
            start=len(faces);plus,minus,split=edge_profile(p,s);lo,hi=info['attach_lo'],info['attach_hi']
            for chain in (plus,minus):
                for j in range(n):add([point_map(p,s,chain[j],lo),point_map(p,s,chain[j],hi),point_map(p,s,chain[j+1],hi),point_map(p,s,chain[j+1],lo)])
            for v in (lo,hi):add([point_map(p,s,q,v) for q in list(plus[:n+1])+list(minus[:n+1][::-1])])
            low,high=info['ends'];overall=info['overall']
            a=[(d,h) for h,d in low['path']];b=[(overall-d,h) for h,d in high['path']]
            top=a[-1][1];breaks=[a[-1][0],b[-1][0]]
            if p[s+'_return']:breaks.extend((info['return_lo'],info['return_hi']))
            wall=a[:1]+b+[(v,top) for v in sorted(set(breaks),reverse=True) if a[-1][0]-1e-12<=v<b[-1][0]-1e-12]+a[-2::-1]
            for skin in (0,1):add([cut_face_point(p,s,q,skin) for q in wall],[[cut_face_point(p,s,q,skin) for q in h[::-1]] for h in cut_loops(p,s)])
            for path in (a,b):
                for u,v in zip(path,path[1:]):add([cut_face_point(p,s,u,0),cut_face_point(p,s,v,0),cut_face_point(p,s,v,1),cut_face_point(p,s,u,1)])
            if p[s+'_return']:
                stations=info['return_stations']
                for chain in (plus,minus):
                    for j in range(len(stations)-1):
                        l0,h0=stations[j];l1,h1=stations[j+1];i=split+j
                        add([point_map(p,s,chain[i],l0),point_map(p,s,chain[i],h0),point_map(p,s,chain[i+1],h1),point_map(p,s,chain[i+1],l1)])
                for end in (0,1):
                    for j in range(len(stations)-1):
                        i=split+j;v0=stations[j][end];v1=stations[j+1][end]
                        ring=[point_map(p,s,plus[i],v0),point_map(p,s,plus[i+1],v1),point_map(p,s,minus[i+1],v1),point_map(p,s,minus[i],v0)]
                        add(ring[:3]);add([ring[0],ring[2],ring[3]])
                rlo,rhi=stations[0];free=((a[-1][0],rlo),(rhi,b[-1][0]));l,h=stations[-1]
                add([point_map(p,s,plus[-1],l),point_map(p,s,plus[-1],h),point_map(p,s,minus[-1],h),point_map(p,s,minus[-1],l)])
            else:free=((a[-1][0],b[-1][0]),)
            for l,h in free:
                if h-l>1e-10:add([cut_face_point(p,s,(l,top),0),cut_face_point(p,s,(h,top),0),cut_face_point(p,s,(h,top),1),cut_face_point(p,s,(l,top),1)])
            for loop in cut_loops(p,s):
                for a,b in zip(loop,loop[1:]+loop[:1]):add([cut_face_point(p,s,a,0),cut_face_point(p,s,b,0),cut_face_point(p,s,b,1),cut_face_point(p,s,a,1)])
            side_ranges[s]=(start,len(faces))
    volume=orient_and_check(faces)
    return dict(params=p,faces=faces,flat=flat,volume=volume,side_face_ranges=side_ranges)


def geometry_signature(mesh):
    records=[]
    for face in mesh.faces:
        rings=[]
        for loop in [face.loop]+list(face.hole_loops or []):
            points=[tuple(round(float(x),4) for x in v.position.toTuple()) for v in loop]
            first=min(range(len(points)),key=lambda i:points[i]);rings.append(points[first:]+points[:first])
        records.append([rings[0]]+sorted(rings[1:]))
    return hashlib.sha256(repr(sorted(records)).encode('ascii')).hexdigest()

def native_group(values,frame=None):
    from core.mesh import Mesh
    from core.group import Group
    from PySide6.QtGui import QVector3D,QVector4D,QMatrix4x4
    geometry=build_geometry(values);p=geometry['params'];mesh=Mesh()
    for rings in geometry['faces']:
        q=[[QVector3D(*(np.asarray(v)*SCALE)) for v in ring] for ring in rings]
        if mesh.add_face(q[0],q[1:] or None) is None:raise PanelError('IngeTrazo rejected a panel face.')
    if any(len(e.faces)!=2 for e in mesh.edges):raise PanelError('Native precision could not preserve this corner. Increase the smallest feature.')
    for edge in mesh.edges:
        a,b=list(edge.faces)
        if QVector3D.dotProduct(a.normal(),b.normal())>.98:edge.soft=True
    g=Group(mesh,name=p['name']);g.component=False;matrix=QMatrix4x4()
    if frame is not None:
        for i,k in enumerate(('x','y','z')):matrix.setColumn(i,QVector4D(*frame[k],0.))
        matrix.setColumn(3,QVector4D(*frame['origin'],1.))
    matrix.scale(1/SCALE);g.xform=matrix
    g.material={'color':[.68,.73,.78] if 'stainless' in p['material'].lower() else [.55,.59,.63]}
    g.ext={KEY:dict(version=VERSION,params=copy.deepcopy(p),geometry_signature=geometry_signature(mesh),
        corner_rule=('Independent panel/mitre/end-tab corners; local wall profiles, separate return ends and physical root reliefs. Tooling clearance unverified.' if p.get('corner_style')=='custom' else 'Full-width through bends, walls and returns; adjacent whole flanges shortened; web tongues and rectangular reliefs of gap width extending gap beyond the side-bend tangent. Tooling clearance unverified.' if full_width_sides(p) else 'Cab butt corners with tapered bend-region cuts and separate seam gap; return ends cut back.' if has_closed_corners(p) else 'Square web reliefs; square return end cutouts.')+' Outside finished dimensions, virtual sharps. Non-90 development uses a constant neutral radius inferred from the shop 90-degree deduction; verify against actual tooling.')}
    return g

def verify_part(g):
    _,data=part_data(g)
    if data.get('geometry_signature')!=geometry_signature(g.mesh):raise PanelError('Direct geometry edits changed this part. Its saved bend definitions no longer match. Keep it as a manual part or create a new sheet-metal part.')

def world_frame(scene,g):
    from core.group import iter_placements
    for root in scene.groups:
        for item,matrix in iter_placements(root):
            if item is g:
                if matrix is None or any(abs(matrix.column(i).toVector3D().length()-1/SCALE)>1e-8 for i in range(3)) or abs(matrix.determinant()-(1/SCALE)**3)>(1/SCALE)**3*1e-4:raise PanelError('Scaled or mirrored panels cannot be regenerated. Use the panel dimensions to resize.')
                return dict(origin=matrix.column(3).toVector3D().toTuple(),**{k:(matrix.column(i).toVector3D()*SCALE).toTuple() for i,k in enumerate(('x','y','z'))})
    raise PanelError('Select a panel still in this drawing.')

def commit_panel(viewport,values,frame=None,target=None,source=None):
    from core.history import Command,InsertGroupCommand
    from PySide6.QtGui import QMatrix4x4
    scene=viewport.scene
    if scene.edit_group is not None:raise PanelError('Exit group editing before placing or updating a panel. Move the completed panel as a whole component.')
    group=native_group(values,frame)
    if target is not None:verify_part(target);world_frame(scene,target);group.xform=QMatrix4x4(target.xform)
    if source is not None:
        face=source['face']
        if source['mesh'] is not scene.mesh or source['edit_group'] is not scene.edit_group or face not in scene.mesh.faces:raise PanelError('The selected flat face changed. Select and capture it again.')
        current=[[tuple(v.position.toTuple()) for v in loop] for loop in [face.loop]+list(face.hole_loops or [])]
        if current!=source['signature']:raise PanelError('The flat source changed. Capture it again before creating the base.')
    if target is not None:
        extra=copy.deepcopy(target.ext or {})
        for key in (KEY,)+LEGACY_KEYS:extra.pop(key,None)
        extra.update(group.ext);group.ext=extra
    class PanelCommand(Command):
        def __init__(self):self.insert=None;self.old=None;self.mesh_before=None;self.pose=QMatrix4x4(group.xform)
        def do(self,sc):
            if target is None:
                if source is not None:
                    mesh=source['mesh'];face=source['face']
                    if self.mesh_before is None:self.mesh_before=mesh.capture_state()
                    edges=set(e for ring in [face.loop]+list(face.hole_loops or []) for v in ring for e in v.edges if face in e.faces)
                    mesh.remove_face(face)
                    for edge in edges:
                        if not edge.faces:mesh.remove_edge(edge)
                group.xform=QMatrix4x4(self.pose)
                if self.insert is None:self.insert=InsertGroupCommand(group)
                self.insert.do(sc)
            else:
                if self.old is None:self.old=(target.mesh,target.ext,target.name,target.material)
                target.mesh,target.ext,target.name,target.material=group.mesh,group.ext,group.name,group.material
                sc.selection.clear();sc.selection.add(target);sc.version+=1
        def undo(self,sc):
            if target is None:
                self.insert.undo(sc)
                if source is not None:source['mesh'].restore_state(self.mesh_before)
            else:target.mesh,target.ext,target.name,target.material=self.old;sc.version+=1
    viewport.history.execute(PanelCommand())
    if viewport.history.last_error:raise PanelError(str(viewport.history.last_error))
    viewport.notify_scene_changed();viewport.update()
    return target if target is not None else group

def rounded_tab_dxf(p,outline):
    """Restore exact circular DXF arcs from the wall tab's sampled preview."""
    if p.get('corner_style')!='custom':return outline,None
    ro=p['radius']+p['thickness'];ba=2*ro-p['deduction'];centres=[]
    for side,info in custom_layout(p).items():
        for end in info['ends']:
            e=end['edit'];r=e['tab_radius']
            if not r or not e['tab_height']:continue
            distance=end['body']-e['tab_projection']+r;s=distance if end['near'] else info['overall']-distance
            for h in (e['tab_bottom']+r,e['tab_bottom']+e['tab_height']-r):centres.append((np.array(flat_point(p,side,ba+h-ro,s)),r))
    if not centres:return outline,None
    def on(point,c,r):return abs(float(np.linalg.norm(np.asarray(point)-c))-r)<2e-10
    start=next((i for i,q in enumerate(outline) if not any(on(q,c,r) for c,r in centres)),0)
    points=outline[start:]+outline[:start];out=[];bulges=[];i=0
    while i<len(points):
        candidate=next(((c,r) for c,r in centres if i+2<len(points) and all(on(q,c,r) for q in points[i:i+3])),None)
        if candidate is None:out.append(points[i]);bulges.append(0.);i+=1;continue
        c,r=candidate;j=i+2
        while j+1<len(points) and on(points[j+1],c,r):j+=1
        angle=0.
        for a,b in zip(points[i:j],points[i+1:j+1]):
            u=np.asarray(a)-c;v=np.asarray(b)-c;angle+=math.atan2(u[0]*v[1]-u[1]*v[0],float(u@v))
        if abs(angle)>math.pi+.000001:out.append(points[i]);bulges.append(0.);i+=1;continue
        out.append(points[i]);bulges.append(math.tan(angle/4));i=j
    return out,bulges


def write_flat_dxf(path,values):
    p=validate(values);d=flat_pattern(p);factor=INCH if p['units']=='in' else .001
    pts=np.asarray(d['outline']);offset=pts.min(0);rows=[]
    def put(code,value):rows.extend((str(code),str(value)))
    put(0,'SECTION');put(2,'HEADER');put(9,'$ACADVER');put(1,'AC1015');put(9,'$INSUNITS');put(70,1 if p['units']=='in' else 4);put(0,'ENDSEC')
    put(0,'SECTION');put(2,'TABLES');put(0,'TABLE');put(2,'LAYER');put(5,'10');put(330,'0');put(100,'AcDbSymbolTable');put(70,3)
    for layer,color in (('CUT',7),('BEND',5),('BEND_NOTES',3)):
        put(0,'LAYER');put(100,'AcDbSymbolTableRecord');put(100,'AcDbLayerTableRecord');put(2,layer);put(70,0);put(62,color);put(6,'CONTINUOUS')
    put(0,'ENDTAB');put(0,'ENDSEC');put(0,'SECTION');put(2,'ENTITIES')
    # Keep old captured boundaries, but export dimensioned rounds as real
    # circles and obrounds as two exact semicircular bulges, not facets.
    old_count=len(p['holes'])+bool(opening(p))
    def polyline(ring,bulges=None):
        put(0,'LWPOLYLINE');put(100,'AcDbEntity');put(8,'CUT');put(100,'AcDbPolyline');put(90,len(ring));put(70,1)
        for i,point in enumerate(ring):
            x,y=(np.asarray(point)-offset)/factor;put(10,f'{x:.9f}');put(20,f'{y:.9f}')
            if bulges is not None and bulges[i]:put(42,bulges[i])
    outer,bulges=rounded_tab_dxf(p,d['outline']);polyline(outer,bulges)
    for ring in d['holes'][:old_count]:polyline(ring)
    for c,centre in cut_instances(p):
        face=c['face'];core,r=cut_core(c,centre)
        if c['kind']=='hole' or (c['kind']=='slot' and c['length']-c['width']<1e-12):
            xy=(cut_flat_point(p,face,centre)-offset)/factor
            put(0,'CIRCLE');put(100,'AcDbEntity');put(8,'CUT');put(100,'AcDbCircle');put(10,f'{xy[0]:.9f}');put(20,f'{xy[1]:.9f}');put(30,0);put(40,f'{r/factor:.9f}')
        elif c['kind']=='slot':
            a=math.radians(c['angle']);n=np.array([-math.sin(a),math.cos(a)])
            ring=[core[0]-r*n,core[1]-r*n,core[1]+r*n,core[0]+r*n]
            origin=cut_flat_point(p,face,(0.,0.));u=cut_flat_point(p,face,(1.,0.))-origin;v=cut_flat_point(p,face,(0.,1.))-origin
            sign=1 if u[0]*v[1]-u[1]*v[0]>0 else -1
            polyline([cut_flat_point(p,face,q) for q in ring],[0,sign,0,sign])
        else:polyline([cut_flat_point(p,face,q) for q in core])
    for line in d['bend_lines']:
        a,b=(np.asarray(line[k])-offset for k in ('a','b'))
        put(0,'LINE');put(100,'AcDbEntity');put(8,'BEND');put(100,'AcDbLine');put(10,f'{a[0]/factor:.9f}');put(20,f'{a[1]/factor:.9f}');put(30,0)
        put(11,f'{b[0]/factor:.9f}');put(21,f'{b[1]/factor:.9f}');put(31,0)
        mid=(a+b)/2;put(0,'TEXT');put(100,'AcDbEntity');put(8,'BEND_NOTES');put(100,'AcDbText');put(10,f'{mid[0]/factor:.9f}');put(20,f'{mid[1]/factor:.9f}');put(30,0)
        put(40,f'{min(.003,min(p["width"],p["length"])/30)/factor:.9f}');put(1,line['label']+(' UP ' if line['direction']>0 else ' DOWN ')+f"{line.get('angle',90):g}")
    put(0,'ENDSEC');put(0,'EOF');Path(path).write_text('\n'.join(rows)+'\n',encoding='ascii')

def display_edges(geometry,view=None):
    edges=defaultdict(list)
    for rings in geometry['faces']:
        q=np.asarray(rings[0]);normal=np.zeros(3)
        for a,b in zip(q,np.roll(q,-1,axis=0)):normal+=np.cross(a,b)
        normal/=np.linalg.norm(normal)
        for ring in rings:
            for a,b in zip(ring,ring[1:]+ring[:1]):
                key=tuple(sorted((tuple(round(x,10) for x in a),tuple(round(x,10) for x in b))))
                edges[key].append(normal)
    return [key for key,normals in edges.items() if (len(normals)!=2 or float(normals[0]@normals[1])<.98)
            and (view is None or any(float(n@view)>1e-9 for n in normals))]


def capture_flat_face(viewport):
    from core.mesh import Face
    selected=[f for f in viewport.scene.selection if isinstance(f,Face)]
    if len(selected)!=1:raise PanelError('Select one drawn flat rectangular face, then choose Base from Face. Open a group first if needed.')
    face=selected[0]
    if face not in viewport.scene.mesh.faces:raise PanelError('Open the source face for editing first.')
    if any(len(e.faces)>1 for v in face.loop for e in v.edges if face in e.faces):
        raise PanelError('This face belongs to an existing solid. Use a drawn 2D face or a flat DXF outline; an existing solid is not replaced.')
    q=np.array([v.position.toTuple() for v in face.loop],dtype=float)
    # Drop collinear subdivision points without changing the outline.
    keep=[]
    for i,point in enumerate(q):
        u=point-q[i-1];v=q[(i+1)%len(q)]-point
        if np.linalg.norm(np.cross(u,v))>1e-6*max(np.linalg.norm(u)*np.linalg.norm(v),1e-15):keep.append(point)
    if len(keep)!=4:raise PanelError('This version requires a rectangular outside outline. Interior holes and slots are supported when clear of the bends.')
    q=np.asarray(keep);x=q[1]-q[0];width=float(np.linalg.norm(x));x/=width
    y=q[2]-q[1];length=float(np.linalg.norm(y));y/=length
    if abs(x@y)>1e-5 or np.linalg.norm(q[3]-(q[0]+y*length))>2e-7:
        raise PanelError('The outside outline is not a planar rectangle.')
    z=np.cross(x,y);z/=np.linalg.norm(z)
    holes=[]
    for h in face.hole_loops or []:
        hp=np.array([v.position.toTuple() for v in h])-q[0]
        if np.max(np.abs(hp@z))>2e-7:raise PanelError('An opening is not in the face plane.')
        holes.append(np.column_stack((hp@x,hp@y)).tolist())
    return dict(face=face,mesh=viewport.scene.mesh,edit_group=viewport.scene.edit_group,
                signature=[[tuple(v.position.toTuple()) for v in loop] for loop in [face.loop]+list(face.hole_loops or [])],
                width=width,length=length,holes=holes,
                frame=dict(origin=q[0].tolist(),x=x.tolist(),y=y.tolist(),z=z.tolist()))

def free_edge_candidates(values,kind='flange'):
    p=validate(values)
    if p['flat_view']:return []
    x0,x1,y0,y1=web_bounds(p);bl,br,tr,tl=corners(p)
    spans={'left':(y0+bl,y1-tl),'right':(y0+br,y1-tr),'front':(x0+bl,x1-br),'back':(x0+tl,x1-tr)}
    layout=side_layout(p);result=[]
    for side in SIDES:
        if not p[side+'_direction']:
            if kind=='return':continue
            lo,hi=spans[side]
            for z in (0.,p['thickness']):result.append((side,np.array(point_map(p,side,(0.,z),lo)),np.array(point_map(p,side,(0.,z),hi))))
        else:
            plus,minus,_=edge_profile(p,side);info=layout[side]
            lo,hi=(info['return_lo'],info['return_hi']) if p[side+'_return'] else (info['lo'],info['hi'])
            if p.get('corner_style')=='custom':lo,hi=info['return_stations'][-1] if p[side+'_return'] else (info['wall_top_lo'],info['wall_top_hi'])
            for curve in (plus,minus):result.append((side,np.array(point_map(p,side,curve[-1],lo)),np.array(point_map(p,side,curve[-1],hi))))
    return result

def match_free_edge(values,a,b,kind='flange',tolerance=2e-6):
    a,b=np.asarray(a),np.asarray(b)
    for side,u,v in free_edge_candidates(values,kind):
        if min(max(np.linalg.norm(a-u),np.linalg.norm(b-v)),max(np.linalg.norm(a-v),np.linalg.norm(b-u)))<=tolerance:return side
    return None

# Angle-aware open-corner construction. The original 90-degree algorithms
# remain the exact compatibility path for every existing corner treatment.
def angled(p):
    return any(p[s+'_direction'] and (abs(p.get(s+'_angle',90)-90)>1e-9 or
        (p[s+'_return'] and abs(p.get(s+'_return_angle',90)-90)>1e-9)) for s in SIDES)

def validate_angles(p):
    for s in SIDES:
        for suffix in ('angle','return_angle'):
            value=p[s+'_'+suffix]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 1<=value<=170:
                raise PanelError('Bend rotation must be from 1 to 170 degrees. Use No flange/return for no bend.')
    if angled(p) and p['corner_style']!='open':
        raise PanelError('Non-90-degree bends need Open corners. Closed, full-width, mitred and tabbed corners retain their 90-degree construction. Choose Open corners before changing the angle.')

def bend_data(p,side,ret=False):
    angle=p.get(side+('_return_angle' if ret else '_angle'),90.)
    a=math.radians(angle);ro=p['radius']+p['thickness']
    setback=ro*math.tan(a/2)
    # Calibrate the effective neutral radius from the existing measured BA90.
    neutral=(2*ro-p['deduction'])/(math.pi/2)
    return setback,neutral*a,2*setback-neutral*a

_web_bounds_90=web_bounds
def web_bounds(p):
    if not angled(p):return _web_bounds_90(p)
    sb={s:bend_data(p,s)[0] if p[s+'_direction'] else 0. for s in SIDES}
    return sb['left'],p['width']-sb['right'],sb['front'],p['length']-sb['back']

_edge_profile_90=edge_profile
def edge_profile(p,side):
    if not angled(p):return _edge_profile_90(p,side)
    t=p['thickness'];rm=p['radius']+t/2;d=p[side+'_direction'];ret=p[side+'_return']
    q=np.array([0.,t/2]);phi=0.;points=[q.copy()];tangents=[(1.,0.)]
    def arc(turn):
        nonlocal q,phi
        sign=1 if turn>0 else -1;normal=np.array([-math.sin(phi),math.cos(phi)])
        centre=q+sign*rm*normal
        for j in range(1,p['segments']+1):
            a=phi+turn*j/p['segments'];n=np.array([-math.sin(a),math.cos(a)])
            points.append(centre-sign*rm*n);tangents.append((math.cos(a),math.sin(a)))
        q=points[-1].copy();phi+=turn
    def line(length):
        nonlocal q
        q=q+length*np.array([math.cos(phi),math.sin(phi)])
        points.append(q.copy());tangents.append((math.cos(phi),math.sin(phi)))
    sb,ba,_=bend_data(p,side);sb2=bend_data(p,side,True)[0] if ret else 0.
    arc(d*math.radians(p[side+'_angle']))
    line(p[side+'_height']-sb-sb2);split=len(points)-1
    if ret:
        arc(d*ret*math.radians(p[side+'_return_angle']))
        line(p[side+'_lip']-sb2)
    q=np.asarray(points);tan=np.asarray(tangents);normal=np.column_stack((-tan[:,1],tan[:,0]))
    return q+t/2*normal,q-t/2*normal,split

_corners_90=corners
def corners(p):
    if not angled(p):return _corners_90(p)
    reach={s:0. for s in SIDES}
    for s in SIDES:
        if p[s+'_direction']:
            a,b,_=edge_profile(p,s);reach[s]=max(0.,-float(min(a[:,0].min(),b[:,0].min())))
    return [p['gap']+max(reach[a],reach[b]) if p[a+'_direction'] and p[b+'_direction'] else 0. for a,b in CORNER_PAIRS]

_side_layout_90=side_layout
def side_layout(p):
    if not angled(p):return _side_layout_90(p)
    x0,x1,y0,y1=web_bounds(p);bl,br,tr,tl=corners(p)
    spans={'left':(y0+bl,y1-tl),'right':(y0+br,y1-tr),'front':(x0+bl,x1-br),'back':(x0+tl,x1-tr)}
    result={}
    for s in SIDES:
        if not p[s+'_direction']:continue
        lo,hi=spans[s];sb,ba,_=bend_data(p,s);ret=p[s+'_return'];sb2,ba2,_=bend_data(p,s,True)
        result[s]=dict(lo=lo,hi=hi,attach_lo=lo,attach_hi=hi,return_lo=lo,return_hi=hi,
            wall_flat=ba+p[s+'_height']-sb-(sb2 if ret else 0.),return_flat=ba2+p[s+'_lip']-sb2 if ret else 0.)
    return result

_cut_face_info_90=cut_face_info
def cut_face_info(p,face):
    if not angled(p) or face=='base':return _cut_face_info_90(p,face)
    if face not in SIDES or not p[face+'_direction']:raise PanelError('A cut needs its original flange.')
    info=side_layout(p)[face];sb=bend_data(p,face)[0];top=p[face+'_height']-(bend_data(p,face,True)[0] if p[face+'_return'] else 0.)
    return dict(label=face.capitalize()+' flange',bounds=(info['lo'],info['hi'],sb,top),
        xrefs=(0.,p['length'] if face in ('left','right') else p['width']),yrefs=(0.,p[face+'_height']),
        xlabels=('Start outside edge','End outside edge'),ylabels=('Base virtual sharp','Top virtual sharp'))

_cut_face_point_90=cut_face_point
def cut_face_point(p,face,xy,skin=0):
    if not angled(p) or face=='base':return _cut_face_point_90(p,face,xy,skin)
    a,b,_=edge_profile(p,face);start=(a if skin==0 else b)[p['segments']]
    angle=p[face+'_direction']*math.radians(p[face+'_angle'])
    q=start+(xy[1]-bend_data(p,face)[0])*np.array([math.cos(angle),math.sin(angle)])
    return point_map(p,face,q,xy[0])

_cut_flat_point_90=cut_flat_point
def cut_flat_point(p,face,xy):
    if not angled(p) or face=='base':return _cut_flat_point_90(p,face,xy)
    sb,ba,_=bend_data(p,face)
    return np.asarray(flat_point(p,face,ba+xy[1]-sb,xy[0]))

_flat_pattern_90=flat_pattern
def flat_pattern(values):
    p=validate(values)
    if not angled(p):return _flat_pattern_90(p)
    polygons=[web_outline(p)];lines=[]
    for s,info in side_layout(p).items():
        lo,hi=info['lo'],info['hi'];end=info['wall_flat'];ret=p[s+'_return']
        total=end+(info['return_flat'] if ret else 0.)
        polygons.append([flat_point(p,s,u,v) for u,v in [(0.,lo),(0.,hi),(total,hi),(total,lo)]])
        ba=bend_data(p,s)[1]
        lines.append(dict(a=flat_point(p,s,ba/2,lo),b=flat_point(p,s,ba/2,hi),label=s.capitalize(),direction=p[s+'_direction'],angle=p[s+'_angle']))
        if ret:
            u=end+bend_data(p,s,True)[1]/2
            lines.append(dict(a=flat_point(p,s,u,lo),b=flat_point(p,s,u,hi),label=s.capitalize()+' return',direction=p[s+'_direction']*ret,angle=p[s+'_return_angle']))
    holes=openings(p)
    for c,centre in cut_instances(p):
        if c['face']!='base':
            loop=[cut_flat_point(p,c['face'],xy).tolist() for xy in cut_loop(c,centre)]
            holes.append(loop if _area(loop)>0 else loop[::-1])
    return dict(outline=trace_net(polygons),holes=holes,bend_lines=lines,layout=side_layout(p))

def validate_angle_clearance(p):
    ranges={}
    for s in SIDES:
        if not p[s+'_direction']:continue
        a,b,_=edge_profile(p,s);loop=clean_loop(list(a)+list(b[::-1]))
        if _boundary_crosses(loop):raise PanelError(s.capitalize()+': the bent cross-section overlaps itself. Increase the leg lengths or reduce the bend/return angle.')
        # Clip every section edge against the interior of the base rectangle.
        # Testing only vertices misses a long return crossing both base skins.
        across=p['width'] if s in ('left','right') else p['length']
        low=np.array([-across,2e-7]);high=np.array([-2e-7,p['thickness']-2e-7])
        for start,end in zip(loop,loop[1:]+loop[:1]):
            start=np.asarray(start);delta=np.asarray(end)-start;t0,t1=0.,1.
            for axis in (0,1):
                if abs(delta[axis])<1e-14:
                    if not low[axis]<=start[axis]<=high[axis]:t0,t1=1.,0.;break
                else:
                    u,v=sorted(((low[axis]-start[axis])/delta[axis],(high[axis]-start[axis])/delta[axis]))
                    t0=max(t0,u);t1=min(t1,v)
            if t0<t1:raise PanelError(s.capitalize()+': the bent flange/return intersects the base. Reduce the angle or length.')
        ranges[s]=max(0.,-min(a[:,0].min(),b[:,0].min()))
    x0,x1,y0,y1=web_bounds(p)
    for a,b,width in [('left','right',x1-x0),('front','back',y1-y0)]:
        if a in ranges and b in ranges and ranges[a]+ranges[b]+p['gap']>=width:
            raise PanelError('Opposite angled flanges/returns enter each other\'s clearance. Increase the base size or reduce the legs/angles.')


def setup(app):
    from PySide6.QtCore import Qt,QPointF,QTimer,QSettings
    from PySide6.QtGui import QVector3D,QColor,QPainter,QPainterPath,QPen,QPixmap,QIcon
    from PySide6.QtWidgets import (QWidget,QVBoxLayout,QGridLayout,QHBoxLayout,QFormLayout,
        QLabel,QPushButton,QTreeWidget,QTreeWidgetItem,QScrollArea,QDialog,QDialogButtonBox,
        QComboBox,QDoubleSpinBox,QSpinBox,QLineEdit,QToolBar,QFileDialog,QTabWidget)
    from tools.base import Tool
    if getattr(app.window,'_rem_sheet_metal_unified',None) is not None:return

    class Preview(QWidget):
        def __init__(self,panel):super().__init__();self.panel=panel;self.setMinimumHeight(190)
        def paintEvent(self,event):
            qp=QPainter(self);qp.setRenderHint(QPainter.Antialiasing);qp.fillRect(self.rect(),QColor('#f3f6f8'));g=self.panel.geometry
            if g is None:qp.drawText(9,25,'Start with New Base or load a sheet-metal part.');return
            flat=g['params']['flat_view'];eye=np.array([0.,0.,1.]) if flat else np.array([.65,1.,.99])
            def project(q):
                x,y,z=q
                return np.array([x,-y]) if flat else np.array([x-.65*y,.6*x+.6*y-z])
            q=np.array([project(v) for f in g['faces'] for ring in f for v in ring]);lo=q.min(0);hi=q.max(0)
            scale=min((self.width()-30)/max(hi[0]-lo[0],1e-9),(self.height()-40)/max(hi[1]-lo[1],1e-9))
            def pixel(q):v=project(q);return QPointF(15+(v[0]-lo[0])*scale,27+(v[1]-lo[1])*scale)
            for rings in sorted(g['faces'],key=lambda f:np.mean(np.asarray(f[0])@eye)):
                q=np.asarray(rings[0]);n=sum((np.cross(a,b) for a,b in zip(q,np.roll(q,-1,axis=0))),np.zeros(3));n/=np.linalg.norm(n)
                if n@eye<=1e-9:continue
                path=QPainterPath();path.setFillRule(Qt.OddEvenFill)
                for ring in rings:
                    path.moveTo(pixel(ring[0]))
                    for v in ring[1:]:path.lineTo(pixel(v))
                    path.closeSubpath()
                light=.65+.35*max(0.,float(n@np.array([.3,.4,.866])));qp.fillPath(path,QColor(*(int(v*light) for v in (184,202,215))))
            qp.setPen(QPen(QColor('#314b5d'),1.))
            for a,b in display_edges(g,eye):qp.drawLine(pixel(a),pixel(b))
            qp.setPen(QColor('#155c40'));qp.drawText(8,17,'FLAT PATTERN' if flat else 'FORMED — adjustable bends')
            if flat:
                qp.setPen(QPen(QColor('#2874c5'),1.,Qt.DashLine))
                for line in g['flat']['bend_lines']:qp.drawLine(pixel((*line['a'],0.)),pixel((*line['b'],0.)))

    class CutPreview(QWidget):
        def __init__(self,panel,face,feature_id):
            super().__init__();self.panel=panel;self.face=face;self.feature_id=feature_id
            self.setObjectName('cut_face_preview');self.setMinimumHeight(215)
        def paintEvent(self,event):
            qp=QPainter(self);qp.setRenderHint(QPainter.Antialiasing);qp.fillRect(self.rect(),QColor('#f3f6f8'))
            try:
                p=self.panel.p;f=cut_face_info(p,self.face);x0,x1,y0,y1=f['bounds'];xr,yr=f['xrefs'],f['yrefs']
                low=np.array([min(x0,xr[0]),min(y0,yr[0])]);high=np.array([max(x1,xr[1]),max(y1,yr[1])])
                sc=min((self.width()-88)/max(high[0]-low[0],1e-9),(self.height()-65)/max(high[1]-low[1],1e-9))
                def pixel(v):return QPointF(45+(v[0]-low[0])*sc,self.height()-35-(v[1]-low[1])*sc)
                qp.setPen(QPen(QColor('#536b7c'),1.));qp.setBrush(QColor('#c5d4de'))
                path=QPainterPath();path.moveTo(pixel((x0,y0)))
                for xy in ((x1,y0),(x1,y1),(x0,y1)):path.lineTo(pixel(xy))
                path.closeSubpath();qp.drawPath(path)
                for c,centre in cut_instances(p,self.face):
                    loop=cut_loop(c,centre);path=QPainterPath();path.moveTo(pixel(loop[0]))
                    for xy in loop[1:]:path.lineTo(pixel(xy))
                    path.closeSubpath();qp.setBrush(QColor('white'));qp.setPen(QPen(QColor('#d98200') if c['id']==self.feature_id else QColor('#536b7c'),1.5));qp.drawPath(path)
                    if c['id']==self.feature_id:
                        q=pixel(centre);qp.drawLine(QPointF(q.x()-4,q.y()),QPointF(q.x()+4,q.y()));qp.drawLine(QPointF(q.x(),q.y()-4),QPointF(q.x(),q.y()+4))
                c=next(c for c in p['cuts'] if c['id']==self.feature_id);_,centre=next((c,q) for c,q in cut_instances(p,self.face) if c['id']==self.feature_id)
                rx=xr[0] if c['x_ref']=='start' else xr[1];ry=yr[0] if c['y_ref']=='start' else yr[1]
                qp.setPen(QPen(QColor('#2874c5'),1.,Qt.DashLine));qp.drawLine(pixel((rx,centre[1])),pixel(centre));qp.drawLine(pixel((centre[0],ry)),pixel(centre))
                factor=INCH if p['units']=='in' else .001;qp.setPen(QColor('#155c40'));qp.drawText(8,17,f['label']+' — outside view / cut centre offsets')
                qp.setPen(QColor('#2874c5'));qp.drawText(8,self.height()-10,f'X {c["x"]/factor:g} · Y {c["y"]/factor:g} {p["units"]} · 0° runs along X')
            except (PanelError,StopIteration,KeyError,ValueError):qp.setPen(QColor('#b12b20'));qp.drawText(8,30,'Enter a valid cut size and centre position.')

    def dialog(title,kind):
        d=QDialog(app.window);d.setObjectName('sheet_metal_'+kind);d.setWindowTitle(title);d.setMinimumWidth(350)
        layout=QVBoxLayout(d);form=QFormLayout();form.setRowWrapPolicy(QFormLayout.WrapLongRows);layout.addLayout(form)
        note=QLabel();note.setWordWrap(True);layout.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);layout.addWidget(buttons)
        buttons.accepted.connect(d.accept);buttons.rejected.connect(d.reject)
        return d,form,note,buttons
    def spin(key,value,minimum=0.,small=False):
        s=QDoubleSpinBox();s.setObjectName(key);s.setDecimals(6);s.setRange(minimum,1e6);s.setKeyboardTracking(False);s.setValue(value);s.setSingleStep(.001 if small else .125);return s
    def combo(key,items,index=0):
        c=QComboBox();c.setObjectName(key);c.addItems(items);c.setCurrentIndex(index);return c

    class Panel(QWidget):
        def __init__(self):
            super().__init__();self.setMinimumWidth(380);self.p=defaults();self.target=None;self.frame=None;self.geometry=None;self.edges=[];self.placing=False;self.preview_active=False;self.in_command=False;self.pending_previous=None;self.highlight=None;self.face_highlight=None
            try:
                saved=QSettings('IngeTrazo','REM Sheet Metal Angles').value('last_settings','')
                if saved:self.p=validate(json.loads(saved))
            except (ValueError,TypeError,KeyError):pass
            layout=QVBoxLayout(self);layout.setContentsMargins(4,4,4,4)
            title=QLabel('Base → Flanges → Cuts → Flatten');layout.addWidget(title)
            self.current=QLabel('No part loaded. Start with New Base.');self.current.setWordWrap(True);self.current.setTextFormat(Qt.PlainText);layout.addWidget(self.current)
            grid=QGridLayout();self.buttons={}
            items=[('base','New Base…'),('face','Base from Face'),('flange','Flange…'),('return','Return…'),
                   ('corners','Corners…'),('material','Material…'),('flat','Flatten'),('export','Export DXF…'),
                   ('load','Load Selected'),('edit','Edit Feature…'),('opening','Opening…'),('cut','Hole / Slot / Cut…')]
            for i,(key,text) in enumerate(items):
                b=QPushButton(text);grid.addWidget(b,i//2,i%2);self.buttons[key]=b
            layout.addLayout(grid)
            self.tree=QTreeWidget();self.tree.setHeaderLabels(['Feature','Definition']);self.tree.setColumnWidth(0,135);self.tree.setMinimumHeight(145);self.tree.setMaximumHeight(180);layout.addWidget(self.tree)
            self.material_label=QLabel();self.material_label.setWordWrap(True);layout.addWidget(self.material_label)
            self.preview=Preview(self);layout.addWidget(self.preview)
            self.report=QLabel();self.report.setWordWrap(True);layout.addWidget(self.report)
            self.error=QLabel();self.error.setWordWrap(True);self.error.setStyleSheet('color:#b12b20');layout.addWidget(self.error);layout.addStretch()
            actions={'base':self.new_base,'face':self.from_face,'flange':lambda:self.start_pick('flange'),
                'return':lambda:self.start_pick('return'),'corners':lambda:self.edit_feature('corners'),
                'material':lambda:self.edit_feature('material'),'flat':self.toggle_flat,'export':self.export_flat,
                'load':self.load_selected,'edit':self.edit_selected,'opening':lambda:self.edit_feature('opening'),'cut':self.start_cut}
            for key,fn in actions.items():self.buttons[key].clicked.connect(lambda _checked=False,fn=fn:self.guarded(fn))
            self.tree.itemDoubleClicked.connect(lambda *_:self.guarded(self.edit_selected));self.refresh()
        def guarded(self,fn):
            try:return fn()
            except (PanelError,OSError) as exc:self.error.setText(str(exc));app.viewport.flash_status(str(exc),7500)
        def snapshot(self):return copy.deepcopy(self.p),self.target,copy.deepcopy(self.frame)
        def restore(self,snapshot):self.p,self.target,self.frame=snapshot;self.refresh()
        def refresh(self):
            self.tree.clear();p=self.p;factor=INCH if p.get('units','in')=='in' else .001;unit=p.get('units','in')
            def row(name,value,kind,side=None):
                r=QTreeWidgetItem([name,value]);r.setToolTip(1,value);r.setData(0,Qt.UserRole,(kind,side));self.tree.addTopLevelItem(r)
            try:
                self.p=validate(self.p);p=self.p;self.geometry=build_geometry(p);self.edges=display_edges(self.geometry);self.error.setText('')
                if self.target is not None:self.frame=world_frame(app.scene,self.target)
                if self.target is not None or self.placing or self.preview_active:
                    row('Base Sheet',f'{p["width"]/factor:g} × {p["length"]/factor:g} {unit}','base')
                    for side in SIDES:
                        if not p[side+'_direction']:continue
                        row(side.capitalize()+' Flange',f'{p[side+"_height"]/factor:g} {unit} '+('UP' if p[side+'_direction']>0 else 'DOWN')+f' {p[side+"_angle"]:g}°','flange',side)
                        if p[side+'_return']:row(side.capitalize()+' Return',f'{p[side+"_lip"]/factor:g} {unit} '+('INWARD' if p[side+'_return']==1 else 'OUTWARD')+f' {p[side+"_return_angle"]:g}°','return',side)
                    if p.get('corner_style')=='custom':
                        for i in range(4):
                            c=corner_feature(p,i);row(CORNER_NAMES[i]+' Corner',dict(open='Open',closed='Closed panel',miter='Mitred returns',tab='Notch / end tab')[c['kind']]+' · '+c['through'].capitalize(),'corners',i)
                    elif full_width_sides(p):row('Full-width Flange',' + '.join(s.capitalize() for s in full_width_sides(p)),'corners')
                    elif has_closed_corners(p):row('Cab Closed Corners',('Left/right' if p['corner_style']=='cab_lr' else 'Front/back')+f' through · gap {p["seam_gap"]/factor:g} {unit}','corners')
                    if any(corners(p)):row('Corner Relief',f'{p["gap"]/factor:g} {unit} · open','corners')
                    captured_count=len(p['holes'])+bool(opening(p))
                    if captured_count:row('Base Openings',str(captured_count)+' captured / centred','opening')
                    for c in p['cuts']:
                        size=(f'Ø {c["diameter"]/factor:g}' if c['kind']=='hole' else f'{c["width"]/factor:g} × {c["length"]/factor:g}')
                        row(c['name'],f'{cut_face_info(p,c["face"])["label"]} · {size} {unit}'+(f' · {c["count"]}×' if c['count']>1 else ''),'cut',c['id'])
                    row('Flat Pattern','Shown' if p['flat_view'] else 'Formed view','flat')
                else:row('Start here','New Base…','new')
                bounds=np.asarray(self.geometry['flat']['outline']);size=(bounds.max(0)-bounds.min(0))/factor
                self.report.setText(f'Flat blank: {size[0]:.4f} × {size[1]:.4f} {unit} · {len(self.geometry["flat"]["bend_lines"])} bends'+('\nNon-90 flat development calibrated from shop 90° data.' if angled(p) else ''))
                if p.get('corner_style')=='custom':self.report.setText(self.report.text()+'\nIndependent wall / return ends. Tooling clearance unverified.')
                elif full_width_sides(p):
                    self.report.setText(self.report.text()+'\nFull-width bend + flange + return. Tooling clearance unverified.')
            except PanelError as exc:self.geometry=None;self.edges=[];self.error.setText(str(exc));self.report.setText('')
            valid=self.geometry is not None;loaded=self.target is not None
            for key in ('flange','return','corners','flat','export','edit','opening','cut'):self.buttons[key].setEnabled(valid and loaded and not self.placing)
            self.buttons['cut'].setEnabled(valid and loaded and not p.get('flat_view') and not self.placing)
            self.buttons['flange'].setEnabled(valid and loaded and not p.get('flat_view') and not self.placing)
            self.buttons['return'].setEnabled(valid and loaded and not p.get('flat_view') and any(p[s+'_direction'] for s in SIDES) and not self.placing)
            for key in ('base','face','load'):self.buttons[key].setEnabled(not self.placing)
            self.buttons['flat'].setText('Show Formed' if p.get('flat_view') else 'Flatten')
            self.current.setText('Click the outside base reference corner. Esc cancels.' if self.placing else ('Part: '+self.target.name if loaded else 'No part loaded. Start with New Base.'))
            self.material_label.setText(f'{p["material"]}\nThickness {p["thickness"]/factor:g} · Inside R {p["radius"]/factor:g} · Deduction {p["deduction"]/factor:g} {unit}/90°')
            self.preview.update();app.viewport.update()
        def root_context(self):
            if app.scene.edit_group is not None:raise PanelError('Exit group editing, then select the whole sheet-metal part.')
        def require_target(self):
            self.root_context();selected=list(app.scene.selection)
            if selected:
                if len(selected)!=1 or not hasattr(selected[0],'mesh'):raise PanelError('Select one whole sheet-metal part, not its individual faces or edges.')
                part_data(selected[0]);verify_part(selected[0]);q=part_parameters(selected[0]);frame=world_frame(app.scene,selected[0])
                changed=selected[0] is not self.target or q!=self.p
                self.target=selected[0];self.p=q;self.frame=frame
                if changed:self.refresh()
            if self.target is None:raise PanelError('Select a sheet-metal part and click Load Selected, or start with New Base.')
            verify_part(self.target);world_frame(app.scene,self.target);return self.target
        def load_selected(self):
            if not app.scene.selection:raise PanelError('Select a whole part made by this tool or either previous extension.')
            self.require_target();self.refresh()
        def apply_current(self,source=None):
            self.in_command=True
            try:
                result=commit_panel(app.viewport,self.p,self.frame,self.target,source)
                self.target=result;self.p=part_parameters(result);self.frame=world_frame(app.scene,result)
                QSettings('IngeTrazo','REM Sheet Metal Angles').setValue('last_settings',json.dumps(self.p))
            finally:self.in_command=False
            self.refresh();return result
        def new_base(self):
            self.root_context();previous=self.snapshot();q=copy.deepcopy(self.p);q.update(name='Sheet metal part',flat_view=False,holes=[],cuts=[],opening_width=0.,opening_length=0.)
            for side in SIDES:q[side+'_direction']=0;q[side+'_return']=0
            self.target=None;self.frame=None;self.preview_active=True
            d,f,n,bb=dialog('New Base — size and starting point','new_base');controls={};state={'factor':INCH if q['units']=='in' else .001,'busy':False}
            name=QLineEdit(q['name']);name.setObjectName('name');f.addRow('Part name',name)
            units=combo('units',['in','mm'],0 if q['units']=='in' else 1);f.addRow('Units',units)
            for k,label,value in [('width','Outside base width',q['width']),('length','Outside base length',q['length']),('pan_height','Pan height',2.4375*INCH),('pan_lip','Return length',.75*INCH)]:
                controls[k]=spin(k,value/state['factor']);f.addRow(label,controls[k])
            plane=combo('plane',['XY — width X, length Y, UP +Z','XZ — width X, length Z, UP −Y','YZ — width Y, length Z, UP +X']);f.addRow('Plane',plane)
            preset=combo('shape',['Flat base — add flanges next','Pan — four flanges','Pan with inward J returns','Panel with outward returns']);f.addRow('Starting shape',preset)
            bb.button(QDialogButtonBox.Ok).setText('Place Base…')
            def preview(*unused):
                if state['busy']:return
                q.update(name=name.text(),units=units.currentText(),width=controls['width'].value()*state['factor'],length=controls['length'].value()*state['factor'])
                shape=preset.currentIndex()
                for side in SIDES:q[side+'_direction']=1 if shape else 0;q[side+'_return']=(0,0,1,-1)[shape];q[side+'_height']=controls['pan_height'].value()*state['factor'];q[side+'_lip']=controls['pan_lip'].value()*state['factor']
                controls['pan_height'].setEnabled(shape>0);controls['pan_lip'].setEnabled(shape>1)
                self.p=copy.deepcopy(q);self.refresh();bb.button(QDialogButtonBox.Ok).setEnabled(self.geometry is not None)
                n.setText(self.error.text() or 'After Place Base, click its outside reference corner in the model. Snaps work; Esc cancels. Material settings come from the main panel.')
            def change_units(text):
                actual={k:w.value()*state['factor'] for k,w in controls.items()};state['factor']=INCH if text=='in' else .001;state['busy']=True
                for k,w in controls.items():w.setValue(actual[k]/state['factor']);w.setSingleStep(.125 if text=='in' else 1.)
                state['busy']=False;preview()
            for w in controls.values():w.valueChanged.connect(preview)
            name.textChanged.connect(preview);preset.currentIndexChanged.connect(preview);units.currentTextChanged.connect(change_units);preview()
            accepted=d.exec()==QDialog.Accepted;self.preview_active=False
            if not accepted:self.restore(previous);return
            self.p=validate(q);self.pending_previous=previous
            axes=[((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),((1.,0.,0.),(0.,0.,1.),(0.,-1.,0.)),((0.,1.,0.),(0.,0.,1.),(1.,0.,0.))][plane.currentIndex()]
            key=KEY+'_place';app.window._tools[key]=PlaceTool(self,axes);app.window._activate_tool(key)
        def from_face(self):
            self.root_context();source=capture_flat_face(app.viewport);previous=self.snapshot();q=copy.deepcopy(self.p)
            q.update(width=source['width'],length=source['length'],holes=copy.deepcopy(source['holes']),cuts=[],opening_width=0.,opening_length=0.,flat_view=False)
            for side in SIDES:q[side+'_direction']=0;q[side+'_return']=0
            try:self.p=validate(q);self.target=None;self.frame=source['frame'];self.apply_current(source)
            except Exception:self.restore(previous);raise
        def picked_cut_face(self,x,y):
            pick=getattr(app.viewport,'pick_face_placement',None) or app.viewport.pick_face_any
            face,owner=pick(x,y)
            if face is None or owner is None or getattr(owner,'mesh',None) is not self.target.mesh:return None
            points=[np.asarray(v.position.toTuple())/SCALE for v in face.loop]
            return match_cut_face(self.p,points)
        def start_cut(self):
            self.require_target()
            if self.p['flat_view']:raise PanelError('Click Show Formed, then select a base or flange face for the cut.')
            key=KEY+'_cut_pick';app.window._tools[key]=CutFaceTool(self);app.window._activate_tool(key)
        def edit_cut(self,face=None,feature_id=None):
            self.require_target();previous=self.snapshot();q=copy.deepcopy(self.p)
            if feature_id is None:
                if face is None:raise PanelError('Click a straight base or flange face first.')
                f=cut_face_info(q,face);xr,yr=f['xrefs'],f['yrefs'];existing={c['id'] for c in q['cuts']};index=1
                while 'cut_'+str(index) in existing:index+=1
                feature_id='cut_'+str(index)
                q['cuts'].append(dict(id=feature_id,name='Cut '+str(index),face=face,kind='hole',x_ref='start',y_ref='end',
                    x=(xr[1]-xr[0])/2,y=(yr[1]-yr[0])/2,diameter=.375*INCH,width=.375*INCH,length=.75*INCH,angle=0.,count=1,pitch=INCH))
                is_new=True
            else:
                c=next((c for c in q['cuts'] if c['id']==feature_id),None)
                if c is None:raise PanelError('That cut feature no longer exists. Reload the part.')
                face=c['face'];is_new=False
            c=next(c for c in q['cuts'] if c['id']==feature_id);info=cut_face_info(q,face)
            self.preview_active=True;factor=INCH if q['units']=='in' else .001;unit=q['units'];state={'delete':False}
            d,f,n,bb=dialog(info['label']+' — dimensioned cut','cut');controls={}
            name=QLineEdit(c['name']);name.setObjectName('cut_name');f.addRow('Feature name',name)
            shape=combo('cut_kind',['Round hole','Obround slot','Rectangular cutout'],('hole','slot','rectangle').index(c['kind']));f.addRow('Cut shape',shape)
            xref=combo('cut_x_ref',list(info['xlabels']),0 if c['x_ref']=='start' else 1);f.addRow('X reference',xref)
            yref=combo('cut_y_ref',list(info['ylabels']),0 if c['y_ref']=='start' else 1);f.addRow('Y reference',yref)
            for key,label in (('x','Centre X offset'),('y','Centre Y offset'),('diameter','Hole diameter'),('width','Slot / cutout width'),('length','Overall length'),('pitch','Row pitch — centre to centre')):
                controls[key]=spin('cut_'+key,c[key]/factor,small=True);f.addRow(label+' ('+unit+')',controls[key])
            angle=spin('cut_angle',c['angle'],minimum=-360.);angle.setMaximum(360.);angle.setSingleStep(15.);f.addRow('Angle (degrees)',angle)
            count=QSpinBox();count.setObjectName('cut_count');count.setRange(1,100);count.setValue(c['count']);f.addRow('Quantity along X',count)
            picture=CutPreview(self,face,feature_id);d.layout().insertWidget(1,picture)
            if not is_new:
                delete=QPushButton('Delete this cut feature');delete.setObjectName('delete_cut');d.layout().insertWidget(d.layout().count()-1,delete)
                def remove():state['delete']=True;d.accept()
                delete.clicked.connect(remove)
            def preview(*unused):
                c.update(name=name.text(),kind=('hole','slot','rectangle')[shape.currentIndex()],x_ref=('start','end')[xref.currentIndex()],y_ref=('start','end')[yref.currentIndex()],angle=angle.value(),count=count.value())
                for key,w in controls.items():c[key]=w.value()*factor
                for key in ('diameter','width','length'):
                    enabled=(key=='diameter') if c['kind']=='hole' else key!='diameter'
                    controls[key].setEnabled(enabled);f.labelForField(controls[key]).setEnabled(enabled)
                angle.setEnabled(c['kind']!='hole');controls['pitch'].setEnabled(c['count']>1)
                self.p=copy.deepcopy(q);self.refresh();picture.update();bb.button(QDialogButtonBox.Ok).setEnabled(self.geometry is not None)
                n.setText(self.error.text() or 'Offsets locate the CUT CENTRE. Flange Y is the outside height datum; TOP means the outside sharp when a return is present. Slot length is tip to tip. A row runs away from the selected X reference. Cuts must clear bends, edges and other cuts.')
            for w in controls.values():w.valueChanged.connect(preview)
            shape.currentIndexChanged.connect(preview);xref.currentIndexChanged.connect(preview);yref.currentIndexChanged.connect(preview)
            name.textChanged.connect(preview);angle.valueChanged.connect(preview);count.valueChanged.connect(preview)
            try:
                preview();accepted=d.exec()==QDialog.Accepted
                if accepted:
                    if state['delete']:q=copy.deepcopy(previous[0]);q['cuts']=[c for c in q['cuts'] if c['id']!=feature_id]
                    self.p=validate(q);self.apply_current()
                else:self.restore(previous)
            except Exception:self.restore(previous);raise
            finally:self.preview_active=False;self.refresh()
        def edit_corner_details(self,initial=None):
            self.require_target();previous=self.snapshot();q=copy.deepcopy(self.p);self.preview_active=True
            d,f,n,bb=dialog('Corners — independent flange and return ends','corners');d.resize(680,760)
            factor=INCH if q['units']=='in' else .001;unit=q['units'];state={'busy':False,'index':initial if isinstance(initial,int) and 0<=initial<4 else 0}
            style=combo('corner_style',[v for k,v in CORNER_OPTIONS],[k for k,v in CORNER_OPTIONS].index(q['corner_style']));f.addRow('Construction mode',style)
            gap=spin('gap',q['gap']/factor,small=True);seam=spin('seam_gap',q['seam_gap']/factor,small=True)
            f.addRow('Earlier modes: relief ('+unit+')',gap);f.addRow('Earlier modes: clearance ('+unit+')',seam)
            choice=combo('corner_index',list(CORNER_NAMES),state['index']);f.addRow('Corner to edit',choice)
            tabs=QTabWidget();d.layout().insertWidget(1,tabs)
            overview=QWidget();form=QFormLayout(overview);tabs.addTab(overview,'Corner')
            kind=combo('corner_kind',['Open relief','Closed panel corner','Mitred return corner','Notched / tabbed end']);form.addRow('Corner treatment',kind)
            through=combo('corner_through',[]);form.addRow('Through flange',through)
            joint=combo('corner_return_joint',['Through return + local trim','Mitred inward returns','Square return cutbacks']);form.addRow('Return-end joint',joint)
            relief=spin('corner_relief',q['gap']/factor,small=True);clearance=spin('corner_clearance',q['seam_gap']/factor,small=True)
            form.addRow('Bend relief ('+unit+')',relief);form.addRow('Joint clearance ('+unit+')',clearance)
            hint=QLabel('Select ONE corner. The next two tabs control its two adjoining flange ends independently. All heights use the base outside plane. The main wall end is the X datum for dimensioned cuts.');hint.setWordWrap(True);form.addRow(hint)
            edits=[]
            labels=(('wall_setback','Main wall end setback'),('return_setback','Extra return-end setback'),
                ('notch_length','End-notch length into wall'),('notch_bottom','Notch lower height'),('notch_height','Notch height'),
                ('tab_projection','Retained tab projection'),('tab_bottom','Tab lower height'),('tab_height','Tab height'),('tab_radius','Tab outer-corner radius'))
            for letter in ('a','b'):
                page=QWidget();ef=QFormLayout(page);tabs.addTab(page,'Flange');fields={}
                for key,label in labels:
                    w=spin('corner_'+letter+'_'+key,0.,small=True);fields[key]=w;ef.addRow(label+' ('+unit+')',w)
                msg=QLabel('Zero notch/tab height and length disables that feature. Tab projection restores stock within the wall setback; it cannot extend beyond the nominal end. Local profiles stay clear of bends.');msg.setWordWrap(True);ef.addRow(msg);edits.append(fields)
            def load_corner():
                state['busy']=True;i=state['index'];c=corner_feature(q,i);pair=CORNER_PAIRS[i]
                kind.setCurrentIndex(('open','closed','miter','tab').index(c['kind']));through.clear();through.addItems([s.capitalize() for s in pair]);through.setCurrentIndex(pair.index(c['through']))
                joint.setCurrentIndex(('butt','miter','open').index(c['return_joint']));relief.setValue(c['relief']/factor);clearance.setValue(c['clearance']/factor)
                for j,s in enumerate(pair):
                    tabs.setTabText(j+1,s.capitalize()+' end')
                    for k,w in edits[j].items():w.setValue(c['ends'][s][k]/factor)
                state['busy']=False
            def preview(*unused):
                if state['busy']:return
                q['corner_style']=CORNER_OPTIONS[style.currentIndex()][0];q['gap']=gap.value()*factor;q['seam_gap']=seam.value()*factor
                custom=q['corner_style']=='custom';choice.setEnabled(custom);tabs.setEnabled(custom);gap.setEnabled(not custom);seam.setEnabled(not custom)
                if custom:
                    i=state['index'];pair=CORNER_PAIRS[i];c=corner_feature(q,i)
                    c.update(kind=('open','closed','miter','tab')[kind.currentIndex()],through=pair[max(0,through.currentIndex())],return_joint=('butt','miter','open')[joint.currentIndex()],relief=relief.value()*factor,clearance=clearance.value()*factor)
                    for j,s in enumerate(pair):c['ends'][s]={k:w.value()*factor for k,w in edits[j].items()}
                    q.setdefault('corner_features',{})[str(i)]=c
                self.p=copy.deepcopy(q);self.refresh();bb.button(QDialogButtonBox.Ok).setEnabled(self.geometry is not None)
                text=('Independent corner: base-bend relief stays automatic; wall setbacks and local profiles affect the straight wall, and return setbacks are separate. Closed panel corners notch only the interfering height band. Mitres use actual lip-width ratios. Tooling clearance and bend sequence remain unverified.' if custom else 'Earlier whole-part corner modes are preserved for compatibility. Choose Independent corners above to edit the three new treatments at each corner.')
                n.setText(self.error.text() or text)
            def choose_corner(index):
                if state['busy']:return
                state['index']=index;load_corner();preview()
            def choose_kind(index):
                if state['busy']:return
                state['busy']=True;joint.setCurrentIndex(1 if index==2 else 0)
                if index==3 and not any(w.value() for w in edits[0].values()):
                    s=CORNER_PAIRS[state['index']][0];ro=q['radius']+q['thickness'];bottom=ro+2*relief.value()*factor
                    top=q[s+'_height']-(ro if q[s+'_return'] else 0.);height=min(.5*INCH,max(0.,(top-bottom)*.75))
                    if height>2e-6:
                        values=dict(wall_setback=.375*INCH,tab_projection=.25*INCH,tab_bottom=bottom,tab_height=height,tab_radius=min(.125*INCH,height/2))
                        for k,v in values.items():edits[0][k].setValue(v/factor)
                state['busy']=False;preview()
            style.currentIndexChanged.connect(preview);gap.valueChanged.connect(preview);seam.valueChanged.connect(preview)
            choice.currentIndexChanged.connect(choose_corner);kind.currentIndexChanged.connect(choose_kind);through.currentIndexChanged.connect(preview);joint.currentIndexChanged.connect(preview)
            relief.valueChanged.connect(preview);clearance.valueChanged.connect(preview)
            for fields in edits:
                for w in fields.values():w.valueChanged.connect(preview)
            try:
                load_corner();preview();accepted=d.exec()==QDialog.Accepted
                if accepted:self.p=validate(q);self.apply_current()
                else:self.restore(previous)
            except Exception:self.restore(previous);raise
            finally:self.preview_active=False;self.refresh()

        def edit_feature(self,kind,side=None):
            if kind=='corners':self.edit_corner_details(side);return
            if kind=='cut':self.edit_cut(feature_id=side);return
            if kind=='new':self.new_base();return
            if kind=='flat':self.toggle_flat();return
            if kind=='material' and self.target is None:
                if app.scene.selection:self.require_target()
            else:self.require_target()
            if kind in ('flange','return') and side not in SIDES:raise PanelError('Choose an edge in the model first.')
            if kind=='return' and not self.p[side+'_direction']:raise PanelError('Add a flange before adding a return on that edge.')
            previous=self.snapshot();q=copy.deepcopy(self.p);self.preview_active=True
            d,f,n,bb=dialog({'base':'Base Sheet','flange':'Edge Flange','return':'Return Flange','corners':'Corners — full width and bend reliefs','material':'Material and Bend Data','opening':'Centred Opening'}[kind]+(' — '+side.capitalize() if side else ''),kind)
            factor=INCH if q['units']=='in' else .001;controls={};state={'factor':factor,'busy':False};unit=q['units'];units=None;preset=None
            def size(k,label,small=False):
                controls[k]=spin(k,q[k]/factor,small=small);controls[k].setSingleStep((.001 if unit=='in' else .01) if small else (.125 if unit=='in' else 1.));f.addRow(label+' ('+unit+')',controls[k])
            def angle_control(key):
                controls[key]=spin(key,q[key],minimum=1.);controls[key].setMaximum(170.);controls[key].setSingleStep(1.);f.addRow('Bend from flat (degrees)',controls[key])
            if kind=='base':
                controls['name']=QLineEdit(q['name']);controls['name'].setObjectName('name');f.addRow('Part name',controls['name']);size('width','Outside base width');size('length','Outside base length')
            elif kind=='flange':
                key=side+'_direction';controls[key]=combo(key,['No flange','UP','DOWN'],(0,1,-1).index(q[key]) if q[key] else 1);f.addRow('Direction',controls[key]);size(side+'_height','Outside leg to virtual sharp'); angle_control(side+'_angle')
            elif kind=='return':
                key=side+'_return';controls[key]=combo(key,['No return','INWARD','OUTWARD'],(0,1,-1).index(q[key]) if q[key] else 1);f.addRow('Direction',controls[key]);size(side+'_lip','Outside return length'); angle_control(side+'_return_angle')
            elif kind=='corners':
                controls['corner_style']=combo('corner_style',[label for key,label in CORNER_OPTIONS],[key for key,label in CORNER_OPTIONS].index(q['corner_style']))
                f.addRow('Corner treatment',controls['corner_style']);size('gap','Bend / corner relief',True);size('seam_gap','Corner / return clearance',True)
            elif kind=='opening':size('opening_width','Opening width (0 = none)');size('opening_length','Opening length (0 = none)')
            else:
                preset=combo('preset',['Custom settings']+[p[0] for p in PRESETS]);f.addRow('Shop preset',preset)
                units=combo('units',['in','mm'],0 if unit=='in' else 1);f.addRow('Units',units)
                controls['material']=QLineEdit(q['material']);controls['material'].setObjectName('material');f.addRow('Material',controls['material'])
                for k,label in [('thickness','Thickness'),('radius','Inside bend radius'),('deduction','Outside deduction / 90°')]:size(k,label,True)
            notes={'base':'Resizes from the fixed reference corner. Captured openings keep their measured size and position.',
                'flange':'Bend angle is rotation from flat: 45° leaves a 135° opening; 90° is square; 135° leaves a 45° opening. Leg length follows the flange to outside virtual sharps, not vertical height. Non-square bends use Open corners with automatic clearance.',
                'return':'INWARD makes a J return; OUTWARD turns the lip away. Choose the second bend rotation from flat. Length uses the outside virtual sharp. Non-square bends use Open corners; sharp closed/mitred/tabbed corner construction remains at 90°.',
                'corners':'Cab closed extends the through flanges and brings adjacent wall ends to a separate butt seam gap. Bend-region end cuts keep the visible base corners intact. Only adjacent flanges bending in the same direction close; opposite bends stay open. Return ends remain cut back. No weld, filler or watertight seal is added.',
                'opening':'Both zero removes the centred opening. Captured holes and slots remain unchanged.',
                'material':'The shop preset sets thickness, inside radius and per-90° outside deduction. It also raises corner relief if thicker stock requires it.'}
            def preview(*unused):
                if state['busy']:return
                for k,w in controls.items():
                    if isinstance(w,QDoubleSpinBox):q[k]=w.value() if k.endswith('_angle') else w.value()*state['factor']
                    elif isinstance(w,QComboBox):q[k]=CORNER_OPTIONS[w.currentIndex()][0] if k=='corner_style' else (0,1,-1)[w.currentIndex()]
                    else:q[k]=w.text()
                if kind=='flange' and not q[side+'_direction']:q[side+'_return']=0
                if kind=='corners':
                    controls['seam_gap'].setEnabled(q['corner_style']!='open')
                    if full_width_sides(q):
                        notes[kind]='Full width keeps the chosen bend, flange and return across the entire finished width. Adjacent whole flanges stop behind that bend and any inward return. Rectangular reliefs are retained: relief width AND extension beyond the shortened bend tangent equal Bend / corner relief, at least thickness. Front/Back/Left/Right are the part edge names, not screen directions. Tooling and bend sequence require shop verification.'
                    else:notes[kind]='Open keeps square corner reliefs. The 3.2 butt-seam options extend wall ends with shaped bend cuts and trimmed returns; they do not make full-width bends or returns. Choose Full width for the new flange construction.'
                if units is not None:q['units']=units.currentText()
                self.p=copy.deepcopy(q);self.refresh();bb.button(QDialogButtonBox.Ok).setEnabled(self.geometry is not None);n.setText(self.error.text() or notes[kind])
                if preset is not None:
                    match=0
                    for i,(_,mat,t,r,bd) in enumerate(PRESETS,1):
                        if q['material']==mat and all(abs(q[k]-v*INCH)<1e-9 for k,v in [('thickness',t),('radius',r),('deduction',bd)]):match=i;break
                    blocked=preset.blockSignals(True);preset.setCurrentIndex(match);preset.blockSignals(blocked)
            def change_units(text):
                actual={k:w.value()*state['factor'] for k,w in controls.items() if isinstance(w,QDoubleSpinBox)};state['factor']=INCH if text=='in' else .001;state['busy']=True
                for k,value in actual.items():
                    controls[k].setValue(value/state['factor']);controls[k].setSingleStep(.001 if text=='in' else .01)
                    label=f.labelForField(controls[k]);label.setText(label.text().rsplit(' (',1)[0]+' ('+text+')')
                state['busy']=False;preview()
            def apply_preset(index):
                if not index:return
                _,material,t,r,bd=PRESETS[index-1];state['busy']=True;controls['material'].setText(material)
                for k,value in [('thickness',t),('radius',r),('deduction',bd)]:controls[k].setValue(value*INCH/state['factor'])
                q['gap']=max(q['gap'],t*INCH);state['busy']=False;preview()
            for w in controls.values():
                if isinstance(w,QDoubleSpinBox):w.valueChanged.connect(preview)
                elif isinstance(w,QComboBox):w.currentIndexChanged.connect(preview)
                else:w.textChanged.connect(preview)
            if units is not None:units.currentTextChanged.connect(change_units);preset.currentIndexChanged.connect(apply_preset)
            preview();accepted=d.exec()==QDialog.Accepted
            try:
                if accepted:
                    self.p=validate(q)
                    if self.target is not None:self.apply_current()
                    else:QSettings('IngeTrazo','REM Sheet Metal Angles').setValue('last_settings',json.dumps(self.p))
                else:self.restore(previous)
            except Exception:self.restore(previous);raise
            finally:self.preview_active=False;self.refresh()
        def edit_selected(self):
            item=self.tree.currentItem()
            if item is None:raise PanelError('Choose a feature in the list, then click Edit Feature.')
            kind,side=item.data(0,Qt.UserRole);self.edit_feature(kind,side)
        def toggle_flat(self):
            self.require_target();previous=self.snapshot();self.p['flat_view']=not self.p.get('flat_view',False)
            try:self.apply_current()
            except Exception:self.restore(previous);raise
        def export_flat(self):
            self.require_target();verify_part(self.target)
            path,_=QFileDialog.getSaveFileName(app.window,'Export sheet-metal flat pattern',self.p['name']+'.dxf','DXF (*.dxf)')
            if path:write_flat_dxf(path,self.p);app.viewport.flash_status('Flat pattern saved: CUT, BEND and BEND_NOTES layers.',6000)
        def picked_side(self,x,y,kind):
            g=self.target;proxy=app.viewport.pick_edge_any(x,y)
            if proxy is None or not getattr(proxy,'in_group',False) or proxy.group.mesh is not g.mesh:return None
            frame=world_frame(app.scene,g)
            def local(v):
                q=np.asarray(v.toTuple())-np.asarray(frame['origin']);return [q@np.asarray(frame[k]) for k in ('x','y','z')]
            return match_free_edge(self.p,local(proxy.a),local(proxy.b),kind)
        def start_pick(self,kind):
            self.require_target()
            if self.p['flat_view']:raise PanelError('Click Show Formed before selecting a flange or return edge.')
            key=KEY+'_pick';app.window._tools[key]=EdgeTool(self,kind);app.window._activate_tool(key)
        def document_changed(self):
            if self.in_command or self.preview_active or self.placing:return
            selected=list(app.scene.selection)
            if len(selected)==1 and hasattr(selected[0],'mesh'):
                try:part_data(selected[0]);g=selected[0];q=part_parameters(g);frame=world_frame(app.scene,g)
                except PanelError:
                    self.target=None;self.frame=None;self.refresh();self.error.setText('Selected component has no usable sheet-metal bend definitions. Start New Base or select a generated sheet-metal part.');return
                if g is not self.target or q!=self.p:self.target=g;self.p=q;self.frame=frame;self.refresh();return
            if self.target is not None:
                try:
                    q=part_parameters(self.target);self.frame=world_frame(app.scene,self.target)
                    if q!=self.p:self.p=q;self.refresh()
                except PanelError:self.target=None;self.frame=None;self.refresh()
            self.preview.update()

    class PlaceTool(Tool):
        name='Place Sheet Metal Base';uses_snap=True
        def __init__(self,panel,axes):self.panel=panel;self.axes=axes;self.done=False
        def on_activate(self,vp):self.scene=vp.scene;self.panel.placing=True;self.panel.refresh();vp.flash_status('Click the outside base reference corner. Snaps work; Esc cancels.',8000)
        def update_point(self,ctx):
            if ctx.world is not None:self.panel.frame=dict(origin=list(ctx.world.toTuple()),x=self.axes[0],y=self.axes[1],z=self.axes[2]);app.viewport.update()
        def on_hover(self,ctx):self.update_point(ctx)
        def on_click(self,ctx):
            if ctx.world is None:return
            def create():
                if app.scene is not self.scene:raise PanelError('The drawing changed. Start New Base again.')
                self.update_point(ctx);self.panel.apply_current();self.done=True;app.window.activate_select_tool()
            self.panel.guarded(create)
        def on_deactivate(self,vp):
            self.panel.placing=False
            if not self.done and self.panel.pending_previous is not None:self.panel.restore(self.panel.pending_previous)
            self.panel.pending_previous=None;self.panel.refresh()
        def on_cancel(self,vp):app.window.activate_select_tool()

    class EdgeTool(Tool):
        name='Pick Sheet Metal Edge';uses_snap=False
        def __init__(self,panel,kind):self.panel=panel;self.kind=kind
        def on_activate(self,vp):vp.flash_status('Click an edge to add/edit its flange.' if self.kind=='flange' else 'Click a flange free edge to add/edit its return.',8000)
        def on_hover(self,ctx):
            self.panel.highlight=None
            try:
                side=self.panel.picked_side(ctx.screen.x(),ctx.screen.y(),self.kind)
                if side:
                    _,a,b=next(e for e in free_edge_candidates(self.panel.p,self.kind) if e[0]==side);self.panel.highlight=(a,b)
            except PanelError:pass
            app.viewport.update()
        def on_click(self,ctx):
            def perform():
                side=self.panel.picked_side(ctx.screen.x(),ctx.screen.y(),self.kind)
                if side is None:raise PanelError('Click a full free edge of the active sheet-metal part. Use Edit Feature to change an existing bend.')
                app.window.activate_select_tool();QTimer.singleShot(0,lambda:self.panel.guarded(lambda:self.panel.edit_feature(self.kind,side)))
            self.panel.guarded(perform)
        def on_deactivate(self,vp):self.panel.highlight=None;vp.update()
        def on_cancel(self,vp):app.window.activate_select_tool()

    class CutFaceTool(Tool):
        name='Pick Sheet Metal Cut Face';uses_snap=False
        def __init__(self,panel):self.panel=panel;self.scene=None
        def on_activate(self,vp):
            self.scene=vp.scene;vp.flash_status('Click the broad straight face of a flange or the base. Esc cancels.',9000)
        def on_hover(self,ctx):
            self.panel.face_highlight=None
            try:
                face=self.panel.picked_cut_face(ctx.screen.x(),ctx.screen.y())
                if face:
                    x0,x1,y0,y1=cut_face_info(self.panel.p,face)['bounds']
                    self.panel.face_highlight=[cut_face_point(self.panel.p,face,xy,1) for xy in ((x0,y0),(x1,y0),(x1,y1),(x0,y1))]
            except PanelError:pass
            app.viewport.update()
        def on_click(self,ctx):
            def perform():
                if app.scene is not self.scene:raise PanelError('The drawing changed. Reload a sheet-metal part.')
                face=self.panel.picked_cut_face(ctx.screen.x(),ctx.screen.y())
                if face is None:raise PanelError('Click a broad straight face of the loaded base or flange. Return lips, bend curves and thickness edges are not cut faces in this version.')
                app.window.activate_select_tool();QTimer.singleShot(0,lambda:self.panel.guarded(lambda:self.panel.edit_cut(face=face)))
            self.panel.guarded(perform)
        def on_deactivate(self,vp):self.panel.face_highlight=None;vp.update()
        def on_cancel(self,vp):app.window.activate_select_tool()

    panel=Panel();scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(panel)
    dock=app.add_panel('Sheet Metal Angles',scroll,start_hidden=True)
    def show():panel.guarded(lambda:panel.load_selected() if app.scene.selection else None);app.show_panel(dock)
    action=app.add_menu_action('Sheet Metal Angles…',show,'Ctrl+Alt+B',tip='Unified sheet metal: base, edge flanges, returns, corners and flat pattern.')
    if action is not None:
        pix=QPixmap(32,32);pix.fill(Qt.transparent);qp=QPainter(pix);qp.setPen(QPen(QColor('#256b87'),3));qp.drawLine(5,24,17,24);qp.drawArc(12,12,12,12,270*16,90*16);qp.drawLine(24,18,24,5);qp.end();action.setIcon(QIcon(pix))
        bar=QToolBar('Sheet Metal Angles',app.window);bar.setObjectName(KEY+'_toolbar');bar.addAction(action);app.window.addToolBar(Qt.TopToolBarArea,bar)
    def overlay(vp,painter):
        if not dock.isVisible() or panel.geometry is None or panel.frame is None:return
        if panel.target is None and not panel.placing and not panel.preview_active:return
        frame=panel.frame
        def world(q):return QVector3D(*(np.asarray(frame['origin'])+sum(q[i]*np.asarray(frame[k]) for i,k in enumerate(('x','y','z')))))
        def line(a,b):
            u,v=vp._world_to_pixel(world(a)),vp._world_to_pixel(world(b))
            if u is not None and v is not None:painter.drawLine(QPointF(*u),QPointF(*v))
        painter.setPen(QPen(QColor(10,155,105,185),1.2))
        for a,b in panel.edges:line(a,b)
        p=panel.p
        if p['flat_view']:
            painter.setPen(QPen(QColor('#2874c5'),1.4,Qt.DashLine))
            for bend in panel.geometry['flat']['bend_lines']:
                line((*bend['a'],p['thickness']),(*bend['b'],p['thickness']))
                xy=(np.asarray(bend['a'])+bend['b'])/2;pos=vp._world_to_pixel(world((*xy,p['thickness'])))
                if pos is not None:painter.drawText(QPointF(pos[0]+4,pos[1]),bend['label']+(' UP ' if bend['direction']>0 else ' DOWN ')+f"{bend.get('angle',90):g}°")
        if panel.highlight is not None:painter.setPen(QPen(QColor('#d98d00'),4));line(*panel.highlight)
        if panel.face_highlight is not None:
            painter.setPen(QPen(QColor('#d98d00'),3));ring=panel.face_highlight
            for a,b in zip(ring,ring[1:]+ring[:1]):line(a,b)
        a=(p['width']/2,p['length']/2,p['thickness']);b=(a[0],a[1],a[2]+min(p['width'],p['length'])*.25)
        painter.setPen(QPen(QColor('#2874c5'),2));line(a,b);xy=vp._world_to_pixel(world(b))
        if xy is not None:painter.drawText(QPointF(xy[0]+5,xy[1]),'UP')
    app.add_overlay(overlay);app.on_document_changed(panel.document_changed)
    def context(menu,selection):menu.addSeparator();menu.addAction('Sheet Metal Angles…',show)
    app.add_context_menu(context)
    def kind(g):
        try:_,data=part_data(g)
        except PanelError:return None
        return {'type':'Sheet metal','props':[('Material',data['params'].get('material','')),('Workflow','Base / Flange / Return / Flatten')]}
    app.add_outliner_kind(kind)
    app.window._rem_sheet_metal_angles=dict(panel=panel,dock=dock,action=action,show=show,version=VERSION)

