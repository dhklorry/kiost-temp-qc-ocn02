import pandas as pd, numpy as np

PRED=['d04','d19','d31','d39','d49']
TGT=['d07','d09','d15']
NOM={'d04':4.19,'d07':7.04,'d09':9.44,'d15':14.74,'d19':19.59,'d31':30.68,'d39':39.45,'d49':49.35}
STEPS=[1,3,6,12,18,36,72,144,288]

def base(W):
    W=W.copy()
    # tide eta from available depths
    zs=[]
    for p in PRED+TGT:
        c='Z_'+p
        if c in W: zs.append(W[c]-W[c].median())
    eta=pd.concat(zs,axis=1).mean(axis=1)
    W['eta']=eta
    for p in PRED+TGT:
        c='Z_'+p
        W[c]=W[c].fillna(NOM[p]+eta)   # reconstruct depth when missing
    return W

def make(W, use_d19=True):
    W=base(W)
    P=[p for p in PRED if use_d19 or p!='d19']
    F=pd.DataFrame(index=W.index)
    F['eta']=W['eta']; F['deta']=W['eta'].diff(6); F['deta2']=W['eta'].diff(36)
    for p in P:
        F['T_'+p]=W['T_'+p]; F['S_'+p]=W['S_'+p]; F['Z_'+p]=W['Z_'+p]
    # differences vs surface
    for p in P[1:]:
        F['dT04_'+p]=W['T_d04']-W['T_'+p]
        F['dS04_'+p]=W['S_d04']-W['S_'+p]
        F['grad04_'+p]=(W['T_d04']-W['T_'+p])/(W['Z_'+p]-W['Z_d04'])
    for a,b in [('d19','d31'),('d31','d39'),('d39','d49'),('d19','d39'),('d31','d49')]:
        if a in P and b in P:
            F['grad_%s_%s'%(a,b)]=(W['T_'+a]-W['T_'+b])/(W['Z_'+b]-W['Z_'+a])
            F['dT_%s_%s'%(a,b)]=W['T_'+a]-W['T_'+b]
    # mixed-layer style indices
    F['strat']=W['T_d04']-W['T_d49']
    F['sstrat']=W['S_d49']-W['S_d04']
    # temporal dynamics of predictors
    for p in P:
        s=W['T_'+p]
        for k in STEPS:
            F['lag_%s_%d'%(p,k)]=s-s.shift(k)
            F['led_%s_%d'%(p,k)]=s-s.shift(-k)
        for w,nm in [(6,'1h'),(36,'6h'),(144,'24h'),(432,'72h')]:
            r=s.rolling(w,center=True,min_periods=max(2,w//3)).mean()
            F['anom_%s_%s'%(p,nm)]=s-r
            if nm in ('24h','72h'): F['rmean_%s_%s'%(p,nm)]=r
        F['rstd_%s_24h'%p]=s.rolling(144,center=True,min_periods=48).std()
    for p in P:
        s=W['S_'+p]
        for k in [6,36,144]:
            F['slag_%s_%d'%(p,k)]=s-s.shift(k)
        F['sanom_%s'%p]=s-s.rolling(144,center=True,min_periods=48).mean()
    # time / tidal harmonics
    t=W.index
    # DatetimeIndex -> UTC 나노초. Index.view 는 일부 pandas 버전에서 제거되었으므로
    # 수치적으로 동일한 asi8 을 우선 사용한다(값 동일함을 확인).
    _ns = np.asarray(getattr(t, 'asi8', None) if getattr(t, 'asi8', None) is not None else t.view('int64'))
    tt=(_ns/1e9/3600.0); tt=tt-tt[0]
    doy=t.dayofyear+t.hour/24.0
    F['doy']=doy
    for k in [1,2,3]:
        F['doy_s%d'%k]=np.sin(2*np.pi*k*doy/365.25); F['doy_c%d'%k]=np.cos(2*np.pi*k*doy/365.25)
    hod=t.hour+t.minute/60.0
    F['hod_s']=np.sin(2*np.pi*hod/24); F['hod_c']=np.cos(2*np.pi*hod/24)
    for nm,per in [('M2',12.4206),('S2',12.0),('K1',23.9345),('O1',25.8193),('M4',6.2103),('MSf',354.367)]:
        F[nm+'_s']=np.sin(2*np.pi*tt/per); F[nm+'_c']=np.cos(2*np.pi*tt/per)
    return F,W

def interp_feats(F,W,zt,use_d19=True):
    """baseline interpolations evaluated at target depth zt (Series)"""
    G=F.copy()
    G['zt']=zt; G['zt_rel']=zt-W['Z_d04']
    pairs=[('d04','d19'),('d04','d31'),('d19','d31'),('d04','d39'),('d19','d39'),('d04','d49')]
    for a,b in pairs:
        if not use_d19 and (a=='d19' or b=='d19'): continue
        za,zb=W['Z_'+a],W['Z_'+b]; ta,tb=W['T_'+a],W['T_'+b]
        G['ip_%s_%s'%(a,b)]=ta+(tb-ta)*(zt-za)/(zb-za)
        sa,sb=W['S_'+a],W['S_'+b]
        G['ips_%s_%s'%(a,b)]=sa+(sb-sa)*(zt-za)/(zb-za)
    return G

from scipy.interpolate import PchipInterpolator
def shape_feats(W, zt, use_d19=True):
    """PCHIP / curvature interpolation over available (z,T) and (z,S) at target depth"""
    P=[p for p in PRED if use_d19 or p!='d19']
    Z=np.column_stack([W['Z_'+p].values for p in P])
    T=np.column_stack([W['T_'+p].values for p in P])
    S=np.column_stack([W['S_'+p].values for p in P])
    zt=np.asarray(zt,dtype=float)
    n=len(zt); outT=np.full(n,np.nan); outS=np.full(n,np.nan); outT2=np.full(n,np.nan)
    for i in range(n):
        z=Z[i]; t=T[i]; s=S[i]
        m=np.isfinite(z)&np.isfinite(t)
        if m.sum()>=3:
            zz=z[m]; tt=t[m]; o=np.argsort(zz); zz,tt=zz[o],tt[o]
            u,ui=np.unique(zz,return_index=True)
            if len(u)>=3:
                f=PchipInterpolator(u,tt[ui],extrapolate=True)
                outT[i]=f(zt[i]); outT2[i]=f(zt[i],2)
        ms=np.isfinite(z)&np.isfinite(s)
        if ms.sum()>=3:
            zz=z[ms]; ss=s[ms]; o=np.argsort(zz); zz,ss=zz[o],ss[o]
            u,ui=np.unique(zz,return_index=True)
            if len(u)>=3:
                outS[i]=PchipInterpolator(u,ss[ui],extrapolate=True)(zt[i])
    return outT,outS,outT2

def add_shape(G,W,zt,use_d19=True,lags=(18,36,-18,-36)):
    a,b,c=shape_feats(W,zt.values,use_d19)
    G=G.copy(); G['pch_T']=a; G['pch_S']=b; G['pch_T2']=c
    G['pch_dev04']=W['T_d04'].values-a
    s=pd.Series(a,index=W.index)
    for k in lags: G['pch_lag%d'%k]=a-s.shift(k).values
    G['pch_anom24']=a-s.rolling(144,center=True,min_periods=48).mean().values
    return G

def iso_feats(W, use_d19=True, deltas=(0.2,0.5,1.0,2.0,3.0,5.0)):
    """Depth of first crossing of T = T_surface - delta, by linear segments over available layers."""
    P=[p for p in PRED if use_d19 or p!='d19']
    Z=np.column_stack([W['Z_'+p].values for p in P])
    T=np.column_stack([W['T_'+p].values for p in P])
    n=len(W); out={}
    Tsurf=T[:,0]
    for d in deltas:
        tgt=Tsurf-d
        z=np.full(n,np.nan); done=np.zeros(n,bool)
        for j in range(Z.shape[1]-1):
            t0,t1=T[:,j],T[:,j+1]; z0,z1=Z[:,j],Z[:,j+1]
            cross=(~done)&np.isfinite(t0)&np.isfinite(t1)&(t0>=tgt)&(t1<tgt)&(t0!=t1)
            zz=z0+(z1-z0)*(t0-tgt)/(t0-t1)
            z[cross]=zz[cross]; done|=cross
        z[~done]=60.0
        out['isoz_%s'%str(d).replace('.','p')]=z
    # gradient ratios / curvature
    for j in range(Z.shape[1]-2):
        g1=(T[:,j]-T[:,j+1])/(Z[:,j+1]-Z[:,j]); g2=(T[:,j+1]-T[:,j+2])/(Z[:,j+2]-Z[:,j+1])
        out['gr_%d'%j]=g1/np.where(np.abs(g2)<1e-3,np.nan,g2)
        out['gd_%d'%j]=g1-g2
    # explicit inter-layer difference ratios (shape descriptors)
    for j in range(Z.shape[1]-1):
        for k in range(j+1,Z.shape[1]-1):
            a=T[:,j]-T[:,j+1]; b=T[:,k]-T[:,k+1]
            out['rat_%d_%d'%(j,k)]=np.clip(a/np.where(np.abs(b)<0.05,np.nan,b),-20,20)
    tot=T[:,0]-T[:,-1]
    for j in range(1,Z.shape[1]-1):
        out['frac_%d'%j]=np.clip((T[:,0]-T[:,j])/np.where(np.abs(tot)<0.2,np.nan,tot),-2,3)
    R=pd.DataFrame(out,index=W.index)
    for c in ['isoz_0p5','isoz_2p0','isoz_5p0']:
        s=R[c]
        R[c+'_a24']=s-s.rolling(144,center=True,min_periods=48).mean()
        for k in (18,-18,36,-36): R[c+'_l%d'%k]=s-s.shift(k)
    return R

def hf_feats(W, use_d19=True):
    """dense short-lag / band-pass features for internal-wave band"""
    P=[p for p in PRED if use_d19 or p!='d19']
    out={}
    for p in P:
        s=W['T_'+p]
        hp=s-s.rolling(37,center=True,min_periods=10).mean()      # <6h
        hp2=s-s.rolling(13,center=True,min_periods=5).mean()      # <2h
        bp=s.rolling(7,center=True,min_periods=3).mean()-s.rolling(37,center=True,min_periods=10).mean()
        out['hp_'+p]=hp; out['hp2_'+p]=hp2; out['bp_'+p]=bp
        for k in list(range(1,13))+[15,18,21,24,30,36]:
            out['hpl_%s_%d'%(p,k)]=hp.shift(k); out['hpd_%s_%d'%(p,k)]=hp.shift(-k)
        out['v_'+p]=s.diff(3); out['v2_'+p]=s.diff(3).diff(3)
        out['vf_'+p]=s.shift(-3)-s.shift(3)
        out['hpstd_'+p]=hp.rolling(72,center=True,min_periods=24).std()
        out['hpstd6_'+p]=hp.rolling(37,center=True,min_periods=12).std()
    if use_d19 and 'd19' in P:
        h4=out['hp_d04']; h19=out['hp_d19']; h31=out['hp_d31']
        for k in range(-6,7):
            out['x1904_%d'%k]=h19.shift(k)-h4
            out['x1931_%d'%k]=h19.shift(k)-h31
    return pd.DataFrame(out,index=W.index)

def _sigmat(S,T):
    """UNESCO-ish sigma-t (density anomaly at p=0), adequate for feature use."""
    T=np.asarray(T,dtype=float); S=np.asarray(S,dtype=float)
    rw=(999.842594+6.793952e-2*T-9.095290e-3*T**2+1.001685e-4*T**3
        -1.120083e-6*T**4+6.536332e-9*T**5)
    A=(8.24493e-1-4.0899e-3*T+7.6438e-5*T**2-8.2467e-7*T**3+5.3875e-9*T**4)
    B=(-5.72466e-3+1.0227e-4*T-1.6546e-6*T**2)
    C=4.8314e-4
    return rw+A*S+B*S**1.5+C*S**2-1000.0

def dens_feats(W, use_d19=True):
    P=[p for p in PRED if use_d19 or p!='d19']
    out={}
    sig={}
    for p in P:
        s=_sigmat(W['S_'+p].values,W['T_'+p].values)
        sig[p]=pd.Series(s,index=W.index); out['sig_'+p]=sig[p]
        out['siga24_'+p]=sig[p]-sig[p].rolling(144,center=True,min_periods=48).mean()
        out['sighp_'+p]=sig[p]-sig[p].rolling(37,center=True,min_periods=10).mean()
    for i in range(len(P)-1):
        a,b=P[i],P[i+1]
        dz=W['Z_'+b]-W['Z_'+a]
        out['N2_%s_%s'%(a,b)]=9.81/1025.0*(sig[b]-sig[a])/dz
        out['dsig_%s_%s'%(a,b)]=sig[b]-sig[a]
    for b in P[1:]:
        out['dsig04_'+b]=sig[b]-sig['d04']
    # pycnocline depth: first crossing of sigma_surface + delta
    Z=np.column_stack([W['Z_'+p].values for p in P])
    G=np.column_stack([sig[p].values for p in P])
    n=len(W)
    for d in (0.1,0.25,0.5,1.0,2.0):
        tgt=G[:,0]+d; z=np.full(n,np.nan); done=np.zeros(n,bool)
        for j in range(G.shape[1]-1):
            g0,g1=G[:,j],G[:,j+1]; z0,z1=Z[:,j],Z[:,j+1]
            cr=(~done)&np.isfinite(g0)&np.isfinite(g1)&(g0<=tgt)&(g1>tgt)&(g0!=g1)
            zz=z0+(z1-z0)*(tgt-g0)/(g1-g0)
            z[cr]=zz[cr]; done|=cr
        z[~done]=60.0
        s=pd.Series(z,index=W.index); out['mld_%s'%str(d).replace('.','p')]=s
        out['mld_%s_a24'%str(d).replace('.','p')]=s-s.rolling(144,center=True,min_periods=48).mean()
        for k in (18,-18,36,-36): out['mld_%s_l%d'%(str(d).replace('.','p'),k)]=s-s.shift(k)
    return pd.DataFrame(out,index=W.index)

def rel_depth_feats(G,W,zt,use_d19=True):
    """explicit differences/ratios between the TARGET depth and all depth-like features.
    Trees cannot form differences, so make them explicit."""
    G=G.copy(); z=zt.values
    dcols=[c for c in G.columns if c.startswith('isoz_') or c.startswith('mld_')]
    dcols=[c for c in dcols if not any(s in c for s in ('_a24','_l'))]
    for c in dcols:
        G['rel_'+c]=z-G[c].values
        G['relr_'+c]=(z-W['Z_d04'].values)/np.maximum(G[c].values-W['Z_d04'].values,0.5)
    z04=W['Z_d04'].values
    for p in (['d19'] if use_d19 else [])+['d31','d39','d49']:
        zp=W['Z_'+p].values
        G['zfrac_'+p]=(z-z04)/(zp-z04)
    return G

def disp_feats(W, use_d19=True, win=144):
    """Isotherm vertical displacement zeta (metres) from each layer's temperature anomaly
    divided by the local background vertical gradient. Trees cannot form this ratio."""
    P=[p for p in PRED if use_d19 or p!='d19']
    out={}
    Tb={p:W['T_'+p].rolling(win,center=True,min_periods=win//3).mean() for p in P}
    Z={p:W['Z_'+p] for p in P}
    zeta={}
    for i,p in enumerate(P):
        # local background gradient using the neighbour below (fallback: above)
        if i+1<len(P): q=P[i+1]; g=(Tb[p]-Tb[q])/(Z[q]-Z[p])
        else: q=P[i-1]; g=(Tb[q]-Tb[p])/(Z[p]-Z[q])
        g=g.where(g.abs()>0.02, np.nan)
        z=(Tb[p]-W['T_'+p])/g          # >0 : isotherms pushed upward
        z=z.clip(-40,40)
        zeta[p]=z; out['zeta_'+p]=z; out['grad_bg_'+p]=g
        out['zeta_hp_'+p]=z-z.rolling(37,center=True,min_periods=10).mean()
        for k in (1,2,3,4,6,9,12,18,24,36):
            out['zeta_%s_l%d'%(p,k)]=z.shift(k); out['zeta_%s_d%d'%(p,k)]=z.shift(-k)
        out['zeta_amp_'+p]=z.rolling(72,center=True,min_periods=24).std()
    if len(P)>=2:
        for a,b in zip(P[:-1],P[1:]):
            out['zeta_diff_%s_%s'%(a,b)]=zeta[a]-zeta[b]
    return pd.DataFrame(out,index=W.index)

def disp_target_feats(G,W,zt,use_d19=True,win=144):
    """How far the target depth sits from displaced isotherm surfaces."""
    P=[p for p in PRED if use_d19 or p!='d19']
    G=G.copy(); z=zt.values
    Tb={p:W['T_'+p].rolling(win,center=True,min_periods=win//3).mean() for p in P}
    for i,p in enumerate(P):
        if i+1<len(P): q=P[i+1]; g=(Tb[p]-Tb[q])/(W['Z_'+q]-W['Z_'+p])
        else: q=P[i-1]; g=(Tb[q]-Tb[p])/(W['Z_'+p]-W['Z_'+q])
        g=g.where(g.abs()>0.02,np.nan)
        zeta=((Tb[p]-W['T_'+p])/g).clip(-40,40)
        # depth of that layer's isotherm after displacement, relative to target depth
        G['dsurf_'+p]=(W['Z_'+p].values-zeta.values)-z
        G['dsurf_n_'+p]=G['dsurf_'+p]/np.maximum(np.abs(W['Z_'+p].values-z),0.5)
    return G
