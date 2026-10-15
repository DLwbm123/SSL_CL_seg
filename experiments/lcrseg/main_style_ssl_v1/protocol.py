SEEDS=(163,164)
ORDERS={'fundus':(('REFUGE','RIM_ONE_r3','Drishti_GS'),('REFUGE','Drishti_GS','RIM_ONE_r3')),
        'prostate':(('RUNMC','BMC','I2CVB','HK','UCL','BIDMC'),),
        'mnms':(('Siemens','Philips','GE','Canon'),)}
COUNTS={'fundus':{'REFUGE':(40,160),'RIM_ONE_r3':(16,63),'Drishti_GS':(10,41)},
        'prostate':{'RUNMC':(80,271),'BMC':(134,439),'I2CVB':(128,576),'HK':(48,120),'UCL':(46,143),'BIDMC':(84,222)},
        'mnms':{'Siemens':(230,872),'Philips':(254,1066),'GE':(110,502),'Canon':(116,530)}}
MAPS={'fundus':(0,1,1),'prostate':(0,1),'mnms':(0,1,1,1)}
SHAPES={'fundus':(3,384,384),'prostate':(1,256,256),'mnms':(1,384,384)}
CLASSES={'fundus':('background','rim','cup'),'prostate':('background','prostate'),'mnms':('background','LV','MYO','RV')}
def steps(dataset,domain):return 100*((max(COUNTS[dataset][domain])+1)//2)
def nodes():
    out=[]
    for ds,orders in ORDERS.items():
        for seed in SEEDS:
            source=f'{ds}_s{seed}_source'
            out.append(dict(id=source,dataset=ds,seed=seed,order=1,stage=0,domain=orders[0][0],predecessor=None,steps=steps(ds,orders[0][0])))
            for order,domains in enumerate(orders,1):
                predecessor=source
                for stage,domain in enumerate(domains[1:],1):
                    key=f'{ds}_s{seed}_o{order}_t{stage}'
                    out.append(dict(id=key,dataset=ds,seed=seed,order=order,stage=stage,domain=domain,predecessor=predecessor,steps=steps(ds,domain)))
                    predecessor=key
    assert sum(n['steps'] for n in out)==511600
    return out

def options(total,source):
    return dict(total_steps=total,warmup_fraction=.2,U_ramp_fraction=.2,lr=.001,weight_decay=4e-5,
                parent_lr_multiplier=1. if source else .5,lr_B_over_A=1.,feature_lr_multiplier=1.,
                lambda_U=.25,lambda_SWD=.2,PAS_confidence=.7,PAS_cosine=.5)
