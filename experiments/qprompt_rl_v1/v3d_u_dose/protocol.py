"""The full finite development grid; no result-dependent scheduling."""
SEEDS=(165,166,167)
DOMAINS=('RIM_ONE_r3','Drishti_GS')
WEIGHTS={'ORIGINAL':0.,'FINE_0125':.125,'FINE_025':.25,'FINE_05':.5}
H=1200
CAPS=dict(source=0,development=0,controller=0,main=28800,smoke=160,replay=1448)
PREDECESSOR_COMMIT='288f162f65fac251d23172a91376b6c0cc9397aa'


def check():
    assert len(SEEDS)*len(DOMAINS)*len(WEIGHTS)*H==CAPS['main']
    assert list(WEIGHTS.values())==[0.,.125,.25,.5]
    assert CAPS['replay']==int((CAPS['main']+CAPS['smoke'])*.05+.99999)
    print('PASS fixed dose grid and finite budgets')


if __name__=='__main__':check()
