"""Summarize saved independent CEM replays without fabricating completion."""
import json, math, statistics, sys
from pathlib import Path
result=json.loads(Path(sys.argv[1]).read_text())
rows=result['episodes']; n=len(rows); wins=sum(bool(r['success']) for r in rows)
z=1.96
if n:
    p=wins/n; d=1+z*z/n; centre=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    ci=[max(0,centre-half),min(1,centre+half)]
else: ci=None
keys=['robot_fall','wrong_contact','base_failure','head_contact','top_fallen','fast_strike']
summary={'complete':result['complete'],'episodes':n,'successes':wins,'success_95ci_wilson':ci,
         'counts':{key:sum(bool(r[key]) for r in rows) for key in keys}}
for key in ['base_max_xy_m','base_max_tilt_rad','strike_head_speed_m_s','strike_contact_duration_s']:
    values=[float(r[key]) for r in rows]
    summary[key]={'mean':statistics.mean(values),'max':max(values)} if values else None
delays=[r['strike_separation_delay_s'] for r in rows if r['success']]
summary['successful_mean_separation_delay_s']=statistics.mean(delays) if delays else None
print(json.dumps(summary,indent=2,allow_nan=False))
