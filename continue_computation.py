import sys
#sys.argv=['rds_test_sources.py',0.2,5,0.1,4]
#sys.argv=['rds_test_sources.py',0.2]
#exec(open('rds_test_sources.py').read())
#sys.argv=['rds_test_sources.py',0.1,5,0.1,4]
#exec(open('rds_test_sources.py').read())
import subprocess
k=1
subprocess.call([sys.executable, 'rds_test_sources_cont.py', '0.02', '0.2','%d'%k])
#k=2
#subprocess.call([sys.executable, 'rds_test_sources.py', '0.5', '5.0','%d'%k])
#k=3
#subprocess.call([sys.executable, 'rds_test_sources.py', '0.5', '5.0','%d'%k])
#k=4
#subprocess.call([sys.executable, 'rds_test_sources.py', '0.1', '5.0','%d'%k])
