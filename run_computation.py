# Powered by FENICS for more info see https://fenicsproject.org

import sys
import subprocess

# Add various parameters of the systems to compute - form is [d1,d2,t_end,'Boundary conditions','Kinetics', 'Sources']
# Options - Boundary conditions: N (Neumann), D (Dirichlet)
# Options - Kinetics: SCH (Schnackenberg 0.1,0.85), T (Thomas), F (FitzHugh-Nagumo), LLM (Liu-Liaw-Maini) 
# Options - Sources: So (System with sources), NoSo (System without sources)

RDSParam = [['4.0','32.4','800','N','SCH','So','0.0'],
            ['4.0','32.315','800','N','SCH','So','0.0'],
            ]
k = 0 #Counts the interations

# make all of the computation
for l in range(0,len(RDSParam)):
    #if sources run rds_fenics_sources
    if RDSParam[l][5] == 'So':
        subprocess.call([sys.executable, 'rds_FENICS_sources.py', RDSParam[l][0], RDSParam[l][1], RDSParam[l][2], RDSParam[l][3], RDSParam[l][4],'%d'%k])
    #if nosources run rds_fenics_no_sources    
    elif RDSParam[l][5] == 'NoSo':
        subprocess.call([sys.executable, 'rds_FENICS_no_sources.py', RDSParam[l][0], RDSParam[l][1],RDSParam[l][2],RDSParam[l][3],RDSParam[l][4],'%d'%k])
    #quit if the input is wrong
    else:
        print("Wrong switch for sources, iteration skipped")
    k+=1
    
