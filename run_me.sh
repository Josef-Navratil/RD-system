systemctl start docker 
docker run -ti -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:current "python3 run_computation.py"
python3 shutdown.py
