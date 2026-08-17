# The library givaro is required !!!
install WSL and ubuntu 24.04.1 LTS;
install givaro: sudo apt install libgivaro-dev;
install g++;
compile (example): g++ -o warx-clrw4 warx-clrw4.C -O3 -march=native -std=gnu++11 -lgivaro -lgmp -lgmpxx
