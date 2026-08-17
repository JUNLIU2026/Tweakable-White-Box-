givaro library is required!!!
install givaro on Raspberry Pi 5; 
compile (example): g++ -O3 -std=c++11 warx-clrw4.C -o warx-clrw4 -I/usr/local/include -L/usr/local/lib -lgivaro -lgmpxx -lgmp -static -flax-vector-conversions;
note: change the block size and test
