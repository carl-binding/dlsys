
## CFLAGS = -O
CFLAGS = -g 

CC = gcc $(CFLAGS) 


SRC = mem.c tensor.c tensor_io.c tensor_p.c mnist.c main.c main_t.c  \
	list.c tuple.c autograd.c nn_basic.c

HDR = mem.h tensor.h tensor_p.h mnist.h list.h tuple.h autograd.h nn_basic.h 

OBJ = mem.o tensor.o tensor_io.o tensor_p.o mnist.o list.o tuple.o autograd.o nn_basic.o


all: $(OBJ) main_t main

%.o: %.c $(HDR)
	$(CC) -c $<

main: main.c $(OBJ)
	$(CC) -o main main.c $(OBJ) -lm

main_t: main_t.c $(OBJ)
	$(CC) -o main_t main_t.c $(OBJ) -lm

comp_graph: comp_graph.c comp_graph.h d_list.o
	$(CC) -o comp_graph comp_graph.c d_list.o -lm

ecg: ecg.c ecg.h list.o
	$(CC) -o ecg ecg.c list.o -lm

autograd: autograd.c list.o tensor.o tensor_p.o tensor_io.o
	$(CC) -o autograd autograd.c list.o tensor.o tensor_p.o tensor_io.o mnist.o -lm

nn_basic: nn_basic.c $(OBJ)
	$(CC) -o nn_basic nn_basic.c $(OBJ) -lm

clean:
	rm -f *.o

