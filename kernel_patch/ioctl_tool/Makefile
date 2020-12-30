.PHONY: all clean

CFLAGS := -Wall -Werror -O3 -g

all: ioctl_tool

ioctl_tool: ioctl_tool.c
	gcc $(CFLAGS) -o ioctl_tool ioctl_tool.c

clean:
	rm -f ioctl_tool

