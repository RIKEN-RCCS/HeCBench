CC = $(KOKKOS_CXX)
KOKKOS_PATH ?= $(HOME)/kokkos
program = main
source = main.cpp w_blocks.cpp attack.cpp utils.cpp
obj = $(source:.cpp=.o)

include $(KOKKOS_PATH)/Makefile.kokkos

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS)
LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)
CFLAGS += -O3

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp bitcracker.h sha256.h aes.h
	$(CC) $(CFLAGS) -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -f hash_pass/img_win8_user_hash.txt -d hash_pass/user_passwords.txt
