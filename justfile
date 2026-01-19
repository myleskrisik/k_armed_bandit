build:
    odin build code/ -debug -out:build/k_armed_bandit.bin

build-fast:
    odin build code/ -debug -o:speed -out:build/k_armed_bandit.bin

run: build-fast
    ./build/k_armed_bandit.bin

gen_images: run
    python code/graph.py
