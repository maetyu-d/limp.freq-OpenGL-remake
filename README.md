# limp.freq (LLD Bot World, OpenGL remake)

OpenGL simulation of a circular world with:
- Diameter: 10,000 pixels (world-space units)
- 1000 green self-navigating bots
- Bots with asymmetrical movement constraints (slower, wobbly, less efficient, easier left turns)
- A goal that respawns at a random in-world position every 30 seconds

## Build (macOS)

This uses GLUT + legacy OpenGL APIs.

```bash
clang++ -std=c++17 -O3 -DGL_SILENCE_DEPRECATION main.cpp -framework OpenGL -framework GLUT -o bots_world
./bots_world
```

## Build (Linux, freeglut)

```bash
g++ -std=c++17 -O3 main.cpp -lGL -lglut -o bots_world
./bots_world
```
