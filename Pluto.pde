
Planet Pluto = new Planet(new PVector(500, 750), new PVector(-1, 0), 20, 15, color(0, 0, 255));
Planet Charon = new Planet(new PVector(500, 250), new PVector(1, 0), 10, 10, color(0, 255, 0));


void setup() {
  size(1000, 1000);
}



void draw() {
  background(0);
  for (int i = 0; i < 100; i++) {
    float distance = distancesq(Pluto, Charon);
    Pluto.acc = Pluto.pos.copy().add(Charon.pos.copy().mult(-1)).mult(-Charon.mass / distance);
    Charon.acc = Charon.pos.copy().add(Pluto.pos.copy().mult(-1)).mult(-Pluto.mass / distance);
    Pluto.update(.01);
    Charon.update(.01);
  }
  Pluto.render();
  Charon.render();
}
