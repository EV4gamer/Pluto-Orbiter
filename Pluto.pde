
Planet Pluto = new Planet(new PVector(500, 450), new PVector(0, 0), 30, 15, color(0, 0, 255));
Planet Charon = new Planet(new PVector(500, 550), new PVector(6, 0), 10, 10, color(0, 255, 0));
Planet Craft = new Planet(new PVector(500, 650), new PVector(5, 0), 0, 2, color(255, 0, 0));

void setup() {
  size(1000, 1000);
}



void draw() {
  background(0);
  for (int i = 0; i < 2000; i++) {
    float distance = distancesq(Pluto, Charon);
    Pluto.acc = Pluto.pos.copy().add(Charon.pos.copy().mult(-1)).mult(-Charon.mass / distance);
    Charon.acc = Charon.pos.copy().add(Pluto.pos.copy().mult(-1)).mult(-Pluto.mass / distance);
    Pluto.update(.001);
    Charon.update(.001);
    
    float CrCh = distancesq(Craft, Charon);
    float CrPl = distancesq(Craft, Pluto);
    Craft.acc.add(Craft.pos.copy().add(Charon.pos.copy().mult(-1)).mult(-Charon.mass / CrCh));
    Craft.acc.add(Craft.pos.copy().add(Pluto.pos.copy().mult(-1)).mult(-Pluto.mass / CrPl));
    Craft.update(.001);
    
  }
  
  
  translate(width/2 - Pluto.pos.x, height/2 - Pluto.pos.y);
  Pluto.render();
  Charon.render();
  Craft.render();
  translate(-width/2 + Pluto.pos.x, -height/2 + Pluto.pos.y);
}
