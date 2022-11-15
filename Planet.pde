class Planet {
  PVector pos;
  PVector vel;
  PVector acc;

  float radius;
  float mass;
  color col;
  
  Planet(PVector p, PVector v, float m, float r, color c) {
    pos = p;
    vel = v;
    acc = new PVector(0, 0);
    mass = m;
    radius = r;
    col = c;
  }

  void applyForce(PVector f) {
    f.div(mass);
    acc.add(f);
  }

  void update(float fac) {
    vel.add(PVector.mult(acc, fac));
    pos.add(PVector.mult(vel, fac));
    acc.mult(0);
  }

  void render() {
      noStroke();
      fill(col);
      circle(pos.x, pos.y, radius * 2);    
  }
}


float distancesq(Planet Q, Planet P) {
  return (Q.pos.x - P.pos.x) * (Q.pos.x - P.pos.x) + (Q.pos.y - P.pos.y) * (Q.pos.y - P.pos.y);
}
