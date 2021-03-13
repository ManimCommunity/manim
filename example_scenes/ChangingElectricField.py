from manim import *

def get_force_field_func(*point_strength_pairs, **kwargs):
    radius = kwargs.get("radius", 0.5)
    
    def func(point):
        result = np.array(ORIGIN)
        for center, strength in point_strength_pairs:
            to_center = center - point
            norm = get_norm(to_center)
            if norm == 0:
                continue
            elif norm < radius:
                to_center /= radius ** 3
            else:
                to_center /= norm ** 3
            to_center *= -strength
            result += to_center
        return(result)
    return(func)

class ElectricParticle(Circle):
    def __init__(self, radius = 0.5, color = WHITE, sign = "+", **kwargs):
        #digest_config(self, kwargs)
        super(ElectricParticle, self).__init__(
            stroke_color = WHITE,
            stroke_width = 0.5,
            fill_color = color,
            fill_opacity = 0.8,
            radius = radius,
            **kwargs
        )
        
        sign = MathTex(sign)
        sign.set_stroke(WHITE, 1)
        sign.set_width(0.5 * self.get_width())
        sign.move_to(self)
        self.add(sign)

class Proton(ElectricParticle):
    def __init__(self, color = RED_E, **kwargs):
        super(Proton, self).__init__(color = color, **kwargs)

class Electron(ElectricParticle):
    def __init__(self, color = BLUE_E, sign = "-", **kwargs):
        super(Electron, self).__init__(color = color, sign = sign, **kwargs)

class ChangingElectricField(Scene):
    CONFIG = {
        "vector_field_config": {},
        "num_particles": 6,
        "anim_time": 5
    }
    def __init__(self, vector_field_config = {}, num_particles = 6, anim_time = 5, **kwargs):
        super(ChangingElectricField, self).__init__(**kwargs)
        self.vector_field_config = vector_field_config
        self.num_particles = num_particles
        self.anim_time = anim_time
    
    def construct(self):
        particles = self.get_particles()
        vector_field = self.get_vector_field()
        
        def update_vector_field(vector_field):
            new_field = self.get_vector_field()
            vector_field.become(new_field)
            vector_field.func = new_field.func
        
        def update_particles(particles, dt):
            func = vector_field.func
            for particle in particles:
                force = func(particles.get_center())
                particle.velocity += force * dt
                particle.shift(particle.velocity * dt)
        
        vector_field.add_updater(update_vector_field)
        particles.add_updater(update_particles)
        
        self.add(vector_field, particles)
        self.wait(self.anim_time)
        for m_obj in vector_field, particles:
            m_obj.suspend_updating()
        self.wait()
        for m_obj in vector_field, particles:
            m_obj.resume_updating()
        self.wait(3)
        
    def get_particles(self):
        particles = self.particles = VGroup()
        for n in range(self.num_particles):
            if n % 2 == 0:
                particle = Proton(radius = 0.2)
                particle.charge = 1
            else:
                particle = Electron(radius = 0.2)
                particle.charge = -1
            particle.velocity = np.random.normal(0, 1.0, 3)
            particles.add(particle)
            particle.shift(np.random.normal(0, 1.0, 3))
        
        particles.arrange_in_grid(buff = LARGE_BUFF)
        return(particles)
    
    def get_vector_field(self):
        func = get_force_field_func(*list(zip(
            list(map(lambda x: x.get_center(), self.particles)), (p.charge for p in self.particles)
        )))
        self.vector_field = VectorField(func, ** self.vector_field_config)
        return(self.vector_field)
