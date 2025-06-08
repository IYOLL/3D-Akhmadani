import plotly.graph_objects as go
import numpy as np

class SceneGraph:
    def __init__(self):
        self.objects = []
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def get_all_meshes(self):
        meshes = []
        for obj in self.objects:
            meshes.extend(obj.get_meshes())
        return meshes

class TransformableObject:
    def __init__(self):
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 0]  # radians
        self.scale = [1, 1, 1]
    
    def translate(self, x, y, z):
        self.translation = [x, y, z]
    
    def rotate(self, x_rot, y_rot, z_rot):
        self.rotation = [x_rot, y_rot, z_rot]
    
    def scale_obj(self, x_scale, y_scale, z_scale):
        self.scale = [x_scale, y_scale, z_scale]
    
    def apply_transform(self, vertices):
        vertices = vertices * np.array(self.scale)
        rx, ry, rz = self.rotation
        
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        rot_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        rot_z = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        vertices = np.dot(vertices, rot_x.T)
        vertices = np.dot(vertices, rot_y.T)
        vertices = np.dot(vertices, rot_z.T)
        
        vertices = vertices + np.array(self.translation)
        return vertices

class Cube(TransformableObject):
    def __init__(self):
        super().__init__()
        self.color = "blue"
    
    def get_meshes(self):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        vertices = self.apply_transform(vertices)
        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7]
        ]
        mesh = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=[face[0] for face in faces],
            j=[face[1] for face in faces],
            k=[face[2] for face in faces],
            flatshading=True,
            color=self.color,
            opacity=1,
            name="Cube"
        )
        return [mesh]

class Cylinder(TransformableObject):
    def __init__(self):
        super().__init__()
        self.color = "silver"
        self.resolution = 20
    
    def get_meshes(self):
        theta = np.linspace(0, 2*np.pi, self.resolution)
        z = np.linspace(0, 1, 2)
        theta, z = np.meshgrid(theta, z)
        
        x = np.cos(theta)
        y = np.sin(theta)
        
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        vertices = self.apply_transform(vertices)
        
        x = vertices[:, 0].reshape(2, self.resolution)
        y = vertices[:, 1].reshape(2, self.resolution)
        z = vertices[:, 2].reshape(2, self.resolution)
        
        mesh = go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, self.color], [1, self.color]],
            opacity=1,
            showscale=False,
            name="Cylinder"
        )
        return [mesh]

class Cone(TransformableObject):
    def __init__(self):
        super().__init__()
        self.color = "red"
        self.resolution = 20
    
    def get_meshes(self):
        theta = np.linspace(0, 2*np.pi, self.resolution)
        z = np.linspace(0, 1, self.resolution)
        theta, z = np.meshgrid(theta, z)
        
        r = 1 - z  # Radius decreases as z increases
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        vertices = self.apply_transform(vertices)
        
        x = vertices[:, 0].reshape(self.resolution, self.resolution)
        y = vertices[:, 1].reshape(self.resolution, self.resolution)
        z = vertices[:, 2].reshape(self.resolution, self.resolution)
        
        mesh = go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, self.color], [1, self.color]],
            opacity=1,
            showscale=False,
            name="Cone"
        )
        return [mesh]

class Torus(TransformableObject):
    def __init__(self):
        super().__init__()
        self.color = "gold"
        self.resolution = 20
        self.major_radius = 1
        self.minor_radius = 0.3
    
    def get_meshes(self):
        theta = np.linspace(0, 2*np.pi, self.resolution)
        phi = np.linspace(0, 2*np.pi, self.resolution)
        theta, phi = np.meshgrid(theta, phi)
        
        x = (self.major_radius + self.minor_radius * np.cos(phi)) * np.cos(theta)
        y = (self.major_radius + self.minor_radius * np.cos(phi)) * np.sin(theta)
        z = self.minor_radius * np.sin(phi)
        
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        vertices = self.apply_transform(vertices)
        
        x = vertices[:, 0].reshape(self.resolution, self.resolution)
        y = vertices[:, 1].reshape(self.resolution, self.resolution)
        z = vertices[:, 2].reshape(self.resolution, self.resolution)
        
        mesh = go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, self.color], [1, self.color]],
            opacity=1,
            showscale=False,
            name="Torus"
        )
        return [mesh]

class Sphere(TransformableObject):
    def __init__(self):
        super().__init__()
        self.color = "lightblue"
        self.resolution = 20
    
    def get_meshes(self):
        theta = np.linspace(0, 2*np.pi, self.resolution)
        phi = np.linspace(0, np.pi, self.resolution)
        theta, phi = np.meshgrid(theta, phi)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        vertices = self.apply_transform(vertices)
        
        x = vertices[:, 0].reshape(self.resolution, self.resolution)
        y = vertices[:, 1].reshape(self.resolution, self.resolution)
        z = vertices[:, 2].reshape(self.resolution, self.resolution)
        
        mesh = go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, self.color], [1, self.color]],
            opacity=1,
            showscale=False,
            name="Sphere"
        )
        return [mesh]

# MAIN PROGRAM - Rocket Model
scene = SceneGraph()

# 1. Main Body (Cylinder)
body = Cylinder()
body.translate(0, 0, 0)
body.scale_obj(0.4, 0.4, 2)  # Slim and tall
body.rotate(0, 0, 0)
scene.add_object(body)

# 2. Nose Cone (Cone)
nose = Cone()
nose.translate(0, 0, 2)  # On top of body
nose.scale_obj(0.5, 0.5, 0.8)
nose.rotate(0, 0, 0)
scene.add_object(nose)

# 3. Fins (3 Cubes)
for i, angle in enumerate([0, 2*np.pi/3, 4*np.pi/3]):
    fin = Cube()
    fin.translate(0.5*np.cos(angle), 0.5*np.sin(angle), 0.5)
    fin.scale_obj(0.1, 0.5, 0.3)
    fin.rotate(0, 0, angle)
    fin.color = "darkblue"
    scene.add_object(fin)

# 4. Decorative Ring (Torus)
ring = Torus()
ring.translate(0, 0, 1.5)
ring.scale_obj(0.3, 0.3, 0.3)
ring.rotate(np.pi/2, 0, 0)  # Horizontal orientation
scene.add_object(ring)

# 5. Window (Sphere)
window = Sphere()
window.translate(0.3, 0, 1.2)
window.scale_obj(0.15, 0.15, 0.15)
scene.add_object(window)

# Create figure
fig = go.Figure(data=scene.get_all_meshes())
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-2, 2], autorange=False),
        yaxis=dict(range=[-2, 2], autorange=False),
        zaxis=dict(range=[0, 3], autorange=False),
        aspectmode='cube',
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=1),
            eye=dict(x=1.5, y=1.5, z=1)
        )
    ),
    title="3D Rocket Model with Scene Graph",
    width=1000,
    height=800
)

# Save to HTML
fig.write_html("rocket_model.html", auto_open=True)