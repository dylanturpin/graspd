
import urdfpy
import math
import numpy as np
import os
import shutil
import time
import tempfile
from subprocess import Popen

import pybullet as p
from hydra.utils import to_absolute_path

import xml.etree.ElementTree as ET

import dflex as df

def urdf_add_collision(builder, link, collisions, shape_ke, shape_kd, shape_kf, shape_mu):
        
    # add geometry
    for collision in collisions:
        
        origin = urdfpy.matrix_to_xyz_rpy(collision.origin)

        pos = origin[0:3]
        rot = df.rpy2quat(*origin[3:6])

        geo = collision.geometry

        if (geo.box):
            builder.add_shape_box(
                link,
                pos, 
                rot, 
                geo.box.size[0]*0.5, 
                geo.box.size[1]*0.5, 
                geo.box.size[2]*0.5,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu)
        
        if (geo.sphere):
            builder.add_shape_sphere(
                link, 
                pos, 
                rot, 
                geo.sphere.radius,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu)
         
        if (geo.cylinder):
            
            # cylinders in URDF are aligned with z-axis, while dFlex uses x-axis
            r = df.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.5)

            builder.add_shape_capsule(
                link, 
                pos, 
                df.quat_multiply(rot, r), 
                geo.cylinder.radius, 
                geo.cylinder.length*0.5,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu)
 
        if (geo.mesh):

            for m in geo.mesh.meshes:
                faces = []
                vertices = []

                for v in m.vertices:
                    vertices.append(np.array(v))
                    
                for f in m.faces:
                    faces.append(int(f[0]))
                    faces.append(int(f[1]))
                    faces.append(int(f[2]))
                                    
                mesh = df.Mesh(vertices, faces)
                
                builder.add_shape_mesh(
                    link,
                    pos,
                    rot,
                    mesh,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                    mu=shape_mu)

def urdf_load(
    builder, 
    filename, 
    xform, 
    floating=False, 
    armature=0.0, 
    shape_ke=1.e+4, 
    shape_kd=1.e+3, 
    shape_kf=1.e+2, 
    shape_mu=0.25,
    limit_ke=100.0,
    limit_kd=10.0):

    robot = urdfpy.URDF.load(filename)

    # maps from link name -> link index
    link_index = {}

    builder.add_articulation()

    # add base
    if (floating):
        root = builder.add_link(-1, df.transform_identity(), (0,0,0), df.JOINT_FREE)

        # set dofs to transform
        start = builder.joint_q_start[root]

        builder.joint_q[start + 0] = xform[0][0]
        builder.joint_q[start + 1] = xform[0][1]
        builder.joint_q[start + 2] = xform[0][2]

        builder.joint_q[start + 3] = xform[1][0]
        builder.joint_q[start + 4] = xform[1][1]
        builder.joint_q[start + 5] = xform[1][2]
        builder.joint_q[start + 6] = xform[1][3]
    else:    
        root = builder.add_link(-1, xform, (0,0,0), df.JOINT_FIXED)

    if len(robot.links[0].visuals) > 0:
        print("x")
    for l in range(len(robot.links[0].visuals)):
        robot.links[0].visuals[l].origin = robot.links[0].inertial.origin
    urdf_add_collision(builder, root, robot.links[0].visuals, shape_ke, shape_kd, shape_kf, shape_mu)
    link_index[robot.links[0].name] = root

    # add children
    for joint in robot.joints:

        type = None
        axis = (0.0, 0.0, 0.0)

        if (joint.joint_type == "revolute" or joint.joint_type == "continuous"):
            type = df.JOINT_REVOLUTE
            axis = joint.axis
        if (joint.joint_type == "prismatic"):
            type = df.JOINT_PRISMATIC
            axis = joint.axis
        if (joint.joint_type == "fixed"):
            type = df.JOINT_FIXED
        if (joint.joint_type == "floating"):
            type = df.JOINT_FREE
        
        parent = -1

        if joint.parent in link_index:
            parent = link_index[joint.parent]

        origin = urdfpy.matrix_to_xyz_rpy(joint.origin)

        pos = origin[0:3]
        rot = df.rpy2quat(*origin[3:6])

        lower = -1.e+3
        upper = 1.e+3
        damping = 0.0

        # limits
        if (joint.limit):
            
            if (joint.limit.lower != None):
                lower = joint.limit.lower
            if (joint.limit.upper != None):
                upper = joint.limit.upper

        # damping
        if (joint.dynamics):
            if (joint.dynamics.damping):
                damping = joint.dynamics.damping

        # add link
        link = builder.add_link(
            parent=parent, 
            X_pj=df.transform(pos, rot), 
            axis=axis, 
            type=type,
            limit_lower=lower,
            limit_upper=upper,
            limit_ke=limit_ke,
            limit_kd=limit_kd,
            damping=damping)
       
        # add collisions
        if len(robot.link_map[joint.child].visuals) > 0:
            print("x")
        for l in range(len(robot.link_map[joint.child].visuals)):
            robot.link_map[joint.child].visuals[l].origin = robot.link_map[joint.child].inertial.origin
        urdf_add_collision(builder, link, robot.link_map[joint.child].visuals, shape_ke, shape_kd, shape_kf, shape_mu)

        # add ourselves to the index
        link_index[joint.child] = link





# build an articulated tree
def build_tree(
    builder, 
    angle,
    max_depth,    
    width=0.05,
    length=0.25,
    density=1000.0,
    joint_stiffness=0.0,
    joint_damping=0.0,
    shape_ke = 1.e+4,
    shape_kd = 1.e+3,
    shape_kf = 1.e+2,
    shape_mu = 0.5,
    floating=False):


    def build_recursive(parent, depth):

        if (depth >= max_depth):
            return

        X_pj = df.transform((length * 2.0, 0.0, 0.0), df.quat_from_axis_angle((0.0, 0.0, 1.0), angle))

        type = df.JOINT_REVOLUTE
        axis = (0.0, 0.0, 1.0)

        if (depth == 0 and floating == True):
            X_pj = df.transform((0.0, 0.0, 0.0), df.quat_identity())
            type = df.JOINT_FREE

        link = builder.add_link(
            parent, 
            X_pj, 
            axis, 
            type,
            stiffness=joint_stiffness,
            damping=joint_damping)
        
        # box
        # shape = builder.add_shape_box(
        #     link, 
        #     pos=(length, 0.0, 0.0),
        #     hx=length, 
        #     hy=width, 
        #     hz=width,
        #     ke=shape_ke,
        #     kd=shape_kd,
        #     kf=shape_kf,
        #     mu=shape_mu)
        
        # capsule
        shape = builder.add_shape_capsule(
            link, 
            pos=(length, 0.0, 0.0), 
            radius=width, 
            half_width=length, 
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
            mu=shape_mu)

        # recurse
        #build_tree_recursive(builder, link, angle, width, depth + 1, max_depth, shape_ke, shape_kd, shape_kf, shape_mu, floating)
        build_recursive(link, depth + 1)

    # 
    build_recursive(-1, 0)



# SNU file format parser

class MuscleUnit:

    def __init__(self):
        
        self.name = ""
        self.bones = []
        self.points = []

class Skeleton:

    def __init__(self, skeleton_file, muscle_file, builder, filter):

        self.parse_skeleton(skeleton_file, builder, filter)
        self.parse_muscles(muscle_file, builder)

    def parse_skeleton(self, filename, builder, filter):
        file = ET.parse(filename)
        root = file.getroot()
        
        self.node_map = {}       # map node names to link indices
        self.xform_map = {}      # map node names to parent transforms
        self.mesh_map = {}       # map mesh names to link indices objects

        self.coord_start = len(builder.joint_q)
        self.dof_start = len(builder.joint_qd)

    
        type_map = { 
            "Ball": df.JOINT_BALL, 
            "Revolute": df.JOINT_REVOLUTE, 
            "Prismatic": df.JOINT_PRISMATIC, 
            "Free": df.JOINT_FREE, 
            "Fixed": df.JOINT_FIXED
        }

        builder.add_articulation()

        for child in root:

            if (child.tag == "Node"):

                body = child.find("Body")
                joint = child.find("Joint")

                name = child.attrib["name"]
                parent = child.attrib["parent"]
                parent_X_s = df.transform_identity()

                if parent in self.node_map:
                    parent_link = self.node_map[parent]
                    parent_X_s = self.xform_map[parent]
                else:
                    parent_link = -1

                body_xform = body.find("Transformation")
                joint_xform = joint.find("Transformation")

                body_mesh = body.attrib["obj"]
                body_size = np.fromstring(body.attrib["size"], sep=" ")
                body_type = body.attrib["type"]
                body_mass = body.attrib["mass"]

                body_R_s = np.fromstring(body_xform.attrib["linear"], sep=" ").reshape((3,3))
                body_t_s = np.fromstring(body_xform.attrib["translation"], sep=" ")

                joint_R_s = np.fromstring(joint_xform.attrib["linear"], sep=" ").reshape((3,3))
                joint_t_s = np.fromstring(joint_xform.attrib["translation"], sep=" ")
            
                joint_type = type_map[joint.attrib["type"]]
                
                #joint_lower = np.fromstring(joint.attrib["lower"], sep=" ")
                #joint_uppper = np.fromstring(joint.attrib["upper"], sep=" ")

                if ("axis" in joint.attrib):
                    joint_axis = np.fromstring(joint.attrib["axis"], sep=" ")
                else:
                    joint_axis = np.array((0.0, 0.0, 0.0))

                body_X_s = df.transform(body_t_s, df.quat_from_matrix(body_R_s))
                joint_X_s = df.transform(joint_t_s, df.quat_from_matrix(joint_R_s))

                mesh_base = os.path.splitext(body_mesh)[0]
                mesh_file = mesh_base + ".usd"

                #-----------------------------------
                # one time conversion, put meshes into local body space (and meter units)

                # stage = Usd.Stage.Open("./assets/snu/OBJ/" + mesh_file)
                # geom = UsdGeom.Mesh.Get(stage, "/" + mesh_base + "_obj/defaultobject/defaultobject")

                # body_X_bs = df.transform_inverse(body_X_s)
                # joint_X_bs = df.transform_inverse(joint_X_s)

                # points = geom.GetPointsAttr().Get()
                # for i in range(len(points)):

                #     p = df.transform_point(joint_X_bs, points[i]*0.01)
                #     points[i] = Gf.Vec3f(p.tolist())  # cm -> meters
                

                # geom.GetPointsAttr().Set(points)

                # extent = UsdGeom.Boundable.ComputeExtentFromPlugins(geom, 0.0)
                # geom.GetExtentAttr().Set(extent)
                # stage.Save()
                
                #--------------------------------------
                link = -1

                if len(filter) == 0 or name in filter:

                    joint_X_p = df.transform_multiply(df.transform_inverse(parent_X_s), joint_X_s)
                    body_X_c = df.transform_multiply(df.transform_inverse(joint_X_s), body_X_s)

                    if (parent_link == -1):
                        joint_X_p = df.transform_identity()

                    # add link
                    link = builder.add_link(
                        parent=parent_link, 
                        X_pj=joint_X_p,
                        axis=joint_axis,
                        type=joint_type,
                        damping=2.0,
                        stiffness=10.0)

                    # add shape
                    shape = builder.add_shape_box(
                        body=link, 
                        pos=body_X_c[0],
                        rot=body_X_c[1],
                        hx=body_size[0]*0.5,
                        hy=body_size[1]*0.5,
                        hz=body_size[2]*0.5,
                        ke=1.e+3*5.0,
                        kd=1.e+2*2.0,
                        kf=1.e+2,
                        mu=0.5)

                # add lookup in name->link map
                # save parent transform
                self.xform_map[name] = joint_X_s
                self.node_map[name] = link
                self.mesh_map[mesh_base] = link

    def parse_muscles(self, filename, builder):

        # list of MuscleUnits
        muscles = []

        file = ET.parse(filename)
        root = file.getroot()

        self.muscle_start = len(builder.muscle_activation)

        for child in root:

                if (child.tag == "Unit"):

                    unit_name = child.attrib["name"]
                    unit_f0 = float(child.attrib["f0"])
                    unit_lm = float(child.attrib["lm"])
                    unit_lt = float(child.attrib["lt"])
                    unit_lmax = float(child.attrib["lmax"])
                    unit_pen = float(child.attrib["pen_angle"])

                    m = MuscleUnit()
                    m.name = unit_name

                    incomplete = False

                    for waypoint in child.iter("Waypoint"):
                    
                        way_bone = waypoint.attrib["body"]
                        way_link = self.node_map[way_bone]
                        way_loc = np.fromstring(waypoint.attrib["p"], sep=" ", dtype=np.float32)

                        if (way_link == -1):
                            incomplete = True
                            break

                        # transform loc to joint local space
                        joint_X_s = self.xform_map[way_bone]

                        way_loc = df.transform_point(df.transform_inverse(joint_X_s), way_loc)

                        m.bones.append(way_link)
                        m.points.append(way_loc)

                    if not incomplete:

                        muscles.append(m)
                        builder.add_muscle(m.bones, m.points, f0=unit_f0, lm=unit_lm, lt=unit_lt, lmax=unit_lmax, pen=unit_pen)

        self.muscles = muscles

def build_rigid(builder, x=True,y=True,z=True,rpy=True, com=(0.0,0.0,0.0)):
    builder.add_articulation()
    rigid = -1
    if rpy:
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform(com, df.quat_identity()),
            axis=(1.0, 0.0, 0.0),
            type=df.JOINT_REVOLUTE,
            armature=0.0)
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0, 0.0, 0.0), df.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            type=df.JOINT_REVOLUTE,
            armature=0.0)
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0, 0.0, 0.0), df.quat_identity()),
            axis=(0.0, 0.0, 1.0),
            type=df.JOINT_REVOLUTE,
            armature=0.0)
    if x:
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0, 0.0, 0.0), df.quat_identity()),
            axis=(1.0, 0.0, 0.0),
            type=df.JOINT_PRISMATIC,
            armature=0.0)
    if y:
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0, 0.0, 0.0), df.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            type=df.JOINT_PRISMATIC,
            armature=0.0)
    if z:
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0, 0.0, 0.0), df.quat_identity()),
            axis=(0.0, 0.0, 1.0),
            type=df.JOINT_PRISMATIC,
            armature=0.0)
    return rigid

# method from https://github.com/AbrarAnwar/cross_section_rl
def pyvistaToTrimeshFaces(cells):
    faces = []
    idx = 0
    while idx < len(cells):
        curr_cell_count = cells[idx]
        curr_faces = cells[idx+1:idx+curr_cell_count+1]
        faces.append(curr_faces)
        idx += curr_cell_count+1
    return np.array(faces)

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

# from ganhand
# 3D model simulations
def run_simulation(
    hand_verts,
    hand_faces,
    obj_verts,
    obj_faces,
    simulation_step=1 / 240,
    num_iterations=35,
    object_friction=1.2,
    hand_friction=1.2,
    hand_restitution=0,
    object_restitution=0.5,
    object_mass=1,
    verbose=False,
    vhacd_resolution=1000,
    #vhacd_exe= '/home/ecorona/GHANds/v-hacd/bin/linux/testVHACD',
    vhacd_exe='/home/dylanturpin/repos/v-hacd/src/build/test/testVHACD',
    wait_time=0,
    save_video=False,
    save_video_path=None,
    save_hand_path=None,
    save_obj_path=None,
    save_simul_folder=None,
    use_gui=False,
):
    if use_gui:
        conn_id = p.connect(p.GUI)
    else:
        conn_id = p.connect(p.DIRECT)

    hand_indicies = hand_faces.flatten().tolist()
    p.resetSimulation(physicsClientId=conn_id)
    p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=conn_id)
    p.setPhysicsEngineParameter(
        numSolverIterations=150, physicsClientId=conn_id
    )
    p.setPhysicsEngineParameter(
        fixedTimeStep=simulation_step, physicsClientId=conn_id
    )
    p.setGravity(0, 9.8, 0, physicsClientId=conn_id)

    # add hand
    base_tmp_dir = to_absolute_path("tmp/objs")
    os.makedirs(base_tmp_dir, exist_ok=True)
    hand_tmp_fname = tempfile.mktemp(suffix=".obj", dir=base_tmp_dir)
    save_obj(hand_tmp_fname, hand_verts, hand_faces)

    if save_hand_path is not None:
        shutil.copy(hand_tmp_fname, save_hand_path)

    hand_collision_id = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=hand_tmp_fname,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        indices=hand_indicies,
        physicsClientId=conn_id,
    )
    hand_visual_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=hand_tmp_fname,
        rgbaColor=[0, 0, 1, 1],
        specularColor=[0, 0, 1],
        physicsClientId=conn_id,
    )

    hand_body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=hand_collision_id,
        baseVisualShapeIndex=hand_visual_id,
        physicsClientId=conn_id,
    )

    p.changeDynamics(
        hand_body_id,
        -1,
        lateralFriction=hand_friction,
        restitution=hand_restitution,
        physicsClientId=conn_id,
    )

    obj_tmp_fname = tempfile.mktemp(suffix=".obj", dir=base_tmp_dir)
    os.makedirs(base_tmp_dir, exist_ok=True)
    # Save object obj
    if save_obj_path is not None:
        final_obj_tmp_fname = tempfile.mktemp(suffix=".obj", dir=base_tmp_dir)
        save_obj(final_obj_tmp_fname, obj_verts, obj_faces)
        shutil.copy(final_obj_tmp_fname, save_obj_path)
    # Get obj center of mass
    obj_center_mass = np.mean(obj_verts, axis=0)
    obj_verts -= obj_center_mass
    # add object
    use_vhacd = True

    if use_vhacd:
        if verbose:
            print("Computing vhacd decomposition")
            time1 = time.time()
        # convex hull decomposition
        save_obj(obj_tmp_fname, obj_verts, obj_faces)

        if not vhacd(obj_tmp_fname, vhacd_exe, resolution=vhacd_resolution):
            raise RuntimeError(
                "Cannot compute convex hull "
                "decomposition for {}".format(obj_tmp_fname)
            )
        else:
            print("Succeeded vhacd decomp")

        obj_collision_id = p.createCollisionShape(
            p.GEOM_MESH, fileName=obj_tmp_fname, physicsClientId=conn_id
        )
        if verbose:
            time2 = time.time()
            print(
                "Computed v-hacd decomposition at res {} {:.6f} s".format(
                    vhacd_resolution, (time2 - time1)
                )
            )
    else:
        obj_collision_id = p.createCollisionShape(
            p.GEOM_MESH, vertices=obj_verts, physicsClientId=conn_id
        )

    obj_visual_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=obj_tmp_fname,
        rgbaColor=[1, 0, 0, 1],
        specularColor=[1, 0, 0],
        physicsClientId=conn_id,
    )
    obj_body_id = p.createMultiBody(
        baseMass=object_mass,
        basePosition=obj_center_mass,
        baseCollisionShapeIndex=obj_collision_id,
        baseVisualShapeIndex=obj_visual_id,
        physicsClientId=conn_id,
    )

    p.changeDynamics(
        obj_body_id,
        -1,
        lateralFriction=object_friction,
        restitution=object_restitution,
        physicsClientId=conn_id,
    )

    # simulate for several steps
    if save_video:
        images = []
        if use_gui:
            renderer = p.ER_BULLET_HARDWARE_OPENGL
        else:
            renderer = p.ER_TINY_RENDERER

    for step_idx in range(num_iterations):
        p.stepSimulation(physicsClientId=conn_id)
        #if save_video:
            #img = take_picture(renderer, conn_id=conn_id)
            #images.append(img)
        if save_simul_folder:
            hand_step_path = os.path.join(
                save_simul_folder, "{:08d}_hand.obj".format(step_idx)
            )
            shutil.copy(hand_tmp_fname, hand_step_path)
            obj_step_path = os.path.join(
                save_simul_folder, "{:08d}_obj.obj".format(step_idx)
            )
            pos, orn = p.getBasePositionAndOrientation(
                obj_body_id, physicsClientId=conn_id
            )
            mat = np.reshape(p.getMatrixFromQuaternion(orn), (3, 3))
            obj_verts_t = pos + np.dot(mat, obj_verts.T).T
            save_obj(obj_step_path, obj_verts_t, obj_faces)
        time.sleep(wait_time)

    pos_end = p.getBasePositionAndOrientation(
        obj_body_id, physicsClientId=conn_id
    )[0]

    #if use_vhacd:
    #    os.remove(obj_tmp_fname)
    if save_obj_path is not None:
        os.remove(final_obj_tmp_fname)
    os.remove(hand_tmp_fname)
    distance = np.linalg.norm(pos_end - obj_center_mass)
    p.disconnect(physicsClientId=conn_id)
    return distance

def save_obj(filename, verticies, faces):
    with open(filename, "w") as fp:
        for v in verticies:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write("f %d %d %d\n" % (f[0], f[1], f[2]))

def vhacd(
    filename,
    vhacd_path,
    resolution=1000,
    concavity=0.001,
    planeDownsampling=4,
    convexhullDownsampling=4,
    alpha=0.05,
    beta=0.0,
    maxhulls=1024,
    pca=0,
    mode=0,
    maxNumVerticesPerCH=64,
    minVolumePerCH=0.0001,
):

    cmd_line = (
        '"{}" --input "{}" --resolution {} --concavity {:g} '
        "--planeDownsampling {} --convexhullDownsampling {} "
        "--alpha {:g} --beta {:g} --maxhulls {:g} --pca {:b} "
        "--mode {:b} --maxNumVerticesPerCH {} --minVolumePerCH {:g} "
        '--output "{}" --log "/dev/null"'.format(
            vhacd_path,
            filename,
            resolution,
            concavity,
            planeDownsampling,
            convexhullDownsampling,
            alpha,
            beta,
            maxhulls,
            pca,
            mode,
            maxNumVerticesPerCH,
            minVolumePerCH,
            filename,
        )
    )
    print(cmd_line)

    devnull = open(os.devnull, "wb")
    vhacd_process = Popen(
        cmd_line,
        bufsize=-1,
        close_fds=True,
        shell=True,
        stdout=devnull,
        stderr=devnull,
    )
    return 0 == vhacd_process.wait()
