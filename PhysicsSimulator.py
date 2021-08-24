import pygame
import numpy as np
import math
from pygame.locals import *
from Box2D import *
from Box2D.b2 import *

DEG2RAD = pi / 180
RAD2DEG = 180 / pi

class Bone:
    def __init__(self, world, position, size, angle=0):
        self.position = position
        self.size = size
        
        bodyDef = b2BodyDef()
        bodyDef.type = b2_dynamicBody
        bodyDef.position.Set(position[0], position[1])
        #bodyDef.angle = (angle * RAD2DEG + 90) * DEG2RAD
        self.body = world.CreateBody(bodyDef)
        
        
        shape = b2PolygonShape()
        shape.SetAsBox(size[0], size[1])

        fixtureDef = b2FixtureDef()
        fixtureDef.shape = shape
        fixtureDef.density = 1.0
        fixtureDef.friction = 0.3
        self.body.CreateFixture(fixtureDef)

class Junction:
    def __init__(self, world, boneA, boneB, referenceAngle, limitAngle):
        jointDef = b2RevoluteJointDef()
        jointDef.bodyA = boneA.body
        jointDef.bodyB = boneB.body
        jointDef.collideConnected = True
        jointDef.localAnchorA.Set(0, boneA.size[1])
        jointDef.localAnchorB.Set(0, -boneB.size[1])
        jointDef.referenceAngle = referenceAngle
        jointDef.enableLimit = True
        jointDef.lowerAngle = -limitAngle * DEG2RAD
        jointDef.upperAngle = limitAngle * DEG2RAD
        #jointDef.maxMotorTorque = 0
        #jointDef.motorToque = 0
        #jointDef.motorSpeed = 0
        #jointDef.enableMotor = False
        self.joint = world.CreateJoint(jointDef)


def my_draw_polygon(polygon, simulator, body, fixture):
    vertices=[(body.transform*v) for v in polygon.vertices]
    vertices=[(v[0], v[1]) for v in vertices]
    simulator.pygame.draw.polygon(simulator.screen, simulator.colors[body.type], vertices)

    text = simulator.font.render(str(int(body.angle * RAD2DEG)), True, (0, 100, 255))
    simulator.screen.blit(text, [body.position.x, body.position.y])
polygonShape.draw = my_draw_polygon

def my_draw_distanceJoint(joint, simulator):
    simulator.pygame.draw.line(
        simulator.screen,
        (0, 255, 0),
        [joint.anchorA.x, joint.anchorA.y], 
        [joint.anchorB.x, joint.anchorB.y],
        5)
distanceJoint.draw = my_draw_distanceJoint

def my_draw_revoluteJoint(joint, simulator):
    simulator.pygame.draw.line(
        simulator.screen,
        (0, 255, 0),
        [joint.bodyA.position.x, joint.bodyA.position.y],
        [joint.anchorA.x, joint.anchorA.y ],
        1)
    simulator.pygame.draw.line(
        simulator.screen,
        (0, 255, 0),
        [joint.anchorB.x, joint.anchorB.y],
        [joint.bodyB.position.x, joint.bodyB.position.y],
        1)
    simulator.pygame.draw.circle(
        simulator.screen,
        (0, 255, 0),
        [joint.anchorA.x, joint.anchorA.y],
        10,
        width = 1)
    
    if simulator.bTemp:
        offset = 20
        simulator.bTemp = False
    else:
        offset = -40
        simulator.bTemp = True
    
    text = simulator.font.render(str(int(joint.angle / pi * 180)), True, (0, 255, 255))
    simulator.screen.blit(text, [joint.anchorA.x + offset, joint.anchorA.y])
    text = simulator.font.render(str(int(joint.GetReferenceAngle() / pi * 180)), True, (0, 255, 0))
    simulator.screen.blit(text, [joint.anchorA.x + offset, joint.anchorA.y + 20])
    
revoluteJoint.draw = my_draw_revoluteJoint

def get_matrix_by_transform(position, angle):
    return np.matrix([[math.cos(angle), -math.sin(angle), position[0]],
                      [math.sin(angle), math.cos(angle), position[1]],
                      [0, 0, 1]])

class Simulator:
    
    def __init__(self, width, height):
        pygame.init()
        self.pygame = pygame
        
        self.SCREEN_WD = width
        self.SCREEN_HT = height
        self.TARGET_FPS = 30
        
        self.screen = pygame.display.set_mode((self.SCREEN_WD, self.SCREEN_HT))
        self.pygame.display.set_caption("PyBox2D_Example")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('consolas', 24)

        self.world = b2World(gravity = (0, 0), doSleep = True)

        self.timeStep = 1.0 / 30
        self.velIters = 100
        self.posIters = 100

        self.colors = {
            staticBody  : (255,255,255,255),
            dynamicBody : (127,127,127,255),
        }

        self.bones = {}
        self.junctions = []

        self.originalBoneMatrix = {}
        self.changedBoneMatrix = {}
        self.matrixOverTime = []
        
        self.bTemp = True
        self.tick = 0

        self.jointQueue = []
        self.jointTime = 0


    def init_world(self):
        ground = Bone(self.world, (160, 0), (160, 10))
        ground.body.type = b2_staticBody
        bone1 = Bone(self.world, (160, 20), (10, 40))
        junction0 = Junction(self.world, ground, bone1, 0, 1)
        bone2 = Bone(self.world, (160, 180), (10, 40))
        junction1 = Junction(self.world, bone1, bone2, 0, 2)
        bone3 = Bone(self.world, (160, 340), (10, 40))
        junction2 = Junction(self.world, bone2, bone3, 0, 3)
        bone4 = Bone(self.world, (160, 340), (5, 40))
        junction3 = Junction(self.world, bone3, bone4, 30, 8)
        bone5 = Bone(self.world, (160, 340), (8, 40))
        junction3 = Junction(self.world, bone3, bone5, -10, 5)
        bone6 = Bone(self.world, (160, 500), (5, 40))
        junction4 = Junction(self.world, bone1, bone6, -30, 8)
        bone7 = Bone(self.world, (160, 500), (8, 40))
        junction5 = Junction(self.world, bone5, bone7, -30, 8)

    def set_world_sequentially(self, bones, junctions):
        if bones is None or len(bones) < 1:
            return

        print("input bone data : ")
        for id in bones.keys():
            print(bones[id])

        self.bones = {}
        for id in bones.keys():
            self.bones[id] = Bone(self.world, bones[id][0], bones[id][1], bones[id][2])
            self.originalBoneMatrix[id] = get_matrix_by_transform(bones[id][0], bones[id][2])

        self.bones[-1].body.type = b2_staticBody
        self.jointQueue = junctions.copy()

        
    
    def set_world(self, bones, junctions):
        if bones is None or len(bones) < 1:
            return

        for id in bones.keys():
            self.bones[id] = Bone(self.world, bones[id][0], bones[id][1], bones[id][2])
            self.originalBoneMatrix[id] = get_matrix_by_transform(bones[id][0], bones[id][2])
            
            bone_keys = self.bones.keys()
            exist = []
            for junction in junctions:
                if (junction[0][0], junction[0][1]) in exist:
                    continue
                
                if (junction[0][0] in bone_keys) and  (junction[0][1] in bone_keys):
                    exist.append((junction[0][0], junction[0][1]))
                    self.junctions.append(Junction(
                        self.world,
                        self.bones[junction[0][0]],
                        self.bones[junction[0][1]],
                        junction[1],
                        10
                    ))
        
        self.bones[-1].body.type = b2_staticBody
        '''
        for junction in junctions:
            self.junctions.append(Junction(
                self.world,
                self.bones[junction[0][0]],
                self.bones[junction[0][1]],
                junction[1],
                10
            ))
        '''
        '''
        for id in bones.keys():
            self.originalBoneMatrix[id] = get_matrix_by_transform(self.bones[id].body.position,
                                                                  self.bones[id].body.angle)
        '''

    def run_sequentially(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    continue
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    running = False
                    continue

            self.screen.fill((0, 0, 0, 0))

            for body in self.world.bodies:
                for fixture in body.fixtures:
                    fixture.shape.draw(self, body, fixture)
            

            for joint in self.world.joints:
                joint.draw(self)


            if len(self.jointQueue) > 0:
                if self.jointTime < pygame.time.get_ticks():
                    self.jointTime += 5000
                    junction = self.jointQueue.pop(0)
                    self.junctions.append(Junction(
                        self.world,
                        self.bones[junction[0][0]],
                        self.bones[junction[0][1]],
                        junction[1],
                        10
                    ))
                    print(junction[1]*RAD2DEG)

            #for id in self.bones.keys():
            #    print(self.bones[id].position)
            
            
            self.world.Step(self.timeStep, self.velIters, self.posIters)
            self.pygame.display.flip()
            self.clock.tick(self.TARGET_FPS)

        self.pygame.quit()
        print("simulator done")
            
    
    def run(self):
        #self.init_world()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    continue
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    running = False
                    continue

            self.screen.fill((0, 0, 0, 0))

            for body in self.world.bodies:
                for fixture in body.fixtures:
                    fixture.shape.draw(self, body, fixture)
            

            for joint in self.world.joints:
                joint.draw(self)

            if pygame.time.get_ticks() > 1000:
                self.tick += 1
                if self.tick > 30:
                    self.tick = 0

                    changeMatrix = {}
                    for id in self.bones.keys():
                        self.changedBoneMatrix[id] = get_matrix_by_transform(self.bones[id].body.position,
                                                                             self.bones[id].body.angle)
                        changeMatrix[id] = np.linalg.inv(self.originalBoneMatrix[id]) * self.changedBoneMatrix[id]
                    self.matrixOverTime.append(changeMatrix)
            
            if (pygame.time.get_ticks() // 2000) % 2 == 0:
                world.gravity = (10, -10)
            else:
                world.gravity = (-10, -10)
                       
            self.world.Step(self.timeStep, self.velIters, self.posIters)
            self.pygame.display.flip()
            self.clock.tick(self.TARGET_FPS)

            

        self.pygame.quit()
        print("simulator done")

simulator = Simulator(658, 856)
simulator.init_world()
simulator.run()
