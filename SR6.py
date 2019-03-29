#Luis Esturban 17256
#Graficas por computadora
#SR6
#29/03/2019
import numpy as np
import random
import struct
from math import *
from operator import attrgetter
from collections import namedtuple
import sys


def char(c):
    return struct.pack("=c", c.encode('ascii'))
def word(c):
    return struct.pack("=h", c)
def dword(c):
    return struct.pack("=l", c)
def color(r, g, b):
    return bytes([b, g, r])


V2 = namedtuple('Vertex2', ['x', 'y'])
V3 = namedtuple('Vertex3', ['x', 'y', 'z'])

def sum(v0, v1):
    return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def res(v0, v1):
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def mul(v0, k):
    return V3(v0.x * k, v0.y * k, v0.z *k)

def pun(v0, v1):
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

def cru(v0, v1):
    return V3(
        v0.y * v1.z - v0.z * v1.y,
        v0.z * v1.x - v0.x * v1.z,
        v0.x * v1.y - v0.y * v1.x,
      )

def lon(v0):
    return (v0.x**2 + v0.y**2 + v0.z**2)**0.5

def norm(v0):
    l = lon(v0)
    if (l == 0):
        return V3(0, 0, 0)
    return V3(v0.x/l, v0.y/l, v0.z/l)


class Texture(object):
    def __init__(self, filename):
        self.path = filename
        self.read()

    def read(self):
        img = open(self.path, "rb")
        img.seek(2+4+4)
        siz = struct.unpack("=l", img.read(4))[0] 
        img.seek(2+4+4+4+4)
        self.width = struct.unpack("=l", img.read(4))[0]
        self.height = struct.unpack("=l", img.read(4))[0] 
        self.pix = []
        img.seek(siz)

        for y in range(self.height):
            self.pix.append([])
            for x in range(self.width):
                b = ord(img.read(1))
                g = ord(img.read(1))
                r = ord(img.read(1))
                self.pix[y].append(color(r, g, b))
        img.close()
        
    def get_color(self, tx, ty, inte=1):
        x = int(tx * self.width)
        y = int(ty * self.height)
        try:
        	return bytes(map(lambda b: round(b*inte) if b *inte > 0 else 0, self.pix[y][x]))
        except:
            return bytes(map(lambda b: round(b*inte) if b *inte > 0 else 0, self.pix[y-1][x-1]))

class Obj(object):
    def __init__(self, filename):
        with open(filename) as f:
            self.lines = f.read().splitlines()

        self.vertex = []
        self.tvertex = []
        self.cars = []
        self.read()
    #Funcion para poder leer los valores de obj
    def read(self):
        for line in self.lines:
            if line:
                prefix, value = line.split(' ', 1)

                if prefix == 'v':
                    self.vertex.append(list(map(float, value.split(' '))))
                if prefix == 'vt':
                    self.tvertex.append(list(map(float, value.split(' ')))) 
                elif prefix == 'f':
                    self.cars.append([list(map(int, car.split('//'))) for car in value.split(' ')])
class Bitmap(object):
    #Contiene los valores de iniciacion 
    def glInit(self):
        self.width = 0
        self.height = 0
        self.color = color(0, 0, 0)
        self.col = color(255, 0, 0)
    #Funcion que asigna el tamano de la escena
    def glCreateWindow(self, width, height):
        self.width = width
        self.height = height
        self.framebuffer = []
        self.glClear()
    #Funcion que crea el punto en el espacio indicado
    def point(self, x, y, color):
	    self.framebuffer[y][x] = color
	#Funcion que crea el limite del area de trabajo permitida
    def glViewPort(self, x, y , width, height):
        self.pX = x
        self.pY = y
        self.vW = width
        self.vH = height
    #indica la pocicion donde se colocara el punto
    def glVertex(self, x, y):
        xF = int((x+1)*(self.vW/2)+self.pX)
        yF = int((y+1)*(self.vH/2)+self.pY)
        self.point(xF, yF, self.col)
    #Funcion que crea la escena en la cual se va a trabajar
    def glClear(self):
        self.framebuffer = [
            [
                self.color
                    for x in range(self.width)
	        ]
	        for y in range(self.height)
	    ]
    
        self.zbuffer = [
            [-float('inf') for x in range(self.width)]
            for y in range(self.height)
        ]
    #Funcion que le asigna el color al fondo menores o iguales a 1
    def glClearColor(self, r, g, b):
        if(r<=1 and g<=1 and b<=1):
            self.color = color((r*255), (g*255), (b*255))
            self.glClear()
    #Asigna el color del punto elegido
    def glColor(self, r, g, b):
       self.col = color(r, g, b)
    #Funcion que calcula el pixel en x
    def pixelX(self,a):
    	resX = 2*((a-self.pX)/self.vW)-1
    	return resX
    #Funcion que calcula el pixel en y
    def pixelY(self,b):
    	resY = 2*((b-self.pY)/self.vH)-1
    	return resY
    #Funcion para poder crear lineas que van de -1 a 1
    def glLine(self,x0, y0, x1, y1):
    	xN0 = round((x0+1)*(self.vW/2)+self.pX)
    	xN1 = round((x1+1)*(self.vW/2)+self.pX)
    	yN0 = round((y0+1)*(self.vH/2)+self.pY)
    	yN1 = round((y1+1)*(self.vH/2)+self.pY)
    	i = 0
    	while i <= 1:
    		x = xN0 + (xN1 - xN0) * i
    		y = yN0 + (yN1 - yN0) * i 
    		self.glVertex(self.pixelX(x),self.pixelY(y))
    		i += 0.01
    #Funcion para poder mover la camara
    def lookAt(self,eye,center, up,translate):
        translate=V3(*translate)
        z = norm(res(eye,center))
        x = norm(cru(up,z))
        y = norm(cru(z, x))
        self.loadViewMatrix(x,y,z,center)
        self.loadProjectionMatrix(eye.z / lon(res(eye, center)))
        self.loadViewPortMatrix(translate)
    ################################################################Bloque de las matrices
    def loadModelMatrix(self, translate, scale, rotate):
        translate = V3(*translate)
        scale = V3(*scale)
        rotate = V3(*rotate)
        translate_matrix = np.matrix ([
            [1,0,0,translate.x],
            [0,1,0,translate.y],
            [0,0,1,translate.z],
            [0,0,0,1]])
        scale_matrix = np.matrix ([
            [scale.x,0,0,0],
            [0,scale.y,0,0],
            [0,0,scale.z,0],
            [0,0,0,1]])
        v = rotate.x
        rotation_matrix_x = np.matrix ([
            [1,0,0,0],
            [0,cos(v),-sin(v),0],
            [0,sin(v),cos(v),0],
            [0,0,0,1]])
        v = rotate.y
        rotation_matrix_y = np.matrix ([
            [cos(v),0,sin(v),0],
            [0,1,0,0],
            [-sin(v),0,cos(v),0],
            [0,0,0,1]])
        v = rotate.z
        rotation_matrix_z = np.matrix ([
            [cos(v),-sin(v),0,0],
            [sin(v),cos(v),0,0],
            [0,0,1,0],
            [0,0,0,1]])
        rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z
        self.Model = translate_matrix @ rotation_matrix @ scale_matrix
    def loadViewMatrix(self,x,y,z,center):
        M = np.matrix ([
            [x.x,x.y,x.z,0],
            [y.x,y.y,y.z,0],
            [z.x,z.y,z.z,0],
            [0,0,0,1]])
        O = np.matrix ([
            [1,0,0,-center.x],
            [0,1,0,-center.y],
            [0,0,1,-center.z],
            [0,0,0,1]])
        self.View = M @ O
    def loadViewPortMatrix(self,translate):
        self.ViewPort = np.matrix ([
            [self.width/2,0,0,translate.x+self.width/2],
            [0,self.height/2,0,translate.y+self.height/2],
            [0,0,500,500],
            [0,0,0,1]])
    def loadProjectionMatrix(self, coeff):
        self.Proyection = np.matrix ([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,coeff,1]])
#########################################################################################################################################
    def flood(self, x, y, col1, col2):
        if (x < 0 or x >= self.width or y < 0 or y >= self.height):
            return
        if (self.framebuffer[x][y] != col1):
            return
        self.point(x, y, col2)

        self.flood(x+1, y, col1, col2)
        self.flood(x-1, y, col1, col2)
        self.flood(x, y+1, col1, col2)
        self.flood(x, y-1, col1, col2)
    def floodFill(self, x, y, colorN):
        colorO = self.framebuffer[x][y]
        self.flood(x, y, colorO, colorN)
    def transform(self, vertex, tra=(0, 0, 0), sca=(1, 1, 1)):
        return V3(
          round((vertex[0] + tra[0]) * sca[0]),
          round((vertex[1] + tra[1]) * sca[1]),
          round((vertex[2] + tra[2]) * sca[2])
        )
    #Funcion para transformar conforme a las matrices
    def transform2(self,vertex):
        aug_vertex=np.matrix([
            [vertex.x],
            [vertex.y],
            [vertex.z],
            [1]
            ])
        trans_vertex = self.Model @ aug_vertex
        trans_vertex = [
            (round(trans_vertex.item(0)/trans_vertex.item(3))),
            (round(trans_vertex.item(1)/trans_vertex.item(3))),
            (round(trans_vertex.item(2)/trans_vertex.item(3)))
            ]
        return V3(*trans_vertex)

    def bbox(self, A, B, C):
        xs = sorted([A.x, B.x, C.x])
        ys = sorted([A.y, B.y, C.y])
        return V2(xs[0], ys[0]), V2(xs[2], ys[2])
    #Coordenadas barycentricas
    def barycentric(self, A, B, C, P):
        cx, cy, cz = cru(
                V3(C.x - A.x, B.x - A.x, A.x - P.x),
                V3(C.y - A.y, B.y - A.y, A.y - P.y)
            )
        if (cz == 0): 
            return -1, -1, -1
        u = cx/cz
        v = cy/cz
        w = 1 - (u + v)
        return w, v, u
    #Funcion de llenado de triangulos
    def triangleS(self, A, B, C, color=None, texture=None, texture_coords=(), inte=1):
        bbox_min, bbox_max = self.bbox(A, B, C)

        for x in range(bbox_min.x, bbox_max.x + 1):
            for y in range(bbox_min.y, bbox_max.y + 1):
                w, v, u = self.barycentric(A, B, C, V2(x, y))

                if (w <0 or v < 0 or u < 0):
                    continue

                if texture:
                    tA, tB, tC = texture_coords
                    tx = tA.x * w + tB.x * v + tC.x * u
                    ty = tA.y * w + tB.y * v + tC.y * u

                    color = texture.get_color(tx, ty, inte)

                z = A.z * w + B.z * v + C.z * u

                if x < 0 or y < 0:
                    continue

                if z > self.zbuffer[x][y]:
                    self.point(x, y, color)
                    self.zbuffer[x][y] = z
    #Funcion para poder multiplicar las matrices para a,b,c
    def mulMat(self):
        a = self.Model @ self.View
        b = self.Proyection @ a
        c = self.ViewPort @ b
        self.hola=c
	#Funcion donde se crea la imagen 
    def glFinish(self, filename):
        f = open(filename, 'wb')
        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(14 + 40 + self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(14 + 40))
        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))
        f.write(dword(0))
        f.write(dword(self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        for x in range(self.height):
            for y in range(self.width):
                f.write(self.framebuffer[x][y])
        f.close()
    def mtl(self,filename):
    	colMTL=[]
    	with open(filename) as lins:
    		for lin in lins:
    			if 'Kd' in lin:
    				colMTL.append(lin.split())
    	return colMTL
     #Funcion para cargar el archivo obj y llenarlo 
    def load2(self,filename,mat, tra, sca,):
        mod = Obj(filename)
        colMTL=self.mtl(mat)
        mat2=colMTL[0]
        luz = V3(0, 0, 1)

        for car in mod.cars:
            vcount = len(car)
            if vcount == 3:
                f1 = car[0][0] - 1
                f2 = car[1][0] - 1
                f3 = car[2][0] - 1

                a = V3(*mod.vertex[f1])
                b = V3(*mod.vertex[f2])
                c = V3(*mod.vertex[f3])
                
                nor = norm(cru(res(b, a), res(c, a)))
                inte = pun(nor, luz)
                gray = round(255 * inte)
                
                a = self.transform(mod.vertex[f1], tra, sca)
                b = self.transform(mod.vertex[f2], tra, sca)
                c = self.transform(mod.vertex[f3], tra, sca)
                if gray < 0:
                    continue
                self.triangleS(a, b, c, color(round(round(float(mat2[1])*255)*inte),round(round(float(mat2[2])*255)*inte),round(round(float(mat2[3])*255)*inte)))
            else:
                f1 = car[0][0] - 1
                f2 = car[1][0] - 1
                f3 = car[2][0] - 1
                f4 = car[3][0] - 1
                ver =  [
                    self.transform(mod.vertex[f1], tra, sca),
                    self.transform(mod.vertex[f2], tra, sca),
                    self.transform(mod.vertex[f3], tra, sca),
                    self.transform(mod.vertex[f4], tra, sca)
                ]
                a, b, c, d = ver
                nor = norm(cru(res(a, b), res(c, d)))
                inte = pun(nor, luz)
                gray = round(255 * inte)
                if gray < 0:
                    continue
                self.triangleS(a, b, c, color(round(round(float(mat2[1])*255)*inte),round(round(float(mat2[2])*255)*inte),round(round(float(mat2[3])*255)*inte)))
                self.triangleS(a, c, d, color(round(round(float(mat2[1])*255)*inte),round(round(float(mat2[2])*255)*inte),round(round(float(mat2[3])*255)*inte)))
    #Funcion para leer texturas
    def load3(self, filename, tra, sca,rot, texture):
        mod = Obj(filename)
        self.loadModelMatrix(tra,sca,rot)
        self.mulMat()
        luz = V3(0, 0, 1)
        for car in mod.cars:
            vcount = len(car)
            if vcount == 3:
                f1 = car[0][0] - 1
                f2 = car[1][0] - 1
                f3 = car[2][0] - 1

                a = self.transform2(V3(mod.vertex[f1][0],mod.vertex[f1][1],mod.vertex[f1][2]))
                b = self.transform2(V3(mod.vertex[f2][0],mod.vertex[f2][1],mod.vertex[f3][2]))
                c = self.transform2(V3(mod.vertex[f3][0],mod.vertex[f3][1],mod.vertex[f3][2]))
                nor = norm(cru(res(b, a), res(c, a)))
                inte = pun(nor, luz)
                #a = self.transform(mod.vertex[f1], tra, sca) 
                #b = self.transform(mod.vertex[f2], tra, sca)
                #c = self.transform(mod.vertex[f3], tra, sca)
                if inte > 255:
                    inte = 255
                if inte < 0:
                    inte = 0
                if not texture:
                    mon = round(255 * inte)
                    if mon < 0:
                        continue
                    self.triangleS(a, b, c, color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                else:
                    mon = round(255 * inte)
                    t1 = car [0][1] - 1
                    t2 = car [1][1] - 1
                    t3 = car [2][1] - 1
                    tA = V2(*mod.tvertex[t1])
                    tB = V2(*mod.tvertex[t2])
                    tC = V2(*mod.tvertex[t3])
                    self.triangleS(a, b, c, color(mon, mon, mon), texture=texture, texture_coords=(tA, tB, tC), inte=inte)
            else:
                f1 = car[0][0] - 1
                f2 = car[1][0] - 1
                f3 = car[2][0] - 1
                f4 = car[3][0] - 1
                vertices =  [
                    self.transform(mod.vertex[f1], tra, sca),
                    self.transform(mod.vertex[f2], tra, sca),
                    self.transform(mod.vertex[f3], tra, sca),
                    self.transform(mod.vertex[f4], tra, sca)
                ]
                a, b, c, d = vertices
                nor = norm(cru(res(a, b), res(c, d)))
                inte = pun(nor, luz)
                gray = round(255 * inte)
                self.triangleS(a, b, c)
                self.triangleS(a, c, d)
r = Bitmap()
r.glInit()
r.glCreateWindow(800, 600)
#t = Texture('Stonewall15_512x512.bmp')
r.lookAt(V3(0, 0, 0), V3(0, 0, -100), V3(0, 1,0),translate=(0,0,0))
r.load3('mono.obj', tra=(400, 300, 0), sca=(70, 70, 70),rot=(0,-1,0),texture=None)
r.glFinish('SR6.bmp')

