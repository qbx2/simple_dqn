#!/usr/bin/env python3
import time
import struct
from multiprocessing import Process, Condition, Value
from socketserver import ThreadingMixIn
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import urllib
import re
import ctypes
import io
import zlib
import numpy as np
import time
import random

DEBUG = False

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
	pass

class HTTPHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		with self.server.counter:
			parsed_path = urllib.parse.urlparse(self.path)

			m = re.match(r'^.*\.png$', parsed_path.path)

			if m is not None:
				data = self.server.current_screen
				self.send_response(200)
				#self.send_header('Content-Type', 'image/bmp')
				#self.send_header('Content-Encoding', 'gzip')
				#self.send_header('Content-Length', str(len(data)))
				self.send_header('Content-Type', 'multipart/x-mixed-replace;boundary=SimpleDQNBoundary')
				self.end_headers()

				try:
					while True:
						data = self.server.current_screen

						if data is not None:
							self.wfile.write(b'--SimpleDQNBoundary\r\n')
							self.wfile.write(b'Content-Type: image/png\r\n')
							self.wfile.write(b'Content-Length: %d\r\n\r\n'%len(data))
							self.wfile.write(data)
							self.wfile.flush()

						with self.server.condition_refresh:
							self.server.condition_refresh.wait()
				except BrokenPipeError:
					pass

				return

			if parsed_path.path == '/':
				self.send_response(302)
				self.send_header('Location', '_.png')
				self.end_headers()
				return

			self.send_response(404)
			return

class Server:
	def __init__(self):
		print('The rendering server is running at :11000, check http://localhost:11000')
		self.http_server = ThreadedHTTPServer(('', 11100), HTTPHandler)

	def run(self):
		threading.Thread(target=self.http_server.serve_forever, daemon=True).start()

class Counter:
	def __init__(self):
		self.value = Value(ctypes.c_int)

	def __enter__(self):
		with self.value.get_lock():
			self.value.value += 1

	def __exit__(self, exc_type, exc_val, exc_tb):
		with self.value.get_lock():
			self.value.value -= 1

	def __repr__(self):
		return str(self.value.value)

class Line:
	def __init__(self, x1, x2, y1, y2):
		self.x1 = x1
		self.x2 = x2
		self.y1 = y1
		self.y2 = y2
		self.direction = 0 if x1 == x2 else 1 # 0 for verical, 1 for horizontal

	def __str__(self):
		return '%s line (%s = %d, %d <= %s <= %d)'%(('vertical', 'x', self.x1, self.y1, 'y', self.y2) if self.direction == 0 else ('horizontal', 'y', self.y1, self.x1, 'x', self.x2))

	def collide(self, x_t, y_t, x_tp1, y_tp1, dx, dy):
		# return new state
		# x_t, y_t is collision point
		
		# 1. calculate collision point
		# 2. calculate new point
		if self.direction == 1: # horizontal, x1 != x2, y1 == y2
			x, y = (self.y1 - y_t) * dx / dy + x_t, self.y1
			collided = self.x1 <= x <= self.x2 and y_t != self.y1 and (y_t <= y <= y_tp1 or y_tp1 <= y <= y_t) and abs(self.y1 - y_t) <= abs(dy)
			if collided:
				if DEBUG:
					print('collision with %s'%self)
				sign = -1 if y_t < self.y1 else 1
				x_t, y_t = x, y
				y_tp1 = sign * abs(y_tp1 - self.y1) + self.y1
			return x_t, y_t, x_tp1, y_tp1, collided
		else: # vertical, y1 != y2, x1 == x2
			x, y = self.x1, (self.x1 - x_t) * dy / dx + y_t
			collided = self.y1 <= y <= self.y2 and x_t != self.x1 and (x_t <= x <= x_tp1 or x_tp1 <= x <= x_t) and abs(self.x1 - x_t) <= abs(dx)
			if collided:
				if DEBUG:
					print('collision with %s'%self)
				sign = -1 if x_t < self.x1 else 1
				x_t, y_t = x, y
				x_tp1 = sign * abs(x_tp1 - self.x1) + self.x1
			return x_t, y_t, x_tp1, y_tp1, collided

class Wall(Line):
	def __init__(self, x1, x2, y1, y2):
		super().__init__(x1, x2, y1, y2)
		self.color = 0x4248c8ff
		self.height = y2-y1+1
		self.width = x2-x1+1
		self.x = self.x1
		self.y = self.y1
		self.buf = np.ones((self.height, self.width)) * self.color

class PlayField:
	def __init__(self, ServerClass=None):
		self.width, self.height = 50, 50
		self.renderer = Renderer(self.width, self.height)
		self.controller = Controller()
		self.init()

		if ServerClass is not None:
			self.server = ServerClass()
			self.server.http_server.counter = Counter()
			self.server.http_server.condition_refresh = Condition()
			self.server.http_server.current_screen = None
			self.server.run()

	def init(self):
		self.score = 0
		#self.walls = [Wall(0, 0, 0, self.height-1), Wall(0, self.width-1, 0, 0), Wall(self.width-1, self.width-1, 0, self.height-1), Wall(0, self.width-1, self.height-1, self.height-1)]
		self.walls = [Wall(0, 0, 0, self.height-1), Wall(0, self.width-1, 0, 0), Wall(self.width-1, self.width-1, 0, self.height-1)]#, Wall(0, self.width//2-20, self.height-1, self.height-1), Wall(self.width//2+20, self.width-1, self.height-1, self.height-1)]
		self.ball = Ball(random.randint(0, self.width-1), 10, random.randint(2,5), random.randint(2,5))
		self.paddle = Paddle(self, random.randint(10, self.width-10), 40)
		#self.timer = Timer()
		#self.controller = Controller()

	def run(self):
		try:
			while True:
				# 1. draw
				data = self.renderer.prepare_rendering_data(self)

				if self.server is not None:
					if self.server.http_server.counter.value.value > 0: # render only when at least one client is connected to the server
						self.server.http_server.current_screen = self.renderer.render(data)

						with self.server.http_server.condition_refresh:
							self.server.http_server.condition_refresh.notify_all()

				# 2. show and input, move
				if 0 <= self.ball.x <= 49 and 0 <= self.ball.y <= 49:			
					key = int(self.controller.input(pf=self, data=data))
					self.paddle.move(key * 3)
				else:
					self.controller.new_episode(self, data)
					self.init()

				# move
				x_t, y_t = self.ball.x, self.ball.y
				self.ball.move()

				paddleLines = self.paddle.getLines()
				lines = self.walls + paddleLines

				if DEBUG:
					print(x_t, y_t, self.ball.x, self.ball.y, self.ball.dx, self.ball.dy)

				while True:
					*tpos, tdx, tdy = x_t, y_t, self.ball.x, self.ball.y, self.ball.dx, self.ball.dy
					min_dist = None
					collided_lines = []

					for line in lines:
						*pos, collided = line.collide(*tpos, tdx, tdy)

						if collided:
							if DEBUG:
								print(pos)
							dist_x, dist_y = (tpos[0] - pos[0]), (tpos[1] - pos[1])
							dist = dist_x * dist_x + dist_y * dist_y

							if min_dist is None or dist == min_dist:
								collided_lines.append(line)
								if line.direction == 0:
									self.ball.dx = -tdx
								else:
									self.ball.dy = -tdy

								x_t, y_t, self.ball.x, self.ball.y, min_dist = *pos, dist
							elif dist < min_dist:
								collided_lines = []
								collided_lines.append(line)
								if DEBUG:
									print('last collision is canceled')
								self.ball.dx, self.ball.dy = (-tdx, tdy) if line.direction == 0 else (tdx, -tdy)
								x_t, y_t, self.ball.x, self.ball.y, min_dist = *pos, dist

					if min_dist is None:
						break

					for line in paddleLines:
						if line in collided_lines:
							self.score += 10

					if DEBUG:
						print('\t',x_t, y_t, self.ball.x, self.ball.y, self.ball.dx, self.ball.dy)

				#input()

				#if self.ball.y > self.height:
				#	self.init()

				#assert 0 <= self.ball.x <= 49 and 0 <= self.ball.y <= 49
				#assert (int(self.ball.x), int(self.ball.y)) == (self.ball.x, self.ball.y)
				#time.sleep(1/30)
		except KeyboardInterrupt:
			self.controller.handle_keyboardinterrupt()
			raise

class Ball:
	width, height = 1, 1

	def __init__(self,  x, y, dx, dy):
		self.x = x
		self.y = y
		self.dx = dx
		self.dy = dy
		self.color = 0xc84848ff
		self.buf = np.ones((self.height, self.width)) * self.color

	def move(self):
		#print('(%d,%d) => '%(self.x,self.y),end='')
		self.x += self.dx
		self.y += self.dy
		#print('(%d,%d)'%(self.x,self.y))

class Paddle:
	width, height = 8, 1

	def __init__(self, pf, x, y):
		self.x = x
		self.y = y
		self.color = 0xc84848ff
		self.buf = np.ones((self.height, self.width)) * self.color
		self.max_x = pf.width - self.width
		self.lines = None
		self.updateLines()

	def move(self, dx):
		self.x = max(0, min(self.max_x, self.x + dx))
		if dx:
			self.updateLines()

	def updateLines(self):
		if self.lines is None:
			self.lines = [Line(self.x, self.x + self.width - 1, self.y, self.y)]
		else:
			line = self.lines[0]
			line.x1, line.x2, line.y1, line.y2 = self.x, self.x + self.width - 1, self.y, self.y

	def getLines(self):
		return self.lines

class Controller:
	def __init__(self):
		pass

	def handle_keyboardinterrupt(self):
		pass

	def new_episode(self, pf, data):
		pass

	def input(self, pf=None, data=None):
		if pf is not None:
			return min(1, max(-1, pf.ball.x - (pf.paddle.x + pf.paddle.width // 2)))

		return None

class Timer:
	def wait(self, sec):
		time.sleep(sec)

class Renderer:
	MAG_POWER = 10

	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.fps = 0
		self.last_update = time.time()
		self.ret = np.zeros((self.height, self.width), dtype='>u4')

	def prepare_rendering_data(self, pf):
		#self.fps += 1

		#t = time.time() - self.last_update
		#if t >= 1:
		#	print('fps:',self.fps/t)
		#	self.last_update = time.time()
		#	self.fps=0

		ret = self.ret
		ret.fill(0x000000ff) # alpha

		for wall in pf.walls:
			ret[wall.y:wall.y+wall.height, wall.x:wall.x+wall.width] = wall.buf

		ret[pf.ball.y:pf.ball.y+pf.ball.height, pf.ball.x:pf.ball.x+pf.ball.width] = pf.ball.buf
		ret[pf.paddle.y:pf.paddle.y+pf.paddle.height, pf.paddle.x:pf.paddle.x+pf.paddle.width] = pf.paddle.buf
		return ret

	def render(self, data):
		#pad = b''#b'\x00' * (self.width * 4 // 4 + 1) * 4
		sig = b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'

		def write_chunk(f, typ, data):
			f.write(struct.pack(">I",len(data)))
			f.write(typ)
			f.write(data)
			f.write(struct.pack(">I",zlib.crc32(typ+data, 0)))

		f = io.BytesIO()
		f.write(sig)

		def write_ihdr(f, width, height, depth, color):
			chunk = b''
			chunk += struct.pack(">iiBB",width,height,depth,color)
			chunk += b'\0\0\0'
			write_chunk(f, b"IHDR", chunk)

		write_ihdr(f, self.width * self.MAG_POWER, self.height * self.MAG_POWER, 8, 6)

		def write_idat(f, pixels):
			if isinstance(pixels,list):
				pixels = bytes(pixels)
			write_chunk(f, b"IDAT", zlib.compress(pixels))

		pixels = b''.join(b'\x00' + col.tostring() for col in data.repeat(self.MAG_POWER, axis=0).repeat(self.MAG_POWER, axis=1))
		write_idat(f, pixels)

		def write_iend(f):
			write_chunk(f, b"IEND", b"")

		write_iend(f)

		return f.getvalue()

		#data = list(zip(*data))
		#data = b''.join((b''.join(struct.pack('<L', col | (0xff << 24)) for col in row) + pad) for row in reversed(data))
		#return b'BM' + struct.pack('<LHHL', len(data) + 24, 0, 0, 32) + struct.pack('<LHHHH', 12, self.width, self.height, 1, 32) + b'\xff' * 6 + data

if __name__ == "__main__":
	pf = PlayField(Server)
	pf.run()
