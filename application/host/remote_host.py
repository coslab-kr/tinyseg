import serial

ser = serial.Serial('/dev/ttyACM0')
array = bytearray(1024 * 1024)

while True:
  msg = ser.read_until(b'TINYSEG_INI')
  print(msg.decode('UTF-8'))

  rw = ser.read()
  pos = int.from_bytes(ser.read(4), byteorder='little')
  size = int.from_bytes(ser.read(4), byteorder='little')

  if rw == b'r':
    ser.write(array[pos:pos+size])
  elif rw == b'w':
    data = ser.read(size)
    array[pos:pos+size] = data
  
  msg = ser.read_until(b'TINYSEG_FIN')
  print(msg.decode('UTF-8'))

