import struct

with open(r'C:\Users\Zwmar\projects\sub1quant\llama.cpp\build\bin\llama-perplexity.exe', 'rb') as f:
    data = f.read()

dlls = set()
for i in range(len(data)):
    if data[i:i+4] in (b'.dll\x00', b'.DLL\x00'):
        start = i
        while start > 0 and data[start-1] not in (0, 0x22, 0x2E, 0x5C):
            start -= 1
        name = data[start:i+4]
        try:
            n = name.decode('utf-8', errors='ignore').strip('\x00').strip()
            if n.endswith('.dll') or n.endswith('.DLL'):
                dlls.add(n.lower())
        except:
            pass

for d in sorted(dlls):
    print(d)
