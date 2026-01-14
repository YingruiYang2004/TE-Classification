import random

## 1..75 core
## 76..100 non-core

ran1 = random.randint(1, 100)
ran2 = -1

if ran1 <= 75:
    print("core")
else:
    print("non-core")
    ran2 = random.randint(1, 75)

print("First random:", ran1)
print("Additional core:", ran2)