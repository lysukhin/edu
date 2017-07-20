n = int(input())
a = map(int, input().split())
x = int(input())

closest = x + 2001
mindiff = 2001

for el in a:
	if abs(x - el) <= mindiff:
		if abs(x - el) == mindiff:
			closest = min(closest, el)
		else:
			closest = el			
			mindiff = abs(x - el)

print(closest)