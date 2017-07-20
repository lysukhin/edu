"""
Контроперация

(Время: 1 сек. Память: 16 Мб Сложность: 17%)
Хакер Василий получил доступ к классному журналу и хочет заменить все свои минимальные оценки на максимальные. 
Напишите программу, которая заменяет оценки Василия, но наоборот: все максимальные – на минимальные.

Входные данные
Первая строка входного файла INPUT.TXT содержит натуральное число N – количество оценок в журнале. Во второй строке записаны N целых чисел Ai – оценки Василия. Все числа во входных данных не превышают 1000 по абсолютной величине.

Выходные данные
В выходной файл OUTPUT.TXT выведите исправленные оценки, сохранив порядок.
"""

n = int(input())
a = list(map(int, input().split()))

min_ = 1001
max_ = -1001

for el in a:
	if el > max_:
		max_ = el
	if el < min_:
		min_ = el

for i in range(n):
	if a[i] == max_:
		a[i] = min_

print(*a)