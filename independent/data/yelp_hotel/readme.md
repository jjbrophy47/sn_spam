label: 0 - not spam (N, NR), 1 - spam (Y, YR)
hotel_accepts_cc: 0 - n/a, 1 - no, 2 - yes
hotel_wifi: 0 - n/a, 1 - free, 2 - no, 3 - paid
hotel_price: 0 - n/a, 1 - $, 2 - $$, 3 - $$$, 4 - $$$$

def p(x):
	r = 0
	if x == '$' or x == '£' or x == '€':
		r = 1
	elif x == '$$' or x == '££' or x == '€€':
		r = 2
	elif x == '$$$' or x == '£££' or x == '€€€':
		r = 3
	elif x == '$$$$' or x == '££££' or x == '€€€€':
		r = 4
	return r