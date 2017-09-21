kid_friendly: 0 - n/a, 1 - no, 2 - yes
accepts_cc: 0 - n/a, 1 - no, 2 - yes
parking (sum): 1 - garage, 2 - onsite, 4 - private, 8 - valet, 16 - street, 32 - validated
attire: 0 - n/a, 1 - casual, 2 - dressy, 3 - formal
group_friendly: 0 - n/a, 1 - no, 2 - yes
rest_price: 0 - n/a, 1 - $, 2 - $$, 3 - $$$, 4 - $$$$
rest_wifi: 0 - n/a, 1 - free, 2 - no, 3 - paid
rest_meal_type (sum): 1 - breakfast, 2 - brunch, 4 - lunch, 8 - dinner, 16 - dessert, 32 - late night
rest_alcohol: 0 - n/a, 1 - beer & wine, 2 - full bar, 3 - no
rest_noise_level: 0 - n/a, 1 - average, 2 - loud, 3 - quiet, 4 - very loud
rest_ambience (sum): 1 - casual, 2 - intimate, 4 - classy, 8 - touristy, 16 - trendy, 32 - upscale, 64 - upmarket, 128 - romantic
rest_has_tv: 0 - n/a, 1 - no, 2 - yes
rest_caters: 0 - n/a, 1 - no, 2 - yes
rest_wheelchair_friendly: 0 - n/a, 1 - no, 2 - yes

def flagged(x):
	r = 0
	if x == 'N' or x == 'NR':
		r = 0
	elif x == 'Y' or x == 'YR':
		r = 1
	return r

def noise(x):
	r = 0
	if x == 'Average':
		r = 1
	elif x == 'Loud':
		r = 2
	elif x == 'Quiet':
		r = 3
	elif x == 'Very Loud':
		r = 4
	return r

def alcohol(x):
	r = 0
	if x == 'Beer & Wine Only':
		r = 1
	elif x == 'Full Bar':
		r = 2
	elif x == 'No':
		r = 3
	return r

def wifi(x):
	r = 0
	if x == 'Free':
		r = 1
	elif x == 'No':
		r = 2
	elif x == 'Paid':
		r = 3
	return r

def attire(x):
	r = 0
	if x == 'Casual':
		r = 1
	elif x == 'Dressy':
		r = 2
	elif x == 'Formal (Jacket Required)':
		r = 3
	return r

def cc(x):
	r = 0
	if x == 'No':
		r = 1
	elif x == 'Yes':
		r = 2
	return r

def park(x):
	r = 0
	x = str(x)
	if 'Garage' in x:
		r += 1
	if 'On-Site' in x:
		r += 2
	if 'Private Lot' in x:
		r += 4
	if 'Valet' in x:
		r += 8
	if 'Street' in x:
		r += 16
	if 'Validated' in x:
		r += 32
	return r

def gf(x):
	r = 0
	x = str(x)
	if 'Breakfast' in x:
		r += 1
	if 'Brunch' in x:
		r += 2
	if 'Lunch' in x:
		r += 4
	if 'Dinner' in x:
		r += 8
	if 'Dessert' in x:
		r += 16
	if 'Late Night' in x:
		r += 32
	return r

def ambience(x):
	r = 0
	x = str(x)
	if 'Casual' in x:
		r += 1
	if 'Intimate' in x:
		r += 2
	if 'Classy' in x:
		r += 4
	if 'Touristy' in x:
		r += 8
	if 'Trendy' in x:
		r += 16
	if 'Upscale' in x:
		r += 32
	if 'Upmarket' in x:
		r += 64
	if 'Romantic' in x:
		r += 128
	return r