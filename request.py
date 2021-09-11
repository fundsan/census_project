import requests
data={'age': 35,
    'workclass': 'Private',
    'fnlgt' : 77516,
    'education' : 'Bachelors',
    'education-num' :13,
    'marital-status': 'Never-married',
    'occupation' : 'Other-service',
    'relationship':'Not-in-family',
    'race':'White',
    'sex':'Female',
    'capital-gain':0,
    'capital-loss':0,
    'hours-per-week':40,
    'native-country':'United-States'}
r = requests.post('https://fast-river-63740.herokuapp.com/infer',  json=data)


print(r.status_code)
print(r.json())