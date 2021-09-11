import requests
data={'age': 43,
    'workclass': 'Private',
    'fnlgt' : 292175,
    'education' : 'Doctorate',
    'education-num' :16,
    'marital-status': 'Married-civ-spouse',
    'occupation' : 'Prof-specialty',
    'relationship':'Husband',
    'race':'White',
    'sex':'Male',
    'capital-gain':0,
    'capital-loss':0,
    'hours-per-week':45,
    'native-country':'United-States'}
r = requests.post('https://fast-river-63740.herokuapp.com/infer',  json=data)


print(r.status_code)
print(r.json())