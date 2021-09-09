import requests
data={'age': 39,
    'workclass': 'State-gov',
    'fnlgt' : 77516,
    'education' : 'Bachelors',
    'education_num' :13,
    'marital_status': 'Never-married',
    'occupation' : 'Adm-clerical',
    'relationship':'Not-in-family',
    'race':'White',
    'sex':'Male',
    'capital_gain':2147,
    'capital_loss':0,
    'hour_per_week':40,
    'native_country':'United-States'}
r = requests.post('https://fast-river-63740.herokuapp.com/infer',  json=data)


print(r.status_code)
print(r.json())