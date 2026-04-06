# data.py

points = {
    'Hospital_Hub': (0, 0),
    'Regional_Depot': (10, 10),
    'Clinic_A': (2, 4), 'Clinic_B': (4, 6), 'Clinic_C': (6, 5),
    'Clinic_D': (7, 2), 'Pharmacy_E': (5, 1), 'Lab_F': (3, 0),
    'Clinic_G': (1, 2), 'Pharmacy_H': (8, 4)
}

weights = {
    'Clinic_A': 4, 'Clinic_B': 3, 'Clinic_C': 2.5, 'Clinic_D': 2,
    'Pharmacy_E': 1.5, 'Lab_F': 1, 'Clinic_G': 0.8, 'Pharmacy_H': 0.5
}

nodes = list(weights.keys())
depots = ['Hospital_Hub', 'Regional_Depot']