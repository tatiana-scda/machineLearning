import pandas as pd
import string as st

# the axis garantees that will drop a colunm and not a row
#df.drop('reports', axis= 1) #drop by name
#df.drop(df.index[2]) # by index 
#df.drop(df.index[[2,3]])

file_path = 'chemical_carriers.csv'
filtered_path = 'chemical_carriers_filtered.csv'

with open(file_path, 'r') as file:
    lines = file.readlines()
    
    header_length = len(lines[0].split(','))
    
    output_file = open(filtered_path, 'w')
    for line in lines:
        if len(line.split(',')) == header_length:
            output_file.write(line)
    output_file.close()

print(header_length)

results = pd.read_csv(filtered_path, low_memory = False)

header_list = list(results)
print('Header list length {}'.format(len(header_list)))

keep_list = ['test_id', 'cas_number', 'chem_name', 'carrier_id']

# chemical_carrier keep_list = ['carrier_id', 'test_id', 'chem_name']

# test keep_list = ['test_id', 'reference_number', 'test_characteristics', 'species_number', 'test_characteristics', 'test_formulation']

drop_list = [identifier for identifier in header_list if identifier not in keep_list]

results = results.drop(drop_list, axis=1)

results.to_csv("final_chemical_carriers.csv", sep=',')

#print(results)
