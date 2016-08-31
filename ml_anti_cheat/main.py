import csv as csv
import numpy as np
import os
import pprint
import pdb
# Set break point
# pdb.set_trace()


# Open up the csv file in to a Python Object
pp = pprint.PrettyPrinter(indent=4)
csv_file_object = csv.reader(open('data/train.csv', 'rb'))

# The 'next()' simply skips the first line (we skip it since its a header)
header = csv_file_object.next()

data=[]
for row in csv_file_object:
	data.append(row)

# Convert data to an array.
data = np.array(data)

# Pretty print data
pp.pprint(data)

# The size() function counts how many elements are in
# in the array and sum() (as you would expects) sums up
# the elements in the array.

#data[0::, N] - 
#		the 0:: represents all data in the array.
#		the N   represents which column to be displayed
number_passengers = np.size(data[0::, 1].astype(np.float))

number_survived = np.sum(data[0::, 1].astype(np.int))

woman_only_stats = data[0::,4] == 'female'
men_only_stats   = data[0::,4] != 'female'
pdb.set_trace()
# Selecting woman only, The '1' in the array is the column
# representing a 1 or a 0 depending on the survival(1) or not(0).
woman_onboard = data[woman_only_stats, 1].astype(np.float)
men_onboard   = data[men_only_stats, 1].astype(np.float)

# Whats the proportion of woman/men who survived
proportion_women_survived = np.sum(woman_onboard)/np.size(woman_only_stats)

proportion_men_survived = np.sum(men_onboard)/np.size(men_only_stats)

print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived

test_file = open('data/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

prediction_file = open("data/genderbasedmodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
test_file.close()
prediction_file.close()

pdb.set_trace()


