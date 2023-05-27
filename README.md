# TermProject
DataScience Term Project

# Our DataSet
https://archive.ics.uci.edu/ml/datasets/adult
Data Set Characteristics:  	Multivariate	Number of Instances:	48842	Area:	Social
Attribute Characteristics:	Categorical, Integer	Number of Attributes:	14	Date Donated	1996-05-01
Associated Tasks:	Classification	Missing Values?	Yes	Number of Web Hits:	2777169
To determine whether a person makes over 50K a year.

# Demonstrate the description of the project
The goal of this project is to predict whether an individual's income exceeds $50,000 per year based on various demographic and employment attributes using machine learning techniques. The project will involve preprocessing, data scaling and encoding, and the application of classification, regression, and clustering algorithms. The team will follow the end-to-end Big Data process (except data curation and deployment) and use k-fold cross-validation for testing classification models.

# Statistical description of the dataset.
The Adult Income dataset contains 48,842 instances and 14 attributes, of which 6 are numerical and 8 are categorical. The dataset is derived from the 1994 US Census Bureau database. The attributes in the dataset are:
```
age: continuous (numerical)
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked (categorical)
fnlwgt: continuous (numerical). It represents the number of people the census believes the entry represents (sampling weight).
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool (categorical)
education-num: continuous (numerical). It represents the number of years of education in total.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse (categorical)
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces (categorical)
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried (categorical)
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black (categorical)
sex: Female, Male (categorical)
capital-gain: continuous (numerical)
capital-loss: continuous (numerical)
hours-per-week: continuous (numerical)
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands (categorical)
The target variable is 'income', which is binary: '>50K' and '<=50K'. The aim is to predict this variable based on the attributes provided.
```
