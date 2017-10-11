Copyright (c) 2017 Kamran Kowsari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this dataset and associated documentation files (the "Dataset"), to deal
in the dataset without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Dataset, and to permit persons to whom the dataset is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Dataset.

If you use this dataset please cite:
Referenced paper: HDLTex: Hierarchical Deep Learning for Text Classification


Description of Dataset: 

Here is three datasets which include WOS-11967 , WOS-46985, and WOS-5736
Each folder contains:
-X.txt 
-Y.txt
-YL1.txt
-YL2.txt

X is input data that include text sequences 
Y is target value 
YL1 is target value of level one (parent label)
YL2 is target value of level one (child label)

Meta-data:
This folder contain on data file as following attribute:
Y1	Y2	Y	Domain	area	keywords	Abstract

Abstract is input data that include text sequences of  46,985 published paper
Y is target value 
YL1 is target value of level one (parent label)
YL2 is target value of level one (child label)
Domain is majaor domain which include 7 labales: {Computer  Science,Electrical  Engineering,  Psychology,  Mechanical  Engineering,Civil  Engineering,  Medical  Science,  biochemistry}
area is subdomain or area of the paper such as CS-> computer graphics which contain 134 labels.
keywords : is authors keyword of the papers




Web of Science Dataset WOS-11967
-This dataset contains 11,967 documents with 35 categories which include 7 parents categories.


Web of Science Dataset WOS-46985
-This dataset contains 46,985 documents with 134 categories which include 7 parents categories.


Web of Science Dataset WOS-5736
-This dataset contains 5,736 documents with 11 categories which include 3 parents categories.





Referenced paper: HDLTex: Hierarchical Deep Learning for Text Classification

Bib:

@inproceedings{kowsari2017HDLTex,
  title={HDLTex: Hierarchical Deep Learning for Text Classification},
  author={Kowsari, Kamran and Brown, Donald E and Heidarysafa, Mojtaba and Jafari Meimandi, Kiana and and Gerber, Matthew S and Barnes, Laura E},
  booktitle={Machine Learning and Applications (ICMLA), 2017 16th IEEE International Conference on},
  year={2017},
  organization={IEEE}
}
