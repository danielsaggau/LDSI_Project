# LDSI_Project

This is my project on language models for legal text generation.

In the 'data.py' file I sample from the courtlistener bulk file. 
In the 'cleaning.py' file i preprocess the data. 
Processing also includes tokenization and creating suitable sequences for our models.
Due to the quadratic scaling of our transformer based langauge models we need to sample for the raw text files to feed the model a sequences, providing the model a many to one problem.
Many in this case refers to the many words leading to the word it is supposed to predict.
'generation.py' includes our fine tuning on the legal data and further also provides a performance benchmark.




