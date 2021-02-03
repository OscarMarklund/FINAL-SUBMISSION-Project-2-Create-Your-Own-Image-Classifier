# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


USER INPUT GUIDE:

running python train.py passes user input into train_function.py which resolves the rest of the code.
running python predict.py passes user input into predict_function.py which resolves the rest of the code.

whilst the code can run without any user alterations (if working with a replica default workspace I was provided with.) it is suggested to at least use the GPU as it is set by default to CPU.


Significant referencing from these sources were used:
- The most thorough explanation of the project and brilliant guidelines for an incompetent like me - https://knowledge.udacity.com/questions/325025
- For the layout of argparsing in the train.py and predict.py code - https://www.youtube.com/watch?v=q94B9n_2nf0
- For revealing I did not need a 'model = blahblah' clause in my load_checkpoint function which gave me SO MUCH GRIEF - https://github.com/yzahr/AIPND-- -YZ/blob/7598f3ed0c02aae83c11a03482d355998b3ab8c2/netfunc.py#L127



Changelog:
- train.py line 9, 29 and 41 introduce the d_dir or data_directory argument
- train.py lines 22-26 introduce the quit() response to unsuitable model architecture requests from user
- train_function.py line 5 and 24 changed to accommodate the new argument d_dir
- train_function.py lines 38-42 altered so that validation images are not skewed etc. but are presented exactly like the test iamges.
- Image Classifier Project.html altered exactly the same way as the above alteration.
