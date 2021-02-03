# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

USER INPUT GUIDE:

running python train.py passes user input into train_function.py which resolves the rest of the code.
running python predict.py passes user input into predict_function.py which resolves the rest of the code.

whilst the code can run without any user alterations to default values effectively, it is suggested to at least use the GPU as it is set by default to CPU.


Significant referencing from these sources were used:
- The most thorough explanation of the project and brilliant guidelines for an incompetent like me - https://knowledge.udacity.com/questions/325025
- For the layout of argparsing in the train.py and predict.py code - https://www.youtube.com/watch?v=q94B9n_2nf0
- For revealing I did not need a 'model = blahblah' clause in my load_checkpoint function which gave me SO MUCH GRIEF - https://github.com/yzahr/AIPND-- -YZ/blob/7598f3ed0c02aae83c11a03482d355998b3ab8c2/netfunc.py#L127