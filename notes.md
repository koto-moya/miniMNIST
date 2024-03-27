# Implementing MNSIT with Numpy Only


Following "Built up Micrograd" a little to help out with implementing backprop.  I realized that the actual implementation of back propagation is not as trivial as I thought.  I looked at a old notebook that I had made following this lecture when it first came out and immediately I saw where I was getting stuck.  I needed to create some idea of an object that can store a value and it's associated gradient.  Karpathy uses the Value object to achieve this in Micrograd.     


class value wraps a single scalar value and keeps track of it.  

Big python moment the dunder methods like \__add__ and  \__mul__ need to be defined in order to do operations with you created object.



A lot of work has been done and I've neglected to tell you about all of it.  I implemented the engine for common ops and .backward() functionality.  Implemented the NN from scratch (based on Micrograd).  Ripped my Trainer from the original MNIST project.  After hacking the pieces together I got stuck.  the model wasn't training, I don't even care if the loss gets worse I just want it to move! The loss doesn't budge regardless of the cycles I run it over

after much frustration I went over to a notebook to see if I could build the model in Pytorch.  Luckily I succeeded, I think this helped rethink how the models work.  Specifically the data loader is where I want to start improving my model.  





