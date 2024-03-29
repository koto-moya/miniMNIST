# Implementing MNSIT with Numpy Only


Following "Built up Micrograd" a little to help out with implementing backprop.  I realized that the actual implementation of back propagation is not as trivial as I thought.  I looked at a old notebook that I had made following this lecture when it first came out and immediately I saw where I was getting stuck.  I needed to create some idea of an object that can store a value and it's associated gradient.  Karpathy uses the Value object to achieve this in Micrograd.     


class value wraps a single scalar value and keeps track of it.  

Big python moment the dunder methods like \__add__ and  \__mul__ need to be defined in order to do operations with you created object.



A lot of work has been done and I've neglected to tell you about all of it.  I implemented the engine for common ops and .backward() functionality.  Implemented the NN from scratch (based on Micrograd).  Ripped my Trainer from the original MNIST project.  After hacking the pieces together I got stuck.  the model wasn't training, I don't even care if the loss gets worse I just want it to move! The loss doesn't budge regardless of the cycles I run it over

after much frustration I went over to a notebook to see if I could build the model in Pytorch.  Luckily I succeeded, I think this helped rethink how the models work.  Specifically the data loader is where I want to start improving my model.

I changed the dataloader to include an iterator to try and get the traing schedule to work with batches.  This proved to be quite a headache and I still had not gotten the model to run at all.  I moved away form the batching for now and went straight to the brute force method.  I implemented a sudo batch training by loading in a shuffled batch of data each epoch which is obviously not optimal but thats no the point of this exercise.  

After falling back on the original dataloader (enhanced with a shuffler) I went forth with the training loop and was finally able to get the model to train, albeit very slowly.  I have been playing around with reducing the "batch size" which speeds up the epoch but will require more epochs to train to a respectable accuracy.  I assume with my current implementation there is a sweet spot of batch size/epochs that will train in an acceptable amount of time.  My goal is to train to ~90% accuracy in <= 2 hours.

Training on a batch size of 16, lr = 0.01 for 200 epochs produced ~20% training accuracy. It took about 20 mins projecting this out I expect this hyper param set to get to 90% 1.5 hours if it ever gets there. 

I went back to the white board and realized that my intuitions of how the NN architecture worked was incorrect.  So now I am going to play around with each level of the module to get a feeling for how it actually works.  Honestly, I've realized that I don't really know how to extend from a neuron to a layer to a model.  Like I don't even know what a neuron really is, I think I've been assuming a neuron is what would actually be considered a layer.  


I get it! A neuron is the weighted sum of the inputs!  This is obviously not groundbreaking and I knew this before technically but I couldn't intuit it. Now I can   





