# Deep-frequency-principle-towards-understanding-why-deeper-learning-is-faster

This is a repository corresponding to the paper 'Deep frequency principle towards understanding why deeper learning is faster'. The repository mainly introduces the experiments carried out in the paper concretely.

Understanding the effect of depth in deep learning is a critical problem. In this work, we utilize the Fourier analysisto empirically provide a promising mechanism to understandwhy deeper learning is faster. To this end, we separate a deep neural network into two parts in the analysis, one is a pre-condition component and the other is a learning component, in which the output of the pre-condition one is the input of the learning one. Based on experiments of deep networksand real dataset, we propose a deep frequency principle, that is, the effective target function for a deeper hidden layer biases towards lower frequency during the training. Therefore,the learning component effectively learns a lower frequency function if the pre-condition component has more layers. Due to the well-studied frequency principle, i.e., deep neural net-works learn lower frequency functions faster, the deep fre-quency principle provides a reasonable explanation to why deeper learning is faster. We believe these empirical studies would be valuable for future theoretical studies of the effect of depth in deep learning.

In this repository, two experiments are mainly discussed. First one is the variants of Resnet18 on CIFAR10, and the second one is fully-connected network on MNIST. The training details for experiments are as follows.

For the experiments of the variants of Resnet18 on CIFAR10, the network structures are shown in Fig. 4. The output layer is equipped with softmax and the network is trained by Adam optimizer with cross-entropy loss and batch size 256. The learning rate is changed as the training proceeds, that is, 10−3 for epoch 1-40 , 10−4 for epoch 41-60, and 10−5 for epoch 61-80. We use 40000 samples of CIFAR10 as training set and 10000 examples as the validation set. The training accuracy and the validation accuracy are shown in Fig. 5. The RDF of the effective target function of the last hidden layer for each variant is shown in Fig. 6. For experiment of fully-connected network on MNIST, we choose the activation function of tanh and size 784 − 500 − 500 − 500 − 500 − 500 − 10. The output layer of the network does not equip any activation function. The network is trained by Adam optimizer with mean squared loss, batch size 256 and learning rate 10−5. The training is stopped when the loss is smaller than 10−2. We use 30000 samples of the MNIST as training set. The RDF of the effective target functions of different hidden layers are shown in Fig. 8. Note that ranges of different dimensions in the input are different, which would result in that for a same δ, different dimensions keeps different frequency ranges when convolving with the Gaussian function. Therefore, we normalized each dimension by its maximum amplitude, thus, each dimension lies in [−1, 1]. Without doing such normalization, one can also obtain similar results of deep frequency principle.
