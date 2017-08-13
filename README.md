*Topics*

* Computer Vision
    * Object Detection & Recognition
        * Inspection & Quality Control
        * IoT
        * Medical Image Processing & Diagnosis
        * Air Surveillance
    * Motion Detection & Analysis
        * Visual Surveillance
        * Autonomous Vehicles / Drones
        * Missle Guidance
    * Scene Reconstruction
    * Computer Recognition
        * Image / Video Captioning        
        * Image Classification
        * 2D and 3D Restoration
        * OCR
    * Super Resolution
    * Image / Video Generation
    * Virtual Reality
* Text / Voice Processing
    * Semantic Segmentation
    * Natural Language Processing (VLP)
    * Voice User Interface (VUI)
    * Smart Personal Assistants
    * Computer Generated Reports (NLG)
    * ChatBots
    * Music / Script Generation
* Clustering & Anomaly Detection
    * Market Basket Analysis
    * Cohort Segmentation
* Prediction and Classification
    * Cardiovascular Events / Diseases / Cancer Detection & Prediction
    * Smart Email Categorization
    * Stock Market Predictions and Algorithmic Trading
    * Trends Prediction: Crime / Suicide / Crop Yields / Revenue
* Recommender Systems
    * Netflix / Amazon / Spotify
    * News Feeds
    * Robo-Advisors 
    * Portfolio Rebalancing

    
*Applying Deep Learning* (Andrew Ng)   
* Trends in DL: 
    * Scale: better performance with large NN with scaled data and computation
    * End-to-end DL: output is not only numbers
        * before: audio -> phonenes -> transcript
            * now: audio -> transcript
        * before: xray hand -> bone lengths -> age
            * now: xray hand -> age (but: lack of samples)
        * image -> caption
        * english -> french
        * parameters -> image
* Bias-Variance-Tradeoff
    * Performance measurements: 
        * optimal error rate (a.k.a Bayes rate, theoretical optimum caused by noice)
        * human-level (estimate of the optimal error rate, e.g. team of expert doctors)
            * while worse than humans, have good ways to progress:
                * labels from humans
                * error analysis
                * estimate bias/variance effects
        * training set
        * training-dev (same distribution as training)
        * dev (a.k.a evaluation)
        * test (same distribution as dev)
    * Problems:
        * High bias: big gap between human-level and training set errors
        * High variance (overfitting): big gap between training and training-dev errors
        * Training-Test mismatch: big gap between training and dev
        * Overfitting of dev: big gap between dev and test
        * Both high bias and high variance
    * Solutions:
        * Training error high
            * bigger model
            * training longer
            * new model architecture
        * Train-Dev error high:
            * more data
            * regularization
            * new model architecture
        * Dev error high:
            * more data similar to test
            * data synthesis
            * new model architecture
        * Test error high:
            * get more dev data
        * DL-specific (weaker bias-variance-coupling):
            * bigger model: high-performance hardware teams
            * more data: 
                * automatic data synthesise, e.g. OCR, speech recognition, NLP, Video Games (RL)
                * unified data warehouse
        * super-human performance: pick subsets where humans are better
* Model Buckets
    * General DL
        * Fully-Connected Densely-Connected Layers
    * Sequence Models
        * 1D-Sequnces: RNN, LSTM, GRU
    * Image Models
        * 2D/3D: CNN
    * Other:
        * Unsupervised Learning, RL
* What can AI do?
    * anything that a typical person can do in less than 1 second
    * predicting outcome of the next in sequence of events (big data)
* AI PM:
    * Shared Vision (Venn-Diagram)
        * Traditional:
            * PM: what users love
            * Engineers: what's feasible
        * AI:
            * PM: collecting traing and test data
            * Engineers:
            * together: provide samples
* Virtuous circle of AI
    * good Product -> more Users -> more Data -> better Product
* Career path in ML
    * ML course
    * DL school
    * PhD student process:
        * read papers (>20)
        * replicate results
    * Dirty work


    
*Efficient BackProp* (Yann LeCun)
* Stochastic versus Batch learning
    * Stochastic: faster, often better results, can be used for tracking changes
    * Batch Learning: understandable conditions of convergence, many acceleration techniques only operate in batch learning, simpler theoretical analysis of weight dynamics and convergence rates
* Shuffling
    * choose examples with maximum information
    * shuffle training set so that successive examples never belong to the same class
    * present examples that produce a large error more frequently
* Normalizing
    * average of each input variable should be close to zero
    * their covariances should about the same
    * they should be uncorrelated
* Sigmoid
    * symmetric sigmoids such as hyperbolic tangent (tanh) often converge faster than the standard logistic function
    * since tanh is sometimes computationally expensive, an approximation of it by a ratio of polynomials can be used instead
    * sometimes add a small linear term, e.g. f(x)=tanh(x)+ax can avoid flat spots
* Targets
    * choose target values at the point of the maximum second derivative on the sigmoid so as to avoid saturating the output units
* Initializing Weights
    * Assumptions: training set normalized and sigmoid with tanh is used
    * Then weights should be randomly drawn from a distribution with mean zero and standard deviation std(W)=m^-.5 where m is the fan-in (the number of connections feeding into the node)
* Learning Rate
    * give each weight its own learning rate
    * proportional to the square root of the number of inputs to the unit
    * weights in lower layers should typically be larger than in the higher layers
* Radial Basis Functions vs Sigmoid Units
    * In RBF, the dot product of the weight and input vector is replaced with a Euclidean distance between the input and wieght and the sigmoid is replaced by an exponential.
    * A single RBF covers only a small local region of the input space:    
        * learning can be faster
        * disadvantage in high dimensional spaces because the RBF units are needed to cover the spaces
    * RBF more appropriate in low dimensional upper layers
    * Sigmoids more approprite in high dimensional lower loayers
    
  
    
    
*[Neural Networks Zoo](http://www.asimovinstitute.org/neural-network-zoo/)*
* Feed forward neural networks (FF or FFNN) and perceptrons (P)
* Radial basis function (RBF)
* Hopfield network (HN)
* Markov chains (MC or discrete time Markov Chain, DTMC)
* Boltzmann machines (BM)
* Restricted Boltzmann machines (RBM)
* Autoencoders (AE)
* Sparse autoencoders (SAE)
* Variational autoencoders (VAE)
* Denoising autoencoders (DAE) 
* Deep belief networks (DBN)
* Convolutional neural networks (CNN or deep convolutional neural networks, DCNN)
* Deconvolutional networks (DN)
* Deep convolutional inverse graphics networks (DCIGN)
* Generative adversarial networks (GAN)Recurrent neural networks (RNN)Long / short term memory (LSTM)
* Gated recurrent units (GRU)
* Neural Turing machines (NTM)
* Bidirectional recurrent neural networks, bidirectional long / short term memory networks and bidirectional gated recurrent units (BiRNN, BiLSTM and BiGRU respectively)
* Deep residual networks (DRN)
* Echo state networks (ESN)
* Extreme learning machines (ELM)
* Liquid state machines (LSM)
* Support vector machines (SVM)
* Kohonen networks (KN, also self organising (feature) map, SOM, SOFM)

<table>
    <tr>
        <td><img width='160' height='240' src='./cheatsheets/neuralnetworks.png' alt='neural networks zoo'/></td>
    </tr>
</table>


*Note*
* [Deep Learning Glossary](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md) (for Korean)
: This documentation is translated from blog post on wildml.com (with author's permission)


*Dependencies*
* [Keras](https://github.com/fchollet/keras)
* [TensorFlow](https://github.com/tensorflow/tensorflow)
