<p align="center">
  <img src="https://user-images.githubusercontent.com/112116530/232979540-e28f53e4-4e4f-48ea-bbb9-1d3a2d1d4a63.png" alt= “” width="300" height="300">
</p>

_We propose worldNET a training-efficient location predictor for sparse geographic data by continuous latitude and longitude coordinates. By taking a continuous approach, we can derive meaningful nonlinear interpolations of feature data and are able to create a more explainable mapping of global geographic features. We show that worldNET can deliver comparable performance (~55\% continent accuracy) for seen locations and triple guessing performance on unseen locations (~21\% continent accuracy)._

### Introduction

For nearly two decades, a popular topic of interest for computer vision researchers has been the prediction of geographic location from image data. Recent developments in Deep Convolutional Neural Networks (DCNNs) have accelerated this development, with state-of-the-art models showing upwards of 70\% accuracy in general location prediction. All of these attempts involve a straightforward process—create a feature extraction model, and then classify overall location based on those features and a discretization of the world map. Cutting edge models also require massive amounts of training data (~100M training samples) in order to accurately make these predictions. 

We propose a model that can infer continuous GPS location via image feature recognition. Specifically, we compare two conventional approaches similar to previous works (nearest neighbors, simple VGG transfer learning) and another approach that includes an interpolation head paired with a stored memory of training coordinates. We examine the effect of scarcity of training data and sparsity in the location distribution on the efficacy of each of these models. Our dataset consists of a small sample size (2300) of only a few truly unique locations (23). A model with this sample efficiency in theory better understands the data, and in larger-scale applications may be more successful than the current state-of-the-art.

### Related Work

There are two dominant classes of approaches that are taken to solve this problem. Before the popularization of Deep Convolutional Neural Networks (DCNN), nearest-neighbors classifiers were the optimal approach (Hays \& Efros, 2008). These compare vector similarity between images from some feature extraction process (tiny image, color histograms, HOG features) and use the k-nearest features to compute some weighted final average of image geoposition. 

Recently, DCNNs have dominated these purely algorithmic approaches. Most notably, Google's PlaNet improved accuracy at the country, region, and street level by more than twofold, setting the stage for several other improvements within this new genre of image feature interpretation (Weyand et al., 2017). This model uses the Inception architecture with an attached classification head of a cell discretization of the world map. However, in order to achieve this level of accuracy (71.3\% correct continent), their model was trained on 91M images from an even distribution of Google StreetView images across the globe.

### Methods

##### Conventional Approaches

We use data from the GSV-Cities dataset, which includes over 560,000 geotagged images from 23 cities. We normalize each coordinate to 0-1 and resize each image to 300 x 400 resolution. We de-bias the data by algorithmically reducing the eurocentric concentration  of training data.

We compute the Mean Haversine Loss to measure absolute distances between geographic coordinates, compensating for distance distortion near the poles.

Our initial approach similar to a K-nearest neighbors (KNN) algorithm for feature clustering, where the K-closest extracted feature descriptors of the training images were used to calculate the weighted mean of the longitude and latitude coordinates. The implementation utilizes feature vectors already extracted from the VGG model, although the other considered alternative was utilizing HOG feature descriptors by building a bag of words for each training image with a vocabulary of defined features. Each training image produced 512 features with each descriptor being 108 dimensions. These (512, 108) features were flattened into one long feature descriptor, and the distances between each feature were calculated to determine the K-nearest for each training image. Utilizing a mean shift clustering algorithm, each training image had its own mean shift model trained with the K-nearest feature descriptors extracted from the training images, and with the model’s calculated clusters, the final prediction is the "dominant" cluster with the highest number of vectors.  

To demonstrate the viability of existing state-of-the-art methods for continuous prediction, we implement a VGG-based direct feed forward network from image to GPS. For this implementation, we use a frozen pertained VGG19 model to process the image into features. These features are then flattened and passed into a trainable regression head that consists of 2 fully-connected layers each with 512 units. Then the output is passed into a final fully-connected layer with 2 output units. The activation function for every fully-connected layer is ReLU and the activation function for the output layer is sigmoid. For the optimizer, we chose to use the Adam optimizer with a learning rate of 0.01.

We had several initial attempts to improve on the modified continuous VGG network. Because this model was never confident enough to deviate from the mean (see Results). We attempted modifications of our loss function, including implementing a continuous version of categorical crossentropy (even penalties far away from the local distribution associated with the guess mean. We also attempted penalties for low-variance batch guesses. We modified the architecture to train instead a "feature location prediction head" and clustering/predicting via coordinate outputs from those clusters. We also tried a variety of feature detection architectures, including Google's Inception architecture which shows cutting-edge success in these tasks for classification-based approaches (Weyand et al., 2017). None of these attempts could meaningfully remedy the convergence of variance in our naive DCNN approach without severely sacrificing average distance and/or location accuracy.

##### worldNET

Our final model to produce high-confidence continuous predictions on sparse data we call worldNET. Our model first clusters locations in the training data by some algorithm (we have only city data, so rely on those as our clusters). With the centroids of those clusters as pseudolabels for the training image dataset, we fit a VGG-based classifier model to predict a probability distribution over those centroids (categorical-crossentropy loss). Importantly, while this probability distribution should represent a likelihood of feature mapping established by our model, it isn't trivial to turn this into a continuous prediction. 

We settle on the idea of an "interpolation head." After achieving satisfactory categorical accuracy ($\sim$70\% on validation data) we freeze our model's weights and append this interpolation head on top of the distribution generated by the classifier. Our goal for this additional structure is to use the latent probability distribution over known clusters (cities, in our case) to be able to infer some interpolation between each of those cluster centroids as a function of their relative feature match. The interpolation head is theoretically sophisticated enough to capture some of the nonlinear relationships that we see influenced by history, globalization, and the spread of ideas and movements that have determined what features exist where. The interpolation head, in our case a two layer network with ReLU activation, produces an output of the same size as the number of known centroids, can then be composed with those centroid coordinates to produce final predictions.

A phenomenon that we introduce by using continuous interpolated predictions instead of categorical is out-of-domain predictions--those that are in the ocean. We fix this by passing our final predicted outputs over a topographical mesh of global landmasses. If the prediction lies outside of this mesh, we assign it point on land from the nearest distance.

### Results

Both of our DCNN-based approaches had similar training times (nearly identical parameter size). Each were trained for $\sim$ 50 epochs over a train/test split of 1600/700 and 1800/600 images on Google's cloud compute units. Final comparative results are shown below:

```text
                  mean distance   <1,000km  
 --------------- --------------- ---------
  mean model      9659521 m       0               
  guess model     10512453 m      0.102599        
  random guess    9749176 m       0               
  VGG19           9175222 m       0.036123        
  nn classifier   5383454 m       0.23            
  worldNET        3102460 m       0.559375 
```

Figure 4a: Performance of models on unseen images of seen cities

```text
                 mean distance   <1,000km  
 -------------- --------------- --------- 
  mean model     6520583 m       0         
  guess model    6851173 m       0.07857
  random guess   11406806 m      0         
  VGG19          6300439 m       0         
  worldNET       5368128 m       0.21875  
```
  
Figure 4b: Performance of models on unseen images of unseen cities

worldNET vastly outperforms control models, most notably the naive adaptation of the current state-of-the-art feature interpretation model (Simple VGG). As we see in graphical representations of this data, the nature of the training dataset causes worldNET to categorize often cities like Montreal and Toronto as Australia, which may cause an inflated mean distance. Nonetheless, we are still able to achieve relatively high continent accuracy (triple guessing) with no knowledge of cities and only sparse understanding of surrounding regions.

Notably, a VGG19-only model, which we would see in an intuitive direct transfer approach to this problem, achieves only mean accuracy performance, as it can never be confident enough to predict away from the mean. Thus, low accuracy is achieved as well.

worldNET often almost exactly categorizes seen cities, only making a few mistakes which are also reasonable. For unseen cities, worldNET also delivers solid performance in inferring location purely relying on an interpolation between known cities. We can infer that this guesswork becomes better with a broader geographic distribution of known training cities. worldNET is often still prone to vast miscategorizations (in this case, incorrectly predicting urban Australia as Toronto, for example), but even these predictions are reasonable. We would further expect that with more data 

We show that, compared with conventional approaches, worldNET performs significantly better in both distance predictions and accuracy. We see reasonable predictions even when we predict over cities which are foreign to the training dataset, and few mistakes for known cities. Conventional approaches notably fail–they can only converge on the mean of the training data

### Technical Discussion

We show at least a proof-of-concept that there is a viable way to learn nonlinear interpolations of feature similarity in order to geolocate images. However, this requires our training dataset to be broad enough to encompass the full testing data. For instance, if we only trained on Europe, our model would have no ability to predict an image from New York--it's restricted to the domain of the training data. Further, we mandate good classification accuracy on the training data to ensure that our model properly interprets the features.

A limitation of our dataset was the variance of the location of image coordinates surrounding each city. We believe that, had there been a broader distribution of images around cities, or perhaps if we had trained the interpolation head on \textit{some} unseen cities, that our interpolation head would better learn the mapping of unseen features worldwide.

Intuitively, though, our model should be better equipped to learn spatial reasoning between features. While complex categorical approaches should be able to make similar relationships between categories in latent space, and so with enough training data learn these spatial relationships, it nonetheless cannot learn on sparse data and cannot learn these relationships efficiently due to the classifying head architecture. Our model, on a larger scale, could contend with these models and perhaps serve better potential for application in the real world where perhaps only sparse data is available.

### SRC

Doxxing poses a clear risk to society and people’s safety, and this is definitely an issue that comes up should the accuracy of worldNET continues to increase. Luckily, worldNET's accuracy is not very good, especially compared to other cutting-edge models and especially on small street-scale geolocating. Our predictions are pretty general. Regardless, any information of even an overall location or city may be beneficial to a stalker, making them a malicious stakeholder in this proposal.

We bring up potential positive beneficiaries of worldNET in our proposal, but we concede that there are malicious stakeholders including stalkers, foreign militaries, and other government organizations who could potentially misuse our model to find the geolocation using images. We acknowledge this possibility, and one potential argument that may soften the blow from these stakeholders is that our dataset comes from images available online, which most people and organizations already have access to, and our dataset contains few images from areas where images are rarely taken or restricted such as in middle Eastern countries. Thus, foreign and military organizations will most likely not make good use of the model if the images that we are using to train worldNET come from generally available sources. Still, if we are successfully able to interpolate with enough accuracy, our model could in theory generate locations that are unseen in publicly available data, though this is unlikely.

Another concern was regarding the dataset having person-identifying information within the images themselves. However, our data (GSV-Cities) contain no traces of any individual’s private data since the images do not contain any people within the dataset. Additionally, consumers who utilize worldNET would be working with the trained model, which means they are unable to look at the training dataset and only have access to the model weights, which most likely remains a black box to most consumers. 

Finally, there is a concern of sample bias in the CMU dataset we initially stated that we'd use for training worldNET, which could impact our platform’s usage and credibility. This is a valid concern that our team has already attempted to address by de-biasing the data by algorithmically suppressing the number of images from geographically dense city in order to distribute the data more evenly around the globe, as many of the images were clustered around a few select Eurocentric areas. One bias that we are unfortunately unable to account for is a lack of images in areas where photos may be restricted, such as Saudi Arabia or China, which remains more of a political rather than technical issue and is completely out of our hands as developers of the model.

### Conclusion

The VGG implementation follows the basic feed forward architecture of a DCNN is successful in a classification version of the task, however, due to a lack of confidence in spatial reasoning,  always converges to repeatedly predicting the mean of the dataset.

Our continuous model, worldNET, can interpret sparsity of geographic locations in our dataset and uses the prediction of a probability distribution of city centroids as a proxy to predict gps coordinates. These results show great promise for the spatial reasoning capacity of a worldNET-like model. It should be noted, though, that this model is dependent on a reasonable distribution of training data, since it will simply be unable to predict coordinates outside of the range that it sees in training. Nonetheless, the capability to predict areas unseen in training data and spatially relate extracted features in data as a geographic distribution is novel, and could be a cutting-edge innovation of img2GPS models. Still, our interpolation head and methodology is simple and proof-of-concept, so future studies can improve the interpolation head of the model to mitigate this. 

